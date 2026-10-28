"""Deterministic answer-group features for training and offline inference."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import math
import re
from typing import Iterable

import numpy as np

from .answer_groups import AnswerGroup


FEATURE_SCHEMA_VERSION = "answer-group-features-v1"
FEATURE_SETS = (
    "consensus",
    "text",
    "embedding",
    "full_no_context",
    "full",
)

_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
_WORD_RE = re.compile(r"[a-z]+")
_UNITS = frozenset(
    {
        "cent", "cents", "dollar", "dollars", "day", "days", "foot", "feet",
        "gram", "grams", "hour", "hours", "inch", "inches", "kilogram",
        "kilograms", "kilometer", "kilometers", "litre", "litres", "liter",
        "liters", "meter", "meters", "mile", "miles", "minute", "minutes",
        "month", "months", "ounce", "ounces", "percent", "percentage", "pound",
        "pounds", "second", "seconds", "week", "weeks", "yard", "yards", "year",
        "years",
    }
)
_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
        "have", "he", "her", "his", "how", "in", "is", "it", "many", "much",
        "of", "on", "or", "she", "that", "the", "their", "there", "they", "to",
        "was", "were", "what", "when", "which", "will", "with",
    }
)

TRACE_FEATURE_NAMES = (
    "log_chars",
    "log_words",
    "log_lines",
    "log_numbers",
    "log_unique_numbers",
    "log_equals",
    "log_operators",
    "log_steps",
    "has_boxed",
    "has_final_answer_phrase",
    "has_answer_phrase",
    "has_percent",
    "has_currency",
    "has_fraction",
    "question_number_coverage",
    "extra_number_ratio",
    "lexical_jaccard",
    "unit_coverage",
    "answer_is_integer",
    "answer_is_negative",
    "answer_is_zero",
    "answer_log_abs",
    "answer_log_denominator",
    "answer_appears_in_question",
    "answer_to_question_max_log_ratio",
    "reasoning_marker_rate",
    "correction_marker_rate",
)

CONSENSUS_FEATURE_NAMES = (
    "support_fraction",
    "support_margin_fraction",
    "is_unique_plurality",
    "is_singleton",
    "distinct_answer_fraction",
    "unparseable_fraction",
)

EMBEDDING_SCALAR_NAMES = (
    "embedding_group_cohesion",
    "embedding_group_to_all_cosine",
    "embedding_group_delta_norm",
    "embedding_group_mean_norm",
)

PARSER_FEATURE_NAMES = (
    "has_hash_marker",
    "hash_marker_starts_numeric",
    "hash_marker_looks_like_heading",
    "has_boxed_answer",
    "boxed_answer_matches_official",
    "boxed_answer_mismatches_official",
    "has_final_answer_segment",
    "final_answer_matches_official",
    "final_answer_mismatches_official",
    "explicit_answer_matches_official",
    "explicit_answer_mismatches_official",
)

CANDIDATE_FEATURE_DIM = 1072


def _safe_fraction(token: str) -> Fraction | None:
    try:
        return Fraction(token.replace(",", ""))
    except (ValueError, ZeroDivisionError):
        return None


def _numeric_tokens(text: str) -> list[Fraction]:
    values: list[Fraction] = []
    for token in _NUMBER_RE.findall(text):
        value = _safe_fraction(token)
        if value is not None:
            values.append(value)
    return values


def _content_words(text: str) -> set[str]:
    return {word for word in _WORD_RE.findall(text.lower()) if word not in _STOPWORDS}


def _safe_cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(left, right) / denominator)


def trace_features(question: str, text: str, parsed_answer: str | None) -> np.ndarray:
    """Extract compact, model-free trace and question-consistency features."""

    lower = text.lower()
    words = _WORD_RE.findall(lower)
    question_words = _content_words(question)
    trace_words = _content_words(text)
    question_numbers = _numeric_tokens(question)
    trace_numbers = _numeric_tokens(text)
    question_number_set = set(question_numbers)
    trace_number_set = set(trace_numbers)

    if question_number_set:
        number_coverage = len(question_number_set & trace_number_set) / len(question_number_set)
    else:
        number_coverage = 1.0
    extra_number_ratio = len(trace_number_set - question_number_set) / max(1, len(trace_number_set))

    union_words = question_words | trace_words
    lexical_jaccard = len(question_words & trace_words) / max(1, len(union_words))
    question_units = question_words & _UNITS
    trace_units = trace_words & _UNITS
    unit_coverage = (
        len(question_units & trace_units) / len(question_units) if question_units else 1.0
    )

    answer = _safe_fraction(parsed_answer) if parsed_answer is not None else None
    answer_is_integer = float(answer is not None and answer.denominator == 1)
    answer_is_negative = float(answer is not None and answer < 0)
    answer_is_zero = float(answer == 0) if answer is not None else 0.0
    answer_log_abs = math.log1p(abs(float(answer))) if answer is not None else 0.0
    answer_log_denominator = math.log1p(answer.denominator) if answer is not None else 0.0
    answer_appears = float(answer is not None and answer in question_number_set)
    max_question = max((abs(float(value)) for value in question_numbers), default=1.0)
    answer_ratio = (
        math.log1p(abs(float(answer)) / max(max_question, 1e-12)) if answer is not None else 0.0
    )

    line_count = max(1, len(text.splitlines()))
    step_count = len(re.findall(r"(?:^|\n)\s*(?:[-*]|\d+[.)])\s+", text))
    operator_count = len(re.findall(r"[+*/=]|(?<!\w)-(?!\w)", text))
    reasoning_markers = len(
        re.findall(r"\b(?:because|therefore|thus|so|hence|then|first|next|finally)\b", lower)
    )
    correction_markers = len(
        re.findall(r"\b(?:actually|instead|correction|reconsider|wait|however)\b", lower)
    )

    values = np.array(
        [
            math.log1p(len(text)),
            math.log1p(len(words)),
            math.log1p(line_count),
            math.log1p(len(trace_numbers)),
            math.log1p(len(trace_number_set)),
            math.log1p(text.count("=")),
            math.log1p(operator_count),
            math.log1p(step_count),
            float("\\boxed{" in text),
            float("final answer" in lower),
            float("answer" in lower),
            float("%" in text or "percent" in lower),
            float("$" in text or "dollar" in lower),
            float(bool(re.search(r"\d+\s*/\s*\d+", text))),
            number_coverage,
            extra_number_ratio,
            lexical_jaccard,
            unit_coverage,
            answer_is_integer,
            answer_is_negative,
            answer_is_zero,
            answer_log_abs,
            answer_log_denominator,
            answer_appears,
            answer_ratio,
            reasoning_markers / max(1, len(words)),
            correction_markers / max(1, len(words)),
        ],
        dtype=np.float32,
    )
    if values.shape != (len(TRACE_FEATURE_NAMES),) or not np.isfinite(values).all():
        raise ValueError("Trace feature extraction produced an invalid vector")
    return values


def _canonical_number(text: str) -> str | None:
    match = _NUMBER_RE.search(text)
    if match is None:
        return None
    value = _safe_fraction(match.group())
    return str(value) if value is not None else None


def parser_consistency_features(text: str, parsed_answer: str | None) -> np.ndarray:
    """Describe whether explicit answer cues agree with the fixed scorer parser.

    The benchmark parser treats every ``####`` occurrence as an answer marker,
    including Markdown headings.  These features do not replace that parser;
    they help the ranker avoid candidates whose visible boxed/final answer is
    inconsistent with the canonical answer that the scorer will actually use.
    """

    hash_parts = text.rsplit("####", 1)
    has_hash = len(hash_parts) == 2
    hash_tail = hash_parts[1] if has_hash else ""
    hash_starts_numeric = bool(
        has_hash
        and re.match(
            r"\s*(?:[$*`]|\\boxed\{)*\s*[-+]?\d",
            hash_tail,
        )
    )

    boxed_matches = re.findall(
        r"\\boxed\{\s*([-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?)\s*\}",
        text,
    )
    boxed_answer = _canonical_number(boxed_matches[-1]) if boxed_matches else None

    final_segments = re.split(r"final\s+answer\s*: ?", text, flags=re.IGNORECASE)
    has_final_segment = len(final_segments) >= 2
    final_answer = (
        _canonical_number(final_segments[-1]) if has_final_segment else None
    )
    explicit_answer = boxed_answer if boxed_answer is not None else final_answer

    def agrees(answer: str | None) -> float:
        return float(answer is not None and parsed_answer is not None and answer == parsed_answer)

    def disagrees(answer: str | None) -> float:
        return float(answer is not None and parsed_answer is not None and answer != parsed_answer)

    values = np.asarray(
        [
            float(has_hash),
            float(hash_starts_numeric),
            float(has_hash and not hash_starts_numeric),
            float(boxed_answer is not None),
            agrees(boxed_answer),
            disagrees(boxed_answer),
            float(has_final_segment and final_answer is not None),
            agrees(final_answer),
            disagrees(final_answer),
            agrees(explicit_answer),
            disagrees(explicit_answer),
        ],
        dtype=np.float32,
    )
    if values.shape != (len(PARSER_FEATURE_NAMES),) or not np.isfinite(values).all():
        raise ValueError("Parser-consistency feature extraction produced an invalid vector")
    return values


def _consensus_features(
    group: AnswerGroup,
    groups: list[AnswerGroup],
    parsed_answers: list[str | None],
) -> np.ndarray:
    other_support = max((item.support for item in groups if item is not group), default=0)
    max_support = max(item.support for item in groups)
    unique_plurality = group.support == max_support and sum(
        item.support == max_support for item in groups
    ) == 1
    return np.asarray(
        [
            group.support / 4.0,
            (group.support - other_support) / 4.0,
            float(unique_plurality),
            float(group.support == 1),
            len(groups) / 4.0,
            sum(answer is None for answer in parsed_answers) / 4.0,
        ],
        dtype=np.float32,
    )


def _embedding_features(
    group: AnswerGroup,
    embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    selected = np.asarray(embeddings[list(group.indices)], dtype=np.float32)
    group_mean = selected.mean(axis=0)
    all_mean = np.asarray(embeddings, dtype=np.float32).mean(axis=0)

    norms = np.linalg.norm(selected, axis=1, keepdims=True)
    unit_selected = selected / np.maximum(norms, 1e-12)
    unit_group_mean = unit_selected.mean(axis=0)
    unit_all_mean = np.asarray(embeddings, dtype=np.float32)
    unit_all_mean = unit_all_mean / np.maximum(
        np.linalg.norm(unit_all_mean, axis=1, keepdims=True), 1e-12
    )
    unit_all_mean = unit_all_mean.mean(axis=0)
    embedding_vector = unit_group_mean - unit_all_mean

    if len(selected) <= 1:
        cohesion = 0.0
    else:
        similarities = unit_selected @ unit_selected.T
        upper = similarities[np.triu_indices(len(selected), k=1)]
        cohesion = float(upper.mean())

    scalars = np.asarray(
        [
            cohesion,
            _safe_cosine(group_mean, all_mean),
            float(np.linalg.norm(group_mean - all_mean)),
            float(np.linalg.norm(group_mean)),
        ],
        dtype=np.float32,
    )
    return embedding_vector.astype(np.float32, copy=False), scalars


def feature_names(feature_set: str, embedding_dim: int = 1024) -> tuple[str, ...]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")
    consensus = tuple(f"consensus:{name}" for name in CONSENSUS_FEATURE_NAMES)
    text = tuple(f"trace_mean:{name}" for name in TRACE_FEATURE_NAMES) + tuple(
        f"trace_max:{name}" for name in TRACE_FEATURE_NAMES
    )
    embedding = tuple(f"embedding_delta:{index}" for index in range(embedding_dim))
    embedding_scalars = tuple(f"embedding:{name}" for name in EMBEDDING_SCALAR_NAMES)
    if feature_set == "consensus":
        return consensus
    if feature_set == "text":
        return text
    if feature_set == "embedding":
        return embedding
    if feature_set == "full_no_context":
        return text + embedding
    return consensus + text + embedding + embedding_scalars


def feature_schema_hash(feature_set: str, embedding_dim: int = 1024) -> str:
    payload = "\n".join((FEATURE_SCHEMA_VERSION, feature_set, *feature_names(feature_set, embedding_dim)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def candidate_feature_schema_hash(embedding_dim: int = 1024) -> str:
    names = (
        *(f"consensus:{name}" for name in CONSENSUS_FEATURE_NAMES),
        *(f"trace:{name}" for name in TRACE_FEATURE_NAMES),
        *(f"embedding_delta:{index}" for index in range(embedding_dim)),
        *(f"embedding:{name}" for name in EMBEDDING_SCALAR_NAMES),
        *(f"parser:{name}" for name in PARSER_FEATURE_NAMES),
    )
    payload = "\n".join((FEATURE_SCHEMA_VERSION, "candidate", *names))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def group_feature_matrix(
    question: str,
    candidate_texts: Iterable[str],
    embeddings: np.ndarray,
    groups: list[AnswerGroup],
    parsed_answers: list[str | None],
    feature_set: str,
) -> np.ndarray:
    """Build one deterministic feature vector per parseable answer group."""

    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")
    candidate_texts = list(candidate_texts)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.ndim != 2 or embeddings.shape[0] != len(candidate_texts):
        raise ValueError("Candidate embeddings and texts are not aligned")

    per_candidate = np.stack(
        [
            trace_features(question, text, parsed_answers[index])
            for index, text in enumerate(candidate_texts)
        ],
        axis=0,
    )
    vectors: list[np.ndarray] = []
    for group in groups:
        trace_selected = per_candidate[list(group.indices)]
        text_block = np.concatenate((trace_selected.mean(axis=0), trace_selected.max(axis=0)))
        consensus_block = _consensus_features(group, groups, parsed_answers)
        embedding_block, embedding_scalars = _embedding_features(group, embeddings)
        if feature_set == "consensus":
            vector = consensus_block
        elif feature_set == "text":
            vector = text_block
        elif feature_set == "embedding":
            vector = embedding_block
        elif feature_set == "full_no_context":
            vector = np.concatenate((text_block, embedding_block))
        else:
            vector = np.concatenate(
                (consensus_block, text_block, embedding_block, embedding_scalars)
            )
        vectors.append(vector.astype(np.float32, copy=False))

    expected_dim = len(feature_names(feature_set, embeddings.shape[1]))
    matrix = np.stack(vectors, axis=0) if vectors else np.empty((0, expected_dim), np.float32)
    if matrix.shape != (len(groups), expected_dim) or not np.isfinite(matrix).all():
        raise ValueError("Answer-group feature matrix is invalid")
    return matrix


def candidate_feature_matrix(
    question: str,
    candidate_texts: Iterable[str],
    embeddings: np.ndarray,
    groups: list[AnswerGroup],
    parsed_answers: list[str | None],
) -> np.ndarray:
    """Build one feature vector per candidate for development experiments."""

    candidate_texts = list(candidate_texts)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if embeddings.shape != (len(candidate_texts), 1024):
        raise ValueError("Candidate embeddings and texts are not aligned")
    group_by_candidate = {
        index: group for group in groups for index in group.indices
    }
    maximum_support = max((group.support for group in groups), default=0)
    unparseable_fraction = sum(answer is None for answer in parsed_answers) / 4.0
    vectors = []
    for index, text in enumerate(candidate_texts):
        group = group_by_candidate.get(index)
        if group is None:
            consensus = np.asarray(
                [
                    0.0,
                    -maximum_support / 4.0,
                    0.0,
                    1.0,
                    len(groups) / 4.0,
                    unparseable_fraction,
                ],
                dtype=np.float32,
            )
        else:
            consensus = _consensus_features(group, groups, parsed_answers)
        singleton = AnswerGroup(
            answer=parsed_answers[index] or "",
            indices=(index,),
        )
        embedding_vector, embedding_scalars = _embedding_features(
            singleton, embeddings
        )
        vectors.append(
            np.concatenate(
                (
                    consensus,
                    trace_features(question, text, parsed_answers[index]),
                    embedding_vector,
                    embedding_scalars,
                    parser_consistency_features(text, parsed_answers[index]),
                )
            ).astype(np.float32, copy=False)
        )
    matrix = np.stack(vectors, axis=0)
    if matrix.shape != (len(candidate_texts), CANDIDATE_FEATURE_DIM) or not np.isfinite(matrix).all():
        raise ValueError("Candidate feature matrix is invalid")
    return matrix
