"""Canonical final-answer grouping aligned with the official scorer."""

from dataclasses import dataclass

from common import parse_answer


@dataclass(frozen=True)
class AnswerGroup:
    """Candidates whose final answers have the same canonical numeric value."""

    answer: str
    indices: tuple[int, ...]

    @property
    def first_index(self) -> int:
        return self.indices[0]

    @property
    def support(self) -> int:
        return len(self.indices)


def parsed_candidate_answers(row: dict) -> list[str | None]:
    """Parse candidates exactly as the independent official scorer does."""

    return [parse_answer(candidate["text"]) for candidate in row["candidates"]]


def build_answer_groups(row: dict) -> tuple[list[AnswerGroup], list[str | None]]:
    """Return parseable answer groups in deterministic candidate order.

    Unparseable candidates cannot score under the official metric and are not
    included in a normal answer group. The caller provides a deterministic
    fallback only for the unexpected case where every candidate is unparseable.
    """

    parsed = parsed_candidate_answers(row)
    members: dict[str, list[int]] = {}
    for index, answer in enumerate(parsed):
        if answer is not None:
            members.setdefault(answer, []).append(index)
    groups = [
        AnswerGroup(answer=answer, indices=tuple(indices))
        for answer, indices in members.items()
    ]
    return groups, parsed


def plurality_group_index(groups: list[AnswerGroup]) -> int | None:
    """Choose the largest group, breaking ties by earliest candidate index."""

    if not groups:
        return None
    return max(
        range(len(groups)),
        key=lambda index: (groups[index].support, -groups[index].first_index),
    )
