"""Validated inference-checkpoint persistence and resource reporting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from common import PROTOCOL

from .features import (
    CANDIDATE_FEATURE_DIM,
    FEATURE_SCHEMA_VERSION,
    candidate_feature_schema_hash,
    feature_names,
    feature_schema_hash,
)
from .ranker import LinearRanker
from .mlp import MLPRanker
from .stacked import STACK_FEATURE_SETS, StackedRanker


CHECKPOINT_FORMAT = "dase7506-mp2-linear-ranker-v1"
STACKED_CHECKPOINT_FORMAT = "dase7506-mp2-stacked-ranker-v1"
CANDIDATE_STACKED_CHECKPOINT_FORMAT = "dase7506-mp2-stacked-ranker-v2"
MLP_CANDIDATE_STACKED_CHECKPOINT_FORMAT = "dase7506-mp2-stacked-ranker-v3"
MODEL_FILENAME = "model.npz"
METADATA_FILENAME = "metadata.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_checkpoint(
    directory: Path,
    model: LinearRanker,
    training_metadata: dict,
) -> dict:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / MODEL_FILENAME
    metadata_path = directory / METADATA_FILENAME
    np.savez(
        model_path,
        feature_mean=model.feature_mean.astype(np.float32),
        feature_scale=model.feature_scale.astype(np.float32),
        weights=model.weights.astype(np.float32),
    )
    metadata = {
        "format": CHECKPOINT_FORMAT,
        "protocol": PROTOCOL,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_schema_hash": feature_schema_hash(model.feature_set),
        "feature_set": model.feature_set,
        "feature_dim": model.feature_dim,
        "embedding_dim": 1024,
        "array_uncompressed_bytes": int(
            model.feature_mean.nbytes + model.feature_scale.nbytes + model.weights.nbytes
        ),
        "training": training_metadata,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return checkpoint_report(directory)


def save_stacked_checkpoint(
    directory: Path,
    model: StackedRanker,
    training_metadata: dict,
) -> dict:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / MODEL_FILENAME
    metadata_path = directory / METADATA_FILENAME
    arrays = {}
    for feature_set, models in model.base_models.items():
        for index, base_model in enumerate(models):
            prefix = f"base__{feature_set}__{index}"
            arrays[f"{prefix}__feature_mean"] = base_model.feature_mean.astype(np.float32)
            arrays[f"{prefix}__feature_scale"] = base_model.feature_scale.astype(np.float32)
            arrays[f"{prefix}__weights"] = base_model.weights.astype(np.float32)
    for index, candidate_model in enumerate(model.candidate_models):
        prefix = f"candidate__{index}"
        arrays[f"{prefix}__feature_mean"] = candidate_model.feature_mean.astype(np.float32)
        arrays[f"{prefix}__feature_scale"] = candidate_model.feature_scale.astype(np.float32)
        arrays[f"{prefix}__weights"] = candidate_model.weights.astype(np.float32)
    for index, candidate_model in enumerate(model.candidate_mlp_models):
        prefix = f"candidate_mlp__{index}"
        arrays[f"{prefix}__feature_mean"] = candidate_model.feature_mean.astype(np.float32)
        arrays[f"{prefix}__feature_scale"] = candidate_model.feature_scale.astype(np.float32)
        arrays[f"{prefix}__input_weights"] = candidate_model.input_weights.astype(np.float32)
        arrays[f"{prefix}__hidden_bias"] = candidate_model.hidden_bias.astype(np.float32)
        arrays[f"{prefix}__output_weights"] = candidate_model.output_weights.astype(np.float32)
        arrays[f"{prefix}__output_bias"] = candidate_model.output_bias.astype(np.float32)
    arrays["stack__feature_mean"] = model.stack_model.feature_mean.astype(np.float32)
    arrays["stack__feature_scale"] = model.stack_model.feature_scale.astype(np.float32)
    arrays["stack__weights"] = model.stack_model.weights.astype(np.float32)
    np.savez(model_path, **arrays)
    metadata = {
        "format": (
            MLP_CANDIDATE_STACKED_CHECKPOINT_FORMAT
            if model.candidate_mlp_models
            else CANDIDATE_STACKED_CHECKPOINT_FORMAT
            if model.candidate_models
            else STACKED_CHECKPOINT_FORMAT
        ),
        "protocol": PROTOCOL,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "base_feature_sets": list(STACK_FEATURE_SETS),
        "base_feature_schema_hashes": {
            feature_set: feature_schema_hash(feature_set)
            for feature_set in STACK_FEATURE_SETS
        },
        "models_per_feature_set": model.models_per_feature_set,
        "candidate_model_count": len(model.candidate_models),
        "candidate_mlp_model_count": len(model.candidate_mlp_models),
        "candidate_feature_schema_hash": (
            candidate_feature_schema_hash() if model.candidate_models else None
        ),
        "stack_feature_dim": int(model.stack_model.weights.size),
        "embedding_dim": 1024,
        "array_uncompressed_bytes": int(sum(array.nbytes for array in arrays.values())),
        "training": training_metadata,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return checkpoint_report(directory)


def load_checkpoint(directory: Path) -> tuple[LinearRanker | StackedRanker, dict]:
    directory = Path(directory)
    metadata_path = directory / METADATA_FILENAME
    model_path = directory / MODEL_FILENAME
    if not metadata_path.is_file() or not model_path.is_file():
        raise FileNotFoundError("Checkpoint must contain metadata.json and model.npz")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("protocol") != PROTOCOL:
        raise ValueError("Checkpoint format or benchmark protocol is incompatible")
    if metadata.get("format") in {
        STACKED_CHECKPOINT_FORMAT,
        CANDIDATE_STACKED_CHECKPOINT_FORMAT,
        MLP_CANDIDATE_STACKED_CHECKPOINT_FORMAT,
    }:
        return _load_stacked_checkpoint(model_path, metadata)
    if metadata.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("Checkpoint format or benchmark protocol is incompatible")
    feature_set = metadata.get("feature_set")
    embedding_dim = int(metadata.get("embedding_dim", -1))
    expected_dim = len(feature_names(feature_set, embedding_dim))
    if metadata.get("feature_schema_hash") != feature_schema_hash(feature_set, embedding_dim):
        raise ValueError("Checkpoint feature schema hash does not match the code")
    if int(metadata.get("feature_dim", -1)) != expected_dim:
        raise ValueError("Checkpoint feature dimension is invalid")
    with np.load(model_path, allow_pickle=False) as arrays:
        required = {"feature_mean", "feature_scale", "weights"}
        if set(arrays.files) != required:
            raise ValueError("Checkpoint arrays do not match the expected schema")
        model = LinearRanker(
            feature_set=feature_set,
            feature_mean=arrays["feature_mean"].astype(np.float32, copy=True),
            feature_scale=arrays["feature_scale"].astype(np.float32, copy=True),
            weights=arrays["weights"].astype(np.float32, copy=True),
        )
    if model.feature_dim != expected_dim:
        raise ValueError("Checkpoint array dimension is invalid")
    return model, metadata


def _load_stacked_checkpoint(
    model_path: Path,
    metadata: dict,
) -> tuple[StackedRanker, dict]:
    if tuple(metadata.get("base_feature_sets", ())) != STACK_FEATURE_SETS:
        raise ValueError("Stacked checkpoint feature sets are invalid")
    if metadata.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("Stacked checkpoint feature schema version is invalid")
    expected_hashes = {
        feature_set: feature_schema_hash(feature_set)
        for feature_set in STACK_FEATURE_SETS
    }
    if metadata.get("base_feature_schema_hashes") != expected_hashes:
        raise ValueError("Stacked checkpoint feature schemas do not match the code")
    model_count = int(metadata.get("models_per_feature_set", 0))
    if model_count <= 0:
        raise ValueError("Stacked checkpoint model count is invalid")
    expected_keys = {"stack__feature_mean", "stack__feature_scale", "stack__weights"}
    for feature_set in STACK_FEATURE_SETS:
        for index in range(model_count):
            prefix = f"base__{feature_set}__{index}"
            expected_keys.update(
                {
                    f"{prefix}__feature_mean",
                    f"{prefix}__feature_scale",
                    f"{prefix}__weights",
                }
            )
    candidate_count = int(metadata.get("candidate_model_count", 0))
    has_candidate = metadata.get("format") in {
        CANDIDATE_STACKED_CHECKPOINT_FORMAT,
        MLP_CANDIDATE_STACKED_CHECKPOINT_FORMAT,
    }
    has_candidate_mlp = metadata.get("format") == MLP_CANDIDATE_STACKED_CHECKPOINT_FORMAT
    if has_candidate:
        if candidate_count != model_count:
            raise ValueError("Candidate checkpoint model count is invalid")
        if metadata.get("candidate_feature_schema_hash") != candidate_feature_schema_hash():
            raise ValueError("Candidate checkpoint feature schema does not match the code")
        for index in range(candidate_count):
            prefix = f"candidate__{index}"
            expected_keys.update(
                {
                    f"{prefix}__feature_mean",
                    f"{prefix}__feature_scale",
                    f"{prefix}__weights",
                }
            )
    elif candidate_count != 0:
        raise ValueError("Legacy stacked checkpoint cannot contain candidate models")
    candidate_mlp_count = int(metadata.get("candidate_mlp_model_count", 0))
    if has_candidate_mlp:
        if candidate_mlp_count != model_count or not has_candidate:
            raise ValueError("Candidate MLP checkpoint model count is invalid")
        for index in range(candidate_mlp_count):
            prefix = f"candidate_mlp__{index}"
            expected_keys.update(
                {
                    f"{prefix}__feature_mean",
                    f"{prefix}__feature_scale",
                    f"{prefix}__input_weights",
                    f"{prefix}__hidden_bias",
                    f"{prefix}__output_weights",
                    f"{prefix}__output_bias",
                }
            )
    elif candidate_mlp_count != 0:
        raise ValueError("Checkpoint format cannot contain candidate MLP models")
    with np.load(model_path, allow_pickle=False) as arrays:
        if set(arrays.files) != expected_keys:
            raise ValueError("Stacked checkpoint arrays do not match the expected schema")
        base_models = {}
        for feature_set in STACK_FEATURE_SETS:
            models = []
            expected_dim = len(feature_names(feature_set, 1024))
            for index in range(model_count):
                prefix = f"base__{feature_set}__{index}"
                model = LinearRanker(
                    feature_set=feature_set,
                    feature_mean=arrays[f"{prefix}__feature_mean"].astype(
                        np.float32, copy=True
                    ),
                    feature_scale=arrays[f"{prefix}__feature_scale"].astype(
                        np.float32, copy=True
                    ),
                    weights=arrays[f"{prefix}__weights"].astype(np.float32, copy=True),
                )
                if model.feature_dim != expected_dim:
                    raise ValueError("A stacked base-model dimension is invalid")
                models.append(model)
            base_models[feature_set] = tuple(models)
        stack_model = LinearRanker(
            feature_set="stack",
            feature_mean=arrays["stack__feature_mean"].astype(np.float32, copy=True),
            feature_scale=arrays["stack__feature_scale"].astype(np.float32, copy=True),
            weights=arrays["stack__weights"].astype(np.float32, copy=True),
        )
        candidate_models = []
        for index in range(candidate_count):
            prefix = f"candidate__{index}"
            model = LinearRanker(
                feature_set="candidate",
                feature_mean=arrays[f"{prefix}__feature_mean"].astype(
                    np.float32, copy=True
                ),
                feature_scale=arrays[f"{prefix}__feature_scale"].astype(
                    np.float32, copy=True
                ),
                weights=arrays[f"{prefix}__weights"].astype(np.float32, copy=True),
            )
            if model.feature_dim != CANDIDATE_FEATURE_DIM:
                raise ValueError("A candidate-model dimension is invalid")
            candidate_models.append(model)
        candidate_mlp_models = []
        for index in range(candidate_mlp_count):
            prefix = f"candidate_mlp__{index}"
            model = MLPRanker(
                feature_mean=arrays[f"{prefix}__feature_mean"].astype(np.float32, copy=True),
                feature_scale=arrays[f"{prefix}__feature_scale"].astype(np.float32, copy=True),
                input_weights=arrays[f"{prefix}__input_weights"].astype(np.float32, copy=True),
                hidden_bias=arrays[f"{prefix}__hidden_bias"].astype(np.float32, copy=True),
                output_weights=arrays[f"{prefix}__output_weights"].astype(np.float32, copy=True),
                output_bias=arrays[f"{prefix}__output_bias"].astype(np.float32, copy=True),
            )
            if model.feature_dim != CANDIDATE_FEATURE_DIM or model.input_weights.shape[1] != 8:
                raise ValueError("A candidate MLP dimension is invalid")
            candidate_mlp_models.append(model)
    if stack_model.feature_dim != len(STACK_FEATURE_SETS) + 1 + has_candidate + has_candidate_mlp:
        raise ValueError("Stacked checkpoint meta-model dimension is invalid")
    return StackedRanker(
        base_models=base_models,
        stack_model=stack_model,
        candidate_models=tuple(candidate_models),
        candidate_mlp_models=tuple(candidate_mlp_models),
    ), metadata


def checkpoint_report(directory: Path) -> dict:
    directory = Path(directory)
    files = []
    disk_bytes = 0
    for name in (MODEL_FILENAME, METADATA_FILENAME):
        path = directory / name
        if path.is_file():
            size = path.stat().st_size
            disk_bytes += size
            files.append({"name": name, "bytes": size, "sha256": _sha256(path)})
    array_bytes = 0
    model_path = directory / MODEL_FILENAME
    if model_path.is_file():
        with np.load(model_path, allow_pickle=False) as arrays:
            array_bytes = int(sum(arrays[name].nbytes for name in arrays.files))
    return {
        "directory": str(directory),
        "files": files,
        "disk_bytes": disk_bytes,
        "array_uncompressed_bytes": array_bytes,
        "limit_bytes": 16 * 1024 * 1024,
        "within_limit": array_bytes <= 16 * 1024 * 1024,
    }
