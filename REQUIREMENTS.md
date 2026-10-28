# DASE7506 MP2 Requirements and Acceptance Criteria

## 1. Objective

Implement an offline method that selects exactly one of four supplied candidate
solutions for each question. The selected candidate must have the correct final
numeric answer under the official `common.parse_answer` rules.

## 2. Inputs and output

The prediction interface receives:

- `bank`: ordered question records with four ordered candidate texts;
- `arrays['embeddings']`: float32 array shaped `[N, 4, 1024]`;
- `checkpoint`: a path to all additional inference assets, or `None`.

It must return a Python `list` of length `N`, containing built-in Python
integers in `[0, 3]`, in bank order.

## 3. Data protocol

- Train: 4,755 questions; the only split allowed to fit weights, statistics,
  transforms, and fusion parameters.
- Validation: 798 questions; development, model selection, diagnostics, and
  ablations only.
- Test: 467 questions; frozen final scoring and reproduction only.
- Audit: possible unseen input-only bank; must run without retraining or
  assumptions about IDs or sample count.
- Candidate IDs are alignment metadata only and must not be model features.
- Test references must not enter training, tuning, feature selection, lookup
  tables, or error-driven changes.

## 4. Fixed benchmark rules

- Do not modify supplied data, the numeric parser, the scorer, or the runner.
- Do not use external datasets or additional pretrained weights.
- Do not generate new solutions or call an online API at evaluation time.
- Select only from the four supplied candidates.
- Any candidate whose canonical parsed answer equals the reference is correct;
  multiple candidates may be correct.

## 5. Resource constraints

- Complete fresh CPU prediction plus independent scoring: no more than 30
  seconds on the reference Xeon Platinum 8457C.
- Total threads across libraries and processes: at most four.
- Additional inference assets: at most 16 MiB uncompressed, including learned
  weights, normalization statistics, transforms, configurations, indexes, and
  ensemble members.
- Training hardware: CPU or one GPU no more powerful than an RTX 4090.
- Inference must run on CPU and offline.

Internal safety targets are 4 MiB or less of inference arrays and 15 seconds or
less for local end-to-end prediction plus scoring.

## 6. Method requirements

- Use scorer-compatible final-answer parsing.
- Group candidates with identical canonical parsed answers.
- Handle one, two, three, or four distinct answer groups.
- Avoid selecting an unparseable candidate while a parseable candidate exists.
- Use deterministic tie-breaking.
- Operate on arbitrary `N` and never hard-code public split sizes.
- Load only the supplied checkpoint at inference; never open references.
- Validate checkpoint schema, feature schema, array shapes, and finite values.

## 7. Experimental requirements

Required baselines:

- fixed candidate position;
- longest trace;
- answer plurality with deterministic tie-breaking.

Required learned comparisons:

- consensus only;
- trace/text features only;
- embeddings only;
- full answer-group ranker;
- at least one ablation of group-relative information.

Report at minimum:

- official full-validation accuracy;
- ambiguous-validation accuracy, defined by at least two distinct valid parsed
  candidate answers;
- accuracy with and without a unique plurality;
- paired corrections and regressions relative to plurality;
- cross-validation mean and variation;
- checkpoint uncompressed size;
- fresh-process prediction and scoring time;
- representative failure categories.

## 8. Reproducibility requirements

- Pin dependencies.
- Record all random seeds.
- Record local hardware and thread variables.
- Keep exact training, prediction, and scoring commands.
- Save the final configuration next to the checkpoint.
- Save SHA-256 hashes for all inference assets.
- Record the submitted source version and matching checkpoint.
- A peer must reproduce the submitted score without retraining.

## 9. Submission requirements

The public submission must include:

- implementation and training code;
- exact installation/training/evaluation commands;
- final inference checkpoint or training-free configuration;
- official evaluation JSON;
- experiment and resource records;
- random seeds and asset hashes;
- report of no more than 10 pages;
- AI-assistance and reused-work disclosure;
- retained dataset license and attribution files.

The course website prepares a GitHub issue; the issue must actually be created
to complete submission. Deadline: 31 October 2026 at 23:59 UTC+8.

## 10. Definition of done

- [x] Official data check passes in Python 3.12.
- [x] Unit and contract tests pass.
- [x] Train-only fitting and fixed-fold experiments complete.
- [x] Validation baselines, learned method, slices, and ablations are recorded.
- [x] Final checkpoint is deterministic and within the uncompressed asset limit.
- [x] Fresh four-thread validation prediction and scoring meet the time limit.
- [x] Current method and configuration are frozen from Train/Validation evidence.
- [ ] Current package selection is test-blind. The 2026-10-28 restoration was
      requested after comparing public-Test scores (linear 382/467; MLP 377/467),
      so Test influenced package-level model selection. This is a documented
      deviation from the fixed benchmark rule; the historical 81.80% score is not
      a test-blind final estimate.
- [x] Clean project-local Python 3.12 environment reproduction matches the recorded
      Validation score.
- [x] Full reports and exact 10-page reports are generated in Chinese and English, with
      solution selection, rejected alternatives, intermediate validation, resource
      checks, Test isolation, limitations, and AI disclosure.
- [x] Commands, disclosure, hashes, and resource records are complete.
- [x] Project overview and requirements documents reflect the final repository.
- [x] Duplicate content and files have been reviewed.
- [x] Temporary outputs, caches, abandoned checkpoints, and build artifacts have
      been removed.
