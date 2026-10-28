# AI Assistance and Reuse Disclosure

Substantive AI assistance was used to inspect the supplied project, clarify the
benchmark contract, design the answer-group ranking method, implement and test
the Python code, structure the experiment protocol, and draft project/report
documentation. The resulting implementation was verified with the supplied
data checker, automated tests, fixed train-only cross-validation, the official
validation scorer, checkpoint-size checks, and fresh-process timing.

The method does not call an AI service at training or inference time. It uses
only the supplied questions, candidate traces, fixed embeddings, and allowed
training references. No external dataset or additional pretrained model weight
is used.

The supplied benchmark code, parser, scorer, runner, data, and attribution come
from the DASE7506 MP2 package. GSM8K and Qwen provenance remains documented in
`code/data/LICENSE.txt`, `code/data/NOTICE.txt`, and `code/data/manifest.json`.

Before submission, the student should review and edit this disclosure so that
it accurately reflects all human and AI contributions made after this version.
