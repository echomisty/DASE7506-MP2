# Previous Four-Model Stack

This directory preserves the exact checkpoint and official Test artifacts for
the previous four-answer-group-model stack. That frozen model scored 381/467
(81.5846%) on the public Test.

These files are historical evidence for the previous four-answer-group stack.
They do not describe the current group-plus-candidate checkpoint in
`code/checkpoint/final`. The current checkpoint received a separate one-time
final Test score of 382/467 (81.7987%) after it was frozen; it was not selected
or tuned using that score. Because this public split had already been used by
earlier methods, the comparison is not unbiased and a future unseen audit bank
is still needed.
