# Improvement Experiments

The original checkpoints were frozen using Train and Validation evidence and each
received one public-Test score. On 2026-10-28, after comparing those scores, the
user requested restoration of the linear candidate. This package-level choice
was therefore Test-informed and does not satisfy the coursework Test-isolation
rule. No Test was rerun during restoration.

| Candidate | Train evidence | Validation evidence | Decision |
|---|---:|---:|---|
| Plurality/confidence gates | Best OOF 82.26% | 743/798 | Rejected: did not transfer |
| Extra pooling, parser cues, lexical hashes | At most 82.31% OOF; most lower | At most 745/798 | Rejected: no robust joint gain |
| Listwise softmax objective | 81.55% ambiguous CV | 741/798 | Rejected |
| Hardest-negative objective | 81.55% ambiguous CV | 742/798 | Rejected |
| Pairwise/listwise/hardest objective stacks | At most 82.17% OOF | At most 745/798 | Rejected: no gain over the smaller stack |
| Quadratic 20-feature stack | 82.12% OOF | 744/798 | Rejected |
| Candidate-level linear ranker alone | 80.56% ambiguous OOF | 739/798 | Kept only as a complementary member |
| Previous four-group-model stack | 82.07% strict nested CV (1735/2114) | 745/798 | Historical checkpoint |
| Four groups + linear candidate ranker | **82.4503% strict nested CV (1743/2114)** | **745/798** | **Selected frozen checkpoint; Test 382/467 (previously scored once)** |
| Four groups + linear candidate ranker + 8-unit MLP | 82.3557% strict nested CV (1741/2114) | 746/798 | Archived experiment; not selected |

The MLP alternative uses 100 epochs, learning rate 0.03, L2 0.001, and seed
7506. Compared with the selected linear-candidate stack, its strict nested Train
CV is two questions lower while Validation is one question higher. Compared
with the four-group stack, it is six questions higher in nested Train CV and
one higher on Validation. We retain the earlier linear-candidate checkpoint
selected by the user. Its checkpoint has 259,032 uncompressed array bytes,
below the 16 MiB limit.

## Final scoring, after freeze

The restored linear-candidate checkpoint has an existing one-time public Test
record of 382/467 (81.7987%). It was not rerun during restoration. The later MLP
alternative scored 377/467 (80.7281%). The user requested the linear checkpoint
after seeing this comparison, so Test influenced package selection. This is a
process deviation from the coursework requirements; an unseen audit bank is
needed for independent comparison.

## Rejected sensitivity and fusion checks

Seed 7507 and 7508 strict nested CV checks scored 1751/2114 and 1752/2114,
but changed fold assignment as well as model seed. Their matching full-data
Validation results were 744/798 and 743/798, below the seed-7506 candidate, so
they were not adopted. Rank-based meta fusion (1717/2114 OOF; 741/798
Validation), quadratic interactions (1742/2114 OOF; 743/798 Validation), and
additional L2 checks were also rejected. The retained selection is the
seed-7506 stack documented in `final_config.json`.
