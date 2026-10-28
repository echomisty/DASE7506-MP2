# Report Generation

`build_reports.py` is the single source for the current bilingual report set.
It validates headline facts against the frozen JSON evidence before writing any
PDF. Sections flow continuously across A4 pages; there are no forced page
breaks at section boundaries. The two submission editions are still checked
to be exactly ten pages after ordinary pagination.

## Outputs

- `../output/pdf/DASE7506_MP2_Full_Report_Chinese.pdf` - 11-page complete technical reference.
- `../output/pdf/DASE7506_MP2_Full_Report_English.pdf` - 11-page complete report.
- `../output/pdf/DASE7506_MP2_10_Page_Report_Chinese.pdf` - exact 10-page submission report.
- `../output/pdf/DASE7506_MP2_10_Page_Report_English.pdf` - exact 10-page submission report.

The full and 10-page editions use the same facts and conclusions. The shorter
editions use denser typography but retain every validation stage: data and
contract checks, baselines, feature ablations, ordinary out-of-fold selection,
rejected alternatives, strict nested cross-validation, final Validation slices,
paired bootstrap, error analysis, resource rechecks, the documented Test-selection deviation, limitations,
and AI disclosure. Full editions additionally include code-level feature,
checkpoint, and claim-to-evidence audits.

Use the exact 10-page Chinese or English edition for the coursework submission.
The longer full editions are supplementary technical references.

## Visual evidence system

The four principal figures are generated as vectors inside the PDF source, so
there are no duplicate image assets to maintain. They share a restrained,
colour-blind-safe blue/teal palette and direct labels:

- the system architecture separates scorer-compatible canonicalisation, the
  ambiguous-question learning path, the single-group bypass, and the final
  deterministic candidate decision;
- the experimental timeline divides boundary protection, model selection, and
  revalidation, and documents the package-level Test comparison and its selection deviation;
- the paired Validation comparison reports percentages and correct counts on
  the marks themselves, including the improvement over plurality;
- the strict nested-CV comparison shows all five outer-fold results, their
  ranges, aggregate means, and the eight-question gain from linear candidate evidence.

## Build

From the repository root:

```bash
python report/build_reports.py
```

The build requires ReportLab and pypdf. The bundled coursework environment can
run the generator directly; `requirements.txt` records standalone versions.

## Evidence and policy

All current-checkpoint claims come from `../experiments/` and
`../results/final/`. The report uses the restored linear checkpoint results: Validation 745/798, strict nested Train CV 1743/2114, and the historical one-time Test record 382/467. It explicitly discloses that package-level restoration on 2026-10-28 followed a comparison with the later MLP Test score (377/467), so the restored package does not satisfy the coursework Test-isolation rule. The Test was not rerun. Raw linear Test predictions are not retained.

Visual QA renders belong only in `../tmp/pdfs/` during review and must be
deleted after verification. Final deliverables are the four PDFs above plus
this reproducible source.
