# LaTeX report (IEEE format, with citations)

This folder is a self-contained LaTeX project for the Assignment 3 report.

```
DM_asg3_314540061.tex   <- main file (the 4 graded questions, in IEEE format)
reference.bib           <- bibliography (real papers, BibTeX)
IEEEtran.cls            <- IEEE conference class
figures/                <- the figures used in the report
```

## How to compile to PDF

**Option A — Overleaf (easiest, no install):**
1. Go to overleaf.com, "New Project" -> "Upload Project", and upload this whole
   folder (or zip it first).
2. Set the main document to `DM_asg3_314540061.tex` (Menu -> Main document).
3. Make sure the compiler is **pdfLaTeX** (Menu -> Compiler).
4. Click **Recompile**. Overleaf runs LaTeX + BibTeX automatically, so the
   citations and References section resolve on the first full recompile.
5. Download the PDF. It is already named `DM_asg3_314540061.pdf`.

**Option B — local (if you have a LaTeX distribution, e.g. MiKTeX/TeX Live):**
```
pdflatex DM_asg3_314540061
bibtex   DM_asg3_314540061
pdflatex DM_asg3_314540061
pdflatex DM_asg3_314540061
```
(Run pdflatex twice after bibtex so the citation numbers settle.)

## Notes
- Every number in the report comes from the code / the Kaggle submission history;
  nothing is invented.
- All nine references are real, verifiable papers for the methods actually used
  (the dataset paper, LightGBM, XGBoost, CatBoost, ROCKET, scikit-learn, LSTM,
  DeepConvLSTM for HAR, and Caruana et al.'s ensemble selection).
