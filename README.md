# Bermuda Sea-Level Practical — Python rewrite

Files:

- `SeaLevelPractical_Python.ipynb` — rewritten teaching practical.
- `environment.yml` — conda-forge environment (Python 3.13).
- `SUBSTANTIVE_CHANGES.md` — analytical/methodological changes from the MATLAB practical.

## Setup

Place these files either inside the extracted original `SLprac` directory or in the directory immediately above `SLprac`, then run:

```bash
conda env create -f environment.yml
conda activate sea-level-practical
jupyter lab
```

The notebook searches for the archived processed MATLAB data in the original directory structure. If later tide-gauge or BATS files use a different format, update the two loader functions/configuration paths near the top of the notebook.
