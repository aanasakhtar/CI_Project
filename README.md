# HP-MOCD Overlapping Extension — Project Setup

## Setup notes

The project now uses the official `pymocd` package for HP-MOCD. The baseline
runner still keeps `MinimalNSGAII` as a fallback so the project remains usable
if the package is unavailable on a given machine.

### Recommended install workflow

1. Create or activate a Conda environment on your `D:` drive. (Optional)
2. Install the project dependencies with `pip install -r requirements.txt`.
3. The HP-MOCD implementation is provided by `pymocd`, so the baseline code
   will use `pymocd.HpMocd(...)` automatically when it is installed.

If `pip` reports a temporary disk-space error, move `PIP_CACHE_DIR`, `TEMP`,
and `TMP` to a folder on `D:` and retry with `--no-cache-dir`.

## Directory Structure

```
hp_mocd_overlap/
├── README.md
├── requirements.txt
├── config.py
├── data/
│   ├── load_lfr.py
│   └── load_dblp.py
├── baseline/
│   └── hp_mocd_baseline.py
└── evaluation/
    └── metrics.py
```
