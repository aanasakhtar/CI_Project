# HP-MOCD Overlapping Extension — Project Setup

## Setup notes

The project now installs cleanly without the optional `hp-mocd` PyPI package.
That package is not always available on Windows/Python combinations, so the
baseline runner falls back to `MinimalNSGAII` automatically when `hp-mocd` is
missing.

### Recommended install workflow

1. Create or activate a Conda environment on your `D:` drive. (Optional)
2. Install the project dependencies with `pip install -r requirements.txt`.
3. If you want to try the official `hp-mocd` package separately, install it
   only after confirming it is available for your platform.

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
