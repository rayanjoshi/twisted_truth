# twisted_truth

<p align="center">
	<img src="https://img.shields.io/github/license/Ileriayo/markdown-badges?style=for-the-badge" alt="license" />
	<img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="python" />
	<img src="https://img.shields.io/badge/yaml-%23ffffff.svg?style=for-the-badge&logo=yaml&logoColor=151515" alt="yaml" />
</p>

Lightweight tools to collect and visualise M1 money supply against equity indices and constituents, and cryptocurrencies.

This small repo fetches M1 Money Supply from FRED, appends market data from Yahoo Finance, and provides
simple utilities to visualise the series and compute long/short-term correlations.

## Features
- Fetch M1 series from FRED and save a single CSV (`data/data.csv`).
- Append index and constituent close prices from Yahoo Finance.
- Visualise M1 vs an index, a constituent, and a cryptocurrency using matplotlib.
- Compute per-year and per-quarter Pearson correlations with optional time-lagging to output a Fisher z-transformed averaged correlation coefficent.

Key concept — Fisher z-transformed average correlation
- Problem: averaging Pearson r values directly is biased.
- Approach (used by `src/calculating_correlation.py`):
	1. Compute Pearson r for each group (e.g. each year or each quarter).
	2. Apply Fisher z-transform: z = arctanh(r) (maps r in (-1,1) to real line).
	3. Average the z values: mean_z = mean(z_i).
	4. Convert back to correlation: avg_r = tanh(mean_z).

	This stabilises variance and gives a sound aggregated correlation estimate.

## Quick start

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

2. Provide secrets in a `.env` file at repository root:

```bash
cp .env.example .env

FRED_API_KEY=your_fred_api_key_here
```

3. Run the srcipts:

```bash
python run.py
```

## Configuration
- All runtime parameters live under the `configs/` directory (Hydra + OmegaConf).
- Common config keys live in `configs/data_loader.yaml` and `configs/calculating_correlation.yaml`.
- You can override any config value on the command line. Example:

```bash
python src/data_loader.py load_data.start_date=2020-01-01 load_data.end_date=2024-12-31
```

## Notes & troubleshooting
- Network: `yfinance` and `fredapi` require outbound internet access.
- Forward-fill / placeholders: if the data loader cannot fetch new M1 values it may forward-fill
	the last value; that will produce zero-variance groups and cause NaNs in per-group Pearson
	correlations. If you see NaNs in correlation output inspect `data/data.csv` for long runs of
	identical `M1_Money_Supply` values and fix the upstream fetch (preferred) or drop/NaN those rows.

## License
- License: MIT (see `LICENSE`)