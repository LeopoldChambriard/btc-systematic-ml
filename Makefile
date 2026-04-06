PY ?= python

.PHONY: install install-tf test test-fast lint format demo data run baseline ablation eval2022 compare readme all clean

install:            ## core + dev dependencies (no TensorFlow)
	$(PY) -m pip install -e ".[dev]"

install-tf:         ## ... plus TensorFlow for the LSTM primary model
	$(PY) -m pip install -e ".[dev,tf]"

test:               ## full test-suite
	$(PY) -m pytest

test-fast:          ## everything except the TensorFlow tests
	$(PY) -m pytest -m "not tf"

lint:
	ruff check .

format:
	ruff format .

demo:               ## whole pipeline on synthetic data, ~1 minute, no download
	sysml -v demo

data:               ## hourly BTCUSDT candles from Binance public data -> data/raw/btc_1h.csv
	sysml download --symbol BTCUSDT --coin BTC --start 2018-01-01 --end 2023-12-31 --out data/raw/btc_1h.csv

run:                ## main experiment (LSTM primary; needs `make install-tf`)
	sysml -v run -c configs/default.yaml

baseline:           ## same pipeline with a logistic-regression primary model
	sysml -v run -c configs/baseline_logreg.yaml

ablation:           ## same as `run` but with the leaky (smoothed, same-day) HMM regimes
	sysml -v run -c configs/ablation_leaky_regimes.yaml

eval2022:           ## train 2018-2020, test on the 2022 bear market
	sysml -v run -c configs/eval_2022_bear.yaml

compare:
	sysml compare results/default results/baseline_logreg results/ablation_leaky_regimes

readme:             ## paste the latest results/default backtest tables into README.md
	$(PY) scripts/update_readme_results.py results/default/summary.md README.md

all: run baseline ablation compare readme

clean:
	rm -rf results/*/ .pytest_cache .ruff_cache build *.egg-info src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
