.PHONY: test lint quick run ablation round2 parity clean

test:
	python3 -m pytest tests/ -q

lint:
	python3 -m ruff check src/ main.py scripts/ tests/

quick:
	python3 main.py --quick

run:
	python3 main.py --outdir results

ablation:
	python3 main.py --ablation --outdir results

round2:
	python3 main.py --round2 --outdir results

parity:
	python3 scripts/check_web_parity.py

clean:
	rm -rf results/__pycache__ src/__pycache__ tests/__pycache__ .pytest_cache
