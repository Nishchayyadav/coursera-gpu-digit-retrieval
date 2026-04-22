PYTHON ?= python3

.PHONY: install run clean

install:
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) src/run_retrieval_gpu.py --output-dir proof --max-images 1797 --query-count 300 --top-k 5 --batch-size 128 --seed 42

clean:
	rm -f proof/run_log.txt proof/retrieval_samples.png proof/retrieval_results.csv
