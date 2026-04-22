# Coursera GPU Specialization Capstone Project: GPU Digit Image Retrieval

## Project Goal
This project demonstrates a practical GPU application: **image retrieval**.
Given a query digit image, the program retrieves the most similar images from a gallery using GPU-accelerated similarity computation.

This aligns with course goals by using GPU computation (PyTorch CUDA tensor operations) instead of CPU-only processing.

---

## Why this is interesting
Similarity search is useful in many domains: visual search, duplicate detection, nearest-neighbor recommendation, and embedding retrieval.
This project shows a clean, reproducible baseline for GPU retrieval on many small images.

---

## Dataset
- Source: `sklearn.datasets.load_digits`
- Size: 1,797 grayscale digit images (`8x8`)
- Labels: digits `0` to `9`
- Included example data file: `data/example_digits_small.csv`

The dataset is lightweight and built-in, which keeps setup simple while still supporting meaningful GPU batch processing.

---

## GPU Method
1. Load and normalize digit images.
2. Split into query and gallery sets.
3. Flatten images and L2-normalize vectors.
4. Compute query-vs-gallery cosine similarity on GPU using matrix multiplication.
5. Retrieve top-k nearest neighbors.
6. Evaluate Top-1 and Top-5 retrieval accuracy.
7. Save proof artifacts (log, image, CSV).

GPU operations are performed through CUDA-backed PyTorch tensor ops.

---

## Code Organization
- **bin/**: placeholder for built executables
- **data/**: example data files
- **lib/**: optional external libraries not installed via package manager
- **src/**:
  - `run_retrieval_gpu.py` (CLI entry point)
  - `retrieval_utils.py` (dataset loading, GPU retrieval, metrics, artifacts)
- **proof/**: output artifacts for peer review (`run_log.txt`, `retrieval_samples.png`, `retrieval_results.csv`)
- **README.md**: project description and usage
- **INSTALL**: installation and environment notes
- **Makefile**: helper commands
- **run.sh**: run wrapper

---

## CLI Usage
Primary run:
```bash
python3 src/run_retrieval_gpu.py --output-dir proof --max-images 1797 --query-count 300 --top-k 5 --batch-size 128 --seed 42
```

Arguments:
- `--output-dir`: artifact output directory
- `--max-images`: number of images to load
- `--query-count`: number of queries
- `--top-k`: neighbors to retrieve
- `--batch-size`: batch size for GPU similarity computation
- `--seed`: deterministic split seed
- `--allow-cpu`: debug-only fallback when CUDA is not available

---

## Build / Run Helpers
Install dependencies:
```bash
make install
```

Run:
```bash
make run
```

Or:
```bash
./run.sh --output-dir proof --max-images 1797 --query-count 300 --top-k 5 --batch-size 128 --seed 42
```

---

## Proof of Execution Artifacts
A successful run generates:
- `proof/run_log.txt` → device details, timing, throughput, Top-1/Top-5 metrics
- `proof/retrieval_samples.png` → visual query + nearest neighbors
- `proof/retrieval_results.csv` → per-query retrieval output

These artifacts are intended for peer-review grading evidence.

Current repository status:
- `proof/run_log.txt` is included.
- `proof/retrieval_samples.png` is included.
- `proof/retrieval_results.csv` is included.
- `data/example_digits_small.csv` is included.

---

## Short Project Description (for submission form)
I built a GPU-based image retrieval pipeline on the sklearn digits dataset. The system compares query images against a gallery using cosine similarity computed on CUDA through PyTorch tensor operations. It retrieves top-k matches, evaluates Top-1 and Top-5 accuracy, and saves proof artifacts (runtime log, retrieval visualization, and CSV results). This project demonstrates meaningful GPU acceleration for a practical nearest-neighbor search workflow.
