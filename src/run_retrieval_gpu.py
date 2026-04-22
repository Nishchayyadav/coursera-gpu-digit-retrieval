#!/usr/bin/env python3
"""Run GPU-based digit image retrieval and generate proof artifacts."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from retrieval_utils import (
    RetrievalStats,
    compute_metrics,
    load_digits_tensors,
    retrieve_topk_gpu,
    save_results_csv,
    save_retrieval_visualization,
    split_query_gallery,
    write_run_log,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="GPU digit retrieval with cosine similarity.")
    parser.add_argument("--output-dir", type=Path, default=Path("proof"), help="Output directory for artifacts.")
    parser.add_argument("--max-images", type=int, default=1797, help="Maximum number of images to load.")
    parser.add_argument("--query-count", type=int, default=300, help="Number of query images.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of nearest neighbors to retrieve.")
    parser.add_argument("--batch-size", type=int, default=128, help="Query batch size for similarity search.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split reproducibility.")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU fallback for local debugging (not valid for final proof).",
    )
    return parser.parse_args()


def main() -> int:
    """Execute retrieval pipeline and write artifacts."""
    args = parse_args()

    has_cuda = torch.cuda.is_available()
    if not has_cuda and not args.allow_cpu:
        print("CUDA GPU is required for assignment proof. Use --allow-cpu only for debugging.")
        return 1

    device = torch.device("cuda" if has_cuda else "cpu")
    device_name = torch.cuda.get_device_name(0) if has_cuda else "CPU (debug mode)"

    images, labels = load_digits_tensors(max_images=args.max_images)

    (
        query_images,
        query_labels,
        gallery_images,
        gallery_labels,
        query_indices,
        gallery_indices,
    ) = split_query_gallery(images=images, labels=labels, query_count=args.query_count, seed=args.seed)

    start = time.perf_counter()
    topk_indices, topk_scores = retrieve_topk_gpu(
        query_images=query_images,
        gallery_images=gallery_images,
        top_k=args.top_k,
        batch_size=args.batch_size,
        device=device,
    )
    if has_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    top1, top5, _ = compute_metrics(
        query_labels=query_labels,
        gallery_labels=gallery_labels,
        topk_indices=topk_indices,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = output_dir / "retrieval_samples.png"
    csv_path = output_dir / "retrieval_results.csv"
    log_path = output_dir / "run_log.txt"

    save_retrieval_visualization(
        query_images=query_images,
        query_labels=query_labels,
        gallery_images=gallery_images,
        gallery_labels=gallery_labels,
        topk_indices=topk_indices,
        output_path=image_path,
        sample_count=6,
        neighbors_to_show=min(3, args.top_k),
    )

    save_results_csv(
        query_indices=query_indices,
        query_labels=query_labels,
        topk_indices=topk_indices,
        topk_scores=topk_scores,
        gallery_indices=gallery_indices,
        gallery_labels=gallery_labels,
        output_path=csv_path,
    )

    queries_per_second = query_images.size(0) / elapsed if elapsed > 0 else 0.0
    stats = RetrievalStats(
        device_name=device_name,
        gpu_enabled=has_cuda,
        query_count=int(query_images.size(0)),
        gallery_count=int(gallery_images.size(0)),
        top1_accuracy=top1,
        top5_accuracy=top5,
        elapsed_seconds=elapsed,
        queries_per_second=queries_per_second,
    )

    write_run_log(
        log_path=log_path,
        stats=stats,
        extra={
            "Dataset": "sklearn.datasets.load_digits",
            "Similarity": "Cosine (GPU matrix multiply)",
            "Output image": str(image_path),
            "Output CSV": str(csv_path),
        },
    )

    print(f"Top-1 accuracy: {top1:.4f}")
    print(f"Top-5 accuracy: {top5:.4f}")
    print(f"Saved: {image_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {log_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
