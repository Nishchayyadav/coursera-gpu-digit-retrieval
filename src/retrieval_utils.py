"""Utilities for GPU-based digit image retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as f
from sklearn.datasets import load_digits


@dataclass
class RetrievalStats:
    """Summary stats for one retrieval run."""

    device_name: str
    gpu_enabled: bool
    query_count: int
    gallery_count: int
    top1_accuracy: float
    top5_accuracy: float
    elapsed_seconds: float
    queries_per_second: float


def load_digits_tensors(max_images: int = 1797) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load digits images and labels as tensors.

    Returns:
        images: Tensor [N, 1, 8, 8] in float32 range [0, 1]
        labels: Tensor [N]
    """
    digits = load_digits()
    images_np = digits.images.astype(np.float32) / 16.0
    labels_np = digits.target.astype(np.int64)

    images_np = images_np[:max_images]
    labels_np = labels_np[:max_images]

    images = torch.from_numpy(images_np).unsqueeze(1)
    labels = torch.from_numpy(labels_np)
    return images, labels


def split_query_gallery(
    images: torch.Tensor,
    labels: torch.Tensor,
    query_count: int,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split images and labels into query and gallery sets with shuffled indices."""
    total = images.size(0)
    if query_count <= 0 or query_count >= total:
        raise ValueError("query_count must be > 0 and < total number of images")

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(total, generator=generator)

    query_indices = permutation[:query_count]
    gallery_indices = permutation[query_count:]

    query_images = images[query_indices]
    query_labels = labels[query_indices]
    gallery_images = images[gallery_indices]
    gallery_labels = labels[gallery_indices]

    return query_images, query_labels, gallery_images, gallery_labels, query_indices, gallery_indices


def _flatten_and_normalize(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Flatten [N,1,H,W] -> [N,D] and l2-normalize for cosine similarity."""
    vectors = images.view(images.size(0), -1).to(device, non_blocking=True)
    vectors = f.normalize(vectors, p=2, dim=1)
    return vectors


def retrieve_topk_gpu(
    query_images: torch.Tensor,
    gallery_images: torch.Tensor,
    top_k: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute top-k nearest neighbors for query images from gallery using cosine similarity on GPU.

    Returns:
        topk_indices: [Q, K] indices into gallery set
        topk_scores: [Q, K] cosine similarity scores
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    gallery_vectors = _flatten_and_normalize(gallery_images, device=device)

    top_indices_chunks = []
    top_scores_chunks = []

    for start in range(0, query_images.size(0), batch_size):
        end = start + batch_size
        query_vectors = _flatten_and_normalize(query_images[start:end], device=device)

        similarities = torch.matmul(query_vectors, gallery_vectors.T)
        scores, indices = torch.topk(similarities, k=top_k, dim=1, largest=True, sorted=True)

        top_indices_chunks.append(indices.detach().cpu())
        top_scores_chunks.append(scores.detach().cpu())

    return torch.cat(top_indices_chunks, dim=0), torch.cat(top_scores_chunks, dim=0)


def compute_metrics(
    query_labels: torch.Tensor,
    gallery_labels: torch.Tensor,
    topk_indices: torch.Tensor,
) -> Tuple[float, float, torch.Tensor]:
    """Compute top-1 and top-5 accuracy from retrieval indices."""
    predicted_labels = gallery_labels[topk_indices]

    top1 = (predicted_labels[:, 0] == query_labels).float().mean().item()

    k_for_top5 = min(5, predicted_labels.size(1))
    top5_hits = (predicted_labels[:, :k_for_top5] == query_labels.unsqueeze(1)).any(dim=1)
    top5 = top5_hits.float().mean().item()

    return top1, top5, predicted_labels


def save_retrieval_visualization(
    query_images: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_images: torch.Tensor,
    gallery_labels: torch.Tensor,
    topk_indices: torch.Tensor,
    output_path: Path,
    sample_count: int = 6,
    neighbors_to_show: int = 3,
) -> None:
    """Save a figure with query images and top retrieved neighbors."""
    sample_count = min(sample_count, query_images.size(0))
    neighbors_to_show = min(neighbors_to_show, topk_indices.size(1))

    cols = 1 + neighbors_to_show
    fig, axes = plt.subplots(sample_count, cols, figsize=(3 * cols, 2.6 * sample_count))

    if sample_count == 1:
        axes = np.expand_dims(axes, axis=0)

    for row in range(sample_count):
        axes[row, 0].imshow(query_images[row, 0], cmap="gray")
        axes[row, 0].set_title(f"Query\nlabel={int(query_labels[row])}")
        axes[row, 0].axis("off")

        for col in range(neighbors_to_show):
            gallery_idx = int(topk_indices[row, col])
            axes[row, col + 1].imshow(gallery_images[gallery_idx, 0], cmap="gray")
            axes[row, col + 1].set_title(f"Top{col+1}\nlabel={int(gallery_labels[gallery_idx])}")
            axes[row, col + 1].axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_results_csv(
    query_indices: torch.Tensor,
    query_labels: torch.Tensor,
    topk_indices: torch.Tensor,
    topk_scores: torch.Tensor,
    gallery_indices: torch.Tensor,
    gallery_labels: torch.Tensor,
    output_path: Path,
) -> None:
    """Save retrieval rows as CSV for proof artifact."""
    rows = []
    topk = topk_indices.size(1)

    for i in range(query_indices.size(0)):
        result = {
            "query_original_index": int(query_indices[i]),
            "query_label": int(query_labels[i]),
            "pred_top1_label": int(gallery_labels[topk_indices[i, 0]]),
            "pred_top1_original_index": int(gallery_indices[topk_indices[i, 0]]),
            "top1_score": float(topk_scores[i, 0]),
        }

        for k in range(topk):
            result[f"top{k+1}_label"] = int(gallery_labels[topk_indices[i, k]])
            result[f"top{k+1}_score"] = float(topk_scores[i, k])

        rows.append(result)

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def write_run_log(log_path: Path, stats: RetrievalStats, extra: Dict[str, str]) -> None:
    """Write summary run log for proof artifact."""
    lines = [
        "GPU Digit Retrieval Run Log",
        "=" * 40,
        f"Device: {stats.device_name}",
        f"GPU enabled: {stats.gpu_enabled}",
        f"Query count: {stats.query_count}",
        f"Gallery count: {stats.gallery_count}",
        f"Top-1 accuracy: {stats.top1_accuracy:.6f}",
        f"Top-5 accuracy: {stats.top5_accuracy:.6f}",
        f"Elapsed seconds: {stats.elapsed_seconds:.6f}",
        f"Queries/second: {stats.queries_per_second:.2f}",
    ]

    for key, value in extra.items():
        lines.append(f"{key}: {value}")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
