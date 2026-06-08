"""Evaluate cmd_regime MoE specialization from collected velocity-task dataset.

Edit CONFIG values in this file, then run directly:

  cd /ws/Jaebeom/pixi-mjlab/mjlab
  uv run python src/mjlab/tasks/velocity/moe/evaluate_data.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from mjlab.tasks.velocity.moe.common import (
  REGIME_NAMES,
  build_velocity_moe_policy,
  pca_2d,
)

# -----------------------------------------------------------------------------
# User config: edit these constants directly.
# -----------------------------------------------------------------------------
CONFIG: dict[str, Any] = {
  "task_id": "Mjlab-Velocity-Flat-KIMM-P2",
  "checkpoint_file": "/log/mjlab/rsl_rl/p2_velocity/2026-04-07_19-49-57_p2_velocity_guidedMoEx2/model_29999.pt",
  "dataset_dir": "/ws/Jaebeom/pixi-mjlab/mjlab/src/mjlab/tasks/velocity/moe/output/p2_cmd_regime_collect_model29999_cmd1-20260406-143343",
  "output_dir": "/ws/Jaebeom/pixi-mjlab/mjlab/src/mjlab/tasks/velocity/moe/output/eval_1/29999",
  "device": "cuda:0",
  "num_envs_for_model_build": 16,
  "max_samples": 200_000,
  "points_per_expert": 10_000,
  "random_seed": 42,
  # If True, remove previous plot outputs in output_dir before writing new ones.
  "clean_output_dir_before_save": True,
}


def _cleanup_previous_outputs(output_dir: Path) -> None:
  """Remove previous evaluation plot files to avoid stale mixed artifacts."""
  patterns = [
    "embedding_*.png",
    "wrong_specialist_l2_heatmap.png",
  ]
  for pattern in patterns:
    for file_path in output_dir.glob(pattern):
      if file_path.is_file():
        file_path.unlink()


def _load_dataset(
  dataset_dir: Path,
  max_samples: int,
) -> dict[str, np.ndarray]:
  """Load dataset shards up to max_samples."""
  part_paths = sorted(dataset_dir.glob("part-*.npz"))
  if not part_paths:
    raise FileNotFoundError(f"No dataset shards found under: {dataset_dir}")

  buffers: dict[str, list[np.ndarray]] = {
    "flat_obs": [],
    "regime_id": [],
    "full_action": [],
  }

  remaining = max_samples
  for part_path in part_paths:
    with np.load(part_path) as part:
      n = int(part["regime_id"].shape[0])
      if n == 0:
        continue
      take = n if remaining <= 0 else min(remaining, n)
      if take <= 0:
        break
      for key in buffers:
        if key not in part:
          raise KeyError(f"Missing key '{key}' in dataset shard: {part_path}")
        buffers[key].append(part[key][:take])

      remaining -= take
      if remaining == 0:
        break

  data = {
    key: np.concatenate(value, axis=0)
    for key, value in buffers.items()
    if len(value) > 0
  }
  if len(data) == 0:
    raise RuntimeError("Loaded dataset is empty.")
  return data


def _feature_to_action(policy, features: torch.Tensor) -> torch.Tensor:
  """Apply MoE post layers and deterministic policy head."""
  x = policy.mlp.post_activation(features)
  x = policy.mlp.output(x)
  if policy.distribution is not None:
    return policy.distribution.deterministic_output(x)
  return x


def _sample_indices_regime_balanced(
  regime_ids: np.ndarray,
  sample_n: int,
  rng: np.random.Generator,
) -> np.ndarray:
  """Sample indices with near-uniform regime coverage, then fill remainder globally."""
  total_n = int(regime_ids.shape[0])
  sample_n = min(int(sample_n), total_n)
  if sample_n <= 0:
    return np.empty((0,), dtype=np.int64)

  num_regimes = len(REGIME_NAMES)
  per_regime_target = sample_n // num_regimes
  picked: list[np.ndarray] = []
  picked_mask = np.zeros(total_n, dtype=bool)

  # Phase 1: equal quota per regime.
  if per_regime_target > 0:
    for rid in range(num_regimes):
      rid_idx = np.flatnonzero(regime_ids == rid)
      if rid_idx.size == 0:
        continue
      take = min(per_regime_target, rid_idx.size)
      chosen = rng.choice(rid_idx, size=take, replace=False).astype(np.int64)
      picked.append(chosen)
      picked_mask[chosen] = True

  selected = (
    np.concatenate(picked, axis=0) if picked else np.empty((0,), dtype=np.int64)
  )

  # Phase 2: fill the remainder from all regimes.
  remaining = sample_n - int(selected.shape[0])
  if remaining > 0:
    pool = np.flatnonzero(~picked_mask)
    if pool.size > 0:
      extra = rng.choice(pool, size=min(remaining, pool.size), replace=False).astype(
        np.int64
      )
      selected = np.concatenate([selected, extra], axis=0)

  rng.shuffle(selected)
  return selected


def _compute_group_separation_stats(
  label_points: dict[str, np.ndarray],
) -> dict[str, Any]:
  """Compute centroid-based inter/intra separation in original feature space."""
  labels = list(label_points.keys())
  if len(labels) == 0:
    return {
      "num_groups": 0,
      "inter_centroid_distance": None,
      "intra_radius": None,
      "separation_ratio": None,
      "separation_margin": None,
    }

  centroids = {label: label_points[label].mean(axis=0) for label in labels}
  intra_per_group = {
    label: float(np.linalg.norm(label_points[label] - centroids[label], axis=1).mean())
    for label in labels
  }
  intra_mean = float(np.mean(list(intra_per_group.values())))

  offdiag: list[float] = []
  for i, a in enumerate(labels):
    for j, b in enumerate(labels):
      if i >= j:
        continue
      offdiag.append(float(np.linalg.norm(centroids[a] - centroids[b])))

  if len(offdiag) == 0:
    return {
      "num_groups": len(labels),
      "intra_radius": intra_mean,
      "intra_radius_by_group": intra_per_group,
      "inter_centroid_distance": None,
      "separation_ratio": None,
      "separation_margin": None,
    }

  inter_mean = float(np.mean(offdiag))
  return {
    "num_groups": len(labels),
    "intra_radius": intra_mean,
    "intra_radius_by_group": intra_per_group,
    "inter_centroid_distance": {
      "mean": inter_mean,
      "min": float(np.min(offdiag)),
      "max": float(np.max(offdiag)),
    },
    "separation_ratio": float(inter_mean / max(intra_mean, 1e-8)),
    "separation_margin": float(inter_mean - intra_mean),
  }


def _compute_shared_regime_separation_stats(
  shared_label_points: dict[str, np.ndarray],
  shared_label_regimes: dict[str, np.ndarray],
) -> dict[str, Any]:
  """Compute regime-level separation stats using pooled shared embeddings."""
  regime_points: dict[str, list[np.ndarray]] = {name: [] for name in REGIME_NAMES}
  for label, points in shared_label_points.items():
    regimes = shared_label_regimes[label]
    for rid, name in enumerate(REGIME_NAMES):
      mask = regimes == rid
      if np.any(mask):
        regime_points[name].append(points[mask])

  pooled: dict[str, np.ndarray] = {}
  for name, chunks in regime_points.items():
    if len(chunks) > 0:
      pooled[name] = np.concatenate(chunks, axis=0)
  return _compute_group_separation_stats(pooled)


def _label_plot_style(label: str) -> tuple[str, str]:
  """Return deterministic (color, marker) for a label."""
  regime_colors = {
    "stand": "#4e4e4e",
    "turn": "#d62728",
    "vx": "#1f77b4",
    "vy": "#2ca02c",
  }
  shared_color = "#111111"
  expert_palette = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
  ]
  expert_markers = ["^", "s", "D", "P", "X", "v", "<", ">", "*", "h"]
  shared_markers = ["o", "8", "p", "H"]

  if label in regime_colors:
    return regime_colors[label], "o"
  if label.startswith("shared"):
    idx = int(label.replace("shared", "") or "0")
    return shared_color, shared_markers[idx % len(shared_markers)]
  if label.startswith("expert"):
    idx = int(label.replace("expert", "") or "0")
    return expert_palette[idx % len(expert_palette)], expert_markers[
      idx % len(expert_markers)
    ]
  return "#1f77b4", "o"


def _regime_plot_style(regime_id: int) -> tuple[str, str]:
  """Return deterministic (color, marker) for regime plotting."""
  regime_styles = {
    0: ("#4e4e4e", "o"),  # stand
    1: ("#d62728", "^"),  # turn
    2: ("#1f77b4", "s"),  # vx
    3: ("#2ca02c", "D"),  # vy
  }
  return regime_styles.get(regime_id, ("#1f77b4", "o"))


def _save_embedding_plots(
  output_dir: Path,
  *,
  label_points: dict[str, np.ndarray],
  label_regimes: dict[str, np.ndarray],
  file_suffix: str = "",
  title_prefix: str = "MoE Expert",
  save_all_plot: bool = True,
  save_by_regime_plot: bool = True,
  all_plot_color_by_regime: bool = False,
  save_centroid_heatmap: bool = True,
  regime_allowed_labels: dict[int, set[str]] | None = None,
) -> dict[str, list[float]]:
  """Create PCA embedding plots with shared legend at the bottom."""
  labels = list(label_points.keys())
  if len(labels) == 0:
    raise ValueError("label_points is empty; cannot draw embedding plots.")
  all_points = np.concatenate([label_points[label] for label in labels], axis=0)
  all_xy, _, _ = pca_2d(all_points)

  label_xy: dict[str, np.ndarray] = {}
  cursor = 0
  for label in labels:
    n = label_points[label].shape[0]
    label_xy[label] = all_xy[cursor : cursor + n]
    cursor += n

  label_to_style = {label: _label_plot_style(label) for label in labels}
  suffix = f"_{file_suffix}" if file_suffix else ""
  legend_handles = [
    Line2D(
      [0],
      [0],
      marker=label_to_style[label][1],
      linestyle="",
      markersize=6,
      markerfacecolor=label_to_style[label][0],
      markeredgecolor="none",
      label=label,
    )
    for label in labels
  ]
  regime_handles = [
    Line2D(
      [0],
      [0],
      marker=_regime_plot_style(rid)[1],
      linestyle="",
      markersize=6,
      markerfacecolor=_regime_plot_style(rid)[0],
      markeredgecolor="none",
      label=REGIME_NAMES[rid],
    )
    for rid in range(len(REGIME_NAMES))
  ]

  if save_all_plot:
    fig, ax = plt.subplots(figsize=(10, 8))
    if all_plot_color_by_regime:
      all_xy_plot = np.concatenate([label_xy[label] for label in labels], axis=0)
      all_regimes_plot = np.concatenate(
        [label_regimes[label] for label in labels], axis=0
      )
      for rid in range(len(REGIME_NAMES)):
        mask = all_regimes_plot == rid
        if not np.any(mask):
          continue
        ax.scatter(
          all_xy_plot[mask, 0],
          all_xy_plot[mask, 1],
          s=10,
          alpha=0.45,
          color=_regime_plot_style(rid)[0],
          marker=_regime_plot_style(rid)[1],
          edgecolors="none",
        )
    else:
      for label in labels:
        xy = label_xy[label]
        ax.scatter(
          xy[:, 0],
          xy[:, 1],
          s=10,
          alpha=0.45,
          color=label_to_style[label][0],
          marker=label_to_style[label][1],
          edgecolors="none",
        )
    ax.set_title(f"{title_prefix} Embeddings (All Regimes, PCA 2D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.2)
    fig.legend(
      handles=(regime_handles if all_plot_color_by_regime else legend_handles),
      loc="lower center",
      bbox_to_anchor=(0.5, -0.01),
      ncol=min(
        max(len(REGIME_NAMES) if all_plot_color_by_regime else len(labels), 1), 6
      ),
      frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
    fig.savefig(output_dir / f"embedding_scatter_all{suffix}.png", dpi=180)
    plt.close(fig)

  if save_by_regime_plot:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    for regime_id, regime_name in enumerate(REGIME_NAMES):
      ax = axes[regime_id // 2, regime_id % 2]
      for label in labels:
        if regime_allowed_labels is not None and label not in regime_allowed_labels.get(
          regime_id, set()
        ):
          continue
        xy = label_xy[label]
        regimes = label_regimes[label]
        mask = regimes == regime_id
        if not np.any(mask):
          continue
        ax.scatter(
          xy[mask, 0],
          xy[mask, 1],
          s=10,
          alpha=0.45,
          color=label_to_style[label][0],
          marker=label_to_style[label][1],
          edgecolors="none",
        )
      ax.set_title(f"Regime: {regime_name}")
      ax.grid(alpha=0.2)
    fig.suptitle(f"{title_prefix} Embeddings by Regime (PCA 2D)")
    fig.legend(
      handles=legend_handles,
      loc="lower center",
      bbox_to_anchor=(0.5, -0.01),
      ncol=min(max(len(labels), 1), 6),
      frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.96))
    fig.savefig(output_dir / f"embedding_scatter_by_regime{suffix}.png", dpi=180)
    plt.close(fig)

  # Plot 3: centroid distance heatmap in PCA space.
  centroids = {label: label_xy[label].mean(axis=0) for label in labels}
  dist = np.zeros((len(labels), len(labels)), dtype=np.float64)
  for i, a in enumerate(labels):
    for j, b in enumerate(labels):
      dist[i, j] = float(np.linalg.norm(centroids[a] - centroids[b]))

  if save_centroid_heatmap:
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(dist, cmap="magma", aspect="auto")
    ax.set_title(f"PCA Centroid Distance ({title_prefix})")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(
      output_dir / f"embedding_centroid_distance_heatmap{suffix}.png", dpi=180
    )
    plt.close(fig)

  return {label: centroids[label].tolist() for label in labels}


def _save_wrong_specialist_heatmap(
  output_dir: Path,
  wrong_specialist_l2: np.ndarray,
  specialist_labels: list[str],
) -> None:
  """Save heatmap for wrong-specialist action deviation."""
  fig, ax = plt.subplots(figsize=(max(8, len(specialist_labels) * 0.6), 4.8))
  im = ax.imshow(wrong_specialist_l2, cmap="viridis", aspect="auto")
  ax.set_title("Wrong Specialist Action Distance (L2)")
  ax.set_xlabel("Forced specialist")
  ax.set_ylabel("Regime")
  ax.set_xticks(np.arange(len(specialist_labels)))
  ax.set_xticklabels(specialist_labels, rotation=45, ha="right")
  ax.set_yticks(np.arange(len(REGIME_NAMES)))
  ax.set_yticklabels(REGIME_NAMES)
  fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
  fig.tight_layout()
  fig.savefig(output_dir / "wrong_specialist_l2_heatmap.png", dpi=180)
  plt.close(fig)


def main() -> None:
  # Import side-effect task registration.
  import mjlab.tasks  # noqa: F401

  task_id = str(CONFIG["task_id"])
  checkpoint_file = str(CONFIG["checkpoint_file"])
  dataset_dir = Path(str(CONFIG["dataset_dir"]))
  output_dir = Path(str(CONFIG["output_dir"]))
  device = str(CONFIG["device"])
  num_envs_for_model_build = int(CONFIG["num_envs_for_model_build"])
  max_samples = int(CONFIG["max_samples"])
  points_per_expert = int(CONFIG["points_per_expert"])
  random_seed = int(CONFIG["random_seed"])
  clean_output_dir_before_save = bool(CONFIG["clean_output_dir_before_save"])

  if checkpoint_file.startswith("/path/to/"):
    raise ValueError("Set CONFIG['checkpoint_file'] to a real checkpoint path.")
  checkpoint_path = Path(checkpoint_file)
  if not checkpoint_path.exists():
    raise FileNotFoundError(f"checkpoint_file does not exist: {checkpoint_path}")
  if checkpoint_path.suffix.lower() != ".pt":
    raise ValueError(
      f"checkpoint_file must be a .pt checkpoint, got: {checkpoint_path.name}"
    )
  if not dataset_dir.exists():
    raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
  if max_samples <= 0:
    raise ValueError(f"max_samples must be positive, got {max_samples}.")

  output_dir.mkdir(parents=True, exist_ok=True)
  if clean_output_dir_before_save:
    _cleanup_previous_outputs(output_dir)

  data = _load_dataset(dataset_dir=dataset_dir, max_samples=max_samples)
  n_samples = int(data["flat_obs"].shape[0])
  print(f"[eval] loaded {n_samples} samples")

  vec_env, _, policy = build_velocity_moe_policy(
    task_id=task_id,
    checkpoint_file=checkpoint_file,
    device=device,
    num_envs=num_envs_for_model_build,
    play=True,
    command_overrides=None,
  )
  vec_env.close()

  flat_obs = torch.from_numpy(data["flat_obs"]).to(device=device, dtype=torch.float32)
  regime_id = torch.from_numpy(data["regime_id"]).to(device=device, dtype=torch.long)
  full_action = torch.from_numpy(data["full_action"]).to(
    device=device, dtype=torch.float32
  )

  rng = np.random.default_rng(random_seed)

  with torch.no_grad():
    latent = policy.obs_normalizer(flat_obs)
    trunk = policy.mlp.pre_mlp(latent)

    shared_each: torch.Tensor | None = None
    if policy.moe.num_shared_experts > 0:
      shared_each = torch.stack(
        [expert(trunk) for expert in policy.moe.shared_experts], dim=1
      )

    specialist_each = torch.stack(
      [expert(trunk) for expert in policy.moe.experts], dim=1
    )

    shared_mean: torch.Tensor | None = None
    if shared_each is not None:
      shared_mean = shared_each.mean(dim=1)

    specialist_per_regime = policy.moe.num_specialist_experts // 4
    feature_dim = specialist_each.shape[-1]

    correct_specialist = torch.zeros(
      (n_samples, feature_dim),
      device=device,
      dtype=specialist_each.dtype,
    )

    for rid in range(len(REGIME_NAMES)):
      block_start = rid * specialist_per_regime
      block_end = block_start + specialist_per_regime
      block_feature = specialist_each[:, block_start:block_end, :].mean(dim=1)
      mask = regime_id == rid
      if torch.any(mask):
        correct_specialist[mask] = block_feature[mask]

    if shared_mean is not None:
      correct_feature = 0.5 * (shared_mean + correct_specialist)
    else:
      correct_feature = correct_specialist

    correct_action = _feature_to_action(policy, correct_feature)

    wrong_specialist_l2 = np.full(
      (len(REGIME_NAMES), policy.moe.num_specialist_experts),
      np.nan,
      dtype=np.float64,
    )

    for expert_idx in range(policy.moe.num_specialist_experts):
      forced_feature = specialist_each[:, expert_idx, :]
      if shared_mean is not None:
        forced_feature = 0.5 * (shared_mean + forced_feature)
      forced_action = _feature_to_action(policy, forced_feature)
      diff = torch.linalg.vector_norm(forced_action - correct_action, dim=-1)

      for rid in range(len(REGIME_NAMES)):
        mask = regime_id == rid
        if torch.any(mask):
          wrong_specialist_l2[rid, expert_idx] = float(diff[mask].mean().item())

    shared_l2 = [None for _ in REGIME_NAMES]
    if shared_mean is not None:
      shared_action = _feature_to_action(policy, shared_mean)
      shared_diff = torch.linalg.vector_norm(shared_action - correct_action, dim=-1)
      for rid in range(len(REGIME_NAMES)):
        mask = regime_id == rid
        if torch.any(mask):
          shared_l2[rid] = float(shared_diff[mask].mean().item())

    full_l2 = [None for _ in REGIME_NAMES]
    full_diff = torch.linalg.vector_norm(full_action - correct_action, dim=-1)
    for rid in range(len(REGIME_NAMES)):
      mask = regime_id == rid
      if torch.any(mask):
        full_l2[rid] = float(full_diff[mask].mean().item())

    # Build regime-balanced sampled embedding points for plotting.
    shared_label_points: dict[str, np.ndarray] = {}
    shared_label_regimes: dict[str, np.ndarray] = {}
    specialist_label_points: dict[str, np.ndarray] = {}
    specialist_label_regimes: dict[str, np.ndarray] = {}
    regime_ids_np = data["regime_id"]

    if shared_each is not None:
      for i in range(policy.moe.num_shared_experts):
        feat = shared_each[:, i, :].detach().cpu().numpy()
        sample_n = min(points_per_expert, feat.shape[0])
        idx = _sample_indices_regime_balanced(regime_ids_np, sample_n, rng)
        label = f"shared{i}"
        shared_label_points[label] = feat[idx]
        shared_label_regimes[label] = regime_ids_np[idx]

    for i in range(policy.moe.num_specialist_experts):
      feat = specialist_each[:, i, :].detach().cpu().numpy()
      sample_n = min(points_per_expert, feat.shape[0])
      idx = _sample_indices_regime_balanced(regime_ids_np, sample_n, rng)
      label = f"expert{i}"
      specialist_label_points[label] = feat[idx]
      specialist_label_regimes[label] = regime_ids_np[idx]

  combined_label_points = {**shared_label_points, **specialist_label_points}
  combined_label_regimes = {**shared_label_regimes, **specialist_label_regimes}
  specialist_matched_label_points: dict[str, np.ndarray] = {}
  specialist_matched_label_regimes: dict[str, np.ndarray] = {}

  correct_specialist_np = correct_specialist.detach().cpu().numpy()
  for rid, regime_name in enumerate(REGIME_NAMES):
    rid_idx_all = np.flatnonzero(regime_ids_np == rid)
    if rid_idx_all.size == 0:
      continue
    take = min(points_per_expert, int(rid_idx_all.size))
    rid_idx = rng.choice(rid_idx_all, size=take, replace=False).astype(np.int64)
    specialist_matched_label_points[regime_name] = correct_specialist_np[rid_idx]
    specialist_matched_label_regimes[regime_name] = regime_ids_np[rid_idx]

  centroids_2d_shared_only: dict[str, list[float]] = {}
  centroid_stats_shared_by_regime: dict[str, Any] = {}
  shared_labels = list(shared_label_points.keys())
  combined_regime_allowed_labels: dict[int, set[str]] = {}
  for rid in range(len(REGIME_NAMES)):
    block_start = rid * specialist_per_regime
    block_end = block_start + specialist_per_regime
    allowed = set(shared_labels)
    for eid in range(block_start, block_end):
      allowed.add(f"expert{eid}")
    combined_regime_allowed_labels[rid] = allowed

  if len(shared_label_points) > 0:
    centroids_2d_shared_only = _save_embedding_plots(
      output_dir,
      label_points=shared_label_points,
      label_regimes=shared_label_regimes,
      file_suffix="shared_only",
      title_prefix="Shared-only",
      save_all_plot=True,
      save_by_regime_plot=False,
      all_plot_color_by_regime=True,
      save_centroid_heatmap=False,
    )
    centroid_stats_shared_by_regime = _compute_shared_regime_separation_stats(
      shared_label_points,
      shared_label_regimes,
    )

  centroids_2d_combined = _save_embedding_plots(
    output_dir,
    label_points=combined_label_points,
    label_regimes=combined_label_regimes,
    file_suffix="combined",
    title_prefix="Shared + Specialist",
    save_all_plot=False,
    save_by_regime_plot=True,
    all_plot_color_by_regime=False,
    save_centroid_heatmap=True,
    regime_allowed_labels=combined_regime_allowed_labels,
  )

  specialist_centroids_2d = _save_embedding_plots(
    output_dir,
    label_points=specialist_label_points,
    label_regimes=specialist_label_regimes,
    file_suffix="specialist_only",
    title_prefix="Specialist-only",
    save_all_plot=False,
    save_by_regime_plot=True,
    all_plot_color_by_regime=False,
    save_centroid_heatmap=True,
  )
  specialist_matched_centroids_2d = _save_embedding_plots(
    output_dir,
    label_points=specialist_matched_label_points,
    label_regimes=specialist_matched_label_regimes,
    file_suffix="specialist_matched_by_regime",
    title_prefix="Specialist (Matched by Regime)",
    save_all_plot=True,
    save_by_regime_plot=False,
    all_plot_color_by_regime=False,
    save_centroid_heatmap=False,
  )

  centroid_stats_combined = _compute_group_separation_stats(combined_label_points)
  centroid_stats_specialist_only = _compute_group_separation_stats(
    specialist_label_points
  )
  centroid_stats_specialist_matched_by_regime = _compute_group_separation_stats(
    specialist_matched_label_points
  )

  specialist_labels = [f"expert{i}" for i in range(policy.moe.num_specialist_experts)]
  _save_wrong_specialist_heatmap(
    output_dir,
    wrong_specialist_l2=wrong_specialist_l2,
    specialist_labels=specialist_labels,
  )

  regime_counts = {
    REGIME_NAMES[i]: int((data["regime_id"] == i).sum())
    for i in range(len(REGIME_NAMES))
  }

  metrics = {
    "task_id": task_id,
    "checkpoint_file": checkpoint_file,
    "dataset_dir": str(dataset_dir),
    "num_samples": n_samples,
    "regime_counts": regime_counts,
    "num_shared_experts": int(policy.moe.num_shared_experts),
    "num_specialist_experts": int(policy.moe.num_specialist_experts),
    "wrong_specialist_l2_by_regime": {
      REGIME_NAMES[rid]: {
        f"expert{eid}": (
          None
          if np.isnan(wrong_specialist_l2[rid, eid])
          else float(wrong_specialist_l2[rid, eid])
        )
        for eid in range(policy.moe.num_specialist_experts)
      }
      for rid in range(len(REGIME_NAMES))
    },
    "shared_only_l2_by_regime": {
      REGIME_NAMES[i]: shared_l2[i] for i in range(len(REGIME_NAMES))
    },
    "full_vs_correct_l2_by_regime": {
      REGIME_NAMES[i]: full_l2[i] for i in range(len(REGIME_NAMES))
    },
    "pca_centroids_2d": centroids_2d_combined,
    "pca_centroids_2d_shared_only": centroids_2d_shared_only,
    "pca_centroids_2d_combined": centroids_2d_combined,
    "pca_centroids_2d_specialist_only": specialist_centroids_2d,
    "pca_centroids_2d_specialist_matched_by_regime": specialist_matched_centroids_2d,
    "centroid_stats_combined": centroid_stats_combined,
    "centroid_stats_specialist_only": centroid_stats_specialist_only,
    "centroid_stats_specialist_matched_by_regime": centroid_stats_specialist_matched_by_regime,
    "centroid_stats_shared_by_regime": centroid_stats_shared_by_regime,
  }

  metrics_path = output_dir / "metrics.json"
  with metrics_path.open("w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

  print("[eval] done")
  print(f"[eval] output_dir: {output_dir}")
  print(f"[eval] metrics: {metrics_path}")


if __name__ == "__main__":
  main()
