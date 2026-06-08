"""Collect velocity-task cmd_regime MoE observations from simulation rollouts.

Edit CONFIG values in this file, then run directly:

  cd /ws/Jaebeom/pixi-mjlab/mjlab
  uv run python src/mjlab/tasks/velocity/moe/collect_data.py
"""

from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mjlab.tasks.velocity.moe.common import (
  REGIME_NAMES,
  build_velocity_moe_policy,
  classify_cmd_regime_from_policy,
  flatten_actor_obs,
  require_router_slice,
)

# -----------------------------------------------------------------------------
# User config: edit these constants directly.
# -----------------------------------------------------------------------------
CONFIG: dict[str, Any] = {
  "task_id": "Mjlab-Velocity-Flat-KIMM-P2",
  "checkpoint_file": "/log/mjlab/rsl_rl/p2_velocity/2026-04-06_18-49-24_p2_velocity_guidedMoE/model_5000.pt",
  "device": "cuda:0",
  "num_envs": 1024,
  "num_steps": 2000,
  "shard_size": 250_000,
  "output_root": "/ws/Jaebeom/pixi-mjlab/mjlab/src/mjlab/tasks/velocity/moe/output",
  "run_name": "p2_cmd_regime_collect_model29999_cmd1",
  # Keep task default when set to None.
  # Example:
  # "cmd_ranges": {"vx": (-1.5, 2.0), "vy": (-0.8, 0.8), "yaw": (-1.2, 1.2)}
  "cmd_ranges": {"vx": (-1.0, 1.0), "vy": (-1.0, 1.0), "yaw": (-1.0, 1.0)},
  # Optional extra override on top of task default twist cfg.
  # Nested dict keys must match twist cfg fields (e.g. "resampling_time_range",
  # "rel_standing_envs", "ranges", ...).
  "twist_cfg_override": None,
}


def _validate_range(name: str, value: Any) -> tuple[float, float]:
  if not isinstance(value, (tuple, list)) or len(value) != 2:
    raise TypeError(
      f"cmd_ranges['{name}'] must be a (min, max) tuple/list, got {value!r}."
    )
  lo = float(value[0])
  hi = float(value[1])
  if lo > hi:
    raise ValueError(f"cmd_ranges['{name}'] min must be <= max, got ({lo}, {hi}).")
  return (lo, hi)


def _build_twist_overrides_from_config() -> dict[str, Any] | Any | None:
  """Build twist command overrides from CONFIG.

  Behavior:
  - Starts from `CONFIG['twist_cfg_override']` (if dict, copied; if object, used as-is).
  - Applies `CONFIG['cmd_ranges']` onto `ranges.lin_vel_x/lin_vel_y/ang_vel_z`.
  """
  twist_override = CONFIG.get("twist_cfg_override", None)
  cmd_ranges = CONFIG.get("cmd_ranges", None)

  if isinstance(twist_override, dict):
    merged: dict[str, Any] = copy.deepcopy(twist_override)
  elif twist_override is None:
    merged = {}
  else:
    if cmd_ranges is not None:
      raise ValueError(
        "When twist_cfg_override is a non-dict object replacement, "
        "cmd_ranges cannot be applied. Set cmd_ranges=None or pass a dict override."
      )
    return twist_override

  if cmd_ranges is not None:
    if not isinstance(cmd_ranges, dict):
      raise TypeError(
        f"cmd_ranges must be a dict or None, got {type(cmd_ranges).__name__}."
      )
    mapped = {
      "vx": "lin_vel_x",
      "vy": "lin_vel_y",
      "yaw": "ang_vel_z",
    }
    ranges_override = merged.setdefault("ranges", {})
    if not isinstance(ranges_override, dict):
      raise TypeError(
        "twist_cfg_override['ranges'] must be a dict when cmd_ranges is used."
      )
    for src_key, dst_key in mapped.items():
      if src_key not in cmd_ranges or cmd_ranges[src_key] is None:
        continue
      ranges_override[dst_key] = _validate_range(src_key, cmd_ranges[src_key])

  return merged if len(merged) > 0 else None


def _jsonable_or_repr(value: Any) -> Any:
  try:
    json.dumps(value)
    return value
  except TypeError:
    return repr(value)


def _extract_effective_cmd_ranges(vec_env: Any) -> dict[str, list[float]] | None:
  """Read applied cmd ranges from env cfg after overrides."""
  try:
    twist_cfg = vec_env.unwrapped.cfg.commands["twist"]
    return {
      "vx": [
        float(twist_cfg.ranges.lin_vel_x[0]),
        float(twist_cfg.ranges.lin_vel_x[1]),
      ],
      "vy": [
        float(twist_cfg.ranges.lin_vel_y[0]),
        float(twist_cfg.ranges.lin_vel_y[1]),
      ],
      "yaw": [
        float(twist_cfg.ranges.ang_vel_z[0]),
        float(twist_cfg.ranges.ang_vel_z[1]),
      ],
    }
  except Exception:
    return None


def _build_onnx_session(
  *,
  onnx_path: Path,
  provider_order: list[str] | None = None,
) -> tuple[Any, str, list[str], int | None]:
  """Create ONNX Runtime session and return (session, input_name, providers)."""
  try:
    import onnxruntime as ort
  except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
      "onnxruntime is required for ONNX collection mode. "
      "Install it in the current environment first."
    ) from exc

  if not onnx_path.exists():
    raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

  available = list(ort.get_available_providers())
  preferred = (
    provider_order
    if provider_order is not None
    else ["CUDAExecutionProvider", "CPUExecutionProvider"]
  )
  providers = [name for name in preferred if name in available]
  if len(providers) == 0:
    raise RuntimeError(
      "No requested ONNX Runtime providers are available. "
      f"requested={preferred}, available={available}"
    )

  session = ort.InferenceSession(str(onnx_path), providers=providers)
  inputs = session.get_inputs()
  if len(inputs) != 1:
    raise RuntimeError(
      f"Expected exactly one ONNX input for policy, got {len(inputs)}."
    )
  input_name = str(inputs[0].name)
  input_shape = inputs[0].shape
  expected_batch_size: int | None = None
  if isinstance(input_shape, list) and len(input_shape) > 0:
    batch_dim = input_shape[0]
    if isinstance(batch_dim, int):
      expected_batch_size = int(batch_dim)
  return session, input_name, providers, expected_batch_size


def _infer_actions_onnx(
  *,
  session: Any,
  input_name: str,
  flat_obs: torch.Tensor,
  device: str,
  expected_batch_size: int | None = None,
) -> torch.Tensor:
  """Run ONNX policy and return action tensor on target device."""
  obs_np = flat_obs.detach().cpu().numpy().astype(np.float32, copy=False)
  if expected_batch_size is None or expected_batch_size == obs_np.shape[0]:
    [actions_np] = session.run(None, {input_name: obs_np})
    return torch.from_numpy(actions_np).to(device=device, dtype=torch.float32)

  if expected_batch_size <= 0:
    raise RuntimeError(f"Invalid ONNX expected batch size: {expected_batch_size}")

  # Fallback for fixed-batch ONNX models (e.g. batch=1 export).
  # Run chunked inference and stitch back.
  parts: list[np.ndarray] = []
  for start in range(0, obs_np.shape[0], expected_batch_size):
    end = min(start + expected_batch_size, obs_np.shape[0])
    chunk = obs_np[start:end]
    if chunk.shape[0] < expected_batch_size:
      # Pad only the last chunk to satisfy fixed-batch input.
      pad_count = expected_batch_size - chunk.shape[0]
      pad = np.repeat(chunk[-1:, :], pad_count, axis=0)
      chunk = np.concatenate([chunk, pad], axis=0)
      [chunk_out] = session.run(None, {input_name: chunk})
      parts.append(chunk_out[: end - start])
    else:
      [chunk_out] = session.run(None, {input_name: chunk})
      parts.append(chunk_out)

  actions_np = np.concatenate(parts, axis=0)
  return torch.from_numpy(actions_np).to(device=device, dtype=torch.float32)


def _export_policy_to_onnx_dynamic_batch(
  *,
  policy: Any,
  onnx_path: Path,
) -> None:
  """Export ONNX policy with dynamic batch axis for vectorized rollout collection."""
  onnx_model = policy.as_onnx(verbose=False)
  onnx_model.to("cpu")
  onnx_model.eval()
  onnx_path.parent.mkdir(parents=True, exist_ok=True)
  torch.onnx.export(
    onnx_model,
    onnx_model.get_dummy_inputs(),
    str(onnx_path),
    export_params=True,
    opset_version=18,
    verbose=False,
    input_names=onnx_model.input_names,
    output_names=onnx_model.output_names,
    dynamic_axes={
      str(onnx_model.input_names[0]): {0: "batch"},
      str(onnx_model.output_names[0]): {0: "batch"},
    },
    dynamo=False,
  )


def _flush_part(
  *,
  output_dir: Path,
  part_idx: int,
  buffers: dict[str, list[np.ndarray]],
) -> int:
  """Flush in-memory buffers to one compressed shard file."""
  if len(buffers["flat_obs"]) == 0:
    return 0

  batch = {key: np.concatenate(value, axis=0) for key, value in buffers.items()}
  sample_count = int(batch["regime_id"].shape[0])
  part_path = output_dir / f"part-{part_idx:04d}.npz"
  np.savez_compressed(part_path, **batch)

  for value in buffers.values():
    value.clear()

  print(f"[collect] wrote {part_path.name}: {sample_count} samples")
  return sample_count


def main() -> None:
  # Import side-effect task registration.
  import mjlab.tasks  # noqa: F401

  task_id = str(CONFIG["task_id"])
  checkpoint_file = str(CONFIG["checkpoint_file"])
  device = str(CONFIG["device"])
  num_envs = int(CONFIG["num_envs"])
  num_steps = int(CONFIG["num_steps"])
  shard_size = int(CONFIG["shard_size"])
  output_root = Path(str(CONFIG["output_root"]))
  run_name = str(CONFIG["run_name"])
  twist_command_overrides = _build_twist_overrides_from_config()

  if checkpoint_file.startswith("/path/to/"):
    raise ValueError(
      "Set CONFIG['checkpoint_file'] to a real checkpoint path before running."
    )
  if shard_size <= 0:
    raise ValueError(f"shard_size must be positive, got {shard_size}.")

  timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
  output_dir = output_root / f"{run_name}-{timestamp}"
  output_dir.mkdir(parents=True, exist_ok=False)

  checkpoint_path = Path(checkpoint_file)
  if not checkpoint_path.exists():
    raise FileNotFoundError(f"checkpoint_file does not exist: {checkpoint_path}")
  ckpt_ext = checkpoint_path.suffix.lower()
  if ckpt_ext != ".pt":
    raise ValueError(
      f"checkpoint_file must be a .pt checkpoint, got: {checkpoint_path.name}"
    )

  # Collection requires .pt policy checkpoint.
  # Action inference still uses ONNX Runtime by exporting the loaded .pt once.
  onnx_source_mode = "exported_from_pt"
  policy_checkpoint_file = checkpoint_file
  onnx_path = output_dir / "policy.onnx"

  vec_env, _, policy = build_velocity_moe_policy(
    task_id=task_id,
    checkpoint_file=policy_checkpoint_file,
    device=device,
    num_envs=num_envs,
    play=True,
    command_overrides=twist_command_overrides,
  )
  effective_cmd_ranges = _extract_effective_cmd_ranges(vec_env)

  print(f"[collect] exporting ONNX policy to {onnx_path}")
  _export_policy_to_onnx_dynamic_batch(
    policy=policy,
    onnx_path=onnx_path,
  )

  onnx_session, onnx_input_name, onnx_providers_used, onnx_expected_batch_size = (
    _build_onnx_session(
      onnx_path=onnx_path,
      provider_order=None,
    )
  )
  print(
    f"[collect] ONNX inference enabled: {onnx_path} "
    f"(providers={onnx_providers_used}, expected_batch={onnx_expected_batch_size})"
  )

  router_slice = require_router_slice(policy)
  router_cfg = policy.router_params

  resolved_router_params: dict[str, Any] = {
    "cmd_threshold": float(getattr(policy.moe, "cmd_threshold", 0.05)),
    "turn_ratio": float(getattr(policy.moe, "turn_ratio", 2.0)),
    "vx_vy_ratio": float(getattr(policy.moe, "vx_vy_ratio", 1.0)),
    "learnable_regime_boundary": bool(
      getattr(policy.moe, "learnable_regime_boundary", False)
    ),
    "regime_lamda": float(getattr(policy, "regime_lamda", 0.0)),
    "regime_turn_ratio_ref": float(
      getattr(policy, "regime_turn_ratio_ref", getattr(policy.moe, "turn_ratio", 2.0))
    ),
    "regime_vx_vy_ratio_ref": float(
      getattr(policy, "regime_vx_vy_ratio_ref", getattr(policy.moe, "vx_vy_ratio", 1.0))
    ),
    "shared_specialist_ratio_lamda": float(
      getattr(policy, "shared_specialist_ratio_lamda", 0.0)
    ),
    "shared_specialist_ratio_init": float(
      getattr(policy.moe, "shared_specialist_ratio_init", 0.5)
    ),
  }
  if resolved_router_params["learnable_regime_boundary"]:
    if policy.moe.regime_logit_t is not None:
      resolved_router_params["learned_turn_ratio"] = float(
        torch.nn.functional.softplus(policy.moe.regime_logit_t).detach().cpu().item()
      )
    if policy.moe.regime_logit_r is not None:
      resolved_router_params["learned_vx_vy_ratio"] = float(
        torch.nn.functional.softplus(policy.moe.regime_logit_r).detach().cpu().item()
      )

  buffers: dict[str, list[np.ndarray]] = {
    "flat_obs": [],
    "router_input": [],
    "command": [],
    "regime_id": [],
    "full_action": [],
  }

  regime_hist = np.zeros(len(REGIME_NAMES), dtype=np.int64)
  total_samples = 0
  part_idx = 0
  pending_samples = 0

  obs = vec_env.get_observations().to(device)

  try:
    with torch.no_grad():
      for step in range(num_steps):
        flat_obs = flatten_actor_obs(obs, policy.obs_groups)
        action = _infer_actions_onnx(
          session=onnx_session,
          input_name=onnx_input_name,
          flat_obs=flat_obs,
          device=device,
          expected_batch_size=onnx_expected_batch_size,
        )

        start, end = router_slice
        router_input = flat_obs[:, start:end]

        regime_id = classify_cmd_regime_from_policy(policy, router_input)

        buffers["flat_obs"].append(flat_obs.detach().cpu().numpy().astype(np.float32))
        buffers["router_input"].append(
          router_input.detach().cpu().numpy().astype(np.float32)
        )
        # For cmd_regime this is identical to router input, but saved explicitly for analysis.
        buffers["command"].append(
          router_input.detach().cpu().numpy().astype(np.float32)
        )
        buffers["regime_id"].append(regime_id.detach().cpu().numpy().astype(np.int64))
        buffers["full_action"].append(action.detach().cpu().numpy().astype(np.float32))

        step_count = int(regime_id.numel())
        pending_samples += step_count
        total_samples += step_count
        regime_hist += np.bincount(
          regime_id.detach().cpu().numpy(),
          minlength=len(REGIME_NAMES),
        )[: len(REGIME_NAMES)]

        obs, _, _, _ = vec_env.step(action)
        obs = obs.to(device)

        if pending_samples >= shard_size:
          written = _flush_part(
            output_dir=output_dir, part_idx=part_idx, buffers=buffers
          )
          part_idx += 1
          pending_samples -= written

        if (step + 1) % 100 == 0 or (step + 1) == num_steps:
          print(
            f"[collect] step {step + 1}/{num_steps} | total_samples={total_samples}"
          )

    if pending_samples > 0:
      _flush_part(output_dir=output_dir, part_idx=part_idx, buffers=buffers)

  finally:
    vec_env.close()

  regime_stats = {
    REGIME_NAMES[i]: {
      "count": int(regime_hist[i]),
      "ratio": float(regime_hist[i] / max(total_samples, 1)),
    }
    for i in range(len(REGIME_NAMES))
  }

  manifest = {
    "task_id": task_id,
    "checkpoint_file": checkpoint_file,
    "device": device,
    "num_envs": num_envs,
    "num_steps": num_steps,
    "total_samples": total_samples,
    "shard_size": shard_size,
    "inference_backend": "onnxruntime",
    "onnx_source_mode": onnx_source_mode,
    "onnx_file": str(onnx_path),
    "onnx_providers_used": onnx_providers_used,
    "onnx_expected_batch_size": onnx_expected_batch_size,
    "policy_checkpoint_file_for_metadata": policy_checkpoint_file,
    "router_mode": policy.router_mode,
    "router_slice": [int(router_slice[0]), int(router_slice[1])],
    "router_params_from_cfg": _jsonable_or_repr(router_cfg),
    "router_params_resolved": resolved_router_params,
    "obs_dim": int(policy.obs_dim),
    "num_shared_experts": int(policy.moe.num_shared_experts),
    "num_specialist_experts": int(policy.moe.num_specialist_experts),
    "num_routed_specialists": int(policy.moe.num_routed_specialists),
    "regime_stats": regime_stats,
    "cmd_ranges_effective": effective_cmd_ranges,
    "cmd_ranges_config": _jsonable_or_repr(CONFIG.get("cmd_ranges", None)),
    "twist_cfg_override_config": _jsonable_or_repr(
      CONFIG.get("twist_cfg_override", None)
    ),
    "twist_cfg_override_applied": _jsonable_or_repr(twist_command_overrides),
  }

  manifest_path = output_dir / "manifest.json"
  with manifest_path.open("w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

  print("[collect] done")
  print(f"[collect] output_dir: {output_dir}")
  print(f"[collect] manifest: {manifest_path}")


if __name__ == "__main__":
  main()
