#!/usr/bin/env python3
import argparse
import math
import os
import re
import struct
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import numpy as np


# ----------------------------
# STL loader (binary/ASCII)
# ----------------------------
def load_stl_vertices(path: str) -> np.ndarray:
  """
  Returns (N,3) vertices array from STL (binary or ASCII).
  Minimal parser, sufficient for collision fitting.
  """
  with open(path, "rb") as f:
    data = f.read()

  # Heuristic: binary STL has 80-byte header + 4-byte tri count + 50 bytes per tri
  if len(data) >= 84:
    tri_count = struct.unpack_from("<I", data, 80)[0]
    expected = 84 + tri_count * 50
    if expected == len(data):
      # binary
      verts = []
      offset = 84
      for _ in range(tri_count):
        # skip normal (12 bytes)
        offset += 12
        v = struct.unpack_from("<9f", data, offset)  # 3 vertices
        offset += 36
        verts.append([v[0], v[1], v[2]])
        verts.append([v[3], v[4], v[5]])
        verts.append([v[6], v[7], v[8]])
        # skip attribute (2 bytes)
        offset += 2
      return np.asarray(verts, dtype=np.float64)

  # ASCII fallback
  text = data.decode("utf-8", errors="ignore")
  verts = []
  for line in text.splitlines():
    line = line.strip()
    if line.startswith("vertex "):
      parts = line.split()
      if len(parts) == 4:
        verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
  if not verts:
    raise RuntimeError(f"Failed to parse STL: {path}")
  return np.asarray(verts, dtype=np.float64)


# ----------------------------
# MuJoCo quaternion (w x y z) -> rotation matrix
# ----------------------------
def quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
  w, x, y, z = q
  # normalize
  n = math.sqrt(w * w + x * x + y * y + z * z)
  if n <= 1e-12:
    return np.eye(3)
  w, x, y, z = w / n, x / n, y / n, z / n
  return np.array(
    [
      [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
      [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
      [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ],
    dtype=np.float64,
  )


def parse_vec(
  attr: Optional[str], n: int, default: Optional[List[float]] = None
) -> np.ndarray:
  if attr is None:
    if default is None:
      return np.zeros((n,), dtype=np.float64)
    return np.array(default, dtype=np.float64)
  parts = [p for p in attr.replace(",", " ").split() if p]
  if len(parts) != n:
    raise ValueError(f"Expected {n} floats, got {len(parts)} from '{attr}'")
  return np.array([float(x) for x in parts], dtype=np.float64)


def fmt_floats(xs: np.ndarray) -> str:
  return " ".join([f"{v:.6g}" for v in xs.tolist()])


# ----------------------------
# Geometry fitting helpers
# ----------------------------
def pca_axis(points: np.ndarray) -> np.ndarray:
  """Return principal axis (unit vector) of points."""
  if points.shape[0] < 10:
    return np.array([0, 0, 1], dtype=np.float64)
  c = points.mean(axis=0)
  X = points - c
  C = (X.T @ X) / max(1, points.shape[0] - 1)
  vals, vecs = np.linalg.eigh(C)  # ascending
  axis = vecs[:, np.argmax(vals)]
  axis = axis / (np.linalg.norm(axis) + 1e-12)
  return axis


def refit_capsule_to_points(
  points: np.ndarray,
  p1: np.ndarray,
  p2: np.ndarray,
  percentile: float = 95.0,
  margin: float = 0.02,
  max_radius_scale: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
  """
  Refit capsule defined by endpoints p1,p2 to point cloud in body frame.
  - Uses current axis to select nearby points.
  - Returns new_p1, new_p2, new_radius, selected_count
  """
  v = p2 - p1
  L = float(np.linalg.norm(v))
  if L < 1e-6:
    # Degenerate: use PCA to create a capsule
    axis = pca_axis(points)
    c = points.mean(axis=0)
    t = (points - c) @ axis
    tmin, tmax = float(t.min()), float(t.max())
    new_p1 = c + tmin * axis
    new_p2 = c + tmax * axis
    # radius from perpendicular distances
    perp = (points - c) - np.outer(t, axis)
    d = np.linalg.norm(perp, axis=1)
    r = float(np.percentile(d, percentile))
    return new_p1, new_p2, r, points.shape[0]

  axis = v / L

  # Project points to axis coordinate t along the segment
  t = (points - p1) @ axis  # scalar per point
  # Perpendicular distances to axis line
  perp = (points - p1) - np.outer(t, axis)
  d = np.linalg.norm(perp, axis=1)

  # Select points close to current segment region
  in_range = (t >= -margin) & (t <= L + margin)
  # Also remove far-out points to keep local region robust
  # (we don't know current radius reliably, so use a loose gate based on median)
  if np.any(in_range):
    d_med = np.median(d[in_range])
  else:
    d_med = np.median(d)
  gate = d <= max_radius_scale * max(d_med, 1e-6)

  sel = in_range & gate
  if int(sel.sum()) < 30:
    # Fallback: broaden selection
    sel = in_range
  if int(sel.sum()) < 10:
    # Fallback to global PCA fit
    axis2 = pca_axis(points)
    c = points.mean(axis=0)
    t2 = (points - c) @ axis2
    tmin, tmax = float(t2.min()), float(t2.max())
    new_p1 = c + tmin * axis2
    new_p2 = c + tmax * axis2
    perp2 = (points - c) - np.outer(t2, axis2)
    d2 = np.linalg.norm(perp2, axis=1)
    r = float(np.percentile(d2, percentile))
    return new_p1, new_p2, r, int(sel.sum())

  t_sel = t[sel]
  d_sel = d[sel]
  tmin, tmax = float(t_sel.min()), float(t_sel.max())

  new_p1 = p1 + tmin * axis
  new_p2 = p1 + tmax * axis
  r = float(np.percentile(d_sel, percentile))
  return new_p1, new_p2, r, int(sel.sum())


# ----------------------------
# XML processing
# ----------------------------
def build_mesh_file_map(
  root: ET.Element, xml_dir: str, meshdir_override: Optional[str]
) -> Dict[str, str]:
  """
  Map mesh name -> absolute STL path
  """
  meshdir = meshdir_override
  # If not overridden, read compiler meshdir
  if meshdir is None:
    comp = root.find("compiler")
    if comp is not None and comp.get("meshdir"):
      meshdir = comp.get("meshdir")
  if meshdir is None:
    meshdir = "."

  meshdir_abs = meshdir
  if not os.path.isabs(meshdir_abs):
    meshdir_abs = os.path.join(xml_dir, meshdir_abs)

  assets = root.find("asset")
  mesh_map: Dict[str, str] = {}
  if assets is None:
    return mesh_map

  for m in assets.findall("mesh"):
    name = m.get("name")
    file_ = m.get("file")
    if not name or not file_:
      continue
    path = file_
    if not os.path.isabs(path):
      path = os.path.join(meshdir_abs, path)
    mesh_map[name] = os.path.abspath(path)
  return mesh_map


def collect_body_visual_points(
  body: ET.Element, mesh_map: Dict[str, str]
) -> np.ndarray:
  """
  Collect vertices from all visual mesh geoms directly under this <body> (not recursively).
  Transforms mesh vertices by geom pos/quat/scale into body frame.
  """
  all_pts = []
  for g in body.findall("geom"):
    mesh_name = g.get("mesh")
    if not mesh_name:
      continue
    # Skip if it's collision mesh (your visual class is 'visual' with mesh)
    # We just use whatever mesh is present; class doesn't matter here.
    if mesh_name not in mesh_map:
      continue

    stl_path = mesh_map[mesh_name]
    if not os.path.exists(stl_path):
      continue

    V = load_stl_vertices(stl_path)

    # geom transform in body frame
    pos = parse_vec(g.get("pos"), 3, default=[0, 0, 0])
    quat = parse_vec(g.get("quat"), 4, default=[1, 0, 0, 0])
    R = quat_wxyz_to_R(quat)

    # optional scale
    if g.get("scale"):
      s = parse_vec(g.get("scale"), 3)
    else:
      s = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    V2 = (V * s[None, :]) @ R.T + pos[None, :]
    all_pts.append(V2)

  if not all_pts:
    return np.zeros((0, 3), dtype=np.float64)
  return np.vstack(all_pts)


def should_skip_geom(geom_name: str, skip_regexes: List[re.Pattern]) -> bool:
  for r in skip_regexes:
    if r.search(geom_name):
      return True
  return False


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--xml", required=True, help="input MuJoCo XML path")
  ap.add_argument("--out", required=True, help="output XML path")
  ap.add_argument(
    "--meshdir",
    default=None,
    help="override meshdir (defaults to <compiler meshdir=...>)",
  )
  ap.add_argument(
    "--percentile",
    type=float,
    default=95.0,
    help="radius percentile of distances (robust)",
  )
  ap.add_argument(
    "--shrink", type=float, default=0.90, help="shrink factor on fitted radius"
  )
  ap.add_argument(
    "--margin", type=float, default=0.02, help="axis-range margin for selecting points"
  )
  ap.add_argument(
    "--skip",
    default=None,
    help="comma-separated regex patterns to skip geom names (in addition to pelvis/foot defaults)",
  )
  ap.add_argument(
    "--check",
    action="store_true",
    help="after writing XML, load with mujoco and report penetrations",
  )
  args = ap.parse_args()

  xml_path = os.path.abspath(args.xml)
  xml_dir = os.path.dirname(xml_path)

  tree = ET.parse(xml_path)
  root = tree.getroot()

  mesh_map = build_mesh_file_map(root, xml_dir, args.meshdir)

  # Default skip: pelvis/base + all foot capsules
  skip_patterns = [
    r"^base_link_collision$",
    r".*foot\d*_collision$",
    r".*foot.*_collision$",
  ]
  if args.skip:
    skip_patterns += [p.strip() for p in args.skip.split(",") if p.strip()]
  skip_regexes = [re.compile(p) for p in skip_patterns]

  # Walk bodies recursively
  worldbody = root.find("worldbody")
  if worldbody is None:
    raise RuntimeError("No <worldbody> found")

  updated = []

  def visit_body(body: ET.Element):
    # Collect body-local visual points
    pts = collect_body_visual_points(body, mesh_map)
    body_name = body.get("name", "(noname)")

    if pts.shape[0] == 0:
      # no mesh under this body
      # still recurse
      for child in body.findall("body"):
        visit_body(child)
      return

    # Update collision geoms directly under this body
    for g in body.findall("geom"):
      gname = g.get("name")
      if not gname:
        continue

      # Skip pelvis/foot etc.
      if should_skip_geom(gname, skip_regexes):
        continue

      # Only target collision-ish geoms with fromto (capsule style)
      if g.get("fromto") is None:
        continue

      # Parse endpoints
      ft = parse_vec(g.get("fromto"), 6)
      p1 = ft[0:3]
      p2 = ft[3:6]

      new_p1, new_p2, r, sel_n = refit_capsule_to_points(
        pts,
        p1,
        p2,
        percentile=args.percentile,
        margin=args.margin,
      )
      r *= float(args.shrink)
      r = max(r, 1e-4)

      g.set("type", "capsule")  # enforce
      g.set("fromto", fmt_floats(np.concatenate([new_p1, new_p2])))
      g.set("size", fmt_floats(np.array([r], dtype=np.float64)))

      updated.append((body_name, gname, sel_n, r))

    for child in body.findall("body"):
      visit_body(child)

  # Start from top-level bodies under worldbody
  for b in worldbody.findall("body"):
    visit_body(b)

  out_path = os.path.abspath(args.out)
  tree.write(out_path, encoding="utf-8", xml_declaration=True)

  print(f"[OK] wrote: {out_path}")
  print(f"[INFO] updated geoms: {len(updated)}")
  for body_name, gname, sel_n, r in updated[:50]:
    print(f"  - {gname} (body={body_name}) sel={sel_n} new_r={r:.6g}")
  if len(updated) > 50:
    print("  ...")

  if args.check:
    try:
      import mujoco

      model = mujoco.MjModel.from_xml_path(out_path)
      data = mujoco.MjData(model)
      mujoco.mj_forward(model, data)

      bad = []
      for i in range(data.ncon):
        c = data.contact[i]
        if c.dist < -1e-5:
          bad.append((c.dist, c.geom1, c.geom2))
      bad.sort(key=lambda x: x[0])

      def gname_id(gid: int) -> str:
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom#{gid}"

      print(f"[CHECK] penetrating contacts at qpos=0: {len(bad)}")
      for dist, g1, g2 in bad[:30]:
        print(f"  {dist: .6f} | {gname_id(g1)} <-> {gname_id(g2)}")
      if len(bad) > 30:
        print("  ...")
    except Exception as e:
      print(f"[WARN] mujoco check failed: {e}")


if __name__ == "__main__":
  main()
