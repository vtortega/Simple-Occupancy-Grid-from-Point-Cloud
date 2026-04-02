#!/usr/bin/env python3
"""
Visualize a point cloud as 2D heatmaps + 3D level-coloured scatter,
detecting building floor levels along the Z axis.

Usage:
    python3 visualize_levels.py <file.pcd> [--fit_plane]
        [--threshold 0.5] [--below 0.30] [--above 1.50]
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ---------------------------------------------------------------------------
# PCD I/O
# ---------------------------------------------------------------------------

def parse_pcd_header(f):
    header = {}
    while True:
        line = f.readline().decode("utf-8", errors="replace").strip()
        if line.startswith("#") or not line:
            continue
        key, *values = line.split()
        header[key] = values
        if key == "DATA":
            break
    return header


def read_pcd(path):
    with open(path, "rb") as f:
        header = parse_pcd_header(f)
        raw = f.read()

    fields = header["FIELDS"]
    sizes  = list(map(int, header["SIZE"]))
    types  = header["TYPE"]
    counts = list(map(int, header["COUNT"]))
    n_pts  = int(header["POINTS"][0])

    if header["DATA"][0] != "binary":
        raise ValueError(f"Only binary PCD supported, got: {header['DATA'][0]}")

    dtype_list = []
    for field, size, typ, count in zip(fields, sizes, types, counts):
        np_type = {"F": "f", "I": "i", "U": "u"}[typ] + str(size)
        dtype_list.append((field, np_type, count) if count > 1 else (field, np_type))

    pts = np.frombuffer(raw, dtype=np.dtype(dtype_list), count=n_pts)
    x = pts["x"].astype(np.float64)
    y = pts["y"].astype(np.float64)
    z = pts["z"].astype(np.float64)

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    return x[mask], y[mask], z[mask]


# ---------------------------------------------------------------------------
# Plane fitting & alignment
# ---------------------------------------------------------------------------

def fit_plane_ransac(x, y, z, n_iter=500, threshold=0.05, subsample=80_000):
    pts = np.stack([x, y, z], axis=1)
    rng = np.random.default_rng(42)
    if len(pts) > subsample:
        pts_sub = pts[rng.choice(len(pts), subsample, replace=False)]
    else:
        pts_sub = pts

    best_n_inliers, best_normal, best_d = 0, np.array([0.0, 0.0, 1.0]), 0.0
    for _ in range(n_iter):
        s = rng.choice(len(pts_sub), 3, replace=False)
        p1, p2, p3 = pts_sub[s]
        normal = np.cross(p2 - p1, p3 - p1)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        d = normal @ p1
        n_inliers = np.sum(np.abs(pts_sub @ normal - d) < threshold)
        if n_inliers > best_n_inliers:
            best_n_inliers, best_normal, best_d = n_inliers, normal, d

    inliers = pts[np.abs(pts @ best_normal - best_d) < threshold]
    print(f"  RANSAC inliers: {len(inliers):,} / {len(pts):,} "
          f"({100 * len(inliers) / len(pts):.1f}%)")

    centroid = inliers.mean(axis=0)
    _, _, Vt = np.linalg.svd(inliers - centroid, full_matrices=False)
    normal = Vt[-1]
    if normal[2] < 0:
        normal = -normal
    return normal, centroid


def align_to_plane(x, y, z, normal, centroid):
    target = np.array([0.0, 0.0, 1.0])
    axis = np.cross(normal, target)
    sin_a = np.linalg.norm(axis)
    cos_a = float(np.dot(normal, target))
    pts = np.stack([x, y, z], axis=1) - centroid

    if sin_a < 1e-9:
        if cos_a < 0:
            pts[:, 2] = -pts[:, 2]
        return pts[:, 0], pts[:, 1], pts[:, 2]

    axis /= sin_a
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = cos_a * np.eye(3) + sin_a * K + (1 - cos_a) * np.outer(axis, axis)
    pts = pts @ R.T
    return pts[:, 0], pts[:, 1], pts[:, 2]


# ---------------------------------------------------------------------------
# Level detection
# ---------------------------------------------------------------------------

def _gaussian_smooth(arr, sigma_bins):
    radius = int(4 * sigma_bins + 0.5)
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    kernel /= kernel.sum()
    return np.convolve(arr, kernel, mode="same")


def _find_peaks(arr, min_dist_bins):
    n = len(arr)
    candidates = np.array(
        [i for i in range(1, n - 1) if arr[i] >= arr[i - 1] and arr[i] >= arr[i + 1]]
    )
    if len(candidates) == 0:
        return np.array([], dtype=int)
    order = np.argsort(arr[candidates])[::-1]
    selected = []
    for i in order:
        idx = candidates[i]
        if all(abs(idx - s) >= min_dist_bins for s in selected):
            selected.append(idx)
    return np.array(sorted(selected))


def detect_levels(z, below, above, threshold, num_levels=None,
                  bin_size=0.05, smooth_sigma_m=0.3):
    """
    Detect floor levels from the Z histogram.

    Peak separation enforced at (below + above) metres so two adjacent level
    bands cannot overlap. Points counted in [z_c - below, z_c + above] for
    each candidate.

    If num_levels is given, exactly that many peaks are kept (the tallest ones
    by slab count) and the threshold filter is ignored.
    Otherwise, levels with fewer than threshold × max count are discarded.
    """
    edges = np.arange(z.min(), z.max() + bin_size, bin_size)
    counts, edges = np.histogram(z, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])

    smoothed = _gaussian_smooth(counts, smooth_sigma_m / bin_size)
    min_dist_bins = max(1, int((below + above) / bin_size))
    peak_bins = _find_peaks(smoothed, min_dist_bins)

    if len(peak_bins) == 0:
        return []

    level_counts = []
    for p in peak_bins:
        z_c = centers[p]
        n_pts = int(np.sum((z >= z_c - below) & (z <= z_c + above)))
        level_counts.append((z_c, n_pts))

    if num_levels is not None:
        # Keep the N peaks with the most points, regardless of threshold
        level_counts.sort(key=lambda t: t[1], reverse=True)
        level_counts = level_counts[:num_levels]
    else:
        max_count = max(c for _, c in level_counts)
        level_counts = [(z_c, cnt) for z_c, cnt in level_counts
                        if cnt >= threshold * max_count]

    level_counts.sort(key=lambda t: t[0])  # sort bottom to top
    return level_counts


def assign_levels(z, levels, below, above):
    """Return per-point level index (0-based) or -1 if outside every level band."""
    assignments = np.full(len(z), -1, dtype=int)
    for i, (z_c, _) in enumerate(levels):
        mask = (z >= z_c - below) & (z <= z_c + above)
        assignments[mask] = i
    return assignments


# ---------------------------------------------------------------------------
# PCD output
# ---------------------------------------------------------------------------

def save_pcd_binary(path, x, y, z):
    """Write x, y, z as a compact binary PCD file (float32)."""
    n = len(x)
    data = np.stack([x.astype(np.float32),
                     y.astype(np.float32),
                     z.astype(np.float32)], axis=1)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA binary\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("utf-8"))
        f.write(data.tobytes())


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Distinct colours for up to ~10 levels; falls back to tab10 beyond that.
_LEVEL_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
    "#fabed4", "#469990",
]

def _level_color(i):
    if i < len(_LEVEL_COLORS):
        return _LEVEL_COLORS[i]
    return plt.cm.tab10(i % 10)


def plot_heatmaps(x, y, z, levels, below, above):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Point cloud projections — log-density heatmap", fontsize=13)

    for ax, a, b, xlabel, ylabel, title in [
        (axes[0], y, z, "Y", "Z", "YZ plane  (X squashed)"),
        (axes[1], x, z, "X", "Z", "XZ plane  (Y squashed)"),
    ]:
        h, xe, ye = np.histogram2d(a, b, bins=400)
        ax.imshow(np.log1p(h).T, origin="lower", aspect="auto",
                  extent=[xe[0], xe[-1], ye[0], ye[-1]], cmap="inferno")

        for i, (z_c, _) in enumerate(levels):
            color = _level_color(i)
            ax.axhspan(z_c - below, z_c + above, color=color, alpha=0.12, linewidth=0)
            ax.axhline(z_c, color=color, linewidth=1.5, alpha=0.9)
            ax.text(xe[0] + 0.01 * (xe[-1] - xe[0]), z_c,
                    f" L{i + 1}", color=color, fontsize=9,
                    va="center", fontweight="bold")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    plt.tight_layout()


def plot_3d(x, y, z, assignments, levels, below, above, max_points=200_000):
    n = len(x)
    if n > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, max_points, replace=False)
        x, y, z, assignments = x[idx], y[idx], z[idx], assignments[idx]

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("3D point cloud coloured by level", fontsize=13)

    # Background points (no level)
    bg = assignments == -1
    if bg.any():
        ax.scatter(x[bg], y[bg], z[bg],
                   c="0.55", s=0.3, alpha=0.08, linewidths=0, depthshade=False)

    # Level points
    for i, (z_c, cnt) in enumerate(levels):
        mask = assignments == i
        if not mask.any():
            continue
        ax.scatter(x[mask], y[mask], z[mask],
                   c=_level_color(i), s=1.0, alpha=0.5, linewidths=0,
                   depthshade=False,
                   label=f"L{i + 1}  Z={z_c:+.2f} m  ({cnt:,} pts)")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if levels:
        ax.legend(loc="upper left", markerscale=8, fontsize=9,
                  framealpha=0.6)

    plt.tight_layout()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="2D heatmaps + 3D level-coloured view of a point cloud."
    )
    parser.add_argument("pcd", help="Path to the .pcd file")
    parser.add_argument("--fit_plane", action="store_true",
                        help="RANSAC plane fit; re-align so that plane becomes Z = 0")
    parser.add_argument("--threshold", type=float, default=0.5, metavar="T",
                        help="Discard levels with fewer than T × max-level points "
                             "(default: 0.5)")
    parser.add_argument("--below", type=float, default=0.30, metavar="M",
                        help="Metres below level plane included in its slab "
                             "(default: 0.30)")
    parser.add_argument("--above", type=float, default=1.50, metavar="M",
                        help="Metres above level plane included in its slab "
                             "(default: 1.50)")
    parser.add_argument("--num_levels", type=int, default=None, metavar="N",
                        help="Force exactly N levels (picks the N tallest peaks; "
                             "ignores --threshold)")
    args = parser.parse_args()

    print(f"Reading {args.pcd} ...")
    x, y, z = read_pcd(args.pcd)
    print(f"Loaded {len(x):,} valid points")

    if args.fit_plane:
        print("Fitting ground plane (RANSAC) ...")
        normal, centroid = fit_plane_ransac(x, y, z)
        angle_deg = np.degrees(np.arccos(np.clip(float(normal @ [0, 0, 1]), -1, 1)))
        print(f"  Plane normal : {normal}")
        print(f"  Tilt from Z  : {angle_deg:.2f} deg")
        x, y, z = align_to_plane(x, y, z, normal, centroid)
        print("  Cloud re-aligned.")

    print(f"  X: [{x.min():.2f}, {x.max():.2f}]")
    print(f"  Y: [{y.min():.2f}, {y.max():.2f}]")
    print(f"  Z: [{z.min():.2f}, {z.max():.2f}]")

    if args.num_levels is not None:
        print(f"Detecting levels  (forced N={args.num_levels}, "
              f"slab=[−{args.below:.2f} m, +{args.above:.2f} m]) ...")
    else:
        print(f"Detecting levels  (threshold={args.threshold}, "
              f"slab=[−{args.below:.2f} m, +{args.above:.2f} m]) ...")
    levels = detect_levels(z, below=args.below, above=args.above,
                           threshold=args.threshold, num_levels=args.num_levels)
    if levels:
        print(f"  Found {len(levels)} level(s):")
        for i, (z_c, cnt) in enumerate(levels, start=1):
            print(f"    L{i}  Z = {z_c:+.2f} m   ({cnt:,} pts in slab)")
    else:
        print("  No levels detected.")

    assignments = assign_levels(z, levels, below=args.below, above=args.above)

    # Save each level as a separate PCD file
    base = os.path.splitext(args.pcd)[0]
    for i, (z_c, cnt) in enumerate(levels):
        mask = assignments == i
        out_path = f"{base}_L{i + 1}.pcd"
        save_pcd_binary(out_path, x[mask], y[mask], z[mask])
        print(f"  Saved L{i + 1} → {out_path}  ({mask.sum():,} pts)")

    plot_heatmaps(x, y, z, levels, args.below, args.above)
    plot_3d(x, y, z, assignments, levels, args.below, args.above)
    plt.show()


if __name__ == "__main__":
    main()
