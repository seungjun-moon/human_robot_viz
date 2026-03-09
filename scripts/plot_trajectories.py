"""
Interactive 3D visualization comparing multiple sample datasets and LeRobot datasets.

Loads raw SE3 transforms from sample *.hdf5 files and extracts joint positions
directly. Plots alongside existing LeRobot dataset trajectories (which go through FK).

Sample datasets live under samples/[DATASET_NAME]/.../*.hdf5.

Usage:
    # Compare two sample datasets
    python scripts/plot_trajectories.py \
        --sample-dirs samples/ego10k samples/egodex --n 1

    # With camera-space conversion
    python scripts/plot_trajectories.py \
        --sample-dirs samples/ego10k --cam_space --n 1

    # Sample datasets + LeRobot categories
    python scripts/plot_trajectories.py \
        --sample-dirs samples/* \
        --arkit-datasets egodex \
        --categories egodex_v4 RLWRLD --n 1
"""

import argparse
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.kinematics_utils import NumpyFK, load_and_compute
from utils.name_utils import (
    CATEGORY_CONFIG,
    ALL_JOINT_NAMES, JOINT_DISPLAY,
    ALL_FINGERTIP_NAMES, FINGERTIP_DISPLAY,
    ALL_HAND_JOINT_NAMES, HAND_JOINT_DISPLAY,
    SKELETON_BONES,
    FINGER_BONES_RIGHT, FINGER_BONES_LEFT,
    FINGER_BONES_RIGHT_TIPS, FINGER_BONES_LEFT_TIPS,
    generate_colors,
)
from utils.data_utils import (
    list_datasets, sample_episodes,
    load_hdf5_episodes,
)


def subsample_traj(traj: dict, max_frames: int) -> dict:
    """Uniformly subsample a trajectory dict to at most max_frames frames."""
    # Find trajectory length from first non-metadata key
    T = None
    for k, v in traj.items():
        if not k.startswith("_") and isinstance(v, dict) and "pos" in v:
            T = v["pos"].shape[0]
            break
    if T is None or T <= max_frames:
        return traj

    indices = np.linspace(0, T - 1, max_frames, dtype=int)
    result = {}
    for k, v in traj.items():
        if k.startswith("_"):
            result[k] = v
        else:
            result[k] = {"pos": v["pos"][indices], "rpy": v["rpy"][indices]}
    return result


def make_3d_html(cat_data: dict[str, tuple[str, str, list[dict]]],
                  keys: list[str], display: dict[str, str],
                  title: str, output: str):
    """Generate interactive 3D plotly HTML with one subplot per key.

    cat_data: {name: (color, label, [traj_dict, ...])}
    """
    n_keys = len(keys)
    n_cols = min(n_keys, 5)
    n_rows = (n_keys + n_cols - 1) // n_cols

    specs = [[{"type": "scatter3d"} for _ in range(n_cols)] for _ in range(n_rows)]
    subplot_titles = [display.get(k, k) for k in keys]
    while len(subplot_titles) < n_rows * n_cols:
        subplot_titles.append("")

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        specs=specs,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.03,
        vertical_spacing=0.06,
    )

    legend_added = set()

    for j_idx, key in enumerate(keys):
        row = j_idx // n_cols + 1
        col = j_idx % n_cols + 1

        for cat_name, (color, label, trajs) in cat_data.items():
            show_legend = cat_name not in legend_added
            if show_legend:
                legend_added.add(cat_name)

            for i, traj in enumerate(trajs):
                pos = traj[key]["pos"]
                opacity = 0.3 + 0.5 / max(len(trajs), 1) * i

                fig.add_trace(
                    go.Scatter3d(
                        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
                        mode="lines",
                        line=dict(color=color, width=2),
                        opacity=opacity,
                        name=label,
                        legendgroup=cat_name,
                        showlegend=(show_legend and i == 0),
                        hovertemplate=(
                            f"{display.get(key, key)} ({label})<br>"
                            "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}"
                            "<extra></extra>"
                        ),
                    ),
                    row=row, col=col,
                )

                fig.add_trace(
                    go.Scatter3d(
                        x=[pos[0, 0]], y=[pos[0, 1]], z=[pos[0, 2]],
                        mode="markers",
                        marker=dict(color=color, size=3),
                        opacity=0.6,
                        legendgroup=cat_name,
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=row, col=col,
                )

    n_total = sum(len(v[2]) for v in cat_data.values())
    cats_str = " vs ".join(v[1] for v in cat_data.values())

    fig.update_layout(
        title=dict(text=f"{title}: {cats_str} ({n_total} total)", font_size=16),
        height=450 * n_rows,
        width=350 * n_cols,
        legend=dict(font_size=11),
        margin=dict(l=0, r=0, t=60, b=0),
    )

    for i in range(n_keys):
        scene_name = f"scene{i + 1}" if i > 0 else "scene"
        fig.update_layout(**{
            scene_name: dict(
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                zaxis_title="Z (m)",
                aspectmode="data",
            )
        })

    fig.write_html(output, include_plotlyjs=True)
    print(f"Saved to {output}")


def _frustum_traces(c2w: np.ndarray, color: str, label: str,
                     legendgroup: str, fov_h: float = 60.0,
                     fov_w: float = 80.0, depth: float = 0.15):
    """Generate Scatter3d traces for a camera frustum pyramid.

    ARKit / OpenGL camera convention: right=+x, up=+y, forward=-z.

    Args:
        c2w: (4, 4) camera-to-world transform.
        color: hex color string.
        label: legend label.
        legendgroup: plotly legend group.
        fov_h: vertical FOV in degrees.
        fov_w: horizontal FOV in degrees.
        depth: frustum depth (distance from apex to near plane).

    Returns:
        List of go.Scatter3d traces.
    """
    pos = c2w[:3, 3]
    right = c2w[:3, 0]
    up = c2w[:3, 1]
    forward = -c2w[:3, 2]  # ARKit/OpenGL: camera looks along -z

    h = depth * np.tan(np.radians(fov_h / 2))
    w = depth * np.tan(np.radians(fov_w / 2))
    center = pos + forward * depth

    tl = center + up * h - right * w
    tr = center + up * h + right * w
    bl = center - up * h - right * w
    br = center - up * h + right * w

    traces = []
    # 4 edges from apex to corners
    for corner in [tl, tr, br, bl]:
        traces.append(go.Scatter3d(
            x=[pos[0], corner[0]], y=[pos[1], corner[1]], z=[pos[2], corner[2]],
            mode="lines",
            line=dict(color=color, width=3),
            opacity=0.7,
            legendgroup=legendgroup,
            showlegend=False,
            hoverinfo="skip",
        ))

    # Near-plane rectangle
    rect = np.array([tl, tr, br, bl, tl])
    traces.append(go.Scatter3d(
        x=rect[:, 0], y=rect[:, 1], z=rect[:, 2],
        mode="lines",
        line=dict(color=color, width=3),
        opacity=0.7,
        legendgroup=legendgroup,
        showlegend=False,
        hoverinfo="skip",
    ))

    # Forward axis line (slightly beyond frustum depth for visibility)
    fwd_end = pos + forward * depth * 1.3
    traces.append(go.Scatter3d(
        x=[pos[0], fwd_end[0]], y=[pos[1], fwd_end[1]], z=[pos[2], fwd_end[2]],
        mode="lines",
        line=dict(color=color, width=2, dash="dash"),
        opacity=0.5,
        legendgroup=legendgroup,
        showlegend=False,
        hoverinfo="skip",
    ))

    # Camera position marker
    traces.append(go.Scatter3d(
        x=[pos[0]], y=[pos[1]], z=[pos[2]],
        mode="markers+text",
        marker=dict(color=color, size=6, symbol="diamond"),
        text=["cam"],
        textposition="top center",
        textfont=dict(size=8, color=color),
        name=f"{label} (camera)",
        legendgroup=legendgroup,
        showlegend=False,
        hovertemplate="Camera<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
    ))

    return traces


def make_skeleton_html(cat_data: dict[str, tuple[str, str, list[dict]]],
                        joints: list[str], fingertips: list[str],
                        title: str, output: str, frame_idx: int = 0):
    """Draw skeleton poses for a sample frame from each category, plus trajectories.

    Shows all joints in a single 3D scene with bones connecting them.
    One skeleton per category (first episode, specified frame).
    Trajectories from all episodes shown faintly in the background.
    Camera frustum shown if camera c2w data is available.
    """
    bones = list(SKELETON_BONES)
    if fingertips:
        ft_right = [ft for ft in fingertips if ft.startswith("R_")]
        ft_left = [ft for ft in fingertips if ft.startswith("L_")]
        # Use full finger chains if intermediate joints are present, else tip-only
        has_intermediate = any("_Knuckle" in ft or "_Base" in ft or "_Mid" in ft
                               for ft in fingertips)
        if ft_right:
            bones += FINGER_BONES_RIGHT if has_intermediate else FINGER_BONES_RIGHT_TIPS
        if ft_left:
            bones += FINGER_BONES_LEFT if has_intermediate else FINGER_BONES_LEFT_TIPS

    all_keys = list(joints) + (fingertips or [])

    fig = go.Figure()

    for cat_name, (color, label, trajs) in cat_data.items():
        if not trajs:
            continue

        # Background trajectories (faint)
        for i, traj in enumerate(trajs):
            for k in all_keys:
                pos = traj[k]["pos"]
                fig.add_trace(go.Scatter3d(
                    x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
                    mode="lines",
                    line=dict(color=color, width=1),
                    opacity=0.08,
                    legendgroup=cat_name,
                    showlegend=False,
                    hoverinfo="skip",
                ))

        # Skeleton from first episode at frame_idx
        traj = trajs[0]
        t = min(frame_idx, traj[all_keys[0]]["pos"].shape[0] - 1)

        # Joint markers
        jx, jy, jz, jtext = [], [], [], []
        for k in all_keys:
            p = traj[k]["pos"][t]
            jx.append(p[0]); jy.append(p[1]); jz.append(p[2])
            disp = JOINT_DISPLAY.get(k, HAND_JOINT_DISPLAY.get(k, FINGERTIP_DISPLAY.get(k, k)))
            jtext.append(disp)

        fig.add_trace(go.Scatter3d(
            x=jx, y=jy, z=jz,
            mode="markers+text",
            marker=dict(color=color, size=5, symbol="circle"),
            text=jtext,
            textposition="top center",
            textfont=dict(size=8, color=color),
            name=f"{label} (skeleton)",
            legendgroup=cat_name,
            showlegend=True,
            hovertemplate="%{text}<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
        ))

        # Bones
        for parent, child in bones:
            if parent not in all_keys or child not in all_keys:
                continue
            pp = traj[parent]["pos"][t]
            cp = traj[child]["pos"][t]
            fig.add_trace(go.Scatter3d(
                x=[pp[0], cp[0]], y=[pp[1], cp[1]], z=[pp[2], cp[2]],
                mode="lines",
                line=dict(color=color, width=4),
                opacity=0.9,
                legendgroup=cat_name,
                showlegend=False,
                hoverinfo="skip",
            ))

        # Camera frustum
        if "_camera_c2w" in traj:
            c2w_all = traj["_camera_c2w"]
            t_cam = min(t, c2w_all.shape[0] - 1)
            for trace in _frustum_traces(c2w_all[t_cam], color, label, cat_name):
                fig.add_trace(trace)

    n_total = sum(len(v[2]) for v in cat_data.values())
    cats_str = " vs ".join(v[1] for v in cat_data.values())

    fig.update_layout(
        title=dict(text=f"{title}: {cats_str} ({n_total} episodes, frame {frame_idx})",
                   font_size=16),
        height=800,
        width=1000,
        legend=dict(font_size=12),
        margin=dict(l=0, r=0, t=60, b=0),
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
        ),
    )

    fig.write_html(output, include_plotlyjs=True)
    print(f"Saved to {output}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare sample HDF5 datasets and/or LeRobot datasets (3D Plotly HTML)")
    parser.add_argument("--sample-dirs", type=str, nargs="+", default=[],
                        help="Directories containing raw *.hdf5 sample files "
                             "(e.g. samples/ego10k samples/egodex)")
    parser.add_argument("--arkit-datasets", nargs="*", default=[],
                        help="Names of sample datasets that need ARKit→ALLEx transform "
                             "(matched against directory name, e.g. 'egodex')")
    parser.add_argument("--categories", nargs="*", default=[],
                        help="LeRobot categories to compare against")
    parser.add_argument("--joints", nargs="+", default=ALL_JOINT_NAMES)
    parser.add_argument("--no-fingertips", action="store_true")
    parser.add_argument("--fingertips-only", action="store_true",
                        help="Plot only fingertips instead of all hand joints")
    parser.add_argument("--max-frames", type=int, default=50,
                        help="Max frames per episode (uniformly subsampled if longer)")
    parser.add_argument("--cam_space", action="store_true", default=False,
                        help="Convert keypoints to camera space via inv(c2w)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory for HTML files (default: vis_html/)")
    parser.add_argument("--n", type=int, default=50,
                        help="Episodes per LeRobot category")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skeleton-frame", type=int, default=0,
                        help="Frame index for skeleton visualization")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Max episodes to load per sample-dir (randomly subsampled)")
    args = parser.parse_args()

    for cat in (args.categories or []):
        if cat not in CATEGORY_CONFIG:
            parser.error(f"Unknown category '{cat}'. "
                         f"Available: {list(CATEGORY_CONFIG.keys())}")

    if args.no_fingertips:
        fingertips = None
    elif args.fingertips_only:
        fingertips = ALL_FINGERTIP_NAMES
    else:
        fingertips = ALL_HAND_JOINT_NAMES

    arkit_set = set(args.arkit_datasets)

    # Collect all dataset names that need colors (samples + categories)
    all_names = []
    for d in args.sample_dirs:
        all_names.append(Path(d).name)
    for cat in (args.categories or []):
        all_names.append(cat)

    colors = generate_colors(len(all_names))
    color_map = {name: color for name, color in zip(all_names, colors)}

    # cat_data: {name: (color, label, [traj_dict, ...])}
    cat_data = {}

    # Load sample HDF5 datasets
    for d in args.sample_dirs:
        sample_dir = Path(d)
        ds_name = sample_dir.name
        if not sample_dir.exists():
            print(f"Warning: {sample_dir} not found, skipping")
            continue

        use_arkit = ds_name in arkit_set
        print(f"[{ds_name}] Loading from {sample_dir}"
              f"{' (ARKit transform)' if use_arkit else ''}")
        trajs = load_hdf5_episodes(sample_dir, args.joints, fingertips,
                                    arkit_transform=use_arkit,
                                    cam_space=args.cam_space,
                                    max_episodes=args.max_episodes,
                                    seed=args.seed)
        trajs = [subsample_traj(t, args.max_frames) for t in trajs]
        if trajs:
            cat_data[ds_name] = (color_map[ds_name], ds_name, trajs)

    # Load LeRobot categories
    if args.categories:
        fk = NumpyFK()
        for cat_name in args.categories:
            base_path = CATEGORY_CONFIG[cat_name][0]
            datasets = list_datasets(base_path)
            print(f"[{cat_name}] Found {len(datasets)} datasets")
            if not datasets:
                print(f"  Skipping")
                continue

            samples = sample_episodes(datasets, args.n, seed=args.seed)
            print(f"  Sampled {len(samples)} episodes")

            trajs = []
            for ds, ep in samples:
                print(f"  FK: {ds.name} ep{ep}...", end="", flush=True)
                traj = load_and_compute(ds, ep, fk, args.joints, fingertips)
                trajs.append(subsample_traj(traj, args.max_frames))
                print(" done")
            label = f"lerobot ({cat_name})"
            cat_data[cat_name] = (color_map[cat_name], label, trajs)

    if not cat_data:
        print("No data loaded.")
        return

    out_dir = Path(args.output) if args.output else Path("vis_html")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build descriptive prefix from dataset names
    ds_names = "_".join(cat_data.keys())
    space = "cam" if args.cam_space else "world"
    prefix = out_dir / f"{ds_names}_{space}"

    # Body joints
    make_3d_html(cat_data, args.joints, JOINT_DISPLAY,
                 "Joint Trajectories", f"{prefix}_joints.html")

    # Hand joints / fingertips
    if fingertips:
        ft_display = FINGERTIP_DISPLAY if args.fingertips_only else HAND_JOINT_DISPLAY
        ft_right = [f for f in fingertips if f.startswith("R_")]
        ft_left = [f for f in fingertips if f.startswith("L_")]

        label = "Fingertip" if args.fingertips_only else "Hand Joint"
        make_3d_html(cat_data, ft_right, ft_display,
                     f"Right {label} Trajectories",
                     f"{prefix}_hand_right.html")
        make_3d_html(cat_data, ft_left, ft_display,
                     f"Left {label} Trajectories",
                     f"{prefix}_hand_left.html")

    # Skeleton view
    make_skeleton_html(cat_data, args.joints, fingertips,
                       "Skeleton + Trajectories", f"{prefix}_skeleton.html",
                       frame_idx=args.skeleton_frame)


if __name__ == "__main__":
    main()
