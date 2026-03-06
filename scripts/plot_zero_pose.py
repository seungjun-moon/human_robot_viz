"""
Visualize ALLEX robot skeleton at zero pose (or any given joint config).

Draws all body keypoints + fingertips connected by skeleton lines.

Usage:
    python plot_zero_pose.py
    python plot_zero_pose.py --output zero_pose.png
    python plot_zero_pose.py --azim -60 --elev 20
"""

import argparse
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from utils.kinematics_utils import NumpyFK
from utils.name_utils import (
    FINGERTIP_LINK_NAMES_RIGHT, FINGERTIP_LINK_NAMES_LEFT,
    FINGER_LINK_NAMES_PER_FINGER_RIGHT, FINGER_LINK_NAMES_PER_FINGER_LEFT,
    FINGER_ORDER,
    BODY_BONES, HAND_BONES_RIGHT, HAND_BONES_LEFT,
    BODY_KEYPOINTS, FINGER_COLORS,
)


def build_finger_bones(finger_links_per_finger: dict, hand_base: str) -> list[tuple]:
    """Build bone pairs for finger chains: base -> Roll -> Proximal -> Middle -> Distal."""
    bones = []
    for finger_name, links in finger_links_per_finger.items():
        # hand_base -> first finger link
        bones.append((hand_base, links[0]))
        # chain through finger links
        for i in range(len(links) - 1):
            bones.append((links[i], links[i + 1]))
    return bones


def plot_skeleton(q48: np.ndarray = None, fk: NumpyFK = None,
                  output: str = None, azim: float = -60, elev: float = 20,
                  title: str = "ALLEX Zero Pose"):
    if fk is None:
        fk = NumpyFK()
    if q48 is None:
        q48 = np.zeros(48)

    lt = fk.forward(q48)

    # Build all bones
    finger_bones_r = build_finger_bones(FINGER_LINK_NAMES_PER_FINGER_RIGHT,
                                         "ALLEX_Right_Hand_base")
    finger_bones_l = build_finger_bones(FINGER_LINK_NAMES_PER_FINGER_LEFT,
                                         "ALLEX_Left_Hand_base")
    # Thumb also connects from palm
    all_bones = (BODY_BONES + HAND_BONES_RIGHT + HAND_BONES_LEFT +
                 finger_bones_r + finger_bones_l)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Draw bones
    for parent, child in all_bones:
        if parent not in lt or child not in lt:
            continue
        p = lt[parent][:3, 3]
        c = lt[child][:3, 3]

        # Color: left=blue, right=red, body=gray, finger=per-finger
        color = "#555555"
        alpha = 0.4
        lw = 1.5
        if "L_Hand" in child or "Left_Hand" in child:
            color = "#1E90FF"
            alpha = 0.5
            lw = 1.0
        elif "R_Hand" in child or "Right_Hand" in child:
            color = "#DC143C"
            alpha = 0.5
            lw = 1.0
        elif child.startswith("L_"):
            color = "#4A90D9"
            alpha = 0.6
            lw = 2.0
        elif child.startswith("R_"):
            color = "#D94A4A"
            alpha = 0.6
            lw = 2.0

        # Check if it's a specific finger
        for fname, fcolor in FINGER_COLORS.items():
            if fname in child:
                color = fcolor
                alpha = 0.6
                lw = 1.0
                break

        ax.plot([p[0], c[0]], [p[1], c[1]], [p[2], c[2]],
                color=color, alpha=alpha, linewidth=lw)

    # Draw body keypoints
    for name, (link, color, size) in BODY_KEYPOINTS.items():
        if link in lt:
            pos = lt[link][:3, 3]
            ax.scatter(*pos, color=color, s=size, zorder=5, edgecolors="white", linewidths=0.5)
            ax.text(pos[0], pos[1], pos[2] + 0.015, name,
                    fontsize=7, ha="center", va="bottom", color=color)

    # Draw fingertips
    for tip_names, side_color, side_label in [
        (FINGERTIP_LINK_NAMES_RIGHT, "#DC143C", "R"),
        (FINGERTIP_LINK_NAMES_LEFT, "#1E90FF", "L"),
    ]:
        for tip_name in tip_names:
            if tip_name in lt:
                pos = lt[tip_name][:3, 3]
                finger = tip_name.split("_")[2]  # Index, Middle, etc.
                fcolor = FINGER_COLORS.get(finger, side_color)
                ax.scatter(*pos, color=fcolor, s=30, zorder=5,
                           marker="^", edgecolors="white", linewidths=0.3)

    # Draw finger link keypoints (small dots)
    for finger_dict, side_color in [
        (FINGER_LINK_NAMES_PER_FINGER_RIGHT, "#DC143C"),
        (FINGER_LINK_NAMES_PER_FINGER_LEFT, "#1E90FF"),
    ]:
        for fname, links in finger_dict.items():
            fcolor = FINGER_COLORS.get(fname, side_color)
            for lname in links:
                if lname in lt:
                    pos = lt[lname][:3, 3]
                    ax.scatter(*pos, color=fcolor, s=10, alpha=0.6, zorder=4)

    # Axis settings
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title, fontsize=13)
    ax.view_init(elev=elev, azim=azim)

    # Equal aspect ratio
    all_pos = np.array([lt[lname][:3, 3] for lname in lt])
    mid = all_pos.mean(axis=0)
    span = (all_pos.max(axis=0) - all_pos.min(axis=0)).max() / 2 * 1.1
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span, mid[2] + span)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#555555", lw=2, label="Body"),
        Line2D([0], [0], color="#4A90D9", lw=2, label="Left Arm"),
        Line2D([0], [0], color="#D94A4A", lw=2, label="Right Arm"),
    ]
    for fname, fcolor in FINGER_COLORS.items():
        legend_elements.append(Line2D([0], [0], color=fcolor, lw=1.5, label=fname))
    ax.legend(handles=legend_elements, fontsize=7, loc="upper left")

    plt.tight_layout()
    if output:
        plt.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved to {output}")
    else:
        plt.show()
    plt.close(fig)


def plot_multi_view(q48: np.ndarray = None, fk: NumpyFK = None,
                    output: str = None, title: str = "ALLEX Zero Pose"):
    """Plot skeleton from 4 viewpoints: front, side, top, 3/4 view."""
    if fk is None:
        fk = NumpyFK()
    if q48 is None:
        q48 = np.zeros(48)

    lt = fk.forward(q48)

    finger_bones_r = build_finger_bones(FINGER_LINK_NAMES_PER_FINGER_RIGHT,
                                         "ALLEX_Right_Hand_base")
    finger_bones_l = build_finger_bones(FINGER_LINK_NAMES_PER_FINGER_LEFT,
                                         "ALLEX_Left_Hand_base")
    all_bones = (BODY_BONES + HAND_BONES_RIGHT + HAND_BONES_LEFT +
                 finger_bones_r + finger_bones_l)

    all_pos = np.array([lt[lname][:3, 3] for lname in lt])
    mid = all_pos.mean(axis=0)
    span = (all_pos.max(axis=0) - all_pos.min(axis=0)).max() / 2 * 1.1

    views = [
        ("Front (YZ)", 0, 0),
        ("Side (XZ)", 0, -90),
        ("Top (XY)", 90, -90),
        ("3/4 View", 20, -60),
    ]

    fig = plt.figure(figsize=(16, 12))

    for v_idx, (view_name, elev, azim) in enumerate(views):
        ax = fig.add_subplot(2, 2, v_idx + 1, projection="3d")

        # Draw bones
        for parent, child in all_bones:
            if parent not in lt or child not in lt:
                continue
            p = lt[parent][:3, 3]
            c = lt[child][:3, 3]

            color = "#555555"
            lw = 1.5
            alpha = 0.5
            if child.startswith("L_") or "Left" in child:
                color = "#4A90D9"
            elif child.startswith("R_") or "Right" in child:
                color = "#D94A4A"
            for fname, fcolor in FINGER_COLORS.items():
                if fname in child:
                    color = fcolor
                    lw = 1.0
                    break

            ax.plot([p[0], c[0]], [p[1], c[1]], [p[2], c[2]],
                    color=color, alpha=alpha, linewidth=lw)

        # Draw body keypoints
        for name, (link, color, size) in BODY_KEYPOINTS.items():
            if link in lt:
                pos = lt[link][:3, 3]
                ax.scatter(*pos, color=color, s=size, zorder=5,
                           edgecolors="white", linewidths=0.5)
                ax.text(pos[0], pos[1], pos[2] + 0.012, name,
                        fontsize=6, ha="center", va="bottom", color=color)

        # Fingertips
        for tip_names in [FINGERTIP_LINK_NAMES_RIGHT, FINGERTIP_LINK_NAMES_LEFT]:
            for tip_name in tip_names:
                if tip_name in lt:
                    pos = lt[tip_name][:3, 3]
                    finger = tip_name.split("_")[2]
                    fcolor = FINGER_COLORS.get(finger, "#888888")
                    ax.scatter(*pos, color=fcolor, s=25, zorder=5,
                               marker="^", edgecolors="white", linewidths=0.3)

        ax.set_xlabel("X", fontsize=8)
        ax.set_ylabel("Y", fontsize=8)
        ax.set_zlabel("Z", fontsize=8)
        ax.set_title(view_name, fontsize=10)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlim(mid[0] - span, mid[0] + span)
        ax.set_ylim(mid[1] - span, mid[1] + span)
        ax.set_zlim(mid[2] - span, mid[2] + span)
        ax.tick_params(labelsize=6)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    if output:
        plt.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved to {output}")
    else:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize ALLEX skeleton at zero pose")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file prefix (generates _single.png and _multi.png)")
    parser.add_argument("--azim", type=float, default=-60, help="Azimuth angle")
    parser.add_argument("--elev", type=float, default=20, help="Elevation angle")
    args = parser.parse_args()

    fk = NumpyFK()
    q48 = np.zeros(48)

    base = args.output or "zero_pose"
    base = base.rsplit(".", 1)[0]

    plot_skeleton(q48, fk, output=f"{base}.png", azim=args.azim, elev=args.elev)
    plot_multi_view(q48, fk, output=f"{base}_views.png")


if __name__ == "__main__":
    main()
