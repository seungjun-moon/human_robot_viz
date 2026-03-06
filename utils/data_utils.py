"""
Data loading utilities for LeRobot and HDF5 sample datasets.

Provides:
- list_datasets / sample_episodes: Discover and sample LeRobot datasets
- load_hdf5_episodes: Load raw HDF5 sample datasets (egodex, ego10k, etc.)
"""

import json
import random
from pathlib import Path

import h5py
import numpy as np

from utils.coordinate_utils import ARKitToAllexConverter
from utils.kinematics_utils import rotation_matrix_to_euler
from utils.name_utils import (
    EGODEX_JOINT_MAP, EGODEX_HAND_JOINT_MAP,
)


def list_datasets(base: Path, required_state_dim: int = 48) -> list[Path]:
    """List dataset dirs that have observation.state with the required dimension."""
    results = []
    if not base.exists():
        return results
    for d in sorted(base.iterdir()):
        info_path = d / "meta" / "info.json"
        if not info_path.exists():
            continue
        with open(info_path) as f:
            info = json.load(f)
        state_feat = info.get("features", {}).get("observation.state", {})
        if state_feat.get("shape") == [required_state_dim]:
            results.append(d)
    return results


def sample_episodes(datasets: list[Path], n: int = 10, seed: int = 42) -> list[tuple[Path, int]]:
    """Sample n (dataset, episode_idx) pairs from different datasets."""
    rng = random.Random(seed)
    if not datasets:
        return []
    if len(datasets) >= n:
        chosen = rng.sample(datasets, n)
    else:
        chosen = [rng.choice(datasets) for _ in range(n)]

    samples = []
    for ds_path in chosen:
        with open(ds_path / "meta" / "info.json") as f:
            info = json.load(f)
        n_eps = info["total_episodes"]
        ep_idx = rng.randint(0, n_eps - 1)
        samples.append((ds_path, ep_idx))
    return samples


def _find_video_for_hdf5(hdf5_path: Path) -> str | None:
    """Find a matching video file for an HDF5 file.

    Tries: {stem}_resized.mp4, {stem}.mp4 in the same directory.
    """
    for suffix in ("_resized.mp4", ".mp4"):
        candidate = hdf5_path.with_name(hdf5_path.stem + suffix)
        if candidate.exists():
            return str(candidate)
    return None


def _find_hdf5_files(directory: Path) -> list[Path]:
    """Find raw HDF5 files, searching subdirectories if the dir itself has none."""
    raw_files = sorted(
        f for f in directory.glob("*.hdf5")
        if not f.name.endswith("_mano.hdf5")
    )
    if raw_files:
        return raw_files
    # Search subdirectories
    return sorted(
        f for f in directory.rglob("*.hdf5")
        if not f.name.endswith("_mano.hdf5")
    )


def load_hdf5_episodes(dataset_dir: Path, joints: list[str],
                        fingertips: list[str] = None,
                        arkit_transform: bool = False,
                        cam_space: bool = False) -> list[dict]:
    """Load raw *.hdf5 sample files, extract positions from SE3 transforms.

    All sample datasets live under samples/[DATASET_NAME]/.../*.hdf5 and share
    the same HDF5 format (transforms group with joint SE3 matrices).

    Args:
        dataset_dir: Path to the dataset directory containing .hdf5 files.
        joints: List of joint keys to extract (e.g. from ALL_JOINT_NAMES).
        fingertips: Optional list of fingertip/hand joint keys.
        arkit_transform: If True, apply ARKit→ALLEx hip-centered coordinate
            conversion (needed for raw egodex data). If False, read transforms
            directly (ego10k, converted datasets, etc.).
        cam_space: If True, convert world-space joints into camera space
            using inv(c2w) from transforms/camera.

    Returns:
        List of trajectory dicts, each: {joint_key: {"pos": (T,3), "rpy": (T,3)},
                                         "_video_path": ..., "_label": ...}
    """
    raw_files = _find_hdf5_files(dataset_dir)
    if not raw_files:
        print(f"  No raw .hdf5 files found in {dataset_dir}")
        return []

    all_keys = list(joints)
    key_map = {k: EGODEX_JOINT_MAP[k] for k in joints}
    if fingertips:
        all_keys += fingertips
        for ft in fingertips:
            key_map[ft] = EGODEX_HAND_JOINT_MAP[ft]

    converter = ARKitToAllexConverter() if arkit_transform else None

    trajs = []
    for rf in raw_files:
        rel = rf.relative_to(dataset_dir)
        print(f"  Loading: {rel}...", end="", flush=True)
        with h5py.File(rf, "r") as f:
            transforms = f["transforms"]

            if arkit_transform:
                # ARKit mode: hip-centered conversion
                T = transforms["hip"].shape[0]
                hip_raw = transforms["hip"][:]  # (T, 4, 4)
                joint_raw = {k: transforms[key_map[k]][:] for k in all_keys}

                result = {}
                for k in all_keys:
                    result[k] = {"pos": np.zeros((T, 3)), "rpy": np.zeros((T, 3))}

                for t in range(T):
                    frame_tfs = {k: joint_raw[k][t] for k in all_keys}
                    converted = converter.convert_frame(hip_raw[t], frame_tfs)
                    for k in all_keys:
                        result[k]["pos"][t] = converted[k][:3, 3]
                        result[k]["rpy"][t] = rotation_matrix_to_euler(converted[k][:3, :3])
            else:
                # Direct mode: read transforms as-is
                # Determine T from first available mapped joint
                direct_keys = [k for k in all_keys if k != "waist"]
                first_mapped = key_map[direct_keys[0]] if direct_keys else "hip"
                T = transforms[first_mapped].shape[0]

                # Load c2w and compute w2c if cam_space requested
                w2c = None
                if cam_space and "camera" in transforms:
                    c2w = transforms["camera"][:]  # (T, 4, 4)
                    w2c = np.linalg.inv(c2w)       # (T, 4, 4)

                result = {}
                # "waist" maps to "hip" which may be dummy; use "camera" if available
                if "waist" in all_keys and "camera" in transforms:
                    cam_tf = transforms["camera"][:]  # (T, 4, 4)
                    if cam_space:
                        # Camera is at origin in cam space
                        cam_pos = np.zeros((T, 3))
                        cam_rpy = np.zeros((T, 3))
                    else:
                        cam_pos = cam_tf[:, :3, 3]
                        cam_rpy = np.zeros((T, 3))
                        for t in range(T):
                            cam_rpy[t] = rotation_matrix_to_euler(cam_tf[t, :3, :3])
                    result["waist"] = {"pos": cam_pos, "rpy": cam_rpy}

                for k in (direct_keys if "waist" in all_keys else all_keys):
                    tf = transforms[key_map[k]][:]  # (T, 4, 4)
                    if w2c is not None:
                        tf = w2c @ tf  # world → camera space
                    pos = tf[:, :3, 3]
                    rpy = np.zeros((T, 3))
                    for t in range(T):
                        rpy[t] = rotation_matrix_to_euler(tf[t, :3, :3])
                    result[k] = {"pos": pos, "rpy": rpy}

        result["_video_path"] = _find_video_for_hdf5(rf)
        result["_label"] = str(rel)
        trajs.append(result)
        print(" done")

    return trajs
