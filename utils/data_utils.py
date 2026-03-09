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
                        max_episodes: int = None,
                        seed: int = 42,
                        undo_geocalib: bool = False) -> list[dict]:
    """Load raw *.hdf5 sample files, extract positions from SE3 transforms.

    Assumes input HDF5 files contain transforms already in the desired
    coordinate frame (e.g. ROS conventions). No axis conversion is applied.

    Args:
        dataset_dir: Path to the dataset directory containing .hdf5 files.
        joints: List of joint keys to extract (e.g. from ALL_JOINT_NAMES).
        fingertips: Optional list of fingertip/hand joint keys.
        max_episodes: If set, randomly subsample to this many episodes.
        seed: Random seed for subsampling.
        undo_geocalib: If True and transforms/gravity exists, undo the
            GeoCalib gravity alignment on camera extrinsics.

    Returns:
        List of trajectory dicts, each: {joint_key: {"pos": (T,3), "rpy": (T,3)},
                                         "_camera_c2w": (T,4,4) if available,
                                         "_video_path": ..., "_label": ...}
    """
    raw_files = _find_hdf5_files(dataset_dir)
    if not raw_files:
        print(f"  No raw .hdf5 files found in {dataset_dir}")
        return []

    if max_episodes is not None and len(raw_files) > max_episodes:
        total = len(raw_files)
        rng = random.Random(seed)
        raw_files = sorted(rng.sample(raw_files, max_episodes))
        print(f"  Subsampled to {max_episodes}/{total} episodes")

    all_keys = list(joints)
    key_map = {k: EGODEX_JOINT_MAP[k] for k in joints}
    if fingertips:
        all_keys += fingertips
        for ft in fingertips:
            key_map[ft] = EGODEX_HAND_JOINT_MAP[ft]

    trajs = []
    for rf in raw_files:
        rel = rf.relative_to(dataset_dir)
        print(f"  Loading: {rel}...", end="", flush=True)
        with h5py.File(rf, "r") as f:
            transforms = f["transforms"]

            # Determine T from first available joint
            first_key = key_map[all_keys[0]]
            T = transforms[first_key].shape[0]

            result = {}
            for k in all_keys:
                tf = transforms[key_map[k]][:]  # (T, 4, 4)
                pos = tf[:, :3, 3]
                rpy = np.zeros((T, 3))
                for t in range(T):
                    rpy[t] = rotation_matrix_to_euler(tf[t, :3, :3])
                result[k] = {"pos": pos, "rpy": rpy}

            # Store camera c2w for frustum visualization.
            # If transforms/gravity exists, it is the GeoCalib R_align rotation
            # that was applied as R_align @ c2w. Undo it to get the original
            # camera extrinsics: original_c2w = R_align.T @ aligned_c2w.
            if "camera" in transforms:
                cam_c2w = transforms["camera"][:]  # (T, 4, 4)
                if undo_geocalib and "gravity" in transforms:
                    R_align = transforms["gravity"][:]  # (3, 3)
                    R_inv = np.eye(4, dtype=cam_c2w.dtype)
                    R_inv[:3, :3] = R_align.T
                    cam_c2w = np.einsum("ij,tjk->tik", R_inv, cam_c2w)
                result["_camera_c2w"] = cam_c2w

        result["_video_path"] = _find_video_for_hdf5(rf)
        result["_label"] = str(rel)
        trajs.append(result)
        print(" done")

    return trajs
