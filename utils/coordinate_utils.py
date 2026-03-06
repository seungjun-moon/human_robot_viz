"""
Coordinate conversion utilities.

Provides:
- ARKitToAllexConverter: Converts ARKit body tracking to ALLEx FK coordinate convention
"""

import numpy as np


class ARKitToAllexConverter:
    """Converts ARKit body tracking to ALLEx FK coordinate convention.

    ARKit body-relative (after hip inverse): X=left, Y=up, Z=forward
    ALLEx FK:                                X=forward, Y=left, Z=up

    Steps:
      1. Apply full hip inverse (rotation + translation) → body-relative frame
      2. Remap axes: ALLEx X = ARKit Z, ALLEx Y = ARKit X, ALLEx Z = ARKit Y
    """

    def __init__(self):
        # Axis remapping rotation (det=+1, proper rotation)
        self.R = np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
        ], dtype=np.float32)

        # 4x4 version
        self.T = np.eye(4, dtype=np.float32)
        self.T[:3, :3] = self.R

    def convert_frame(self, hip_tf: np.ndarray,
                      joint_tfs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Convert a full frame of joint transforms.

        Args:
            hip_tf: (4, 4) hip world transform for this frame
            joint_tfs: {name: (4, 4)} world transforms

        Returns:
            {name: (4, 4)} transforms in ALLEx convention (hip-centered)
        """
        hip_inv = np.linalg.inv(hip_tf)
        result = {}
        for name, tf in joint_tfs.items():
            body_rel = hip_inv @ tf          # body-relative (ARKit axes)
            result[name] = self.T @ body_rel  # remap to ALLEx axes
        return result
