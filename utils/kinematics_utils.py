"""
Forward Kinematics module for ALLEX robot (48-DOF + 12 mimic joints).

Provides:
- NumpyFK: CPU-based FK for data precomputation
- Helper functions for skeleton position extraction and finger force computation
- Constants for joint names, mimic joints, and finger joints
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from utils.name_utils import (  # noqa: F401 — re-exported for backwards compatibility
    JOINT_NAMES_48,
    MIMIC_JOINTS,
    FINGER_JOINT_NAMES_RIGHT,
    FINGER_JOINT_NAMES_LEFT,
    FINGER_MIMIC_JOINT_NAMES_RIGHT,
    FINGER_MIMIC_JOINT_NAMES_LEFT,
    FINGER_JOINT_NAMES_ALL_RIGHT,
    FINGER_JOINT_NAMES_ALL_LEFT,
    FINGER_JOINT_NAMES,
    N_FINGER_JOINTS_PER_HAND,
    FINGER_ORDER,
    FINGER_LINK_NAMES_PER_FINGER_RIGHT,
    FINGER_LINK_NAMES_PER_FINGER_LEFT,
    PALM_LINK_RIGHT,
    PALM_LINK_LEFT,
    FINGER_LINK_NAMES_ORDERED_RIGHT,
    FINGER_LINK_NAMES_ORDERED_LEFT,
    FINGERTIP_LINK_NAMES_RIGHT,
    FINGERTIP_LINK_NAMES_LEFT,
    BODY_LINK_NAMES,
    WRIST_LINK_NAMES,
    PALM_LINK_NAMES,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

URDF_PATH = Path(__file__).parent.parent / "assets" / "allex_v2_urdf" / "allex.urdf"

# Frame transform: URDF frame -> pad-normal frame
# Redefines z-axis as pad normal direction (-y of URDF frame)
#   new x = old x (rotation axis, lateral)
#   new y = old z (bone direction, proximal -> distal)
#   new z = -old y (pad normal, toward palm/object)
_URDF_TO_PAD_NORMAL_FRAME = np.array([
    [1,  0,  0],
    [0,  0, -1],
    [0,  1,  0],
], dtype=np.float64)


# ─────────────────────────────────────────────────────────────────────────────
# URDF Parsing Utilities
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class JointInfo:
    name: str
    joint_type: str  # "revolute", "fixed"
    parent_link: str
    child_link: str
    origin_xyz: np.ndarray  # (3,)
    origin_rpy: np.ndarray  # (3,)
    axis: np.ndarray  # (3,)
    mimic_joint: Optional[str] = None
    mimic_multiplier: float = 1.0
    mimic_offset: float = 0.0


@dataclass
class KinematicTree:
    joints: dict  # name -> JointInfo
    links: list  # all link names
    root_link: str
    children: dict  # link_name -> [(joint_name, child_link), ...]
    topo_order: list = field(default_factory=list)  # topologically sorted joint names


def parse_urdf(urdf_path: str) -> KinematicTree:
    """Parse URDF XML and build kinematic tree."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    joints = {}
    all_links = set()
    child_links = set()
    children = {}  # parent_link -> [(joint_name, child_link)]

    for joint_elem in root.findall("joint"):
        name = joint_elem.get("name")
        jtype = joint_elem.get("type")

        parent_link = joint_elem.find("parent").get("link")
        child_link = joint_elem.find("child").get("link")

        origin = joint_elem.find("origin")
        if origin is not None:
            xyz = np.array([float(x) for x in origin.get("xyz", "0 0 0").split()])
            rpy = np.array([float(x) for x in origin.get("rpy", "0 0 0").split()])
        else:
            xyz = np.zeros(3)
            rpy = np.zeros(3)

        axis_elem = joint_elem.find("axis")
        if axis_elem is not None:
            axis = np.array([float(x) for x in axis_elem.get("xyz", "1 0 0").split()])
        else:
            axis = np.array([1.0, 0.0, 0.0])

        mimic_elem = joint_elem.find("mimic")
        mimic_joint = None
        mimic_mult = 1.0
        mimic_off = 0.0
        if mimic_elem is not None:
            mimic_joint = mimic_elem.get("joint")
            mimic_mult = float(mimic_elem.get("multiplier", "1.0"))
            mimic_off = float(mimic_elem.get("offset", "0.0"))

        ji = JointInfo(
            name=name,
            joint_type=jtype,
            parent_link=parent_link,
            child_link=child_link,
            origin_xyz=xyz,
            origin_rpy=rpy,
            axis=axis,
            mimic_joint=mimic_joint,
            mimic_multiplier=mimic_mult,
            mimic_offset=mimic_off,
        )
        joints[name] = ji

        all_links.add(parent_link)
        all_links.add(child_link)
        child_links.add(child_link)

        if parent_link not in children:
            children[parent_link] = []
        children[parent_link].append((name, child_link))

    root_link = (all_links - child_links).pop()

    # Topological sort via BFS from root
    topo = []
    queue = [root_link]
    visited = set()
    while queue:
        link = queue.pop(0)
        if link in visited:
            continue
        visited.add(link)
        for jname, clink in children.get(link, []):
            topo.append(jname)
            queue.append(clink)

    return KinematicTree(
        joints=joints,
        links=sorted(all_links),
        root_link=root_link,
        children=children,
        topo_order=topo,
    )


def rpy_to_matrix(r, p, y):
    """RPY (roll-pitch-yaw / XYZ extrinsic) to 3x3 rotation matrix."""
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def axis_angle_to_matrix(axis, angle):
    """Rodrigues' formula: axis-angle to 3x3 rotation matrix."""
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def make_transform(R, t):
    """Create 4x4 homogeneous transform from 3x3 R and 3-vec t."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def joint_origin_transform(ji: JointInfo) -> np.ndarray:
    """Static 4x4 transform from joint origin (xyz, rpy)."""
    R = rpy_to_matrix(*ji.origin_rpy)
    return make_transform(R, ji.origin_xyz)


def expand_48_to_full(q48: np.ndarray) -> dict:
    """Expand 48 independent joint values to all revolute joints including mimic."""
    joint_vals = {}
    for i, name in enumerate(JOINT_NAMES_48):
        joint_vals[name] = q48[i]
    for mimic_name, (parent_name, mult, off) in MIMIC_JOINTS.items():
        joint_vals[mimic_name] = joint_vals[parent_name] * mult + off
    return joint_vals


# ─────────────────────────────────────────────────────────────────────────────
# NumpyFK: CPU-based Forward Kinematics
# ─────────────────────────────────────────────────────────────────────────────

class NumpyFK:
    """CPU-based forward kinematics using NumPy. Used for data precomputation."""

    def __init__(self, urdf_path: str = None):
        if urdf_path is None:
            urdf_path = str(URDF_PATH)
        self.ktree = parse_urdf(urdf_path)
        # Pre-compute static origin transforms for each joint
        self.origin_transforms = {}
        for jname, ji in self.ktree.joints.items():
            self.origin_transforms[jname] = joint_origin_transform(ji)

        # Build revolute joint name list in topological order
        self.revolute_names = []
        for jname in self.ktree.topo_order:
            ji = self.ktree.joints[jname]
            if ji.joint_type == "revolute":
                self.revolute_names.append(jname)

    def forward(self, q48: np.ndarray) -> dict:
        """
        Compute FK for a single configuration.
        Returns: dict of link_name -> 4x4 transform (numpy).
        """
        joint_vals = expand_48_to_full(q48)
        link_transforms = {self.ktree.root_link: np.eye(4)}

        for jname in self.ktree.topo_order:
            ji = self.ktree.joints[jname]
            parent_T = link_transforms[ji.parent_link]
            T_origin = self.origin_transforms[jname]

            if ji.joint_type == "revolute":
                angle = joint_vals.get(jname, 0.0)
                R_joint = axis_angle_to_matrix(ji.axis, angle)
                T_joint = make_transform(R_joint, np.zeros(3))
                link_transforms[ji.child_link] = parent_T @ T_origin @ T_joint
            else:  # fixed
                link_transforms[ji.child_link] = parent_T @ T_origin

        return link_transforms

    def forward_with_joint_transforms(self, q48: np.ndarray) -> tuple:
        """
        Compute FK and return both link transforms and per-joint transforms.
        Returns: (link_transforms, joint_transforms)
            - link_transforms: dict of link_name -> 4x4
            - joint_transforms: dict of joint_name -> 4x4 (child link frame)
        """
        joint_vals = expand_48_to_full(q48)
        link_transforms = {self.ktree.root_link: np.eye(4)}
        joint_transforms = {}

        for jname in self.ktree.topo_order:
            ji = self.ktree.joints[jname]
            parent_T = link_transforms[ji.parent_link]
            T_origin = self.origin_transforms[jname]

            if ji.joint_type == "revolute":
                angle = joint_vals.get(jname, 0.0)
                R_joint = axis_angle_to_matrix(ji.axis, angle)
                T_joint = make_transform(R_joint, np.zeros(3))
                child_T = parent_T @ T_origin @ T_joint
                link_transforms[ji.child_link] = child_T
                joint_transforms[jname] = child_T
            else:  # fixed
                child_T = parent_T @ T_origin
                link_transforms[ji.child_link] = child_T
                joint_transforms[jname] = child_T

        return link_transforms, joint_transforms


# ─────────────────────────────────────────────────────────────────────────────
# Skeleton Position Extraction
# ─────────────────────────────────────────────────────────────────────────────

def get_skeleton_positions(fk: NumpyFK, q48: np.ndarray) -> np.ndarray:
    """
    Extract xyz positions of all revolute joints from FK result.

    Args:
        fk: NumpyFK instance
        q48: (48,) joint angles

    Returns:
        positions: (n_revolute, 3) array of xyz positions
            n_revolute = 60 (48 independent + 12 mimic)
    """
    _, joint_transforms = fk.forward_with_joint_transforms(q48)

    positions = []
    for jname in fk.revolute_names:
        T = joint_transforms[jname]
        positions.append(T[:3, 3])

    return np.array(positions)  # (n_revolute, 3)


def get_skeleton_positions_flat(fk: NumpyFK, q48: np.ndarray) -> np.ndarray:
    """
    Get skeleton positions as a flat vector.

    Returns:
        positions_flat: (n_revolute * 3,) = (180,) array
    """
    return get_skeleton_positions(fk, q48).flatten()


# ─────────────────────────────────────────────────────────────────────────────
# Finger Force Computation
# ─────────────────────────────────────────────────────────────────────────────

def get_finger_orientations(fk: NumpyFK, q48: np.ndarray, hand: str = "right") -> np.ndarray:
    """
    Extract rotation matrices for finger joints from FK result.

    Returns orientations in pad-normal frame convention:
      - x-axis = joint rotation axis (lateral)
      - y-axis = bone direction (proximal -> distal)
      - z-axis = pad normal (toward palm/grasped object)

    Args:
        fk: NumpyFK instance
        q48: (48,) joint angles
        hand: "right" or "left"

    Returns:
        orientations: (15, 3, 3) rotation matrices for independent finger joints
            (in pad-normal frame)
    """
    _, joint_transforms = fk.forward_with_joint_transforms(q48)

    if hand == "right":
        finger_names = FINGER_JOINT_NAMES_RIGHT
    else:
        finger_names = FINGER_JOINT_NAMES_LEFT

    orientations = []
    for jname in finger_names:
        T = joint_transforms[jname]
        orientations.append(T[:3, :3] @ _URDF_TO_PAD_NORMAL_FRAME)

    return np.array(orientations)  # (15, 3, 3)


def compute_finger_forces(orientations: np.ndarray, effort_scalars: np.ndarray) -> np.ndarray:
    """
    Compute finger force vectors from FK orientations and effort scalars.

    Force direction = z-axis of each joint's rotation matrix.
    After frame transform, z-axis = pad normal direction:
      - x-axis = joint rotation axis (lateral)
      - y-axis = bone direction (proximal -> distal)
      - z-axis = pad normal (toward palm/grasped object)

    Args:
        orientations: (15, 3, 3) rotation matrices (in pad-normal frame)
        effort_scalars: (15,) effort values per finger joint

    Returns:
        forces: (15, 3) force vectors in pad normal direction
    """
    z_axes = orientations[:, :, 2]  # z-axis = pad normal (after frame transform)
    forces = effort_scalars[:, None] * z_axes
    return forces


def compute_finger_forces_flat(fk: NumpyFK, q48: np.ndarray,
                                right_effort: np.ndarray,
                                left_effort: np.ndarray) -> np.ndarray:
    """
    Compute finger forces for both hands as a flat vector.
    (Joint-based: 15 independent joints per hand)

    Args:
        fk: NumpyFK instance
        q48: (48,) joint angles
        right_effort: (15,) right hand effort values
        left_effort: (15,) left hand effort values

    Returns:
        forces_flat: (90,) = 2 hands x 15 joints x 3 xyz
    """
    right_orientations = get_finger_orientations(fk, q48, hand="right")
    left_orientations = get_finger_orientations(fk, q48, hand="left")

    right_forces = compute_finger_forces(right_orientations, right_effort)
    left_forces = compute_finger_forces(left_orientations, left_effort)

    return np.concatenate([right_forces.flatten(), left_forces.flatten()])  # (90,)


# ─────────────────────────────────────────────────────────────────────────────
# Link-based Finger Force Computation
# ─────────────────────────────────────────────────────────────────────────────

# Number of links per hand for link-based forces
N_FINGER_LINKS_PER_HAND = 21  # 1 palm + 5 fingers x 4 links

# Effort-to-link mapping for 15 independent effort values per hand.
# The 15 effort values correspond to:
#   Index:  [0]=Roll, [1]=MCP, [2]=PIP
#   Middle: [3]=Roll, [4]=MCP, [5]=PIP
#   Ring:   [6]=Roll, [7]=MCP, [8]=PIP
#   Little: [9]=Roll, [10]=MCP, [11]=PIP
#   Thumb:  [12]=Yaw, [13]=CMC, [14]=MCP
#
# Link effort mapping per finger (4 links each):
#   Roll/Yaw link:  effort from Roll/Yaw joint
#   Proximal link:  effort from MCP/CMC joint
#   Middle link:    effort from PIP/MCP joint
#   Distal link:    effort from DIP/IP mimic joint
#
# For palm: effort = 0 (no direct actuation)

# Mimic multiplier for DIP/IP joints
_MIMIC_MULT_FINGER = 0.98  # DIP = PIP * 0.98
_MIMIC_MULT_THUMB_L = 0.77   # L_Thumb: IP = MCP * 0.77
_MIMIC_MULT_THUMB_R = 0.98   # R_Thumb: IP = MCP * 0.98


def _build_link_effort_vector(effort15: np.ndarray, hand: str = "right") -> np.ndarray:
    """
    Build 21-element effort vector for link-based forces from 15 independent efforts.

    Args:
        effort15: (15,) effort values in order:
            [0-2] Index (Roll, MCP, PIP),
            [3-5] Middle (Roll, MCP, PIP),
            [6-8] Ring (Roll, MCP, PIP),
            [9-11] Little (Roll, MCP, PIP),
            [12-14] Thumb (Yaw, CMC, MCP)
        hand: "right" or "left"

    Returns:
        effort21: (21,) effort values mapped to links in order:
            [0] Palm (0.0),
            [1-4] Index (Roll, Proximal=MCP, Middle=PIP, Distal=DIP_mimic),
            [5-8] Middle (...),
            [9-12] Ring (...),
            [13-16] Little (...),
            [17-20] Thumb (Yaw, Proximal=CMC, Middle=MCP, Distal=IP_mimic)
    """
    effort21 = np.zeros(21, dtype=effort15.dtype)
    # Palm: index 0, effort = 0
    effort21[0] = 0.0

    # For each of 4 regular fingers (Index, Middle, Ring, Little)
    for finger_idx in range(4):
        base_in = finger_idx * 3   # start index in effort15
        base_out = 1 + finger_idx * 4  # start index in effort21
        roll_effort = effort15[base_in]
        mcp_effort = effort15[base_in + 1]
        pip_effort = effort15[base_in + 2]
        dip_effort = pip_effort * _MIMIC_MULT_FINGER

        effort21[base_out] = roll_effort       # Roll link
        effort21[base_out + 1] = mcp_effort    # Proximal link
        effort21[base_out + 2] = pip_effort    # Middle link
        effort21[base_out + 3] = dip_effort    # Distal link

    # Thumb (finger_idx = 4)
    base_in = 12
    base_out = 17
    thumb_mult = _MIMIC_MULT_THUMB_R if hand == "right" else _MIMIC_MULT_THUMB_L
    yaw_effort = effort15[base_in]
    cmc_effort = effort15[base_in + 1]
    mcp_effort_thumb = effort15[base_in + 2]
    ip_effort = mcp_effort_thumb * thumb_mult

    effort21[base_out] = yaw_effort            # Yaw link
    effort21[base_out + 1] = cmc_effort        # Proximal link
    effort21[base_out + 2] = mcp_effort_thumb  # Middle link
    effort21[base_out + 3] = ip_effort         # Distal link

    return effort21


def get_finger_link_orientations(fk: NumpyFK, q48: np.ndarray,
                                  hand: str = "right") -> np.ndarray:
    """
    Extract rotation matrices for finger link frames from FK result.

    Uses link transforms (not joint transforms) to get the orientation at
    each link's frame in world coordinates. Returns orientations in
    pad-normal frame convention:
      - x-axis = joint rotation axis (lateral)
      - y-axis = bone direction (proximal -> distal)
      - z-axis = pad normal (toward palm/grasped object)

    Args:
        fk: NumpyFK instance
        q48: (48,) joint angles
        hand: "right" or "left"

    Returns:
        orientations: (21, 3, 3) rotation matrices (in pad-normal frame) for:
            [0] Palm, [1-4] Index, [5-8] Middle, [9-12] Ring,
            [13-16] Little, [17-20] Thumb
    """
    link_transforms = fk.forward(q48)

    if hand == "right":
        link_names = FINGER_LINK_NAMES_ORDERED_RIGHT
    else:
        link_names = FINGER_LINK_NAMES_ORDERED_LEFT

    orientations = []
    for lname in link_names:
        T = link_transforms[lname]
        orientations.append(T[:3, :3] @ _URDF_TO_PAD_NORMAL_FRAME)

    return np.array(orientations)  # (21, 3, 3)


def get_finger_link_positions(fk: NumpyFK, q48: np.ndarray,
                               hand: str = "right") -> np.ndarray:
    """
    Extract xyz positions for finger link frames from FK result.

    Args:
        fk: NumpyFK instance
        q48: (48,) joint angles
        hand: "right" or "left"

    Returns:
        positions: (21, 3) xyz positions for:
            [0] Palm, [1-4] Index, [5-8] Middle, [9-12] Ring,
            [13-16] Little, [17-20] Thumb
    """
    link_transforms = fk.forward(q48)

    if hand == "right":
        link_names = FINGER_LINK_NAMES_ORDERED_RIGHT
    else:
        link_names = FINGER_LINK_NAMES_ORDERED_LEFT

    positions = []
    for lname in link_names:
        T = link_transforms[lname]
        positions.append(T[:3, 3])

    return np.array(positions)  # (21, 3)


def compute_finger_link_forces(orientations: np.ndarray,
                                effort_scalars: np.ndarray) -> np.ndarray:
    """
    Compute finger force vectors from link orientations and effort scalars.

    Force direction = z-axis of each link's rotation matrix.
    After frame transform, z-axis = pad normal direction:
      - x-axis = joint rotation axis (lateral)
      - y-axis = bone direction (proximal -> distal)
      - z-axis = pad normal (toward palm/grasped object)

    Args:
        orientations: (21, 3, 3) rotation matrices (in pad-normal frame)
        effort_scalars: (21,) effort values mapped to links

    Returns:
        forces: (21, 3) force vectors in pad normal direction
    """
    z_axes = orientations[:, :, 2]  # z-axis = pad normal (after frame transform)
    forces = effort_scalars[:, None] * z_axes
    return forces


def compute_finger_link_forces_flat(fk: NumpyFK, q48: np.ndarray,
                                     right_effort: np.ndarray,
                                     left_effort: np.ndarray) -> np.ndarray:
    """
    Compute link-based finger forces for both hands as a flat vector.

    Uses link frames instead of joint frames. Includes Palm, all finger
    phalanges (Roll/Yaw, Proximal, Middle, Distal), and Tip links.
    DIP/IP mimic efforts are computed automatically from PIP/MCP efforts.

    Args:
        fk: NumpyFK instance
        q48: (48,) joint angles
        right_effort: (15,) right hand effort values (independent joints only)
        left_effort: (15,) left hand effort values (independent joints only)

    Returns:
        forces_flat: (126,) = 2 hands x 21 links x 3 xyz
            Per hand (63): Palm(3) + Index(12) + Middle(12) + Ring(12) +
                           Little(12) + Thumb(12)
    """
    # Get link orientations
    right_orientations = get_finger_link_orientations(fk, q48, hand="right")
    left_orientations = get_finger_link_orientations(fk, q48, hand="left")

    # Build link effort vectors (15 -> 21 with mimic expansion + palm)
    right_effort21 = _build_link_effort_vector(right_effort, hand="right")
    left_effort21 = _build_link_effort_vector(left_effort, hand="left")

    # Compute forces
    right_forces = compute_finger_link_forces(right_orientations, right_effort21)
    left_forces = compute_finger_link_forces(left_orientations, left_effort21)

    return np.concatenate([right_forces.flatten(), left_forces.flatten()])  # (126,)


# ─────────────────────────────────────────────────────────────────────────────
# Main API: Get Keypoints from 48-dim Joint Actions
# ─────────────────────────────────────────────────────────────────────────────

def get_keypoints(q48: np.ndarray, urdf_path: str = None) -> dict:
    """
    Compute 3D keypoint positions from 48-dim joint actions.

    This is the main API function. Given 48 joint angles in env_cfg action order,
    returns fingertip positions, all joint positions, wrist positions, etc.

    Args:
        q48: (48,) joint angles in env_cfg action order
        urdf_path: path to ALLEX URDF file (uses default if None)

    Returns:
        dict with keys:
            'fingertips_right': (5, 3) - right hand fingertip xyz
                [Index, Middle, Ring, Little, Thumb]
            'fingertips_left': (5, 3) - left hand fingertip xyz
                [Index, Middle, Ring, Little, Thumb]
            'fingertips_all': (10, 3) - all fingertips [right(5), left(5)]
            'joint_positions': (n_revolute, 3) - all revolute joint xyz
            'joint_names': list of joint names matching joint_positions
            'wrist_right': (3,) - right wrist xyz
            'wrist_left': (3,) - left wrist xyz
            'palm_right': (3,) - right palm xyz
            'palm_left': (3,) - left palm xyz
            'body_keypoints': (n_body, 3) - body link xyz
            'body_names': list of body link names
            'link_transforms': dict of link_name -> 4x4 transform
    """
    fk = NumpyFK(urdf_path)
    link_transforms = fk.forward(q48)

    # Fingertip positions
    right_tips = np.array([
        link_transforms[name][:3, 3] for name in FINGERTIP_LINK_NAMES_RIGHT
    ])
    left_tips = np.array([
        link_transforms[name][:3, 3] for name in FINGERTIP_LINK_NAMES_LEFT
    ])

    # All revolute joint positions
    _, joint_transforms = fk.forward_with_joint_transforms(q48)
    joint_positions = []
    joint_names = []
    for jname in fk.revolute_names:
        T = joint_transforms[jname]
        joint_positions.append(T[:3, 3])
        joint_names.append(jname)

    # Wrist & palm positions
    wrist_r = link_transforms[WRIST_LINK_NAMES["right"]][:3, 3]
    wrist_l = link_transforms[WRIST_LINK_NAMES["left"]][:3, 3]
    palm_r = link_transforms[PALM_LINK_NAMES["right"]][:3, 3]
    palm_l = link_transforms[PALM_LINK_NAMES["left"]][:3, 3]

    # Body keypoints
    body_positions = []
    body_names = []
    for lname in BODY_LINK_NAMES:
        if lname in link_transforms:
            body_positions.append(link_transforms[lname][:3, 3])
            body_names.append(lname)

    return {
        "fingertips_right": right_tips,                       # (5, 3)
        "fingertips_left": left_tips,                         # (5, 3)
        "fingertips_all": np.concatenate([right_tips, left_tips]),  # (10, 3)
        "joint_positions": np.array(joint_positions),         # (n_revolute, 3)
        "joint_names": joint_names,
        "wrist_right": wrist_r,                               # (3,)
        "wrist_left": wrist_l,                                # (3,)
        "palm_right": palm_r,                                 # (3,)
        "palm_left": palm_l,                                  # (3,)
        "body_keypoints": np.array(body_positions),           # (n_body, 3)
        "body_names": body_names,
        "link_transforms": link_transforms,
    }


def get_keypoints_batch(q48_batch: np.ndarray, urdf_path: str = None) -> dict:
    """
    Compute 3D keypoints for a batch of joint configurations.

    Args:
        q48_batch: (N, 48) joint angles
        urdf_path: path to ALLEX URDF file

    Returns:
        dict with same keys as get_keypoints but with batch dimension:
            'fingertips_right': (N, 5, 3)
            'fingertips_left': (N, 5, 3)
            'fingertips_all': (N, 10, 3)
            'joint_positions': (N, n_revolute, 3)
            'wrist_right': (N, 3)
            'wrist_left': (N, 3)
            'palm_right': (N, 3)
            'palm_left': (N, 3)
            'body_keypoints': (N, n_body, 3)
            'joint_names': list (same for all)
            'body_names': list (same for all)
    """
    fk = NumpyFK(urdf_path)
    N = q48_batch.shape[0]

    all_tips_r, all_tips_l = [], []
    all_joints = []
    all_wrist_r, all_wrist_l = [], []
    all_palm_r, all_palm_l = [], []
    all_body = []
    joint_names = None
    body_names = None

    for i in range(N):
        q48 = q48_batch[i]
        link_transforms = fk.forward(q48)

        # Fingertips
        tips_r = np.array([link_transforms[n][:3, 3] for n in FINGERTIP_LINK_NAMES_RIGHT])
        tips_l = np.array([link_transforms[n][:3, 3] for n in FINGERTIP_LINK_NAMES_LEFT])
        all_tips_r.append(tips_r)
        all_tips_l.append(tips_l)

        # All joints
        _, jt = fk.forward_with_joint_transforms(q48)
        positions = []
        names = []
        for jname in fk.revolute_names:
            positions.append(jt[jname][:3, 3])
            names.append(jname)
        all_joints.append(np.array(positions))
        if joint_names is None:
            joint_names = names

        # Wrist & palm
        all_wrist_r.append(link_transforms[WRIST_LINK_NAMES["right"]][:3, 3])
        all_wrist_l.append(link_transforms[WRIST_LINK_NAMES["left"]][:3, 3])
        all_palm_r.append(link_transforms[PALM_LINK_NAMES["right"]][:3, 3])
        all_palm_l.append(link_transforms[PALM_LINK_NAMES["left"]][:3, 3])

        # Body keypoints
        bpos = []
        bnames = []
        for lname in BODY_LINK_NAMES:
            if lname in link_transforms:
                bpos.append(link_transforms[lname][:3, 3])
                bnames.append(lname)
        all_body.append(np.array(bpos))
        if body_names is None:
            body_names = bnames

    tips_r = np.array(all_tips_r)
    tips_l = np.array(all_tips_l)
    return {
        "fingertips_right": tips_r,
        "fingertips_left": tips_l,
        "fingertips_all": np.concatenate([tips_r, tips_l], axis=1),
        "joint_positions": np.array(all_joints),
        "wrist_right": np.array(all_wrist_r),
        "wrist_left": np.array(all_wrist_l),
        "palm_right": np.array(all_palm_r),
        "palm_left": np.array(all_palm_l),
        "body_keypoints": np.array(all_body),
        "joint_names": joint_names,
        "body_names": body_names,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FK Trajectory Computation
# ─────────────────────────────────────────────────────────────────────────────

def rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to roll-pitch-yaw (XYZ extrinsic) in radians."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return np.array([roll, pitch, yaw])


def _build_reorder_indices(dataset_joint_names: list[str]) -> np.ndarray:
    """
    Build index mapping: reorder[i] = dataset column index for FK JOINT_NAMES_48[i].

    Args:
        dataset_joint_names: joint names from dataset info.json (may have _Pos suffix)

    Returns:
        reorder: (48,) int array where q48_fk[i] = q48_dataset[reorder[i]]
    """
    from utils.name_utils import DATASET_TO_FK_NAME

    clean_names = [n.replace("_Pos", "") for n in dataset_joint_names]
    name_to_idx = {name: i for i, name in enumerate(clean_names)}

    reorder = np.zeros(48, dtype=int)
    for fk_idx, fk_name in enumerate(JOINT_NAMES_48):
        if fk_name in name_to_idx:
            reorder[fk_idx] = name_to_idx[fk_name]
        else:
            found = False
            for ds_name, mapped_fk_name in DATASET_TO_FK_NAME.items():
                if mapped_fk_name == fk_name and ds_name in name_to_idx:
                    reorder[fk_idx] = name_to_idx[ds_name]
                    found = True
                    break
            if not found:
                raise ValueError(f"FK joint '{fk_name}' not found in dataset names")

    return reorder


def reorder_to_fk(q48_dataset: np.ndarray, reorder: np.ndarray) -> np.ndarray:
    """Reorder dataset joint vector(s) to FK JOINT_NAMES_48 order."""
    return q48_dataset[..., reorder]


def compute_joint_trajectories(states: np.ndarray, info: dict,
                                fk: NumpyFK, joints: list[str],
                                fingertips: list[str] = None) -> dict:
    """
    Compute position + orientation trajectories for specified joints and fingertips.

    Returns dict: name -> {"pos": (T,3), "rpy": (T,3)}
    """
    from utils.name_utils import JOINT_NAMES, JOINT_LINKS, FINGERTIP_LINKS, HAND_JOINT_LINKS

    try:
        dataset_joint_names = info["features"]["observation.state"]["names"]
    except (KeyError, TypeError):
        dataset_joint_names = JOINT_NAMES
    if dataset_joint_names is None:
        dataset_joint_names = JOINT_NAMES

    reorder = _build_reorder_indices(dataset_joint_names)

    all_keys = list(joints)
    link_names = {j: JOINT_LINKS[j] for j in joints}
    if fingertips:
        all_keys += fingertips
        for ft in fingertips:
            link_names[ft] = HAND_JOINT_LINKS.get(ft, FINGERTIP_LINKS.get(ft))

    T = states.shape[0]
    result = {}
    for k in all_keys:
        result[k] = {"pos": np.zeros((T, 3)), "rpy": np.zeros((T, 3))}

    for t in range(T):
        q48 = reorder_to_fk(states[t], reorder)
        lt = fk.forward(q48)
        for k in all_keys:
            result[k]["pos"][t] = lt[link_names[k]][:3, 3]
            result[k]["rpy"][t] = rotation_matrix_to_euler(lt[link_names[k]][:3, :3])

    return result


def load_and_compute(ds_path, ep_idx: int, fk: NumpyFK,
                      joints: list[str], fingertips: list[str] = None) -> dict:
    """Load a single episode from a LeRobot dataset and compute FK trajectories."""
    import json
    import pyarrow.parquet as pq
    from pathlib import Path

    ds_path = Path(ds_path)
    with open(ds_path / "meta" / "info.json") as f:
        info = json.load(f)
    chunks_size = info.get("chunks_size", 1000)
    chunk_idx = ep_idx // chunks_size
    parquet = ds_path / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{ep_idx:06d}.parquet"
    table = pq.read_table(parquet)
    states = np.array([row.as_py() for row in table.column("observation.state")],
                       dtype=np.float32)
    return compute_joint_trajectories(states, info, fk, joints, fingertips)


if __name__ == "__main__":
    q48 = np.zeros(48)
    result = get_keypoints(q48)

    print("=== ALLEX Forward Kinematics Keypoints ===")
    print(f"\nRight fingertips ({result['fingertips_right'].shape}):")
    for name, pos in zip(FINGERTIP_LINK_NAMES_RIGHT, result["fingertips_right"]):
        finger = name.split("_")[2]
        print(f"  {finger:8s}: [{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}]")

    print(f"\nLeft fingertips ({result['fingertips_left'].shape}):")
    for name, pos in zip(FINGERTIP_LINK_NAMES_LEFT, result["fingertips_left"]):
        finger = name.split("_")[2]
        print(f"  {finger:8s}: [{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}]")

    print(f"\nRight wrist: {result['wrist_right']}")
    print(f"Left wrist:  {result['wrist_left']}")
    print(f"Right palm:  {result['palm_right']}")
    print(f"Left palm:   {result['palm_left']}")

    print(f"\nAll revolute joints: {result['joint_positions'].shape[0]}")
    print(f"Body keypoints: {result['body_keypoints'].shape[0]}")
    print(f"\nBody links:")
    for name, pos in zip(result["body_names"], result["body_keypoints"]):
        print(f"  {name:30s}: [{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}]")
