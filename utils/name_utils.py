"""
Centralized joint, link, and display name definitions for the ALLEX robot.

All name constants used across forward kinematics, data loading, and
visualization scripts are defined here.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Dataset joint names (LeRobot format, with _Pos suffix)
# ─────────────────────────────────────────────────────────────────────────────

JOINT_NAMES = [
    'R_Shoulder_Pitch_Joint_Pos', 'R_Shoulder_Roll_Joint_Pos', 'R_Shoulder_Yaw_Joint_Pos',
    'R_Elbow_Joint_Pos', 'R_Wrist_Yaw_Joint_Pos', 'R_Wrist_Roll_Joint_Pos',
    'R_Wrist_Pitch_Joint_Pos', 'L_Shoulder_Pitch_Joint_Pos', 'L_Shoulder_Roll_Joint_Pos',
    'L_Shoulder_Yaw_Joint_Pos', 'L_Elbow_Joint_Pos', 'L_Wrist_Yaw_Joint_Pos', 'L_Wrist_Roll_Joint_Pos',
    'L_Wrist_Pitch_Joint_Pos', 'R_Thumb_Yaw_Joint_Pos', 'R_Thumb_CMC_Joint_Pos', 'R_Thumb_MCP_Joint_Pos',
    'R_Index_Roll_Joint_Pos', 'R_Index_MCP_Joint_Pos', 'R_Index_PIP_Joint_Pos', 'R_Middle_Roll_Joint_Pos',
    'R_Middle_MCP_Joint_Pos', 'R_Middle_PIP_Joint_Pos', 'R_Ring_Roll_Joint_Pos', 'R_Ring_MCP_Joint_Pos',
    'R_Ring_PIP_Joint_Pos', 'R_Little_Roll_Joint_Pos', 'R_Little_MCP_Joint_Pos', 'R_Little_PIP_Joint_Pos',
    'L_Thumb_Yaw_Joint_Pos', 'L_Thumb_CMC_Joint_Pos', 'L_Thumb_MCP_Joint_Pos', 'L_Index_Roll_Joint_Pos',
    'L_Index_MCP_Joint_Pos', 'L_Index_PIP_Joint_Pos', 'L_Middle_Roll_Joint_Pos', 'L_Middle_MCP_Joint_Pos',
    'L_Middle_PIP_Joint_Pos', 'L_Ring_Roll_Joint_Pos', 'L_Ring_MCP_Joint_Pos', 'L_Ring_PIP_Joint_Pos',
    'L_Little_Roll_Joint_Pos', 'L_Little_MCP_Joint_Pos', 'L_Little_PIP_Joint_Pos', 'Head_Pan_Joint_Pos',
    'Head_Tilt_Joint_Pos', 'Waist_Roll_Joint_Pos', 'Waist_Pitch_Joint_Pos',
]

# Dataset -> FK name aliases (LeRobot uses different names for head/waist)
DATASET_TO_FK_NAME = {
    "Head_Pan_Joint": "Neck_Yaw_Joint",
    "Head_Tilt_Joint": "Neck_Pitch_Joint",
    "Waist_Roll_Joint": "Waist_Yaw_Joint",
    "Waist_Pitch_Joint": "Waist_Pitch_Lower_Joint",
}

# ─────────────────────────────────────────────────────────────────────────────
# FK joint names (48 independent joints in env_cfg action order)
# ─────────────────────────────────────────────────────────────────────────────

JOINT_NAMES_48 = [
    # waist + neck + shoulders interleaved (indices 0-17)
    "Waist_Yaw_Joint",            # 0
    "Waist_Pitch_Lower_Joint",    # 1
    "Neck_Pitch_Joint",           # 2
    "L_Shoulder_Pitch_Joint",     # 3
    "R_Shoulder_Pitch_Joint",     # 4
    "Neck_Yaw_Joint",             # 5
    "L_Shoulder_Roll_Joint",      # 6
    "R_Shoulder_Roll_Joint",      # 7
    "L_Shoulder_Yaw_Joint",       # 8
    "R_Shoulder_Yaw_Joint",       # 9
    "L_Elbow_Joint",              # 10
    "R_Elbow_Joint",              # 11
    "L_Wrist_Yaw_Joint",          # 12
    "R_Wrist_Yaw_Joint",          # 13
    "L_Wrist_Roll_Joint",         # 14
    "R_Wrist_Roll_Joint",         # 15
    "L_Wrist_Pitch_Joint",        # 16
    "R_Wrist_Pitch_Joint",        # 17
    # hands: roll/yaw layer (indices 18-27)
    "L_Thumb_Yaw_Joint",          # 18
    "L_Index_Roll_Joint",         # 19
    "L_Middle_Roll_Joint",        # 20
    "L_Ring_Roll_Joint",          # 21
    "L_Little_Roll_Joint",        # 22
    "R_Thumb_Yaw_Joint",          # 23
    "R_Index_Roll_Joint",         # 24
    "R_Middle_Roll_Joint",        # 25
    "R_Ring_Roll_Joint",          # 26
    "R_Little_Roll_Joint",        # 27
    # hands: MCP/CMC layer (indices 28-37)
    "L_Thumb_CMC_Joint",          # 28
    "L_Index_MCP_Joint",          # 29
    "L_Middle_MCP_Joint",         # 30
    "L_Ring_MCP_Joint",           # 31
    "L_Little_MCP_Joint",         # 32
    "R_Thumb_CMC_Joint",          # 33
    "R_Index_MCP_Joint",          # 34
    "R_Middle_MCP_Joint",         # 35
    "R_Ring_MCP_Joint",           # 36
    "R_Little_MCP_Joint",         # 37
    # hands: PIP/MCP layer (indices 38-47)
    "L_Thumb_MCP_Joint",          # 38
    "L_Index_PIP_Joint",          # 39
    "L_Middle_PIP_Joint",         # 40
    "L_Ring_PIP_Joint",           # 41
    "L_Little_PIP_Joint",         # 42
    "R_Thumb_MCP_Joint",          # 43
    "R_Index_PIP_Joint",          # 44
    "R_Middle_PIP_Joint",         # 45
    "R_Ring_PIP_Joint",           # 46
    "R_Little_PIP_Joint",         # 47
]

# Mimic joints: {mimic_joint: (parent_joint, multiplier, offset)}
MIMIC_JOINTS = {
    "R_Index_DIP_Joint": ("R_Index_PIP_Joint", 0.98, 0.0),
    "R_Little_DIP_Joint": ("R_Little_PIP_Joint", 0.98, 0.0),
    "R_Middle_DIP_Joint": ("R_Middle_PIP_Joint", 0.98, 0.0),
    "R_Ring_DIP_Joint": ("R_Ring_PIP_Joint", 0.98, 0.0),
    "R_Thumb_IP_Joint": ("R_Thumb_MCP_Joint", 0.98, 0.0),
    "L_Index_DIP_Joint": ("L_Index_PIP_Joint", 0.98, 0.0),
    "L_Little_DIP_Joint": ("L_Little_PIP_Joint", 0.98, 0.0),
    "L_Middle_DIP_Joint": ("L_Middle_PIP_Joint", 0.98, 0.0),
    "L_Ring_DIP_Joint": ("L_Ring_PIP_Joint", 0.98, 0.0),
    "L_Thumb_IP_Joint": ("L_Thumb_MCP_Joint", 0.77, 0.0),
    "Waist_Pitch_Dummy_Joint": ("Waist_Pitch_Lower_Joint", 1.0, 0.0),
    "Waist_Pitch_Upper_Joint": ("Waist_Pitch_Lower_Joint", -1.0, 0.0),
}

# ─────────────────────────────────────────────────────────────────────────────
# Finger joint names (independent + mimic, per hand)
# ─────────────────────────────────────────────────────────────────────────────

FINGER_JOINT_NAMES_RIGHT = [
    "R_Index_Roll_Joint", "R_Index_MCP_Joint", "R_Index_PIP_Joint",
    "R_Middle_Roll_Joint", "R_Middle_MCP_Joint", "R_Middle_PIP_Joint",
    "R_Ring_Roll_Joint", "R_Ring_MCP_Joint", "R_Ring_PIP_Joint",
    "R_Little_Roll_Joint", "R_Little_MCP_Joint", "R_Little_PIP_Joint",
    "R_Thumb_Yaw_Joint", "R_Thumb_CMC_Joint", "R_Thumb_MCP_Joint",
]
FINGER_JOINT_NAMES_LEFT = [
    "L_Index_Roll_Joint", "L_Index_MCP_Joint", "L_Index_PIP_Joint",
    "L_Middle_Roll_Joint", "L_Middle_MCP_Joint", "L_Middle_PIP_Joint",
    "L_Ring_Roll_Joint", "L_Ring_MCP_Joint", "L_Ring_PIP_Joint",
    "L_Little_Roll_Joint", "L_Little_MCP_Joint", "L_Little_PIP_Joint",
    "L_Thumb_Yaw_Joint", "L_Thumb_CMC_Joint", "L_Thumb_MCP_Joint",
]

FINGER_MIMIC_JOINT_NAMES_RIGHT = [
    "R_Index_DIP_Joint", "R_Middle_DIP_Joint", "R_Ring_DIP_Joint",
    "R_Little_DIP_Joint", "R_Thumb_IP_Joint",
]
FINGER_MIMIC_JOINT_NAMES_LEFT = [
    "L_Index_DIP_Joint", "L_Middle_DIP_Joint", "L_Ring_DIP_Joint",
    "L_Little_DIP_Joint", "L_Thumb_IP_Joint",
]

FINGER_JOINT_NAMES_ALL_RIGHT = FINGER_JOINT_NAMES_RIGHT + FINGER_MIMIC_JOINT_NAMES_RIGHT
FINGER_JOINT_NAMES_ALL_LEFT = FINGER_JOINT_NAMES_LEFT + FINGER_MIMIC_JOINT_NAMES_LEFT
FINGER_JOINT_NAMES = FINGER_JOINT_NAMES_ALL_RIGHT + FINGER_JOINT_NAMES_ALL_LEFT

N_FINGER_JOINTS_PER_HAND = 15

# ─────────────────────────────────────────────────────────────────────────────
# Finger link names (URDF links, per finger per hand)
# ─────────────────────────────────────────────────────────────────────────────

FINGER_ORDER = ["Index", "Middle", "Ring", "Little", "Thumb"]

FINGER_LINK_NAMES_PER_FINGER_RIGHT = {
    "Index": [
        "R_Hand_Index_Roll",
        "R_Hand_Index_Proximal",
        "R_Hand_Index_Middle",
        "R_Hand_Index_Distal",
    ],
    "Middle": [
        "R_Hand_Middle_Roll",
        "R_Hand_Middle_Proximal",
        "R_Hand_Middle_Middle",
        "R_Hand_Middle_Distal",
    ],
    "Ring": [
        "R_Hand_Ring_Roll",
        "R_Hand_Ring_Proximal",
        "R_Hand_Ring_Middle",
        "R_Hand_Ring_Distal",
    ],
    "Little": [
        "R_Hand_Little_Roll",
        "R_Hand_Little_Proximal",
        "R_Hand_Little_Middle",
        "R_Hand_Little_Distal",
    ],
    "Thumb": [
        "R_Hand_Thumb_Yaw",
        "R_Hand_Thumb_Proximal",
        "R_Hand_Thumb_Middle",
        "R_Hand_Thumb_Distal",
    ],
}

FINGER_LINK_NAMES_PER_FINGER_LEFT = {
    "Index": [
        "L_Hand_Index_Roll",
        "L_Hand_Index_Proximal",
        "L_Hand_Index_Middle",
        "L_Hand_Index_Distal",
    ],
    "Middle": [
        "L_Hand_Middle_Roll",
        "L_Hand_Middle_Proximal",
        "L_Hand_Middle_Middle",
        "L_Hand_Middle_Distal",
    ],
    "Ring": [
        "L_Hand_Ring_Roll",
        "L_Hand_Ring_Proximal",
        "L_Hand_Ring_Middle",
        "L_Hand_Ring_Distal",
    ],
    "Little": [
        "L_Hand_Little_Roll",
        "L_Hand_Little_Proximal",
        "L_Hand_Little_Middle",
        "L_Hand_Little_Distal",
    ],
    "Thumb": [
        "L_Hand_Thumb_Yaw",
        "L_Hand_Thumb_Proximal",
        "L_Hand_Thumb_Middle",
        "L_Hand_Thumb_Distal",
    ],
}

# Palm link names
PALM_LINK_RIGHT = "ALLEX_Right_Hand_Palm"
PALM_LINK_LEFT = "ALLEX_Left_Hand_Palm"

# Ordered list of all finger link names per hand (for flat output)
# Order: Palm(1) + Index(4) + Middle(4) + Ring(4) + Little(4) + Thumb(4) = 21
FINGER_LINK_NAMES_ORDERED_RIGHT = [PALM_LINK_RIGHT]
for _f in FINGER_ORDER:
    FINGER_LINK_NAMES_ORDERED_RIGHT.extend(FINGER_LINK_NAMES_PER_FINGER_RIGHT[_f])

FINGER_LINK_NAMES_ORDERED_LEFT = [PALM_LINK_LEFT]
for _f in FINGER_ORDER:
    FINGER_LINK_NAMES_ORDERED_LEFT.extend(FINGER_LINK_NAMES_PER_FINGER_LEFT[_f])

# ─────────────────────────────────────────────────────────────────────────────
# Fingertip link names (distal links = last link in each finger chain)
# ─────────────────────────────────────────────────────────────────────────────

FINGERTIP_LINK_NAMES_RIGHT = [
    "R_Hand_Index_Distal",
    "R_Hand_Middle_Distal",
    "R_Hand_Ring_Distal",
    "R_Hand_Little_Distal",
    "R_Hand_Thumb_Distal",
]
FINGERTIP_LINK_NAMES_LEFT = [
    "L_Hand_Index_Distal",
    "L_Hand_Middle_Distal",
    "L_Hand_Ring_Distal",
    "L_Hand_Little_Distal",
    "L_Hand_Thumb_Distal",
]

# ─────────────────────────────────────────────────────────────────────────────
# Body and wrist link names
# ─────────────────────────────────────────────────────────────────────────────

BODY_LINK_NAMES = [
    "Waist_Yaw_link",
    "Neck_Pitch_link",
    "Neck_Yaw_link",
    # Left arm chain
    "L_Shoulder_Pitch_link",
    "L_Shoulder_Roll_link",
    "L_Upperarm_link",
    "L_Elbow_link",
    "L_Forearm_link",
    "L_Wrist_Roll_link",
    "L_Wrist_Pitch_link",
    # Right arm chain
    "R_Shoulder_Pitch_link",
    "R_Shoulder_Roll_link",
    "R_Upperarm_link",
    "R_Elbow_link",
    "R_Forearm_link",
    "R_Wrist_Roll_link",
    "R_Wrist_Pitch_link",
]

WRIST_LINK_NAMES = {
    "left": "L_Wrist_Pitch_link",
    "right": "R_Wrist_Pitch_link",
}

PALM_LINK_NAMES = {
    "left": PALM_LINK_LEFT,
    "right": PALM_LINK_RIGHT,
}

HEAD_LINK_NAME = "Neck_Yaw_link"

# ─────────────────────────────────────────────────────────────────────────────
# Plot joint name mappings (simplified names -> URDF link names)
# ─────────────────────────────────────────────────────────────────────────────

FINGER_NAMES = ["Index", "Middle", "Ring", "Little", "Thumb"]

FINGER_COLORS_LIST = ["#e74c3c", "#e67e22", "#2ecc71", "#3498db", "#9b59b6"]

FINGER_COLORS = {
    "Index": "#e74c3c", "Middle": "#e67e22", "Ring": "#2ecc71",
    "Little": "#3498db", "Thumb": "#9b59b6",
}

JOINT_LINKS = {
    "waist":           "Waist_Yaw_link",
    "neck":            "Neck_Pitch_link",
    "head":            "Neck_Yaw_link",
    "shoulder_right":  "R_Shoulder_Pitch_link",
    "shoulder_left":   "L_Shoulder_Pitch_link",
    "elbow_right":     "R_Elbow_link",
    "elbow_left":      "L_Elbow_link",
    "wrist_right":     "R_Wrist_Pitch_link",
    "wrist_left":      "L_Wrist_Pitch_link",
}

ALL_JOINT_NAMES = list(JOINT_LINKS.keys())

JOINT_DISPLAY = {
    "waist": "Waist", "neck": "Neck", "head": "Head",
    "shoulder_right": "R Shoulder", "shoulder_left": "L Shoulder",
    "elbow_right": "R Elbow", "elbow_left": "L Elbow",
    "wrist_right": "R Wrist", "wrist_left": "L Wrist",
}

# Fingertip plot mappings
FINGERTIP_LINKS = {}
for _i, _fname in enumerate(FINGER_NAMES):
    FINGERTIP_LINKS[f"R_{_fname}"] = FINGERTIP_LINK_NAMES_RIGHT[_i]
    FINGERTIP_LINKS[f"L_{_fname}"] = FINGERTIP_LINK_NAMES_LEFT[_i]

ALL_FINGERTIP_NAMES = list(FINGERTIP_LINKS.keys())

FINGERTIP_DISPLAY = {k: k.replace("_", " ") for k in ALL_FINGERTIP_NAMES}

# Full hand joint mappings (Knuckle, Base, Mid, Tip per finger)
_JOINT_LEVELS = ["Knuckle", "Base", "Mid"]
_FK_LEVEL_SUFFIX = {"Knuckle": "Roll", "Base": "Proximal", "Mid": "Middle"}
_FK_THUMB_KNUCKLE = "Yaw"

HAND_JOINT_LINKS = {}
HAND_JOINT_LINKS.update(FINGERTIP_LINKS)

for _side in ("R", "L"):
    for _fname in FINGER_NAMES:
        for _level in _JOINT_LEVELS:
            _key = f"{_side}_{_fname}_{_level}"
            if _fname == "Thumb" and _level == "Knuckle":
                _fk_suffix = _FK_THUMB_KNUCKLE
            else:
                _fk_suffix = _FK_LEVEL_SUFFIX[_level]
            HAND_JOINT_LINKS[_key] = f"{_side}_Hand_{_fname}_{_fk_suffix}"

ALL_HAND_JOINT_NAMES = list(HAND_JOINT_LINKS.keys())

HAND_JOINT_DISPLAY = {k: k.replace("_", " ") for k in ALL_HAND_JOINT_NAMES}

# ─────────────────────────────────────────────────────────────────────────────
# Skeleton bone definitions (parent_link, child_link pairs)
# ─────────────────────────────────────────────────────────────────────────────

BODY_BONES = [
    # Torso
    ("Waist_Base", "Waist_Yaw_link"),
    ("Waist_Yaw_link", "Waist_Pitch_Lower_link"),
    ("Waist_Pitch_Lower_link", "Waist_Pitch_Upper_link"),
    # Neck
    ("Waist_Pitch_Upper_link", "Neck_Pitch_link"),
    ("Neck_Pitch_link", "Neck_Yaw_link"),
    # Left arm
    ("Waist_Pitch_Upper_link", "L_Shoulder_Pitch_link"),
    ("L_Shoulder_Pitch_link", "L_Shoulder_Roll_link"),
    ("L_Shoulder_Roll_link", "L_Upperarm_link"),
    ("L_Upperarm_link", "L_Elbow_link"),
    ("L_Elbow_link", "L_Forearm_link"),
    ("L_Forearm_link", "L_Wrist_Roll_link"),
    ("L_Wrist_Roll_link", "L_Wrist_Pitch_link"),
    # Right arm
    ("Waist_Pitch_Upper_link", "R_Shoulder_Pitch_link"),
    ("R_Shoulder_Pitch_link", "R_Shoulder_Roll_link"),
    ("R_Shoulder_Roll_link", "R_Upperarm_link"),
    ("R_Upperarm_link", "R_Elbow_link"),
    ("R_Elbow_link", "R_Forearm_link"),
    ("R_Forearm_link", "R_Wrist_Roll_link"),
    ("R_Wrist_Roll_link", "R_Wrist_Pitch_link"),
]

HAND_BONES_RIGHT = [
    ("R_Wrist_Pitch_link", "ALLEX_Right_Hand_Palm"),
    ("ALLEX_Right_Hand_Palm", "ALLEX_Right_Hand_base"),
]
HAND_BONES_LEFT = [
    ("L_Wrist_Pitch_link", "ALLEX_Left_Hand_Palm"),
    ("ALLEX_Left_Hand_Palm", "ALLEX_Left_Hand_base"),
]

BODY_KEYPOINTS = {
    "Waist":       ("Waist_Yaw_link",          "#8B4513", 60),
    "Neck":        ("Neck_Pitch_link",          "#FF8C00", 50),
    "Head":        ("Neck_Yaw_link",            "#FF4500", 80),
    "L Shoulder":  ("L_Shoulder_Pitch_link",    "#1E90FF", 50),
    "L Elbow":     ("L_Elbow_link",             "#1E90FF", 45),
    "L Wrist":     ("L_Wrist_Pitch_link",       "#1E90FF", 55),
    "R Shoulder":  ("R_Shoulder_Pitch_link",     "#DC143C", 50),
    "R Elbow":     ("R_Elbow_link",              "#DC143C", 45),
    "R Wrist":     ("R_Wrist_Pitch_link",        "#DC143C", 55),
}

# Simplified skeleton bones (using plot joint names, for online visualization)
SKELETON_BONES = [
    ("waist", "neck"),
    ("neck", "head"),
    ("waist", "shoulder_right"),
    ("waist", "shoulder_left"),
    ("shoulder_right", "elbow_right"),
    ("shoulder_left", "elbow_left"),
    ("elbow_right", "wrist_right"),
    ("elbow_left", "wrist_left"),
]

# Finger bone chains (using plot joint names)
FINGER_BONES_RIGHT = []
FINGER_BONES_LEFT = []
for _fname in FINGER_NAMES:
    _chain_r = [f"R_{_fname}_Knuckle", f"R_{_fname}_Base", f"R_{_fname}_Mid", f"R_{_fname}"]
    _chain_l = [f"L_{_fname}_Knuckle", f"L_{_fname}_Base", f"L_{_fname}_Mid", f"L_{_fname}"]
    FINGER_BONES_RIGHT.append(("wrist_right", _chain_r[0]))
    FINGER_BONES_LEFT.append(("wrist_left", _chain_l[0]))
    for _i in range(len(_chain_r) - 1):
        FINGER_BONES_RIGHT.append((_chain_r[_i], _chain_r[_i + 1]))
        FINGER_BONES_LEFT.append((_chain_l[_i], _chain_l[_i + 1]))

FINGER_BONES_RIGHT_TIPS = [
    ("wrist_right", f"R_{f}") for f in FINGER_NAMES
]
FINGER_BONES_LEFT_TIPS = [
    ("wrist_left", f"L_{f}") for f in FINGER_NAMES
]

# ─────────────────────────────────────────────────────────────────────────────
# EgoDex / ARKit joint name mappings
# ─────────────────────────────────────────────────────────────────────────────

EGODEX_JOINT_MAP = {
    "waist":          "hip",
    "neck":           "neck1",
    "head":           "neck4",
    "shoulder_right": "rightShoulder",
    "shoulder_left":  "leftShoulder",
    "elbow_right":    "rightForearm",
    "elbow_left":     "leftForearm",
    "wrist_right":    "rightHand",
    "wrist_left":     "leftHand",
}

EGODEX_FINGERTIP_MAP = {
    "R_Index":  "rightIndexFingerTip",
    "R_Middle": "rightMiddleFingerTip",
    "R_Ring":   "rightRingFingerTip",
    "R_Little": "rightLittleFingerTip",
    "R_Thumb":  "rightThumbTip",
    "L_Index":  "leftIndexFingerTip",
    "L_Middle": "leftMiddleFingerTip",
    "L_Ring":   "leftRingFingerTip",
    "L_Little": "leftLittleFingerTip",
    "L_Thumb":  "leftThumbTip",
}

_FINGER_ARKIT = {
    "Index": "IndexFinger", "Middle": "MiddleFinger",
    "Ring": "RingFinger", "Little": "LittleFinger", "Thumb": "Thumb",
}
_LEVEL_ARKIT = {"Knuckle": "Knuckle", "Base": "IntermediateBase", "Mid": "IntermediateTip"}

EGODEX_HAND_JOINT_MAP = dict(EGODEX_FINGERTIP_MAP)
for _side_short, _side_long in [("R", "right"), ("L", "left")]:
    for _fname, _arkit_fname in _FINGER_ARKIT.items():
        for _level, _arkit_level in _LEVEL_ARKIT.items():
            _key = f"{_side_short}_{_fname}_{_level}"
            EGODEX_HAND_JOINT_MAP[_key] = f"{_side_long}{_arkit_fname}{_arkit_level}"

# ─────────────────────────────────────────────────────────────────────────────
# Category config for LeRobot datasets: name -> (base_path, color, display_label)
# ─────────────────────────────────────────────────────────────────────────────

from pathlib import Path as _Path

LEROBOT_BASE = _Path.home() / ".cache" / "huggingface" / "lerobot"

CATEGORY_CONFIG = {
    "egodex_v4": (LEROBOT_BASE / "egodex_v4", "#3498db", "Human (egodex)"),
    "RLWRLD":    (LEROBOT_BASE / "RLWRLD",    "#e74c3c", "Robot (RLWRLD)"),
    "sim_v4":    (LEROBOT_BASE / "sim_v4",     "#2ecc71", "Sim (sim_v4)"),
    "sim_v3":    (LEROBOT_BASE / "sim_v3",     "#9b59b6", "Sim (sim_v3)"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Color generation for dynamic dataset plotting
# ─────────────────────────────────────────────────────────────────────────────

# Hand-picked palette of 20 distinguishable colors (colorblind-friendly mix).
# Used first before falling back to HSL generation.
PALETTE = [
    "#e6194b",  # red
    "#3cb44b",  # green
    "#4363d8",  # blue
    "#f58231",  # orange
    "#911eb4",  # purple
    "#42d4f4",  # cyan
    "#f032e6",  # magenta
    "#bfef45",  # lime
    "#fabed4",  # pink
    "#469990",  # teal
    "#dcbeff",  # lavender
    "#9a6324",  # brown
    "#fffac8",  # beige
    "#800000",  # maroon
    "#aaffc3",  # mint
    "#808000",  # olive
    "#ffd8b1",  # apricot
    "#000075",  # navy
    "#a9a9a9",  # grey
    "#e6beff",  # light purple
]


def generate_colors(n: int) -> list[str]:
    """Generate n visually distinguishable hex colors.

    Uses the hand-picked PALETTE for the first 20 colors, then generates
    additional colors by spacing hues evenly around the HSL wheel.
    """
    if n <= len(PALETTE):
        return PALETTE[:n]

    colors = list(PALETTE)
    # Generate additional colors via evenly-spaced hues
    import colorsys
    extra = n - len(PALETTE)
    for i in range(extra):
        h = (len(PALETTE) + i) / n  # spread across full hue circle
        r, g, b = colorsys.hls_to_rgb(h, 0.5, 0.75)
        colors.append(f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}")
    return colors
