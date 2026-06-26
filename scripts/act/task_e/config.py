"""Task E demo-collection constants.
"""

from atec_rl_lab.tasks.task_e.env_cfg import (
    BASKET_CENTER_X, BASKET_CENTER_Y,
    TABLE_CENTER_X, TABLE_CENTER_Y, TABLE_TOP_Z, TABLE_HALF_X,
    BASKET_EXCL_HALF_X, BASKET_EXCL_HALF_Y,
)

__all__ = [
    "BASKET_CENTER_X", "BASKET_CENTER_Y",
    "TABLE_CENTER_X", "TABLE_CENTER_Y", "TABLE_TOP_Z", "TABLE_HALF_X",
    "BASKET_EXCL_HALF_X", "BASKET_EXCL_HALF_Y",
    # robot
    "EE_BODY_NAME", "ARM_JOINT_NAMES", "GRIPPER_JOINT_NAMES",
    "GRIPPER_OPEN_POS", "GRIPPER_CLOSE_POS", "ACTION_SCALE",
    # state machine
    "STEPS", "OPTIMIZED_STEPS", "STATE_ORDER", "OBJECT_STATE_STEPS", "OBJECT_GRASP_Z_OFFSETS",
    # geometry
    "PRE_GRASP_CLEARANCE", "GRASP_Z_OFFSET", "TOOL_CENTER_OFFSET_LOCAL",
    "OBJECT_TOOL_CENTER_OFFSETS_LOCAL", "OBJECT_BASKET_TARGET_OFFSETS",
    "CARRY_Z", "PLACE_HEIGHT", "BASKET_DROP_Z",
    "RETRACT_POS_X", "RETRACT_POS_Y",
    "DEFAULT_PLACE_QUAT_W",
    # success check
    "BASKET_IN_X", "BASKET_IN_Y",
    # spawn regions
    "OBJ_SPAWN_X_MIN", "OBJ_SPAWN_X_MAX", "OBJ_SPAWN_Z", "OBJ_SPAWN_Y_BANDS",
    "OBJ_HALF_EXTENTS", "OBJ_BBOX_MARGIN",
    # camera
    "CAM_POS", "CAM_ROT", "CAM_H", "CAM_W",
    # actuator
    "ACT_STIFFNESS", "ACT_DAMPING", "ACT_EFFORT_LIMIT", "ACT_VEL_LIMIT",
    # warm-up
    "WARMUP_STEPS", "SETTLE_STEPS",
]

# ------------------------------------------------------------------ #
# Robot
# ------------------------------------------------------------------ #
EE_BODY_NAME        = "gripper_base"
ARM_JOINT_NAMES     = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
GRIPPER_JOINT_NAMES = ["joint7", "joint8"]
GRIPPER_OPEN_POS    = [0.05, -0.05]   # joint7, joint8
GRIPPER_CLOSE_POS   = [-0.01,  0.01]

# Must match ActionsCfg: scale=0.5, use_default_offset=True
#   env_action = (joint_target - default_joint_pos) / ACTION_SCALE
ACTION_SCALE = 0.5

# ------------------------------------------------------------------ #
# State-machine
# ------------------------------------------------------------------ #
STEPS: dict[str, int] = {
    "INIT":         100,
    "PRE_GRASP":    200,
    "REACH":        100,
    "CLOSE":         40,
    "LIFT":         160,
    "TRANSPORT":    200,
    "PLACE":         80,
    "OPEN":          60,
    "LIFT_RETRACT":  80,
    "RETRACT":       80,
}
OPTIMIZED_STEPS: dict[str, int] = {
    "INIT":          10,
    "PRE_GRASP":   40,
    "REACH":       50,
    "CLOSE":        20,
    "LIFT":        35,
    "TRANSPORT":    25,
    "PLACE":        25,
    "OPEN":         10,
    "LIFT_RETRACT": 10,
    "RETRACT":      10,
}

# OPTIMIZED_STEPS: dict[str, int] = {
#     "INIT":          60,
#     "PRE_GRASP":   180,
#     "REACH":       150,
#     "CLOSE":        80,
#     "LIFT":        150,
#     "TRANSPORT":    90,
#     "PLACE":        40,
#     "OPEN":         25,
#     "LIFT_RETRACT": 30,
#     "RETRACT":      35,
# }

STATE_ORDER = ["INIT", "PRE_GRASP", "REACH", "CLOSE", "LIFT",
               "TRANSPORT", "PLACE", "OPEN", "LIFT_RETRACT", "RETRACT"]

OBJECT_STATE_STEPS: dict[int, dict[str, int]] = {
    1: {
        "PRE_GRASP": 50,
        "ALIGN_GRIPPER": 40,
        "REACH": 100,
        "CLOSE": 20,
        "LIFT": 50,
        "TRANSPORT": 25,
        "PLACE": 15,
        "OPEN": 10,
        "LIFT_RETRACT": 10,
        "RETRACT": 10,
    },
    2: {
        "PRE_GRASP": 50,
        "REACH": 40,
        "CLOSE": 15,
        "LIFT": 70,
    },
    3: {
        "INIT": 20,
        "PRE_GRASP": 20,
        "REACH": 50,
        "CLOSE": 10,
        "LIFT": 20,
        "TRANSPORT": 20,
        "PLACE": 15,
        "OPEN": 10,
        "LIFT_RETRACT": 10,
        "RETRACT": 10,
    },
}

# ------------------------------------------------------------------ #
# Geometry
# ------------------------------------------------------------------ #
PRE_GRASP_CLEARANCE = 0.12   # metres above object before descent
GRASP_Z_OFFSET      = 0.09   # metres: gripper approach height above object centre
OBJECT_GRASP_Z_OFFSETS: dict[int, float] = {
    1: 0.08,
    2: 0.08,
    3: 0.08,
}
TOOL_CENTER_OFFSET_LOCAL = [0.0, 0.0, 0.0]  # local xyz offset from gripper_base to jaw center
OBJECT_TOOL_CENTER_OFFSETS_LOCAL: dict[int, list[float]] = {
    1: [0.0, -0.05, 0.0],
    2: [-0.04, -0.01, 0.0],
    3: [0.0, 0.0, 0.0],
}
OBJECT_BASKET_TARGET_OFFSETS: dict[int, tuple[float, float]] = {
    1: (0.2, 0.0),
    2: (0.0, 0.0),
    3: (-0.05, 0.05),
}

CARRY_Z       = TABLE_TOP_Z + 0.40   # safe carry height
PLACE_HEIGHT  = TABLE_TOP_Z + 0.15   # baseline low release height
BASKET_DROP_Z = TABLE_TOP_Z + 0.25   # optimized medium release height

RETRACT_POS_X = TABLE_CENTER_X + TABLE_HALF_X - 0.05
RETRACT_POS_Y = TABLE_CENTER_Y
DEFAULT_PLACE_QUAT_W = [0.0, 1.0, 0.0, 0.0]   # top-down orientation (w,x,y,z)

# ------------------------------------------------------------------ #
# Object spawn regions
#
#   object_1  Y ∈ [0.21, 0.28]   (top band)
#   object_2  Y ∈ [0.12, 0.19]   (middle band)
#   object_3  Y ∈ [0.03, 0.10]   (bottom band, closest to basket)
# ------------------------------------------------------------------ #
OBJ_SPAWN_X_MIN = TABLE_CENTER_X - 0.10
OBJ_SPAWN_X_MAX = TABLE_CENTER_X + 0.10
OBJ_SPAWN_Z     = TABLE_TOP_Z + 0.03

# Per-object Y-bands: {object_idx: (y_min, y_max)}
OBJ_SPAWN_Y_BANDS = {
    1: (TABLE_CENTER_Y + 0.20, TABLE_CENTER_Y + 0.25),
    2: (TABLE_CENTER_Y + 0.12, TABLE_CENTER_Y + 0.19),
    3: (TABLE_CENTER_Y + 0.03, TABLE_CENTER_Y + 0.10),
}

# Per-object 2-D bounding-box half-extents (metres, world XY plane, scale=1).
# Used for AABB overlap rejection during randomisation.
OBJ_HALF_EXTENTS: dict[int, tuple[float, float]] = {
    1: (0.050, 0.044),   # Sugar box   (half_x, half_y)
    2: (0.050, 0.030),   # Mustard bottle
    3: (0.100, 0.040),   # Banana
}
OBJ_BBOX_MARGIN = 0.015  # extra clearance between object bounding boxes

# ------------------------------------------------------------------ #
# Success / basket bounds
# ------------------------------------------------------------------ #
BASKET_IN_X = 0.20   # ± metres in X around BASKET_CENTER_X
BASKET_IN_Y = 0.11   # ± metres in Y around BASKET_CENTER_Y

# ------------------------------------------------------------------ #
# Camera (for --save_video / --save_images)
# ------------------------------------------------------------------ #
CAM_H, CAM_W = 480, 640
CAM_POS = (TABLE_CENTER_X - 1.2, TABLE_CENTER_Y, TABLE_TOP_Z + 0.8)
CAM_ROT = (0.957, 0.0, 0.290, 0.0)   # ~34° around +Y → looks forward-down

# ------------------------------------------------------------------ #
# Actuator overrides for data collection (high stiffness for fast tracking)
# ------------------------------------------------------------------ #
ACT_STIFFNESS    = 800.0
ACT_DAMPING      = 80.0
ACT_EFFORT_LIMIT = 100.0
ACT_VEL_LIMIT    = 100.0

# ------------------------------------------------------------------ #
# Warm-up / settle steps before recording starts
# ------------------------------------------------------------------ #
WARMUP_STEPS = 150
SETTLE_STEPS = 30
