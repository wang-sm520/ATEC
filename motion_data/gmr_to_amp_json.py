"""Convert a GMR-retargeted motion pickle to our AMP motion JSON format.

Pipeline:
    LAFAN1 BVH → GMR `bvh_to_robot.py --robot unitree_g1 --save_path X.pkl` → THIS SCRIPT → X.json

The output JSON matches `MotionLoaderG1.FRAME_DIM = 73` and is consumed by AMPPPO.

Each frame layout (73 floats):
    [0  : 29) joint_pos       (in `G1_BODY_29_JOINT_NAMES` order, == GMR mujoco joint order)
    [29 : 58) joint_vel       (finite-difference of dof_pos, scaled by `fps`)
    [58 : 73) ee_pos_in_base  (5 EE bodies × 3, expressed in the pelvis frame)

EE bodies (must match `G1_AMP_EE_BODIES` in `rough_env_cfg.py`):
    left_ankle_roll_link, right_ankle_roll_link, waist_yaw_link, left_wrist_yaw_link, right_wrist_yaw_link

Run in the `gmr` conda env (has mujoco). Example::

    python motion_data/gmr_to_amp_json.py \\
        --pkl /tmp/walk1_subject1.pkl \\
        --output motion_data/g1_29dof/walk1_subject1.json \\
        --motion_weight 1.0
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import mujoco
import numpy as np

# Order MUST match `G1_AMP_EE_BODIES` in
# source/atec_rl_lab/atec_rl_lab/train/locomotion/velocity/config/humanoid/unitree_g1/rough_env_cfg.py
EE_BODIES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)
ROOT_BODY = "pelvis"


def load_g1_mujoco_model(gmr_root: Path) -> mujoco.MjModel:
    xml = gmr_root / "assets" / "unitree_g1" / "g1_mocap_29dof.xml"
    if not xml.exists():
        raise FileNotFoundError(f"GMR G1 mujoco XML not found at {xml}")
    return mujoco.MjModel.from_xml_path(str(xml))


def compute_ee_in_base(
    model: mujoco.MjModel, data: mujoco.MjData,
    root_pos: np.ndarray, root_quat_wxyz: np.ndarray, dof_pos: np.ndarray,
    root_body_id: int, ee_body_ids: list[int],
) -> np.ndarray:
    """Return shape (15,) ee positions in pelvis frame, given a single frame's qpos parts."""
    # qpos for a free-base 29 dof model: [x, y, z, qw, qx, qy, qz, j1, ..., j29]
    data.qpos[:3] = root_pos
    data.qpos[3:7] = root_quat_wxyz
    data.qpos[7 : 7 + dof_pos.shape[0]] = dof_pos
    mujoco.mj_kinematics(model, data)

    root_xpos = data.xpos[root_body_id].copy()
    root_xmat = data.xmat[root_body_id].reshape(3, 3).copy()  # row-major rotation matrix

    rel = np.zeros(len(ee_body_ids) * 3, dtype=np.float32)
    for i, bid in enumerate(ee_body_ids):
        diff_w = data.xpos[bid] - root_xpos
        # Express in base (pelvis) frame: R^T @ diff
        diff_b = root_xmat.T @ diff_w
        rel[i * 3 : (i + 1) * 3] = diff_b
    return rel


def convert(
    pkl_path: Path,
    output_path: Path,
    motion_weight: float,
    gmr_root: Path,
    loop_mode: str = "Wrap",
    enable_cycle_offset_position: bool = True,
    enable_cycle_offset_rotation: bool = True,
) -> None:
    with open(pkl_path, "rb") as f:
        m = pickle.load(f)

    dof_pos = np.asarray(m["dof_pos"], dtype=np.float32)        # (N, 29)
    root_pos = np.asarray(m["root_pos"], dtype=np.float32)      # (N, 3)
    root_rot_xyzw = np.asarray(m["root_rot"], dtype=np.float32) # (N, 4) -- GMR saves as xyzw
    # Convert to mujoco's wxyz convention for setting qpos.
    root_rot = root_rot_xyzw[:, [3, 0, 1, 2]].copy()
    fps = float(m["fps"])
    n = dof_pos.shape[0]
    if dof_pos.shape[1] != 29:
        raise ValueError(f"Expected dof_pos of width 29, got {dof_pos.shape}")

    # Joint velocity via finite difference; pad first frame by repeating second-frame velocity.
    dt = 1.0 / fps
    jvel = np.zeros_like(dof_pos)
    jvel[1:] = (dof_pos[1:] - dof_pos[:-1]) / dt
    jvel[0] = jvel[1] if n > 1 else 0.0

    # Forward kinematics for EE positions in base frame.
    model = load_g1_mujoco_model(gmr_root)
    data = mujoco.MjData(model)
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ROOT_BODY)
    if root_id < 0:
        raise RuntimeError(f"Body '{ROOT_BODY}' not found in GMR G1 model.")
    ee_ids: list[int] = []
    for name in EE_BODIES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise RuntimeError(f"Body '{name}' not found in GMR G1 model.")
        ee_ids.append(bid)

    # Sanity check qpos size
    expected = 7 + 29
    if model.nq != expected:
        raise RuntimeError(f"Model has nq={model.nq}, expected {expected} (free joint + 29 dof).")

    ee_pos_b = np.zeros((n, 15), dtype=np.float32)
    for i in range(n):
        ee_pos_b[i] = compute_ee_in_base(model, data, root_pos[i], root_rot[i], dof_pos[i], root_id, ee_ids)

    frames = np.concatenate([dof_pos, jvel, ee_pos_b], axis=1)  # (N, 73)
    assert frames.shape[1] == 73

    out = {
        "LoopMode": loop_mode,
        "FrameDuration": dt,
        "EnableCycleOffsetPosition": bool(enable_cycle_offset_position),
        "EnableCycleOffsetRotation": bool(enable_cycle_offset_rotation),
        "MotionWeight": float(motion_weight),
        "Frames": frames.tolist(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f)
    print(
        f"[gmr_to_amp_json] Wrote {n} frames @ {fps:.1f} fps -> {output_path} "
        f"(jp range: [{dof_pos.min():.3f}, {dof_pos.max():.3f}], "
        f"ee_pos range: [{ee_pos_b.min():.3f}, {ee_pos_b.max():.3f}])"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", type=Path, required=True, help="GMR retargeting output .pkl")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path")
    parser.add_argument("--motion_weight", type=float, default=1.0)
    parser.add_argument(
        "--gmr_root",
        type=Path,
        default=Path("/home/hpf/wsm/GMR"),
        help="Root of the GMR install (used to find the G1 mujoco XML).",
    )
    args = parser.parse_args()
    convert(args.pkl, args.output, args.motion_weight, args.gmr_root)


if __name__ == "__main__":
    main()
