# Offline head-depth box detector + accuracy eval vs god-view truth.
# Camera looks along base heading (yaw, from odometry) pitched DOWN ~PHI. No sim.
# Usage: python scripts/eval_box_detector.py [out] [phi_deg]

import json
import os
import sys
import math
import torch

OUT = sys.argv[1] if len(sys.argv) > 1 else "outputs/percep_capture"
PHI = math.radians(float(sys.argv[2])) if len(sys.argv) > 2 else math.radians(35.0)
meta = json.load(open(os.path.join(OUT, "meta.json")))


def depth_to_world(depth, intr, cam_pos, yaw, phi=PHI):
    H, W = depth.shape
    fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
    vv, uu = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                            torch.arange(W, dtype=torch.float32), indexing="ij")
    d = depth
    valid = torch.isfinite(d) & (d > 0.05) & (d < 12.0)
    xr = ((uu - cx) / fx * d)[valid]      # optical right
    yd = ((vv - cy) / fy * d)[valid]      # optical down
    zf = d[valid]                          # optical forward
    c, s = math.cos(yaw), math.sin(yaw)
    f = torch.tensor([c, s, 0.0]); right = torch.tensor([s, -c, 0.0]); u = torch.tensor([0., 0., 1.])
    fwd = math.cos(phi) * f - math.sin(phi) * u
    down = math.sin(phi) * f - math.cos(phi) * u
    cp = torch.tensor(cam_pos)
    return cp + xr[:, None] * right + yd[:, None] * down + zf[:, None] * fwd


def detect_box_xy(P, robot_xy, yaw):
    z = P[:, 2]
    p = P[(z > 0.06) & (z < 0.55)]           # box surface; excludes floor(~0) & platform-top
    if p.shape[0] < 50:
        return None, 0
    rx, ry = robot_xy
    c, s = math.cos(yaw), math.sin(yaw)
    rel = p[:, :2] - torch.tensor([rx, ry])
    fwd = rel[:, 0] * c + rel[:, 1] * s
    lat = -rel[:, 0] * s + rel[:, 1] * c
    m = (fwd > 0.2) & (fwd < 4.0) & (lat.abs() < 1.0)
    p, fwd = p[m], fwd[m]
    if p.shape[0] < 50:
        return None, 0
    near = torch.quantile(fwd, 0.05)
    pc = p[fwd < near + 0.5]                  # near-face slab
    cx = float(pc[:, 0].mean()) + 0.40 * c    # near face -> +half depth to box centre
    cy = float(pc[:, 1].mean()) + 0.40 * s
    return (cx, cy), int(pc.shape[0])


print(f"phi={math.degrees(PHI):.0f}deg")
print(f"{'idx':>3} {'dist':>5} {'true_box':>14} {'est_box':>14} {'err_m':>6} {'npts':>6}")
errs = []
for m in meta:
    depth = torch.load(os.path.join(OUT, f"depth_{m['idx']}.pt")).float()
    intr = torch.load(os.path.join(OUT, f"intr_{m['idx']}.pt")).float()
    P = depth_to_world(depth, intr, m["cam_pos"], m["robot_yaw"])
    est, n = detect_box_xy(P, (m["robot_w"][0], m["robot_w"][1]), m["robot_yaw"])
    tb = m["box_w"]
    if est is None:
        print(f"{m['idx']:>3} {m['dist']:>5.2f}  (no detection)")
        continue
    err = math.hypot(est[0] - tb[0], est[1] - tb[1])
    errs.append(err)
    print(f"{m['idx']:>3} {m['dist']:>5.2f} ({tb[0]:>6.2f},{tb[1]:>5.2f}) ({est[0]:>6.2f},{est[1]:>5.2f}) {err:>6.3f} {n:>6}")
if errs:
    print(f"\nmean err = {sum(errs)/len(errs):.3f} m   max err = {max(errs):.3f} m   (n={len(errs)}/{len(meta)})")
