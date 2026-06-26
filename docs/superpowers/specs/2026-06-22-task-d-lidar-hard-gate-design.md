# Task D LiDAR Hard Gate Design

## Goal

Improve the final Task D crossing attempt by changing the current LiDAR confirmation from a soft timeout gate into a hard safety gate. The robot should not enter `forward` / `policy_climb.pt` unless LiDAR has confirmed that the box has formed a usable pit bridge. If confirmation fails, the controller should return to `push_x_pit` and continue pushing for a short debounce interval before trying confirmation again.

This targets the diagnosed missing score component: the current solution reliably reaches 16/36 by collecting box target +14 and robot `x > -1.4` +2, but fails the final +20 because the robot does not reach `x > 2.0`.

## Current Failure

The current late-stage flow is:

```text
push_x_pit
  ↓ odometry rx >= PIT_EDGE_X
confirm_pit_box
  ↓ LiDAR confirmed OR timeout
forward
  ↓ select("climb")
```

Two failure patterns were observed in probe output:

1. `confirm_pit_box` can continue moving forward with `vx=0.45` while `lidar_box_in_pit=false`, causing the robot to approach the pit without a verified bridge.
2. `confirm_pit_box` and `align_climb` currently have timeout paths that enter `forward`, so failure to observe the bridge can still trigger the climb handoff.

The result is that the robot may reach `forward` or terminate near the pit edge with root height around the fall threshold, while still far short of `x > 2.0`.

## Proposed Behavior

Use a two-level LiDAR gate:

```text
may_have_bridge      # loose signal: enough to stop pushing and enter confirmation
confirmed_bridge     # strict signal: enough to enter align/forward
```

### `push_x_pit`

`push_x_pit` should no longer transition to `confirm_pit_box` solely because odometry reports `rx >= PIT_EDGE_X`.

It should transition only when all of these are true:

1. Odometry is at or beyond `PIT_EDGE_X`.
2. The retry cooldown has expired.
3. LiDAR reports a bridge candidate (`may_have_bridge == true`).

If odometry is at the edge but LiDAR does not show a bridge candidate, the controller remains in `push_x_pit` and continues pushing.

### `confirm_pit_box`

`confirm_pit_box` becomes a hard gate:

- It should use near-zero forward speed (`vx=0.0` or a very small creep value) and avoid probing deeper into the pit.
- It may use LiDAR alignment `vy` and yaw correction to stabilize posture.
- If strict LiDAR confirmation succeeds for `CONFIRM_REQUIRED_FRAMES`, transition to:
  - `align_climb` when the alignment angle is outside tolerance.
  - `forward` when already aligned.
- If strict confirmation fails until timeout, transition back to `push_x_pit`, not `forward`.
- When returning to `push_x_pit`, set a retry cooldown so the controller must continue pushing for at least `MIN_REPUSH_STEPS` before it can try confirmation again.

### `align_climb`

`align_climb` should also be a hard gate:

- If LiDAR remains confirmed and alignment is stable for `ALIGN_REQUIRED_FRAMES`, transition to `forward`.
- If LiDAR loses confirmation or alignment times out, transition back to `push_x_pit`, not `forward`.
- Returning to `push_x_pit` should also set the retry cooldown.

### Retry Cooldown

Add a small debounce interval after failed confirmation or failed alignment:

```text
MIN_REPUSH_STEPS = 30
```

At `dt=0.02s`, this is about 0.6 seconds. Its purpose is to prevent rapid oscillation:

```text
push_x_pit → confirm_pit_box → failed → push_x_pit → confirm_pit_box → failed → ...
```

During cooldown, `push_x_pit` continues to execute its push command but is not allowed to re-enter `confirm_pit_box`.

## Detector Interface

Keep the detector self-contained in `demo/solution.py`. Extend `_LidarClimbObservation` with a loose `may_have_bridge` boolean, while keeping existing fields:

```python
valid: bool
may_have_bridge: bool
box_in_pit: bool
alignment_angle: float
alignment_vy: float
confidence: float
reason: str
```

`box_in_pit` remains the strict confirmation signal. `may_have_bridge` can be derived from a looser version of the same pit/top-bin evidence, for example:

```text
may_have_bridge = valid and (pit_bins >= loose_min_pit_bins or top_bins >= loose_min_top_bins)
box_in_pit = pit_bins >= strict_min_pit_bins and top_bins >= strict_min_top_bins
```

The exact thresholds should start conservative enough to avoid early false positives but loose enough to let `push_x_pit` enter confirmation when the bridge is plausibly visible. The implementation should keep the strict `box_in_pit` thresholds at least as strong as the current detector.

## Data Flow

Per control step:

```text
obs["proprio"] + obs.get("extero")
  ↓
_WallPushController.update(row, extero)
  ↓
Odometry update
  ↓
_TaskDLidarClimbDetector.measure(extero)
  ↓
_update_box()
  ↓
_update_phase() uses odometry + may_have_bridge + box_in_pit + cooldown
  ↓
_control() emits velocity command
  ↓
_G1VelocityPolicyBridge.act(proprio, cmd)
```

`AlgSolution.predicts()` keeps selecting `climb` only when `controller.phase == "forward"`.

## Error Handling and Fallback

- Missing, malformed, or non-5760 `extero` should produce an invalid LiDAR observation.
- Invalid LiDAR must not allow transition into `confirm_pit_box` or `forward`.
- If LiDAR remains invalid or unconfirmed, the controller should stay in or return to `push_x_pit`.
- There should be no timeout path from `confirm_pit_box` or `align_climb` directly to `forward`.

This may sacrifice the risky late +20 attempt in cases where LiDAR cannot confirm the bridge, but it avoids the known failure mode where the robot probes into the pit without confirmation.

## Testing

Add or update tests in `tests/test_demo_task_d_lidar_climb_switch.py`:

1. `push_x_pit` with odometry beyond `PIT_EDGE_X` but `may_have_bridge=false` remains in `push_x_pit`.
2. `push_x_pit` with odometry beyond `PIT_EDGE_X`, cooldown expired, and `may_have_bridge=true` enters `confirm_pit_box`.
3. `confirm_pit_box` with consecutive strict confirmations enters `forward` when alignment is within tolerance.
4. `confirm_pit_box` timeout without strict confirmation returns to `push_x_pit` and sets retry cooldown.
5. `confirm_pit_box` command uses near-zero forward velocity.
6. `align_climb` timeout or lost strict confirmation returns to `push_x_pit` and sets retry cooldown.
7. During cooldown, `push_x_pit` cannot re-enter `confirm_pit_box` even if `may_have_bridge=true`.

Runtime verification should run a Task D probe and inspect phase records to ensure there is no transition:

```text
confirm_pit_box --timeout/unconfirmed--> forward
align_climb --timeout/unconfirmed--> forward
```

If the run still scores 16, the diagnosis should distinguish between “safe hard gate prevented unsafe forward” and “confirmed bridge entered forward but climb policy failed.”

## Scope

In scope:

- Modify `demo/solution.py` state transitions and LiDAR observation object.
- Modify/add focused tests for the hard gate behavior.
- Run focused pytest and at least one Task D runtime probe.

Out of scope:

- Retraining `policy_climb.pt`.
- Replacing height-scan detector with full point-cloud reconstruction.
- Adding precise footstep planning.
- Reworking the whole push route before `push_x_pit`.
