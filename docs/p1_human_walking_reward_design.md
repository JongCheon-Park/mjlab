# P1 Humanoid — Human-Inspired Walking Reward Design

휴머노이드(P1)에 인간 보행을 모방한 자연스러운 보행을 학습시키기 위한 reward function 설계 문서. 참고 논문: Human-Inspired Adaptive Gait Learning, SRA-GAIL, Automatic Reward Learning (bilevel), Statistical Reward Shaping (H1), SKATER.

기존 mjlab `Mjlab-Velocity-Flat-P1-MoE` task에 통합하는 형태로 작성하되, legged_gym/IsaacLab 스타일 reference 코드도 함께 제공.

---

## 0. 설계 철학 (우선순위)

```
1. 넘어지지 않기                ← 모든 학습의 기반
2. 명령 속도 추종              ← task 목표
3. 몸통을 사람처럼 세우기        ← stability
4. 좌우 다리 대칭적 운동         ← natural gait
5. hip/knee/ankle 위상관계 매칭  ← human gait pattern
6. 자연스러운 발 접촉 패턴       ← cadence
7. action/torque 매끄러움        ← smoothness
8. (선택) MoCap style imitation  ← human-likeness boost
```

핵심 원칙: **velocity tracking만 강하게 주면 사람처럼 걷기보다 앞으로만 가는 이상한 gait가 나옴**. 따라서 `posture_stability ≥ velocity_tracking > symmetry > gait_pattern > motion_similarity` 순서로 가중치 분배.

---

## 1. 전체 reward 수식

```
total_reward =
    w_vel       * r_velocity_tracking         (linear)
  + w_yaw       * r_yaw_tracking              (angular)
  + w_posture   * r_posture_stability         (orientation + ang_vel)
  + w_height    * r_base_height               (target height)
  + w_sym       * r_bilateral_symmetry        (L/R joint mirror)
  + w_gait      * r_human_gait_pattern        (frequency + foot clearance)
  + w_contact   * r_contact_pattern           (alternation + no-double-flight)
  + w_style     * r_motion_similarity         (MoCap or AMP, 옵션)
  + w_smooth    * r_smoothness                (action_rate + dof_acc)
  + w_energy    * r_energy_efficiency         (torque · joint_vel)
  + w_limit     * r_joint_limit               (soft joint limit)
  + w_alive     * r_alive                     (생존 보너스)
  + w_fall      * r_fall_penalty              (넘어짐 페널티)
```

---

## 2. 현재 mjlab과의 매핑

| 설계 항목              | 현재 mjlab                                                                 | 상태         | 비고                               |
| ---------------------- | -------------------------------------------------------------------------- | ------------ | ---------------------------------- |
| `r_velocity_tracking`  | `track_linear_velocity`                                                    | ✓            | std 0.2 권장                       |
| `r_yaw_tracking`       | `track_angular_velocity`                                                   | ✓            | low_std 0.32                       |
| `r_posture_stability`  | `upright` (`flat_orientation_multi`) + `body_ang_vel` + `angular_momentum` | ✓            | 통합되어 있음                      |
| `r_base_height`        | `base_height`                                                              | ✓            | target=0.84m                       |
| `r_bilateral_symmetry` | (gait pattern reward들이 부분 처리)                                        | ⚠ **불완전** | 명시적 L/R mirror reward 부재      |
| `r_human_gait_pattern` | `biped_air_time`, `biped_double_support_time`, `heelstrike/toeoff_*` 8개   | ✓            | gait frequency 직접 측정 없음      |
| `r_contact_pattern`    | `biped_air_time`, `foot_slip`                                              | ✓            | feet_alternation 명시 없음         |
| `r_motion_similarity`  | 없음                                                                       | ✗ **부재**   | MoCap reference 또는 AMP 별도 필요 |
| `r_smoothness`         | `action_rate_l2`, `joint_torque_rate`, `body_ang_vel`                      | ✓            | dof_acc 없음 (덜 중요)             |
| `r_energy_efficiency`  | `joint_effort_limit`, `hip_roll_torque_l2`                                 | ⚠ **부분**   | 전체 power penalty 부재            |
| `r_joint_limit`        | `dof_pos_limits`                                                           | ✓            | weight -1.0                        |
| `r_alive`              | (암묵적: episode_length이 길수록 reward 누적)                              | ⚠            | 명시적 +1/step 없음                |
| `r_fall_penalty`       | `fell_over` termination                                                    | ✓            | -inf 효과 (episode 종료)           |

**결론**: 큰 틀은 다 있고 **추가/보강 필요한 항목**:
1. `r_bilateral_symmetry` — 명시적 좌우 mirror reward (지금은 gait pattern으로 부분 처리)
2. `r_motion_similarity` — MoCap 또는 AMP-style (별도 reference 데이터 필요)
3. `r_energy_efficiency` — 전체 torque·power penalty
4. `r_alive` — 명시적 생존 보너스

---

## 3. 각 reward 함수 구현 (legged_gym 스타일)

mjlab의 manager 구조에 wrap하기 쉽게 함수 시그니처는 `(env, asset_cfg, **params)` 형태로도 변환 가능. 아래는 legged_gym style의 self-method 형태.

### 3.1 `_reward_tracking_lin_vel()`

```python
def _reward_tracking_lin_vel(self):
    """Reward for tracking commanded linear velocity in xy plane.

    Uses tight std (0.2) so that standing still under non-zero command gives
    near-zero reward — prevents standing-local-minimum exploit.
    """
    lin_vel_error = torch.sum(
        torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / self.cfg.tracking_sigma_lin**2)
```

### 3.2 `_reward_tracking_ang_vel()`

```python
def _reward_tracking_ang_vel(self):
    """Reward for tracking yaw rate command."""
    ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
    return torch.exp(-ang_vel_error / self.cfg.tracking_sigma_ang**2)
```

### 3.3 `_reward_base_orientation()`

```python
def _reward_base_orientation(self):
    """Penalize torso tilt + angular velocity. Y-axis (roll) is stricter than
    X-axis (pitch) because lateral lean is more dangerous for humanoids.
    """
    # projected gravity xy: 0 when upright
    gravity_xy = self.projected_gravity[:, :2]
    # asymmetric std: roll tighter than pitch
    weighted = (
        torch.square(gravity_xy[:, 0]) / self.cfg.sigma_pitch**2
        + torch.square(gravity_xy[:, 1]) / self.cfg.sigma_roll**2
    )
    r_orient = torch.exp(-weighted)
    # angular velocity penalty (penalty form, will be subtracted)
    p_ang_vel = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    return r_orient - self.cfg.w_ang_vel * p_ang_vel
```

### 3.4 `_reward_base_height()`

```python
def _reward_base_height(self):
    """Maintain base height near target. Target should be slightly below
    standing height for natural walking knee bend.
    """
    height_error = torch.square(self.base_pos[:, 2] - self.cfg.target_base_height)
    return torch.exp(-height_error / self.cfg.sigma_h**2)
```

### 3.5 `_reward_bilateral_symmetry()`

```python
def _reward_bilateral_symmetry(self):
    """Encourage human-like left/right joint relationships.

    - hip_pitch: anti-phase (one forward while other back) → (q_L + q_R) should be small
    - hip_roll: similar pattern, but mirror sign depends on convention
    - knee: roughly anti-phase via flexion-extension cycle (use velocity sym)
    - ankle_pitch: anti-phase like hip_pitch

    Sign convention WARNING: this assumes a "standard" humanoid where
    left/right are reflections about the sagittal plane with opposite signs
    on roll axes. Verify with `robot.dof_names` and joint axis directions.
    """
    dof = self.dof_pos

    # hip pitch: anti-phase → sum near 0
    sym_hp = torch.square(dof[:, self.idx_lhp] + dof[:, self.idx_rhp])

    # hip roll: mirror (most robots: same numerical sign for symmetric motion)
    # If your robot has axis-flipped roll, use + instead of -
    sym_hr = torch.square(dof[:, self.idx_lhr] - dof[:, self.idx_rhr])

    # knee: by anti-phase principle, (q_L - q_R) is what swings
    # Use velocity to capture phase instead of position
    vL = self.dof_vel[:, self.idx_l_knee]
    vR = self.dof_vel[:, self.idx_r_knee]
    sym_knee_vel = torch.square(vL + vR)  # anti-phase → sum near 0

    # ankle pitch: anti-phase
    sym_ankle = torch.square(dof[:, self.idx_l_ap] + dof[:, self.idx_r_ap])

    # weighted sum (smaller = more symmetric)
    error = (
        self.cfg.w_sym_hp * sym_hp
        + self.cfg.w_sym_hr * sym_hr
        + self.cfg.w_sym_knee * sym_knee_vel
        + self.cfg.w_sym_ankle * sym_ankle
    )

    # adaptive weight: relax during turning commands (turning breaks symmetry)
    yaw_cmd_abs = torch.abs(self.commands[:, 2])
    turning_factor = torch.clamp(1.0 - yaw_cmd_abs / self.cfg.turn_threshold, 0.0, 1.0)

    return turning_factor * torch.exp(-error / self.cfg.sigma_sym**2)
```

### 3.6 `_reward_human_gait_pattern()`

```python
def _reward_human_gait_pattern(self):
    """Encourage cadence and foot clearance close to human ranges.

    Human walking cadence: ~1.5–2.0 Hz (steps per second per foot).
    Foot clearance: 5–15 cm peak swing height.
    """
    # gait frequency: estimate from recent contact-toggle events
    # requires history buffer; here we use a running rate stored externally
    estimated_freq = self.gait_freq_estimator()  # custom helper, see below

    target_freq = self.cfg.target_gait_freq  # e.g., 1.8 Hz
    r_freq = torch.exp(
        -torch.square(estimated_freq - target_freq) / self.cfg.sigma_freq**2
    )

    # foot clearance (peak swing height)
    peak_z = torch.maximum(self.foot_peak_z_L, self.foot_peak_z_R)
    target_clear = self.cfg.target_foot_clearance  # e.g., 0.10 m
    r_clear = torch.exp(
        -torch.square(peak_z - target_clear) / self.cfg.sigma_clear**2
    )

    return 0.5 * r_freq + 0.5 * r_clear
```

`gait_freq_estimator` 구현 메모: 매 step `air_time` 카운터를 토대로 한 발의 contact-toggle 빈도를 추정. 또는 last N step 동안 contact transitions 수를 세서 frequency 환산.

### 3.7 `_reward_motion_similarity()`

```python
def _reward_motion_similarity(self):
    """Optional: reward similarity with human MoCap reference trajectory.

    Two flavors:
    A) Explicit joint tracking — direct L2 to reference joints (morphology-
       sensitive, can destabilize). Use small weight.
    B) AMP-style discriminator — learn-based, captures distribution rather
       than exact trajectory. Requires separate discriminator training.

    Below: explicit form (A).
    """
    if self.cfg.reference_motion is None:
        return torch.zeros(self.num_envs, device=self.device)

    # Selected joints to match (avoid wrist/shoulder details that morphology differs)
    selected = [
        self.idx_lhp, self.idx_rhp,
        self.idx_lhr, self.idx_rhr,
        self.idx_l_knee, self.idx_r_knee,
        self.idx_l_ap, self.idx_r_ap,
    ]
    q_robot = self.dof_pos[:, selected]
    q_ref = self.reference_motion.sample(self.gait_phase)  # phase ∈ [0, 1]

    joint_error = torch.sum(
        self.cfg.joint_weights * torch.square(q_robot - q_ref), dim=1
    )
    return torch.exp(-joint_error / self.cfg.sigma_motion**2)
```

AMP-style의 경우 reward는 `r_style = log(D(s_t, s_{t+1}))` 형태. mjlab에 AMP 통합은 별도 작업.

### 3.8 `_reward_feet_contact()`

```python
def _reward_feet_contact(self):
    """Reward alternating contact pattern. No double-flight (both feet in air
    for too long) and no quadruple-stance shuffle.
    """
    left_contact = self.contact_left  # bool tensor
    right_contact = self.contact_right
    
    # No double flight
    both_air = ~(left_contact | right_contact)
    p_double_flight = both_air.float()

    # Contact alternation: at least one foot should be in contact most of the time
    r_no_double_flight = 1.0 - p_double_flight

    # Optional: encourage single-support during walking (cmd > threshold)
    cmd_speed = torch.norm(self.commands[:, :2], dim=1)
    is_walking = cmd_speed > self.cfg.walking_threshold
    single_support = left_contact ^ right_contact  # XOR
    r_single = single_support.float() * is_walking.float() * 0.5  # smaller weight

    return r_no_double_flight + r_single
```

### 3.9 `_reward_feet_slide()`

```python
def _reward_feet_slide(self):
    """Penalize foot sliding while in contact. Force foot to be planted
    firmly during stance phase.
    """
    contact = torch.stack([self.contact_left, self.contact_right], dim=1).float()
    foot_xy_vel = torch.stack(
        [self.foot_lin_vel_L[:, :2], self.foot_lin_vel_R[:, :2]], dim=1
    )
    foot_speed = torch.norm(foot_xy_vel, dim=-1)  # (n_envs, 2)
    return torch.sum(contact * foot_speed, dim=1)  # penalty (will be negated)
```

### 3.10 `_reward_action_rate()`

```python
def _reward_action_rate(self):
    """Penalize action jumps step-to-step. Smoothness."""
    return torch.sum(torch.square(self.actions - self.last_actions), dim=1)
```

### 3.11 `_reward_dof_acc()`

```python
def _reward_dof_acc(self):
    """Penalize joint accelerations. Avoid jerky motion."""
    dof_acc = (self.dof_vel - self.last_dof_vel) / self.dt
    return torch.sum(torch.square(dof_acc), dim=1)
```

### 3.12 `_reward_energy()`

```python
def _reward_energy(self):
    """Cost of transport proxy. Penalize mechanical power consumption.

    Note: in evaluation, use CoT = total_work / (m * g * distance) as the
    canonical metric. In training, step-level power penalty suffices.
    """
    power = torch.abs(self.torque * self.dof_vel)
    return torch.sum(power, dim=1)  # penalty (negative weight applied externally)
```

### 3.13 `_reward_joint_limits()`

```python
def _reward_joint_limits(self):
    """Soft penalty for approaching joint limits."""
    margin = self.cfg.joint_limit_margin  # e.g., 0.05 rad
    exceed_upper = torch.clamp(
        self.dof_pos - (self.dof_pos_upper - margin), min=0.0
    )
    exceed_lower = torch.clamp(
        (self.dof_pos_lower + margin) - self.dof_pos, min=0.0
    )
    return torch.sum(torch.square(exceed_upper) + torch.square(exceed_lower), dim=1)
```

### 3.14 `_reward_alive()`

```python
def _reward_alive(self):
    """Constant survival bonus. Helps PPO early when other rewards are noisy."""
    return torch.ones(self.num_envs, device=self.device)
```

### 3.15 `_reward_fall_penalty()`

```python
def _reward_fall_penalty(self):
    """One-shot penalty when fall conditions trigger."""
    base_too_low = self.base_pos[:, 2] < self.cfg.fall_height_threshold
    too_rolled = torch.abs(self.base_rpy[:, 0]) > self.cfg.fall_roll_limit
    too_pitched = torch.abs(self.base_rpy[:, 1]) > self.cfg.fall_pitch_limit

    fallen = base_too_low | too_rolled | too_pitched
    return fallen.float()  # multiplied by negative weight externally
```

---

## 4. 초기 weight 값 (튜닝 출발점)

```python
reward_scales = {
    # === core tracking ===
    "tracking_lin_vel": 1.0,
    "tracking_ang_vel": 0.5,

    # === stability (priority 1: no fall) ===
    "base_orientation": 1.5,     # >= velocity, prevents standing-pose collapse
    "base_height":      0.8,

    # === human-likeness ===
    "bilateral_symmetry": 0.5,
    "human_gait_pattern": 0.4,
    "feet_contact":       0.3,
    "motion_similarity":  0.2,   # very low until verified

    # === penalties (negative weights) ===
    "action_rate":  -0.05,
    "dof_acc":      -0.02,
    "torque_power": -0.0005,
    "joint_limit":  -1.0,
    "feet_slide":   -0.2,
    "collision":    -1.0,

    # === episode shaping ===
    "alive":  0.2,
    "fall":  -5.0,
}
```

**원칙**:
- `posture_stability ≥ velocity_tracking > symmetry > gait_pattern > motion_similarity`
- penalty 절대값 합 ≈ positive reward 합의 1/5 ~ 1/3 (균형)
- `motion_similarity`는 처음엔 0.2로 작게. MoCap-tracking exploit 위험.

---

## 5. Curriculum schedule (4-stage)

### Stage 1 — Stable standing + slow walking (iter 0–1k)

```python
strong = ["base_orientation", "base_height", "alive", "fall"]
medium = ["tracking_lin_vel"]
weak   = ["bilateral_symmetry", "feet_contact"]
off    = ["motion_similarity"]
```

목표: 안 넘어지고, 천천히 (cmd ±0.5m/s) 앞으로 이동. 무릎 과도하게 꺾지 않기, 발 슬라이드 안 하기.

**Weight 조정**:
```python
reward_scales["motion_similarity"]   = 0.0
reward_scales["bilateral_symmetry"]  = 0.2  # 약하게
reward_scales["human_gait_pattern"]  = 0.1
reward_scales["feet_contact"]        = 0.2
```

### Stage 2 — Natural left/right alternation (iter 1k–5k)

```python
strong = ["base_orientation", "tracking_lin_vel"]
medium = ["bilateral_symmetry", "feet_contact", "human_gait_pattern"]
weak   = ["motion_similarity"]
```

목표: 좌우 다리 교대, hip pitch 반대 위상, 발을 끌지 않는 swing, 적당한 step length.

**Weight 조정**:
```python
reward_scales["bilateral_symmetry"]  = 0.5
reward_scales["human_gait_pattern"]  = 0.4
reward_scales["feet_contact"]        = 0.3
reward_scales["motion_similarity"]   = 0.1  # 미세하게 도입
```

### Stage 3 — Human-like style refinement (iter 5k–15k)

```python
strong = ["base_orientation", "tracking_lin_vel", "bilateral_symmetry"]
medium = ["human_gait_pattern", "motion_similarity"]
weak   = ["torque_power", "action_rate"]  # 강도 점진 증가
```

목표: 사람과 유사한 joint coordination, 몸통 흔들림 감소, CoT 감소, 자연스러운 cadence.

**Weight 조정**:
```python
reward_scales["bilateral_symmetry"]  = 0.7
reward_scales["motion_similarity"]   = 0.3
reward_scales["torque_power"]        = -0.001  # 2× 강화
reward_scales["action_rate"]         = -0.08
```

### Stage 4 — Speed and direction variation (iter 15k–30k)

```python
strong = ["base_orientation", "tracking_lin_vel", "tracking_ang_vel"]
medium = ["bilateral_symmetry (adaptive)", "human_gait_pattern"]
```

목표: cmd 범위 확장 (±1.5 m/s, yaw ±2.0 rad/s), terrain perturbation, push 외란 대응.

**Adaptive symmetry**: turning command가 크면 symmetry weight 자동 감소.
```python
yaw_cmd_abs = abs(commands[:, 2])
sym_weight_scale = max(0, 1 - yaw_cmd_abs / 1.5)
```

---

## 6. mjlab 통합 (구현 가이드)

현재 mjlab의 manager 구조에선 위 함수들을 `mjlab.tasks.velocity.mdp.rewards` 모듈에 `RewardTermCfg`-호환 형태로 wrap.

### 6.1 추가할 새 reward 함수

```python
# src/mjlab/tasks/velocity/mdp/human_walking_rewards.py
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.scene_entity_config import SceneEntityCfg


def bilateral_symmetry(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    sigma_sym: float = 0.3,
    w_hp: float = 1.0,
    w_hr: float = 0.5,
    w_knee_vel: float = 0.3,
    w_ankle: float = 0.5,
    turn_threshold: float = 1.0,
    cmd_name: str = "twist",
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    # joint indices: resolved from name pattern
    idx_lhp = asset.joint_indices_by_name("left_hip_pitch_joint")
    idx_rhp = asset.joint_indices_by_name("right_hip_pitch_joint")
    idx_lhr = asset.joint_indices_by_name("left_hip_roll_joint")
    idx_rhr = asset.joint_indices_by_name("right_hip_roll_joint")
    idx_lkn = asset.joint_indices_by_name("left_knee_joint")
    idx_rkn = asset.joint_indices_by_name("right_knee_joint")
    idx_lap = asset.joint_indices_by_name("left_ankle_pitch_joint")
    idx_rap = asset.joint_indices_by_name("right_ankle_pitch_joint")

    dof = asset.data.joint_pos
    dvel = asset.data.joint_vel

    sym_hp = torch.square(dof[:, idx_lhp] + dof[:, idx_rhp]).squeeze(-1)
    sym_hr = torch.square(dof[:, idx_lhr] - dof[:, idx_rhr]).squeeze(-1)
    sym_knee = torch.square(dvel[:, idx_lkn] + dvel[:, idx_rkn]).squeeze(-1)
    sym_ankle = torch.square(dof[:, idx_lap] + dof[:, idx_rap]).squeeze(-1)

    err = (
        w_hp * sym_hp + w_hr * sym_hr
        + w_knee_vel * sym_knee + w_ankle * sym_ankle
    )

    yaw_cmd = env.command_manager.get_command(cmd_name)[:, 2]
    turning = torch.clamp(1.0 - torch.abs(yaw_cmd) / turn_threshold, 0.0, 1.0)

    return turning * torch.exp(-err / sigma_sym**2)


def joint_power(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Energy proxy: |torque * joint_vel| summed over selected joints."""
    asset = env.scene[asset_cfg.name]
    torque = asset.data.applied_torque  # (n_envs, n_joints)
    dvel = asset.data.joint_vel
    return torch.sum(torch.abs(torque * dvel), dim=1)


def alive_bonus(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Constant survival bonus."""
    return torch.ones(env.num_envs, device=env.device)
```

### 6.2 P1 task에 등록

```python
# tasks/velocity/config/p1/env_cfgs.py 끝부분에 추가
from mjlab.tasks.velocity.mdp import human_walking_rewards

cfg.rewards["bilateral_symmetry"] = RewardTermCfg(
    func=human_walking_rewards.bilateral_symmetry,
    weight=0.5,
    params={
        "asset_cfg": SceneEntityCfg("robot"),
        "sigma_sym": 0.3,
        "w_hp": 1.0, "w_hr": 0.5, "w_knee_vel": 0.3, "w_ankle": 0.5,
    },
)

cfg.rewards["energy_power"] = RewardTermCfg(
    func=human_walking_rewards.joint_power,
    weight=-0.0005,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*_(hip|knee|ankle)_.*_joint",))},
)

cfg.rewards["alive_bonus"] = RewardTermCfg(
    func=human_walking_rewards.alive_bonus,
    weight=0.2,
    params={},
)
```

---

## 7. Reward Debugging 가이드

### 7.1 항목별 로그 확인 (필수)

학습 중 매 iter마다 다음 메트릭을 분리해서 봐야 함:

```bash
uv run python /home/park/mjlab/scripts/check_rewards.py
```

또는 TensorBoard에서:
- `Episode_Reward/bilateral_symmetry`
- `Episode_Reward/human_gait_pattern`
- `Episode_Reward/feet_contact`
- `Episode_Reward/motion_similarity`
- `Episode_Reward/base_orientation`
- 등 모든 reward term

### 7.2 문제 진단 매트릭스

| 증상                                 | 의심 reward                                             | 조치                                         |
| ------------------------------------ | ------------------------------------------------------- | -------------------------------------------- |
| 안 걷고 가만히 서있음                | `tracking_lin_vel` std 너무 큼 (partial-credit exploit) | std 0.5 → 0.2                                |
| 발 끌면서 shuffle gait               | `feet_slide` 약함, `gait_pattern` 약함                  | feet_slide weight ×2, gait_pattern weight ×2 |
| 몸통 크게 흔들림                     | `base_orientation` 약함, `body_ang_vel` 약함            | orientation weight ×2, ang_vel weight ×4     |
| 무릎 너무 굽힘 (crouching)           | `base_height` 약함 또는 target 낮음                     | target_height +0.05                          |
| 좌우 비대칭, 한쪽 다리만 움직임      | `bilateral_symmetry` 약함                               | sym weight ×2                                |
| 미친 듯이 점프 (양발 동시 비행)      | `feet_contact` 약함, `no_double_flight` 없음            | contact weight ×2                            |
| 자세 너무 뻣뻣 (사람답지 않음)       | `motion_similarity` 부재 또는 약함                      | MoCap reference 추가, weight 0.3             |
| 빠른 진동/떨림                       | `action_rate`, `dof_acc` 약함                           | action_rate weight ×2                        |
| 너무 느림, 발 거의 안 듦             | `gait_pattern` 약함, `smoothness` 너무 강함             | smoothness 페널티 ×0.5                       |
| 학습 자체가 안 됨 (reward 음수 지속) | penalty 너무 강함                                       | 모든 negative weight ×0.5                    |

### 7.3 추가 진단 신호

- **per-step reward 분해**: `total = Σ w_i * r_i`의 각 항 출력. 어느 항이 dominate하는지 확인.
- **상관관계 체크**: `track_lin_vel`이 높지만 `feet_contact` 0이면 정지 정책 의심.
- **gradient norm 모니터링**: 너무 크면 (>10) → learning_rate 감소 또는 reward magnitude 정규화.

---

## 8. 평가 지표 (학습 reward와 별개)

학습 중에는 reward를 maximize하지만, 평가는 **사람 같은 보행인지를 정량화**.

```python
eval_metrics = {
    # === stability ===
    "fall_rate":               # 일정 거리 내 fell_over 비율
    "torso_pitch_std":         # episode 중 torso pitch 표준편차
    "torso_roll_std":          # 동일 (roll)
    "lateral_displacement":    # 직진 명령에서 y축 누적 변위

    # === tracking ===
    "forward_distance":        # 일정 시간 내 x 진행 거리
    "v_tracking_error":        # mean |v_cmd - v_actual|
    "yaw_tracking_error":      # mean |yaw_cmd - yaw_actual|

    # === gait quality ===
    "gait_symmetry_index":     # Robinson SI: 100·(2·|qL-qR|)/(qL+qR)
    "step_length_avg":         # 평균 보폭
    "step_length_std":         # 보폭 표준편차 (작을수록 일정)
    "cadence_hz":              # 측정된 보행 주파수
    "single_support_ratio":    # 한 발만 닿아 있는 시간 비율

    # === human similarity ===
    "human_joint_similarity":  # selected joint trajectories 대 MoCap reference 코사인 유사도
    "amp_discriminator_score": # AMP 사용 시 D(robot_motion) 평균

    # === foot quality ===
    "feet_slip_distance":      # 접지 중 발 미끄러진 누적 거리
    "max_foot_clearance":      # swing 중 발 최대 높이
    "double_support_time":     # 양발 동시 접지 시간 (per stride)
    "double_flight_time":      # 양발 동시 비행 시간 (낮아야 함)

    # === energy ===
    "CoT":                     # cost of transport = work / (m·g·d)
    "torque_rms":              # 토크 RMS
    "energy_per_meter":        # 거리당 에너지 소비

    # === smoothness ===
    "action_jerk":             # action 3차 미분 norm
    "joint_vel_std":           # 관절 속도 표준편차
}
```

**핵심 지표 (사람 같은 보행)**:
- `gait_symmetry_index < 5` (Robinson SI: 5% 미만이면 매우 대칭)
- `cadence_hz ∈ [1.5, 2.0]` (사람 normal walking)
- `torso_pitch_std < 5°`, `torso_roll_std < 3°`
- `CoT < 0.3` (사람 보행 CoT ≈ 0.2)
- `single_support_ratio > 0.6` (한 발 stance가 보행 대부분)
- `feet_slip_distance < 0.01m` (거의 안 미끄러짐)

---

## 9. 현재 P1-MoE 셋업 vs 본 설계 — 차이 / 권장 조치

### 9.1 이미 잘 맞는 부분

- velocity tracking std=0.2 (빡빡): ✓ standing 회피
- biped_air_time curriculum: 0 → 30 → 20: ✓ 명시적 walking 보상
- upright weight=1.0, std_xy=(sin10°, sin1°): ✓ 측면 기울기 엄격
- 8개 gait pattern reward (heel/toe × hip/knee/ankle): ✓ phase 가이드
- dof_pos_limits, action_rate_l2 적정 페널티: ✓

### 9.2 부족한 부분 — 추가 권장

1. **`bilateral_symmetry` (명시적)** 추가 → 위 6.1 코드 그대로 등록. weight 0.5로 시작.
2. **`alive_bonus`** 추가 → weight 0.2. PPO 초기 stabilization.
3. **`joint_power` (에너지)** 추가 → weight -0.0005. Stage 3에서 강화.
4. **`gait_frequency` 측정 함수** → contact-toggle history buffer로 구현. Stage 2부터 reward에 포함.
5. **(선택) AMP discriminator** → 별도 reference dataset + 모듈 통합 작업. 현 P1 동작이 안정된 후 도입.

### 9.3 조정 권장

- 현재 `pose` reward weight 2.0 + `weight_standing=5.0` (cmd=0 시 5×) → standing 시 pose=10.0이 dominate. cmd≠0인데도 partial-credit exploit 가능성. **Stage 1 통과 후엔 `weight_standing` 3.0으로 낮추기 권장.**
- `biped_air_time` curriculum 6k step → 30 weight는 강한 신호. walking 학습 후엔 `weight=15` 정도로 낮춰 다른 reward (symmetry, energy)에 자리 양보.

---

## 10. 실행 체크리스트

학습 시작 전:

- [ ] joint mirror sign convention 검증 (`_reward_bilateral_symmetry` 부호)
- [ ] `target_base_height` 측정 (P1 standing pose에서 base z)
- [ ] `target_gait_freq` 결정 (사람 normal: 1.8 Hz, P1 leg length 따라 조정 가능)
- [ ] `fall_height_threshold` 측정 (standing height 60%)
- [ ] reward weight 초기값 설정 (§4)
- [ ] curriculum schedule 코드 반영 (§5)

학습 중:

- [ ] iter 100, 500, 1000에서 reward term별 로그 점검
- [ ] gait visualization으로 발 swing 시작 시점 확인
- [ ] symmetry index 측정 (5% 이내 목표)
- [ ] CoT 측정 (0.3 이하 목표)

학습 후 평가:

- [ ] §8의 모든 metric 측정
- [ ] 동일 명령으로 baseline (track_lin_vel only) vs current 비교
- [ ] real robot zero-shot transfer (sim2real gap 측정)

---

## 11. 참고 문헌 매핑

| 논문                                  | 본 설계의 반영 항목                                                                            |
| ------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Human-Inspired Adaptive Gait Learning | `r_bilateral_symmetry`, `r_human_gait_pattern`, 4-stage curriculum                             |
| SRA-GAIL                              | `r_motion_similarity` (joint trajectory similarity 형식)                                       |
| Automatic Reward Learning (bilevel)   | reward 구성 요소 catalog (대부분 본 설계가 흡수)                                               |
| Statistical Reward Shaping (H1)       | weight tuning principles, posture/fall/forward trade-off                                       |
| SKATER                                | `r_bilateral_symmetry`, `r_smoothness`, `r_energy` 만으로도 주기적 gait 가능 — fallback 디자인 |

---

## 부록 A — 최소 구현 (mjlab에 바로 들어갈 reward function 패키지)

(코드: `src/mjlab/tasks/velocity/mdp/human_walking_rewards.py` 별도 파일로 작성)

부록 A는 빠르게 P1-MoE에 통합할 수 있는 **최소 추가 reward** 3개를 담은 단일 .py 파일이며, 위 §6.1을 그대로 따른다. 본 .md 적용 후 `tasks/velocity/config/p1/env_cfgs.py`에 §6.2의 등록 코드 3블록 추가하면 끝.

---

## 12. 다음 단계

1. **부록 A 파일 생성** (`src/mjlab/tasks/velocity/mdp/human_walking_rewards.py`) — 즉시 가능
2. **P1 task config에 등록** — `bilateral_symmetry`, `alive_bonus`, `joint_power` 3개 추가
3. **curriculum stage 1**부터 새 run — 약 1k iter 관찰
4. **stage 2 진입 시 weight 조정** — symmetry 0.2→0.5, gait_pattern 0.1→0.4
5. **stage 3 진입 시** — motion_similarity 도입 검토 (MoCap or AMP)
6. **stage 4 진입 시** — adaptive symmetry weight (turning command 클 때 감소)

각 stage의 종료 기준은 §8 평가 지표가 만족될 때.

---

## 부록 B — r3 적용 이력 (자연 보행 튜닝)

### B.1 문제: 이전 run에서 reward collapse 관측 (`2026-05-11_15-36-15`)

|    iter |    mean_reward | 진단             |
| ------: | -------------: | ---------------- |
|     100 |            -10 | 페널티 dominant  |
|     400 |             99 | 학습 빠르게 상승 |
| **700** | **145 (peak)** | 최정점           |
|    1000 |            113 | 하락 시작        |
|    1500 |             65 | 계속 하락        |
|    1688 |             55 | 현재             |

ep_length는 789로 유지되어 **안 넘어지지만 reward는 계속 깎이는** 상태. iter 1688의 reward 분해:

- 양수: biped_air_time(7.96) + pose(1.41) + upright(1.31) + track_lin(1.30) + base_height(0.67) + track_ang(0.27) ≈ **+12.6/sec**
- 음수: action_rate_l2(-3.82) + angular_momentum(-2.16) + biped_standing_stability(-2.05) + 기타(-1.8) ≈ **-9.8/sec**

→ 정책이 `biped_air_time` (weight=30) 최대화하려고 **거칠게 walking** → action 점프(`action_rate -3.82`)로 페널티가 거의 같은 금액 갉아먹음. **자연 보행이 아닌 격렬 보행**으로 수렴.

### B.2 자연 보행을 위한 weight 재설계

세 논문 통합 원칙:
1. **HI**: 명시적 bilateral symmetry (anti-phase hip pitch + in-phase hip roll) — 양수 reward로 자연스러움 강제
2. **Statistical (H1)**: `feet_air_time` 양수 기여 1위지만 너무 강하면 다른 항이 무너짐
3. **SKATER**: bilateral symmetry 만 명시해도 periodic gait emerge

### B.3 r3 적용 변경

| 항목                                    |      이전 (r2) |          변경 (r3) | 이유                                               |
| --------------------------------------- | -------------: | -----------------: | -------------------------------------------------- |
| `biped_air_time_weight` curriculum peak | 30 (sustained) | **15 (sustained)** | 격렬 보행 유발 막음; bilateral_symmetry로 보강     |
| `action_rate_l2`                        |          -0.05 |          **-0.02** | 보폭 자유 — per-sec 페널티 -3.82 → ~-1.5 예상      |
| `angular_momentum`                      |           -0.1 |          **-0.03** | walking 죽이지 않도록 — per-sec -2.16 → ~-0.6 예상 |
| `biped_standing_stability`              |            1.0 |            **0.5** | walking 중 -2.05 까지 누적되던 문제 절반으로       |
| `body_ang_vel`                          |           -0.2 |           **-0.1** | 약간 완화                                          |
| `bilateral_symmetry` (NEW)              |              — |           **+0.5** | HI 핵심: hip pitch anti-phase + hip roll in-phase  |
| `energy_power` (NEW)                    |              — |          **-5e-4** | HI/SKATER CoT proxy                                |
| `alive_bonus` (NEW)                     |              — |           **+0.2** | PPO 초기 안정화                                    |

### B.4 `bilateral_symmetry` reward 설계 (HI 논문 정확 반영)

```python
sym_hp     = (q_L_hip_pitch + q_R_hip_pitch)²   # anti-phase → sum near 0
sym_hr     = (q_L_hip_roll - q_R_hip_roll)²      # in-phase mirror
sym_knee   = (q̇_L_knee + q̇_R_knee)²              # anti-phase (velocity)
sym_ankle  = (q_L_ankle_p + q_R_ankle_p)²        # anti-phase
error      = 1.0·sym_hp + 0.5·sym_hr + 0.3·sym_knee + 0.5·sym_ankle
reward     = turning_factor · exp(-error / 0.4²)
```

- `turning_factor = clip(1 - |ω_z_cmd|/1.0, 0, 1)`: yaw cmd가 크면 자동 감쇠 (turning은 symmetry 깨도 OK)
- `sigma_sym = 0.4`: HI 논문의 사람 hip pitch L+R 합 분산(~0.09)에 맞춘 partial credit 허용

### B.5 예상 reward (per-sec)

| 항목                            |         예상값 | 비고                   |
| ------------------------------- | -------------: | ---------------------- |
| biped_air_time (weight 15)      |      +1.5~+3.0 | curriculum 발동 후     |
| bilateral_symmetry              |      +0.3~+0.7 | 학습 진행 따라 상승    |
| pose                            |           +1.4 | 유지                   |
| upright                         |           +1.3 | 유지                   |
| track_lin_vel                   |      +1.3~+2.0 | symmetry 도움으로 개선 |
| base_height                     |           +0.7 | 유지                   |
| track_ang_vel                   |      +0.3~+0.5 | turning 학습 후        |
| alive_bonus                     |           +0.2 | 상수                   |
| **positive 합**                 | **+6.7~+10.0** |                        |
| action_rate_l2 (페널티 ½)       |           -1.5 | -3.82 → -1.5 예상      |
| angular_momentum (페널티 ⅓)     |           -0.6 | -2.16 → -0.6 예상      |
| biped_standing_stability (½)    |           -1.0 | -2.05 → -1.0 예상      |
| 기타 (foot, dof, energy 등)     |           -1.0 | 거의 유지              |
| **negative 합**                 |       **-4.1** |                        |
| **net per-sec**                 |  **+2.6~+5.9** |                        |
| **expected mean_reward (×20s)** |     **52~118** | 현 55 대비 0~2배 회복  |

walking이 부드러워지면 action_rate/angular_momentum 페널티가 더 작아져서 실제 net은 +6~+9/sec → **mean_reward 120~180**까지 회복 예상.

### B.6 다음 모니터링 신호

성공:
- `Episode_Reward/bilateral_symmetry > 0.5`: 좌우 대칭 패턴 emerge
- `Episode_Reward/action_rate_l2 > -2.0`: 부드러운 walking
- `Episode_Reward/angular_momentum > -1.0`: torso 흔들림 감소
- `Mean reward > 200`: 자연 보행 달성

실패 → 다음 단계:
- bilateral_symmetry 가 0에 머무름 → joint mirror sign convention 확인 (P1 axis 방향)
- mean_reward 100 미만 → biped_air_time을 8로 더 낮춰서 부드러움에 집중
- 여전히 거친 보행 → `bilateral_symmetry` weight 0.5 → 1.0으로 강화

---

## 부록 C — t1·t2 분리 학습 + t3 하이브리드 (2026-05-12)

### C.1 t1 vs t2 결과 관측

| run    |        iter | 사용자 평가                           |
| ------ | ----------: | ------------------------------------- |
| **t1** | 30,000 완주 | 전진 보행 ✓ / 사이드·회전 ✗           |
| **t2** | 30,000 완주 | 사이드·회전 ✓ / 전진은 t1보다 덜 정밀 |

### C.2 t1 ↔ t2 결정적 차이

`compare_runs.py`로 추출한 핵심 diff:

| 항목                          |       t1 |          t2 |
| ----------------------------- | -------: | ----------: |
| `track_lin_vel_std` 시작      | **0.20** |        0.28 |
| `pose.std_walking[hip_pitch]` | **0.30** |        0.50 |
| `pose.std_walking[hip_yaw]`   | **0.15** |        0.30 |
| `upright.std_xy[y]`           |    sin1° | **sin0.7°** |
| `lateral_step_lead.weight`    |   (없음) |    **0.30** |
| `min_first_swing_air_time`    |     0.20 |    **0.35** |

**원인 진단**:
- t1 전진 잘됨 → `track_lin_vel_std=0.20` (tight curve) 가 forward 정밀도 강제
- t2 회전 잘됨 → `pose[hip_yaw]=0.30` (loose) + `upright.y` tight가 옆 기울기 막고 hip yaw 사용 허용
- t2 옆걸음 잘됨 → `lateral_step_lead` 명시 보상 + `upright.y` tight

### C.3 t3 하이브리드 설계

각 항목별 출처:

| 항목                        |        t3 값 | 출처 / 의도                  |
| --------------------------- | -----------: | ---------------------------- |
| `track_lin_vel_std` 시작    |     **0.20** | t1 — forward 정밀도          |
| `track_lin_vel_std` 최종    |     **0.15** | 더 빡빡한 final stage        |
| `pose[hip_pitch]` walking   |     **0.40** | t1(0.30)↔t2(0.50) 중간       |
| `pose[hip_yaw]` walking     |     **0.30** | t2 — 회전 자유               |
| `pose[knee]` walking        |     **0.70** | t2                           |
| `pose[ankle_pitch]` walking |     **0.30** | t2                           |
| `upright.std_xy[y]`         | **sin0.85°** | 중간 — 너무 빡빡하지 않게    |
| `lateral_step_lead.weight`  |     **0.30** | t2 — 옆걸음 leading-leg lift |
| `min_first_swing_air_time`  |     **0.30** | 중간 — 잔발걸음 적당히       |

---

## 부록 D — t4 forward-only (직진 정밀 학습)

### D.1 t3 학습 후 관측 문제

전진 보행 시:
- hip_roll 사용하여 약간 knock-kneed gait
- hip_yaw 사용하여 약간 pigeon-toed (발끝 안쪽)

원인: `pose.std_walking[hip_roll]=0.15`, `[hip_yaw]=0.30`이 다리 옆쪽 자유도를 허용. multi-direction 학습엔 필요했지만, 직진 정밀 학습엔 noise.

### D.2 HI 논문의 사람 forward 보행 통계

- hip_pitch L/R: 강한 anti-phase (큰 swing ±0.4 rad)
- hip_roll L/R: 작은 in-phase swing (~3-5° max)
- hip_yaw: **거의 0** (직진 시)
- ankle_roll: **거의 0**

→ 사람은 직진 시 hip_roll·hip_yaw·ankle_roll 거의 안 씀. 우리도 그렇게 강제하면 됨.

### D.3 t4 변경 (forward-only 전용)

| 항목                            |         t3 |     **t4** | 효과                          |
| ------------------------------- | ---------: | ---------: | ----------------------------- |
| `cmd_vel` y 범위 (모든 stage)   | (-0.5~1.0) | **(0, 0)** | y 명령 차단                   |
| `cmd_vel` ω_z 범위 (모든 stage) | (-0.5~2.0) | **(0, 0)** | yaw 명령 차단                 |
| `pose.std_walking[hip_roll]`    |       0.15 |   **0.08** | ±4.6° 자유 — knock-knee 차단  |
| `pose.std_walking[hip_yaw]`     |       0.30 |   **0.08** | ±4.6° 자유 — pigeon-toed 차단 |
| `pose.std_walking[ankle_roll]`  |       0.10 |   **0.05** | ±2.9° 자유 — foot edge 차단   |
| `bilateral_symmetry.weight`     |        0.5 |    **0.8** | forward 대칭 핵심 신호 격상   |
| `lateral_step_lead.weight`      |        0.3 |    **0.0** | 비활성 (y 명령 없음)          |

### D.4 설계 근거 매핑

| t4 변경                    | 참고                                                                                   |
| -------------------------- | -------------------------------------------------------------------------------------- |
| `cmd_vel` forward only     | HI 논문의 "stage 1: 안정 standing + 느린 걷기"에 해당. 다방향 학습 전에 직진부터       |
| `pose` 옆쪽 자유도 빡빡    | HI 논문 사람 forward 보행 측정치 (hip_yaw ≈ 0)                                         |
| `bilateral_symmetry` 강화  | SKATER: bilateral symmetry만 강화해도 periodic gait emerge. forward는 좌우 대칭이 정답 |
| `lateral_step_lead` 비활성 | Statistical(H1): 불필요 신호 제거. y cmd 없으니 자연스레 0이지만 명시적 0 weight       |

### D.5 예상 결과 + 모니터링

**성공 신호**:
- `Episode_Reward/pose > 3.5` (3.0 이상 — hip_roll/yaw 안 쓰니 reward 챙기기 쉬워짐)
- `Episode_Reward/bilateral_symmetry > 0.5` (현재 0.15, weight 0.8 + forward 전용으로 빠르게 상승)
- `Episode_Reward/track_linear_velocity > 4.0` (forward 정밀도)
- 시각: 발끝 정면, 무릎 모이지 않음, 평행 보행

**실패 신호 및 다음 조치**:
- 학습 자체 안 됨 → pose 너무 빡빡, hip_roll/yaw std 0.10으로 완화
- 보행 너무 뻣뻣 → bilateral_symmetry weight 0.8 → 0.6으로 조정
- forward 안 됨 → track_lin_vel std 더 빡빡하게 (0.18)

### D.6 다음 단계 (forward 완성 후)

forward-only 학습이 만족스러우면:
1. 그 run의 `source/` 폴더를 **forward foundation**으로 백업
2. cmd_vel curriculum에 y/yaw 단계적 활성 추가
3. `lateral_step_lead.weight` 0.3 복원
4. `pose[hip_yaw]` 0.08 → 0.30 (회전 자유 회복)
5. `pose[hip_roll]` 0.08 → 0.15 (옆걸음 시 약간 자유)

이렇게 **계단식 확장**으로 forward 능력 유지하면서 multi-direction 추가.

---

## 자동 기록 시스템 (2026-05-12 추가)

매 학습 시작 시 `log_dir/`에 자동 저장:

| 파일                | 내용                                               |
| ------------------- | -------------------------------------------------- |
| `params/env.yaml`   | 전체 env config (mjlab 기본)                       |
| `params/agent.yaml` | RL config (mjlab 기본)                             |
| `git/mjlab.diff`    | commit + uncommitted diff (rsl-rl 기본)            |
| `source/`           | reward 관련 13개 .py snapshot (NEW)                |
| `summary.md`        | reward·curriculum·RL config 사람 읽기용 요약 (NEW) |

도구:
- `scripts/check_rewards.py` — 학습 중 reward 모니터링
- `scripts/compare_runs.py` — 두 run env/source 비교
- `scripts/rollback_run.py` — 특정 run의 source 복원

→ 모든 run의 reward 설정·코드 상태가 영구 기록되므로, 어떤 reward로 어떤 결과 나왔는지 추적·롤백 가능.
