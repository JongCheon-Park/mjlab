# P1 MoE — KIMM Beom Reward Stack Adoption

P1 휴머노이드 Mjlab task의 보상함수를 `20260511_beom` 내부 fork의 P1/P2급으로
업그레이드하면서, actor 네트워크는 mjlab의 `MoEActorModel`(rule-based hard
routing MoE)로 유지하기 위한 통합 작업 정리.

## 1. 동기

P1 baseline 학습이 정체된 원인을 분석한 결과 두 가지 문제 확인:

1. **action scale 부족** (hip_pitch 0.073 rad — g1 대비 7.5× 작음) → 정책이 다리를
   walking에 필요한 만큼 휘두를 수 없어 standing-still에 수렴.
2. **reward 정의 부족** — 기본 mjlab의 12개 reward로는 휴머노이드 보행 학습이
   어렵다. 비교 대상인 P2 task는 33+ reward, biped-specific shaping, curriculum,
   domain randomization을 갖춤.

`20260511_beom` 폴더(KIMM 내부 fork 스냅샷)에 풀세트가 있어서, 그쪽 보상 인프라를
mjlab core 코드에 통합 (Option C). actor 네트워크는 mjlab의 `MoEActorModel`을
계속 쓰기 위해 beom의 `MoEMLPModel`(rsl-rl fork 의존)은 사용하지 않음.

## 2. 폴더 변경 요약

```
mjlab/
├── 20260511_beom/                # KIMM fork snapshot (read-only ref)
└── src/mjlab/
    ├── actuator/__init__.py            # ← Fourbar* stub 추가
    ├── envs/mdp/                       # ← beom 전체로 교체
    │   ├── actions/actions.py          #   JointPositionHoldActionCfg 추가
    │   ├── curriculums.py              #   reward_curriculum 등
    │   ├── dr/                         #   pseudo_inertia / motor_bandwidth / RFI
    │   ├── events.py
    │   └── __init__.py
    ├── tasks/velocity/
    │   ├── mdp/                        # ← beom 전체로 교체
    │   │   ├── rewards.py              #   ~25개 추가 reward 함수
    │   │   ├── curriculums.py          #   reward_weight, reward_param
    │   │   ├── observations.py
    │   │   ├── drag_wrench.py
    │   │   └── ...
    │   ├── velocity_env_cfg.py         # ← beom 버전 (33 reward, 13 event)
    │   └── config/
    │       ├── g1/env_cfgs.py          #   defensive guards (event/obs 부재 시 skip)
    │       ├── go1/env_cfgs.py         #   동일
    │       └── p1/                     # ← beom의 P1 cfg 적용 + MoE 매핑
    │           ├── env_cfgs.py
    │           ├── rl_cfg.py
    │           ├── symmetry.py
    │           └── __init__.py
    └── rl/moe_model.py                 # 기존 MoEActorModel 유지
```

## 3. 등록된 task

| Task ID                            | Actor   | Joints 제어  | 용도                            |
| ---------------------------------- | ------- | ------------ | ------------------------------- |
| `Mjlab-Velocity-Flat-P1`           | MLP     | legs only    | baseline                        |
| `Mjlab-Velocity-Flat-P1-MoE`       | **MoE** | legs only    | MoE 비교 대상 (본 문서 주 대상) |
| `Mjlab-Velocity-Flat-P1-Wholebody` | MLP     | all (27 DoF) | whole-body (실험용)             |

`legs only` 모드에서 arms·waist는 `JointPositionHoldActionCfg`로 default
포즈에 PD-hold. 보행 학습이 안정될 때까지 상체 자유도를 잠가두는 전략.

## 4. P1-MoE 환경 사양

런타임 측정값.

| 항목                          | 값                                               |
| ----------------------------- | ------------------------------------------------ |
| Actor obs dim                 | **235**                                          |
| Critic obs dim                | **310**                                          |
| Num actions                   | **12** (legs only)                               |
| Sim timestep                  | 0.005s                                           |
| Decimation                    | 4 (env-step = 20ms)                              |
| Episode length                | 20.0s                                            |
| `num_steps_per_env` (rollout) | 24                                               |
| `max_iterations`              | 30,000                                           |
| `save_interval`               | 500                                              |
| Experiment name               | `p1_velocity_moe` (로그 분리)                    |
| Cmd 초기 범위                 | v_x·v_y ∈ ±0.5, ω_z ∈ ±0.5 (curriculum으로 확장) |

## 5. Reward terms (전체 34개)

### 5.1 Tracking & posture (메인 신호)

| Reward                   |   Weight | 함수                     | 설명                                                                        |
| ------------------------ | -------: | ------------------------ | --------------------------------------------------------------------------- |
| `track_linear_velocity`  | **+5.0** | `track_linear_velocity`  | 명령 v_x, v_y 추종 (speed-dep std 곡선)                                     |
| `track_angular_velocity` | **+2.5** | `track_angular_velocity` | 명령 ω_z 추종, low-speed std는 curriculum                                   |
| `upright`                |     +1.0 | `flat_orientation_multi` | torso 똑바로, `weight_xy=(1,5)`, `std_xy=(sin10°, sin1°)` 로 측면이 더 빡빡 |
| `base_height`            |     +1.0 | `base_height_l2`         | target_height=0.84m                                                         |
| `pose`                   |     +2.0 | `variable_posture`       | 4-mode std (standing/turning/walking/running), `weight_standing=5.0`        |

### 5.2 Smoothness & limits (negative shaping)

| Reward               |   Weight | 함수                            | 설명                                                        |
| -------------------- | -------: | ------------------------------- | ----------------------------------------------------------- |
| `body_ang_vel`       |    -0.05 | `body_angular_velocity_penalty` | torso 각속도 페널티                                         |
| `angular_momentum`   |    -0.05 | `angular_momentum_penalty`      | 전체 각운동량 페널티                                        |
| `dof_pos_limits`     | **-5.0** | `joint_pos_limits`              | soft joint limit 위반 페널티 (강함)                         |
| `action_rate_l2`     |     -0.1 | `action_rate_l2`                | action 점프 페널티                                          |
| `joint_effort_limit` |    -0.01 | `joint_effort_limits`           | soft_ratio=0.7, power=2.0; curriculum으로 -1e-3→-2e-3→-4e-3 |
| `joint_torque_rate`  |    -0.05 | `joint_torque_rate_l2`          | 토크 미분 페널티                                            |
| `hip_roll_torque_l2` |    -1e-4 | `joint_torques_l2`              | hip_roll만 특별히 토크 L2 페널티 (좌우 흔들림 억제)         |
| `self_collisions`    | **-2.0** | `self_collision_cost`           | force_threshold=10N 이상 self-contact 페널티                |
| `soft_landing`       |   -0.001 | `soft_landing`                  | 발 착지 충격 페널티                                         |

### 5.3 Foot patterns

| Reward              |   Weight | 함수                    | 설명                                  |
| ------------------- | -------: | ----------------------- | ------------------------------------- |
| `foot_clearance`    | **-5.0** | `feet_clearance`        | target_height=0.10m, swing 중 발 높이 |
| `foot_swing_height` | **-5.0** | `feet_swing_height`     | target_height=0.10m                   |
| `foot_slip`         |     -0.1 | `feet_slip`             | 접지 중 발 미끄러짐 페널티            |
| `foot_flat`         |     +0.1 | `feet_orientation_flat` | target_height=0.05m, 발바닥 수평 유지 |

### 5.4 Biped gait shaping (양발 보행 전용)

| Reward                      |         Weight | 함수                        | 설명                                                                          |
| --------------------------- | -------------: | --------------------------- | ----------------------------------------------------------------------------- |
| `biped_standing_stability`  |           +1.0 | `biped_standing_stability`  | standing 명령 시 안정                                                         |
| `biped_air_time`            | **curriculum** | `biped_air_time`            | target=0.4s. weight stages: 0 → 20 (12k step) → 15 (18k step) → 10 (24k step) |
| `biped_first_swing_foot`    | **curriculum** | `biped_first_swing_foot`    | weight 0 → 2.0 at 12k step                                                    |
| `biped_double_support_time` | **curriculum** | `biped_double_support_time` | target=0.1s. weight 0 → 1.0 at 24k step                                       |

### 5.5 Gait-phase patterns (heel-strike / toe-off)

발의 phase에 맞춰 hip/knee/ankle 각도가 의도된 패턴을 따르도록 유도. 각 reward는
`axis_signs`로 좌우 대칭 부호 처리.

| Reward                              | Weight | Target body/joint    | Axis signs                 |
| ----------------------------------- | -----: | -------------------- | -------------------------- |
| `heelstrike_foot_pitch_pattern`     |   +0.1 | `.*ankle_roll.*`     | —                          |
| `toeoff_foot_pitch_pattern`         |   +0.1 | `.*ankle_roll.*`     | —                          |
| `heelstrike_knee_pattern`           |   +0.1 | `.*_knee_joint`      | L/R = +1 / +1              |
| `toeoff_knee_pattern`               |   +0.1 | `.*_knee_joint`      | L/R = +1 / +1              |
| `heelstrike_hip_pitch_pattern`      |   +0.1 | `.*_hip_pitch_joint` | L/R = +1 / -1              |
| `toeoff_hip_pitch_pattern`          |   +0.1 | `.*_hip_pitch_joint` | L/R = +1 / -1              |
| `heelstrike_hip_roll_pattern`       |   +0.1 | `.*_hip_roll_joint`  | L/R = +1 / -1              |
| `toeoff_hip_roll_pattern`           |   +0.1 | `.*_hip_roll_joint`  | L/R = +1 / -1, target=2.5° |
| `heelstrike_shoulder_pitch_pattern` |    0.0 | (arms off)           | —                          |
| `toeoff_shoulder_pitch_pattern`     |    0.0 | (arms off)           | —                          |
| `toeoff_elbow_pattern`              |    0.0 | (arms off)           | —                          |

(상체 reward는 weight=0 — legs-only 학습이라 비활성. whole-body 학습에선 활성됨.)

### 5.6 Stride shaping

| Reward               | Weight | 함수                 | 설명                                                     |
| -------------------- | -----: | -------------------- | -------------------------------------------------------- |
| `thigh_swing_target` |   +0.5 | `thigh_swing_target` | 대퇴(hip_roll_link) 좌우 스윙 패턴, axis_signs L/R=+1/-1 |

## 6. Pose 모드별 std (`variable_posture`)

속도/회전 조건에 따라 다른 std로 자세 페널티를 적용. 작은 std = 더 빡빡.

| Joint           | standing | turning | walking | running |
| --------------- | -------: | ------: | ------: | ------: |
| `*hip_pitch*`   |     0.15 |    0.40 |    0.30 |    0.50 |
| `*hip_roll*`    |     0.05 |    0.20 |    0.15 |    0.20 |
| `*hip_yaw*`     |     0.05 |    0.60 |    0.15 |    0.20 |
| `*knee*`        |     0.20 |    0.60 |    0.60 |    0.80 |
| `*ankle_pitch*` |     0.10 |    0.25 |    0.25 |    0.35 |
| `*ankle_roll*`  |     0.10 |    0.10 |    0.10 |    0.15 |

(arms·waist 항목은 legs-only 학습이라 `_filter_posture_std`로 자동 제거됨.)

`weight_standing=5.0` → standing 모드일 때 pose 가중치 추가로 5× 강화.

## 7. Domain Randomization (13 events)

| Event                     | Type     | Notes                                                |
| ------------------------- | -------- | ---------------------------------------------------- |
| `reset_base`              | reset    | 기본                                                 |
| `reset_robot_joints`      | reset    | 기본                                                 |
| `push_robot`              | interval | 외란 push                                            |
| `foot_friction`           | startup  | 마찰 randomize                                       |
| `encoder_bias`            | reset    | 관절 인코더 bias                                     |
| `encoder_bias_ankle_roll` | reset    | 발목 roll 특별 처리                                  |
| `torso_pseudo_inertia`    | startup  | torso_link mass/inertia DR, alpha=(0.1, 0.13)        |
| `link_pseudo_inertia`     | startup  | hip_yaw / knee / ankle_roll link DR, alpha=(0, 0.05) |
| `joint_armature`          | startup  | ankle 제외 (fourbar용 별도)                          |
| `joint_friction`          | startup  | 관절 마찰                                            |
| `joint_damping`           | startup  | 관절 댐핑                                            |
| `actuator_rfi`            | startup  | Rotor Friction Injection, actuator_ids=[0,1,2]       |
| `fourbar_motor_armature`  | startup  | 4-bar 발목 motor armature (P1 stub 적용)             |

또한 actuator delay (`delay_min_lag=0, delay_max_lag=4, hold_prob=0.5,
update_period=20`)가 actuator_ids=[0,1,2]에 자동 적용.

## 8. Curriculum (7 항목)

| Curriculum                               | 동작                                           |
| ---------------------------------------- | ---------------------------------------------- |
| `command_vel`                            | cmd 범위 단계적 확장 — 아래 표                 |
| `biped_air_time_weight`                  | air_time reward weight: 0 → 20 → 15 → 10       |
| `biped_air_time_post_landing_mask_steps` | post-landing mask 단계적 활성 (저속/고속 분리) |
| `biped_double_support_time_weight`       | 0 → 1.0 (24k step)                             |
| `biped_first_swing_foot_weight`          | 0 → 2.0 (12k step)                             |
| `joint_effort_limit_weight`              | -1e-3 → -2e-3 → -4e-3                          |
| `track_angular_velocity_low_std`         | 0.707 → 0.5 (120k step)                        |
| `track_linear_velocity_std`              | 0.548 (고정, curriculum hook만 등록)           |

### Command 범위 단계별 확장 (`command_vel`)

|    Step | v_x 범위 | v_y 범위 | ω_z 범위 |
| ------: | -------- | -------- | -------- |
|       0 | ±0.50    | ±0.50    | ±0.50    |
|  60,000 | ±0.75    | ±0.75    | ±1.00    |
| 120,000 | ±1.00    | ±1.00    | ±1.50    |
| 240,000 | ±1.25    | ±1.00    | ±1.50    |
| 480,000 | ±1.25    | ±1.00    | ±2.00    |

(step = env-step 수. 4096 env × 24 num_steps_per_env = 98,304 env-step / iter이라
60k step ≈ iter 610 부근.)

## 9. Terminations

| Termination             | 조건                                |
| ----------------------- | ----------------------------------- |
| `time_out`              | episode_length_s (20s) 초과         |
| `fell_over`             | torso projected gravity 임계치 위반 |
| `base_height_low`       | base 높이가 너무 낮음               |
| `nan_detected`          | obs/reward에 NaN                    |
| `out_of_terrain_bounds` | 지형 경계 밖 (flat에선 비활성)      |

## 10. Actions

| Action           | Type                         | 대상 관절                                                     |
| ---------------- | ---------------------------- | ------------------------------------------------------------- |
| `joint_pos`      | `JointPositionActionCfg`     | leg 12 DoF (hip pitch/roll/yaw, knee, ankle pitch/roll × L/R) |
| `joint_pos_hold` | `JointPositionHoldActionCfg` | waist_yaw + arms 14 DoF (default 포즈에서 PD-hold)            |

학습 가능한 action 차원은 **12**. arms·waist는 정책이 출력하지 않음.

## 11. Actor / Critic 네트워크

### MoE actor (`MoEActorModel`)

```
obs (235-dim, normalized) ─┬──→ shared MLP ─────┐
                            │  (256, 128)         cat ──→ head ──→ μ (12)
                            └──→ expert_k ───────┘  (128,)
                                 (192, 128)
                                 k ∈ {0=vx, 1=vy, 2=yaw}
```

- Routing rule: `cmd = obs_norm[..., -3:]`, `expert_idx = cmd.abs().argmax(dim=-1)`
- 선택되지 않은 expert는 one-hot mask로 0이 되어 head 입력에서 제거
- `cmd_start=-3` (actor obs 마지막 3 dim이 cmd: lin_vel_x, lin_vel_y, ang_vel_z)
- σ는 baseline과 동일한 scalar Gaussian

### Critic (baseline 그대로)

| 항목        | 값                                |
| ----------- | --------------------------------- |
| Type        | MLP                               |
| Hidden dims | (512, 256, 128)                   |
| Input dim   | 310 (critic obs, privileged 포함) |
| Output      | 1 (V)                             |

### PPO (둘 다 동일)

| Param                 | Value           |
| --------------------- | --------------- |
| `clip_param`          | 0.2             |
| `entropy_coef`        | 0.01            |
| `num_learning_epochs` | 5               |
| `num_mini_batches`    | 4               |
| `learning_rate`       | 1e-3 (adaptive) |
| `gamma`               | 0.99            |
| `lam`                 | 0.95            |
| `desired_kl`          | 0.01            |
| `max_grad_norm`       | 1.0             |
| `value_loss_coef`     | 1.0             |

## 12. Baseline vs MoE 동일성

```
diff /tmp/env_Mjlab-Velocity-Flat-P1.yaml /tmp/env_Mjlab-Velocity-Flat-P1-MoE.yaml
→ no differences
```

env config (reward, observation, command, event, termination, scene, curriculum)
바이트단위 동일. **유일한 차이는 actor 네트워크** (MLP vs MoE) + `experiment_name`
(로그 분리).

## 13. mjlab 본체 변경 (g1 호환성)

beom의 base velocity_env_cfg는 `base_com`, `air_time`, `height_scan` 등을 다른
이름으로 대체하거나 제거. g1·go1 config가 죽지 않도록 방어 코드 추가:

```python
# config/g1/env_cfgs.py, config/go1/env_cfgs.py
if "base_com" in cfg.events:
    cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
if "torso_pseudo_inertia" in cfg.events:
    cfg.events["torso_pseudo_inertia"].params["asset_cfg"].body_names = ("torso_link",)

if "air_time" in cfg.rewards:
    cfg.rewards["air_time"].weight = 0.0

for group in ("actor", "critic"):
    cfg.observations[group].terms.pop("height_scan", None)
```

## 14. Fourbar actuator stub

beom DR 모듈이 `mjlab.actuator.FourbarPdActuator` 등에 의존하지만 mjlab core에
없음. `mjlab/actuator/__init__.py`에 isinstance-only stub 추가:

```python
class FourbarPdActuator:
  motor_armature: object
  default_motor_armature: object
  force_limit: object
  last_motor_torque: object
  def set_motor_armature(self, *args, **kwargs) -> None: ...

class FourbarPdActuatorCfg: pass
class FourbarAnkleGroupCfg: pass
```

P1은 실제로는 `BuiltinPositionActuatorCfg` (또는 `IdealPdActuatorCfg`)를 쓰므로
`isinstance(actuator, FourbarPdActuator)` 분기는 절대 True가 되지 않음 → DR
로직은 안전하게 dormant.

## 15. 실행

### 학습
```sh
# baseline
CUDA_VISIBLE_DEVICES=1 uv run train Mjlab-Velocity-Flat-P1 \
  --env.scene.num-envs 4096

# MoE (본 비교 대상)
CUDA_VISIBLE_DEVICES=1 uv run train Mjlab-Velocity-Flat-P1-MoE \
  --env.scene.num-envs 4096
```

`logs/rsl_rl/p1_velocity/...` vs `logs/rsl_rl/p1_velocity_moe/...` 로 분리 저장.

### 시각 확인
```sh
# 가장 최근 체크포인트 자동 로드
uv run play Mjlab-Velocity-Flat-P1-MoE --viewer viser

# 특정 체크포인트
uv run play Mjlab-Velocity-Flat-P1-MoE \
  --checkpoint-file logs/rsl_rl/p1_velocity_moe/<timestamp>/model_N.pt \
  --viewer viser
```

### 비교 시 모니터링할 메트릭

| 메트릭                                        | 의미                                       | 기대 추이                           |
| --------------------------------------------- | ------------------------------------------ | ----------------------------------- |
| `Mean reward`                                 | 총 reward                                  | 상승, 수만 step 이후 양수           |
| `Mean episode length`                         | episode 길이                               | 800+ (timeout=time_out 우세)        |
| `Episode_Reward/track_linear_velocity`        | x/y 추종                                   | ↑                                   |
| `Episode_Reward/track_angular_velocity`       | yaw 추종                                   | ↑                                   |
| `Episode_Reward/pose`                         | 자세 페널티 (양수 reward는 -error 형태)    | ↑                                   |
| `Episode_Reward/biped_air_time`               | 발 swing 시간                              | curriculum 12k step 이후 큰 양수    |
| `Episode_Reward/action_rate_l2`               | action 점프                                | MoE는 routing 경계에서 더 음수 가능 |
| `Episode_Termination/time_out` vs `fell_over` | timeout이 압도해야 함                      |                                     |
| iter당 wall-time                              | MoE는 ~1.3× 느림 (expert 3개 항상 forward) |                                     |

## 16. 알려진 한계 / 후속 작업

1. **4-bar 발목 모델 없음**: stub만 있고 실제 4-bar 링키지 구현 X. mjlab core에
   `FourbarPdActuator` 추가하고 `kimm_p1_constants.py` (wo_4bar 아닌 버전)로 전환
   필요.
2. **mirror symmetry 미적용**: beom에 `MirrorFn`이 있지만 현재 cfg에선 주석처리.
   양다리 대칭으로 sample efficiency 올리려면 활성화.
3. **MoE는 mjlab 자체 구현** — beom의 `MoEMLPModel`(cmd_regime routing + DINO
   EMA teacher + GRL regime invariance)은 rsl-rl fork 의존이라 도입 안 함. 더
   강력한 MoE 원하면 beom rsl-rl 빌드 필요.
4. **placeholder 모터 PD gain**: `FN_HZ=5.0, ZETA=0.7`로 계산된 stiffness/damping.
   실 P1 데이터시트와 차이 있을 수 있음.
5. **action scale은 자동 PD 계산값 그대로 사용** — beom flow 따라감. 만약 다리가
   여전히 못 움직이면 `joint_pos_action.scale`을 직접 override 필요.
