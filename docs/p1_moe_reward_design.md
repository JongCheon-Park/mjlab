# P1-MoE 보상함수 설계 분석

`Mjlab-Velocity-Flat-P1-MoE` task의 reward·curriculum·event 전체 구성과 **각 항목의 설계 의도**를 정리. 진단·튜닝 이력 포함.

---

## 0. 한눈에 보는 설계 철학

P1 휴머노이드를 평지에서 velocity-tracking 보행시키는 task. 보상함수는 **5개의 layer**로 구성:

1. **Tracking signal**: 명령 추종 (linear/angular velocity)
2. **Posture / stability**: 똑바로 서있기, 일정한 base 높이, 자세 유지
3. **Smoothness / safety**: action 점프·관절 한계·자가 충돌 페널티
4. **Gait shaping**: 보행 phase별 발/관절 패턴 reward (heel-strike, toe-off)
5. **Biped curriculum**: standing → walking → 더 빠른 walking 으로 단계적 유도

**핵심 원리**: 단순 tracking reward만 주면 robot이 "그냥 서있고 부분 점수만 챙기는" local minimum에 빠진다 (실제로 첫 학습에서 관측됨). 따라서 **walking을 명시적으로 보상**하고 **standing을 명시적으로 약화**해야 함.

---

## 1. 환경 사양

| 항목                 | 값                                                                                  |
| -------------------- | ----------------------------------------------------------------------------------- |
| Sim timestep         | 0.005s                                                                              |
| Decimation           | 4 (control dt = 20ms = 50Hz)                                                        |
| Episode length       | 20s (= 1000 control steps)                                                          |
| Actor obs dim        | 235                                                                                 |
| Critic obs dim       | 310                                                                                 |
| Num actions          | 12 (legs only — 6 DoF × 좌우)                                                       |
| Arms/waist           | `JointPositionHoldActionCfg`로 default 자세 PD-hold                                 |
| Episode terminations | `time_out`, `fell_over`, `base_height_low`, `nan_detected`, `out_of_terrain_bounds` |
| Command resampling   | 매 episode 내 (3.0, 8.0)s 간격으로 cmd 재추첨                                       |
| `rel_standing_envs`  | 0.1 (10% 확률로 cmd=0)                                                              |

cmd 분포: `lin_vel_x`, `lin_vel_y`, `ang_vel_z` 각 축에 대해 step별로 curriculum 단계 따라 점진적 확장 (§7 참고).

---

## 2. RL config (PPO + MoE actor)

| 항목                | 값                                   |
| ------------------- | ------------------------------------ |
| Algorithm           | PPO (rsl-rl)                         |
| Actor               | `MoEActorModel` (shared + 3 experts) |
| Critic              | MLP (512, 256, 128)                  |
| `learning_rate`     | 1e-3 (adaptive)                      |
| `entropy_coef`      | 0.01                                 |
| `clip_param`        | 0.2                                  |
| `gamma`             | 0.99                                 |
| `lam`               | 0.95                                 |
| `desired_kl`        | 0.01                                 |
| `num_steps_per_env` | 24                                   |
| `max_iterations`    | 30,000                               |
| `save_interval`     | **50** (변경 — 이전 500)             |
| `experiment_name`   | `p1_velocity_moe`                    |

MoE actor 구조: shared MLP `(256, 128)` + 3 expert `(192, 128)` (vx / vy / yaw 담당) + head `(128,)`. rule-based hard routing: `cmd.abs().argmax(dim=-1)`. cmd가 0이면 expert 0 (vx) 활성. (cf. `docs/p1_moe_design.md` §11 참고.)

---

## 3. 보상함수 전체 카탈로그 (34개)

### 3.1 Tracking signal (메인 신호)

| Reward                   | Weight | std (현재) | 의도                                                                               |
| ------------------------ | -----: | ---------- | ---------------------------------------------------------------------------------- |
| `track_linear_velocity`  |   +5.0 | **0.20**   | 명령 (v_x, v_y) 추종. `exp(-‖err‖²/std²)` 형태. std가 작을수록 정확한 추종만 보상. |
| `track_angular_velocity` |   +2.5 | **0.32**   | 명령 ω_z 추종. low-speed에선 별도 `low_std` 사용 (저속에서도 빡빡).                |

**설계 의도**:
- 가중치 5:2.5 비율 = lin이 ang보다 2배 더 중요. 보행 task에서 직진/측면 이동이 회전보다 본 task의 메인 신호.
- **std는 의도적으로 빡빡하게** (0.20 / 0.32). 이전 run에서 std=0.55였을 때 cmd=0.5 명령에 robot이 정지해도 reward 0.44 (`exp(-0.5²/0.55²)`)를 챙기는 partial-credit exploit이 관측됨. std=0.2면 같은 상황에서 reward = `exp(-0.5²/0.2²) = 0.0019` → 거의 0이 되어서 정지 정책이 reward를 챙길 수 없음.

### 3.2 Posture / stability

| Reward                     |   Weight | 함수                       | 의도                                                                                    |
| -------------------------- | -------: | -------------------------- | --------------------------------------------------------------------------------------- |
| `upright`                  | **+0.5** | `flat_orientation_multi`   | torso 똑바로 유지. weight_xy=(1.0, 5.0), std_xy=(sin10°, sin1°). 측면 기울기에 더 엄격. |
| `base_height`              |     +1.0 | `base_height_l2`           | `target_height=0.84m` 유지. (P1 standing 높이.)                                         |
| `pose`                     |     +2.0 | `variable_posture`         | 자세별 std 곡선. cmd에 따라 standing/turning/walking/running 자동 전환.                 |
| `biped_standing_stability` |     +1.0 | `biped_standing_stability` | cmd≈0일 때 정적 안정성. cmd≠0이면 비활성.                                               |

**`pose` reward의 4-mode std (속도/회전에 따라 동적 전환)**:

| Joint pattern   | standing | turning | walking | running |
| --------------- | -------: | ------: | ------: | ------: |
| `*hip_pitch*`   |     0.15 |    0.40 |    0.30 |    0.50 |
| `*hip_roll*`    |     0.05 |    0.20 |    0.15 |    0.20 |
| `*hip_yaw*`     |     0.05 |    0.60 |    0.15 |    0.20 |
| `*knee*`        |     0.20 |    0.60 |    0.60 |    0.80 |
| `*ankle_pitch*` |     0.10 |    0.25 |    0.25 |    0.35 |
| `*ankle_roll*`  |     0.10 |    0.10 |    0.10 |    0.15 |

`weight_standing=5.0` → cmd=0일 때 pose weight를 5× 부스트하여 standing 자세를 더 강하게 끌어당김 (반대로 walking 모드에선 자세 자유도가 풀림).

**설계 의도**:
- `upright` weight 1.0 → **0.5로 축소** (튜닝됨): 1.0에서는 standing이 매우 reward-rich한 상태가 되어 robot이 movement를 안 함. 0.5로 줄여 walking 시 자세 변화도 허용.
- 측면(y-axis) std가 sin1° = 0.017로 매우 빡빡: 휴머노이드는 옆으로 기울면 빠르게 넘어지므로 **y 기울기는 엄격**, 전후(x)는 sin10° = 0.17로 풀어줘 보행 중 자연스러운 흔들림 허용.
- 4-mode std는 **명령에 맞는 자세 자유도 분배**: standing 시엔 작은 std (자세 강제), walking 시엔 큰 std (다리 swing 허용), running 시엔 더 큰 std.

### 3.3 Smoothness / safety penalties

| Reward               |    Weight | 의도                                                                |
| -------------------- | --------: | ------------------------------------------------------------------- |
| `body_ang_vel`       |     -0.05 | torso 각속도 페널티. 큰 회전 흔들림 방지.                           |
| `angular_momentum`   |     -0.05 | 전체 각운동량 페널티. 격렬한 회전 방지.                             |
| `dof_pos_limits`     |  **-1.0** | soft joint limit 위반 페널티. (튜닝됨 — 이전 -5.0)                  |
| `action_rate_l2`     | **-0.05** | action 점프 페널티. (튜닝됨 — 이전 -0.1)                            |
| `joint_effort_limit` |     -0.01 | soft effort limit (soft_ratio=0.7, power=2.0). 모터 토크 한계 보호. |
| `joint_torque_rate`  |     -0.05 | 토크 미분 페널티. 진동·고주파 명령 억제.                            |
| `hip_roll_torque_l2` |     -1e-4 | hip_roll 토크 L2. 좌우 흔들림 토크 사용량 억제 (특수 페널티).       |
| `self_collisions`    |      -2.0 | force_threshold=10N 이상 self-contact 페널티.                       |
| `soft_landing`       |    -0.001 | 발 착지 충격 페널티. 부드럽게 내리도록 유도.                        |

**설계 의도**:
- `dof_pos_limits` -5.0 → -1.0: 발 들기·다리 스윙은 본질적으로 관절을 한계까지 휘둘러야 함. -5.0이면 stride 시도할 때마다 큰 페널티 → robot이 발을 안 듬. -1.0으로 약화하여 stride 자유도 확보.
- `action_rate_l2` -0.1 → -0.05: 같은 이유. 보폭이 크려면 action이 step마다 변해야 함. -0.1이면 점프 페널티가 stride를 못 만들게 함.
- `hip_roll_torque_l2`: 좌우 흔들림 토크가 자주 sign-flip 하면 보행이 부자연스러움 → 이 특정 관절만 토크 L2 적용 (다른 관절은 토크 L2 안 함).
- `self_collisions` weight -2.0: 휴머노이드는 다리가 서로 부딪힐 위험이 큼. 강하게 페널티.
- `soft_landing` weight -0.001: 매우 약함. 충격 자체를 막기보단 "장기적으로 부드러운 착지를 유도" 정도. 강하면 발 들기 자체를 막음.

### 3.4 Foot patterns

| Reward              | Weight | 의도                                                                                 |
| ------------------- | -----: | ------------------------------------------------------------------------------------ |
| `foot_clearance`    |   -5.0 | swing phase에서 발이 `target_height=0.10m`보다 낮으면 페널티.                        |
| `foot_swing_height` |   -5.0 | swing 정점에서 `target_height=0.10m` 도달 못 하면 페널티.                            |
| `foot_slip`         |   -0.1 | 접지 중 발이 미끄러지면 페널티.                                                      |
| `foot_flat`         |   +0.1 | 접지/swing 시 발바닥 수평 유지하면 보상. `target_height=0.05m` (foot rotation 임계). |

**설계 의도**:
- foot_clearance / foot_swing_height는 **stride 형태를 만드는 핵심 페널티**. weight -5.0으로 강함: 발을 안 들면 큰 페널티가 누적되어 robot이 결국 발을 들도록 학습 유도.
- 단, 이 페널티는 cmd≠0일 때만 의미 있음 (cmd=0이면 발을 안 드는 게 정상). `command_threshold=0.05`로 small cmd 영역은 보호.
- foot_slip은 약함 (-0.1): 너무 강하면 접지 시 발이 미세하게 움직이는 것도 페널티가 되어 학습 방해. 정도껏만.
- foot_flat은 양수 보상: 발이 평평하게 닿아야 안정적 접지 → 보상으로 유도.

### 3.5 Gait-phase patterns (8개 + 3개 비활성)

| Reward                              | Weight | 활성? | 대상 joint                                         |
| ----------------------------------- | -----: | :---: | -------------------------------------------------- |
| `heelstrike_foot_pitch_pattern`     |   +0.1 |   ✓   | `*_ankle_roll_link` (foot rotation at heel-strike) |
| `toeoff_foot_pitch_pattern`         |   +0.1 |   ✓   | 동일 (at toe-off)                                  |
| `heelstrike_knee_pattern`           |   +0.1 |   ✓   | `*_knee_joint`, axis_signs L/R = +1/+1             |
| `toeoff_knee_pattern`               |   +0.1 |   ✓   | 동일                                               |
| `heelstrike_hip_pitch_pattern`      |   +0.1 |   ✓   | `*_hip_pitch_joint`, axis_signs L/R = +1/-1        |
| `toeoff_hip_pitch_pattern`          |   +0.1 |   ✓   | 동일                                               |
| `heelstrike_hip_roll_pattern`       |   +0.1 |   ✓   | `*_hip_roll_joint`, axis_signs L/R = +1/-1         |
| `toeoff_hip_roll_pattern`           |   +0.1 |   ✓   | 동일, `target_angle_deg=2.5`                       |
| `heelstrike_shoulder_pitch_pattern` |    0.0 |   ✗   | arms 비활성 (legs-only 학습)                       |
| `toeoff_shoulder_pitch_pattern`     |    0.0 |   ✗   | 동일                                               |
| `toeoff_elbow_pattern`              |    0.0 |   ✗   | 동일                                               |

**설계 의도**:
- gait pattern reward는 **contact event-gated** — 발이 lift-off 하거나 touch-down 하는 순간에만 트리거. 그러므로 robot이 발을 들지 않으면 신호 자체가 안 나옴 (관측된 dormancy).
- weight 0.1로 약하게 둠 (총 8 × 0.1 = 0.8 per-second 잠재력): 메인 신호가 아니라 **보행이 시작된 뒤 자연스러운 gait를 다듬는 보조 신호**.
- 좌우 axis_signs로 대칭성 강제: 휴머노이드 보행은 좌우가 reversed 패턴 (오른발 toe-off ↔ 왼발 heel-strike). 부호 매핑으로 한 reward로 양쪽 처리.
- arms 시리즈는 legs-only 학습에서는 weight 0 (action도 없으므로 의미 없음). whole-body 변형에서 활성화.

### 3.6 Stride shaping

| Reward               |   Weight | 의도                                                                                          |
| -------------------- | -------: | --------------------------------------------------------------------------------------------- |
| `thigh_swing_target` | **+0.5** | `*_hip_roll_link` 좌우 스윙 패턴 (axis_signs L/R=+1/-1), `target_angle_deg=2.5`, `std≈0.044`. |

**설계 의도**: 대퇴(thigh)의 좌우 흔들림 폭을 명시적으로 안내. lateral motion (v_y) 중에 thigh를 충분히 흔드는 걸 보상. `std=0.044` 매우 빡빡 — 정확한 각도 도달해야 보상 받음.

### 3.7 Biped curriculum-driven rewards

| Reward                      |  Weight (현재) | 의도                                                                            |
| --------------------------- | -------------: | ------------------------------------------------------------------------------- |
| `biped_air_time`            | **curriculum** | 발 swing 시간 (target=0.5s)이 목표값에 가까울수록 보상. 보행의 핵심 신호.       |
| `biped_first_swing_foot`    | **curriculum** | 명령 받은 직후 첫 swing이 빨리 시작될수록 보상. `min_first_swing_air_time=0.2`. |
| `biped_double_support_time` | **curriculum** | 양발이 동시에 닿아 있는 시간 (target=0.1s). 너무 짧거나 길면 페널티.            |

**`biped_air_time` weight curriculum (튜닝됨)**:

| Step (env-step) |   Weight | 비고                                       |
| --------------: | -------: | ------------------------------------------ |
|               0 |      0.0 | 학습 초기 비활성                           |
|           6,000 | **30.0** | 적극 활성화 (이전 12k step에서 20.0이었음) |
|          24,000 |     30.0 | 30 유지 (이전엔 18k에서 15로 감소)         |
|          72,000 |     20.0 | 후반엔 약간 감소 (정교화)                  |

설계 의도:
- weight 30 = 1초에 최대 30 reward 가능 → episode 20초 × 30 = 600 reward의 큰 잠재력. **이게 standing-local-minimum을 부수는 결정타.**
- step 6,000에 빠르게 활성화: 이전 12k step은 너무 늦음. robot이 standing local minimum에 안정화되기 전에 walking 신호 주입.
- step 72k 이후 weight 20으로 감소: 발이 안정적으로 들리기 시작하면 reward 비중을 살짝 낮추어 다른 reward (gait pattern, foot_flat 등)가 보강할 여지 줌.

**`biped_first_swing_foot` params**:
- `wrong_swing_penalty=0.5` — 잘못된 발 swing 시작 시 페널티
- `timeout_steps=15` — cmd 들어온 뒤 15 step 내 swing 없으면 페널티
- `no_swing_penalty=1.0`
- `min_first_swing_air_time=0.2` — 첫 swing은 최소 0.2s 떠야 인정
- `load_transfer_weight=0.5` — 무게중심 전환을 같이 평가

**`biped_double_support_time_weight` curriculum**:

|   Step | Weight |
| -----: | -----: |
|      0 |    0.0 |
| 24,000 |    1.0 |

step 24k에 활성화 — 발이 들리기 시작한 뒤 양발 접지 시간을 다듬는 후행 신호.

### 3.8 다음 reward들의 비활성 상태 (`biped_first_swing_foot`)

현재 `biped_first_swing_foot.weight = 0` (curriculum dict에서 빠짐). 가능성:
- 의도된 비활성 (P1에선 아직 안 씀)
- 또는 미설정 (추후 추가 가능)

비활성이라 effectively 의미 없음. 추후 swing 시작 신호가 필요할 때 활성화.

---

## 4. Domain Randomization (13 events)

| Event                     | Mode     | 목적                                                   |
| ------------------------- | -------- | ------------------------------------------------------ |
| `reset_base`              | reset    | episode 시작 시 base 위치/자세 초기화                  |
| `reset_robot_joints`      | reset    | 관절 위치/속도 초기화 (KNEES_BENT_KEYFRAME 기준)       |
| `push_robot`              | interval | 외란 push (학습 robustness)                            |
| `foot_friction`           | startup  | 발 마찰계수 DR                                         |
| `encoder_bias`            | reset    | 관절 인코더 bias DR (sim2real)                         |
| `encoder_bias_ankle_roll` | reset    | ankle roll 특별 처리                                   |
| `torso_pseudo_inertia`    | startup  | torso 질량/관성 DR (alpha=(0.1, 0.13))                 |
| `link_pseudo_inertia`     | startup  | hip_yaw / knee / ankle_roll 관성 DR (alpha=(0, 0.05))  |
| `joint_armature`          | startup  | 관절 armature DR (ankle 제외 — fourbar용 별도)         |
| `joint_friction`          | startup  | 관절 마찰 DR                                           |
| `joint_damping`           | startup  | 관절 댐핑 DR                                           |
| `actuator_rfi`            | startup  | Rotor Friction Injection (actuator_ids=[0,1,2])        |
| `fourbar_motor_armature`  | startup  | 4-bar 발목 motor armature DR (현재 stub만, 활성 안 됨) |

**설계 의도**: sim2real gap 줄이기. 학습된 정책이 실 로봇에서도 동작하려면 시뮬레이션이 너무 깔끔하면 안 됨. 모터 마찰, 인코더 bias, 관성 추정 오차 등을 학습 중 무작위로 변화시켜 정책이 robust하게 만들어짐.

또한 actuator delay:
- `delay_min_lag=0, delay_max_lag=4` — 0~4 step 지연
- `delay_hold_prob=0.5` — 50% 확률로 지연 hold
- `delay_update_period=20` — 20 step마다 지연값 갱신

→ 실 로봇 컨트롤러 통신 지연 시뮬레이션.

---

## 5. Curriculum 학습 (7개)

### 5.1 Command velocity 범위 단계적 확장

| Step (env-step) | v_x 범위 | v_y 범위 | ω_z 범위 |
| --------------: | -------: | -------: | -------: |
|               0 |     ±0.5 |     ±0.5 |     ±0.5 |
|          60,000 |    ±0.75 |    ±0.75 |     ±1.0 |
|         120,000 |     ±1.0 |     ±1.0 |     ±1.5 |
|         240,000 |    ±1.25 |     ±1.0 |     ±1.5 |
|         480,000 |    ±1.25 |     ±1.0 |     ±2.0 |

설계 의도: **느린 cmd부터 시작하여 점진적으로 빠른 cmd로 확장**. 처음부터 ±2.0 같은 큰 명령을 주면 robot이 따라가지 못해 학습 불가. 학습 초기엔 작은 영역에서 안정적으로 학습한 뒤 점차 어려운 명령으로 확장.

step 480k = iter ~5,000 (4096 envs × 24 steps/iter / 24 step counter unit ≈ 4096 step/iter). 30k iter 학습 중 5천 iter까지 cmd 범위 확장 완료.

### 5.2 Track velocity std curriculum (튜닝됨)

| Curriculum                       |   step 0 | step 120,000 |
| -------------------------------- | -------: | -----------: |
| `track_linear_velocity_std/std`  | **0.20** |         0.16 |
| `track_angular_velocity_low_std` | **0.32** |         0.22 |

설계 의도: std를 초기부터 빡빡하게 (이전엔 0.55에서 시작했다가 standing 이용 exploit 됨). 보행 학습이 안정되면 더 빡빡하게 (0.16) 최종 정교화.

### 5.3 Biped reward weight curriculum

이미 §3.7에서 다룸. 핵심:
- `biped_air_time_weight`: 0 → 30 (step 6k) → 30 (step 24k) → 20 (step 72k)
- `biped_double_support_time_weight`: 0 → 1.0 (step 24k)

### 5.4 기타

- `biped_air_time_post_landing_mask_steps`: 착지 직후 N step은 air_time reward 마스킹 (post-landing transient 제거).
  - low_speed: 0 → 10 (step 60k)
  - high_speed: 0 → 5 (step 60k)
- `joint_effort_limit_weight`: 현재 stage 1개만 (-0.001 고정). 추후 단계적 강화 가능.

---

## 6. Anti-standing-local-minimum 튜닝 이력

### 6.1 첫 시도 (`2026-05-11_14-39-20`)

- `Mean reward = 193, ep_length = 1000 (timeout 100%), fell_over = 0`
- 표면적으로는 학습 양호.
- 실제로는 **robot이 발을 들지 않고 정지 상태에서 reward 챙김.**

증거:
- `biped_air_time = 0.09` per-sec (weight 10일 때 최대 10 가능) → 99% 미달
- `biped_first_swing_foot = 0` → 발이 단 한 번도 lift-off 안 함
- `track_linear_velocity = 3.57` 인데, std=0.55였음. cmd=0.5에 std=0.55면 `exp(-0.25/0.30) = 0.44` → standing해도 절반 reward 챙김.

문제 진단 (§7.1):
1. track std 너무 관대 → standing exploit 가능
2. biped_air_time 신호 너무 약함 (curriculum 마지막 weight 10) → walking 인센티브 부족
3. action_rate_l2 / dof_pos_limits 페널티 → stride 시도 시 추가 페널티

| Side                                     | Reward 합 (per-sec) |
| ---------------------------------------- | ------------------: |
| Standing-friendly (upright, pose, track) |             **~15** |
| Walking-bonus (biped_air_time 등)        |           **~0.11** |

→ 비율 **135:1**. 정책 입장에서 standing이 우월 전략.

### 6.2 적용된 튜닝

| 변경                                    | 값                              |
| --------------------------------------- | ------------------------------- |
| `upright.weight`                        | 1.0 → **0.5**                   |
| `dof_pos_limits.weight`                 | -5.0 → **-1.0**                 |
| `action_rate_l2.weight`                 | -0.1 → **-0.05**                |
| `track_linear_velocity_std` step 0      | 0.55 → **0.20**                 |
| `track_angular_velocity_low_std` step 0 | 0.71 → **0.32**                 |
| `biped_air_time_weight` curriculum      | (0→20→15→10) → **(0→30→30→20)** |
| `biped_air_time_weight` 첫 활성 step    | 12,000 → **6,000** (절반)       |

### 6.3 예상 재균형

| Side                                                                    | Reward 합 (튜닝 후 추정, per-sec) |
| ----------------------------------------------------------------------- | --------------------------------: |
| Standing-friendly (upright 0.5×2.7 + pose 3 + track 0.2 partial 거의 0) |                            **~6** |
| Walking-bonus (biped_air_time weight 30 → 잠재 3, 실제 도달 시)         |                            **~3** |

비율 135:1 → **~2:1** 정도. 정책 입장에서 walking이 reward 챙기는 길이 됨.

---

## 7. 모니터링 메트릭 — 성공/실패 신호

### 7.1 성공 신호 (학습 잘 풀리는 중)

- `Episode_Reward/biped_air_time` > 0.5 (per-sec): 발이 실제로 들리기 시작
- `Episode_Reward/biped_first_swing_foot` ≠ 0: 첫 swing이 트리거됨
- `Episode_Reward/track_linear_velocity` > 3.0: 실제 추종이 됨 (std가 빡빡한데도 reward 받는다는 건 실제 robot이 움직이고 있다는 뜻)
- `Episode_Termination/time_out` ≫ `fell_over`: 잘 서있음
- `Mean episode length` 800+: 안 넘어짐
- `Mean reward` 단조증가 (튜닝 후 시작 시점에 잠시 감소 후 다시 증가)

### 7.2 실패 신호 (튜닝 더 필요)

- `biped_air_time` < 0.3 이라도 정체: walking 신호가 여전히 약함 → weight 50+ 시도
- `track_linear_velocity` 정체 (0.5 미만): std 0.2도 빡빡함 → 0.3으로 약간 푸는 게 나음
- `fell_over` 폭증: 페널티 너무 약함 → dof_pos_limits를 -1.5로 약간 강화
- `dof_pos_limits` 페널티 폭증 (-5 이하): 관절 자주 한계 닿음 → joint range 자체를 늘리거나 keyframe 자세 조정
- `Mean reward` 음수에 머무름: exploration 실패 → `entropy_coef` 0.01 → 0.02

---

## 8. 보상함수 layer 간 상호작용 (설계 핵심)

각 layer는 서로 trade-off 관계:

```
Tracking signal          ← 학습의 큰 목표
   │
   ↑ 강화하면              ↓ 약화하면
   stride 동기 부여        standing 이용 가능

Posture / stability      ← 자세 유지
   │
   ↑ 강화하면              ↓ 약화하면
   안정적 but standing 편향  walking에 자유도

Smoothness penalties    ← action 점프 억제
   │
   ↑ 강화하면              ↓ 약화하면
   부드러운 but standing  stride 시도 자유 but 진동

Gait shaping            ← phase 패턴 가이드
   │
   ↑ 강화하면              ↓ 약화하면
   특정 gait 강제          자유로운 gait

Biped curriculum        ← walking 적극 보상
   │
   ↑ 강화하면              ↓ 약화하면
   walking 강요          standing 회피 가능
```

**튜닝 핵심**: 4개 노브의 균형:

1. **Tracking std**: 작을수록 standing exploit 어려움 (현재 0.20 — 빡빡)
2. **Biped air_time weight**: 클수록 walking 보상 큼 (현재 curriculum 30 — 강함)
3. **Smoothness penalty weights**: 작을수록 stride 시도 자유 (현재 dof_pos_limits=-1.0, action_rate=-0.05 — 풀어줌)
4. **Posture weight**: 작을수록 standing 매력 적음 (현재 upright=0.5 — 축소)

이전 첫 시도에서 1·2·3·4 모두 standing-favorable한 방향이어서 학습 실패. 현재는 모두 walking-favorable한 방향으로 조정.

---

## 9. 향후 개선 여지

1. **`biped_first_swing_foot` weight curriculum 추가**: 현재 weight=0. step 6k부터 weight 2.0 정도로 활성화하면 swing 시작 자체에 직접 보상 가능.
2. **standing 페널티 추가**: `walking_threshold`(0.5) 이상 cmd인데 robot이 안 움직이면 명시적 페널티. 현재는 페널티가 없고 reward를 못 받는 것뿐.
3. **`track_linear_velocity` exp → quadratic 전환 검토**: exp는 partial credit이 풍부함 (긴 꼬리). quadratic은 sharp drop-off → standing exploit 더 어려움.
4. **`foot_clearance` target_height 조정**: 현재 0.10m. 더 높이면 발 들기 강제 (0.15m 시도).
5. **MoE routing에 standing expert 추가**: 현재 cmd=0 시 vx-expert가 standing 떠맡음. 4-expert 구조로 standing 전용 expert 분리하면 standing/walking 경계 매끄러워질 가능성.

---

## 10. 코드 위치 참조

| 파일                                                                                             | 역할                                                |
| ------------------------------------------------------------------------------------------------ | --------------------------------------------------- |
| [src/mjlab/tasks/velocity/velocity_env_cfg.py](src/mjlab/tasks/velocity/velocity_env_cfg.py)     | 33개 reward + 13 event base 정의                    |
| [src/mjlab/tasks/velocity/mdp/rewards.py](src/mjlab/tasks/velocity/mdp/rewards.py)               | 각 reward 함수 구현 (track, biped, gait pattern 등) |
| [src/mjlab/tasks/velocity/config/p1/env_cfgs.py](src/mjlab/tasks/velocity/config/p1/env_cfgs.py) | P1 task-specific override + curriculum 정의         |
| [src/mjlab/tasks/velocity/config/p1/rl_cfg.py](src/mjlab/tasks/velocity/config/p1/rl_cfg.py)     | PPO + MoE actor cfg                                 |
| [src/mjlab/rl/moe_model.py](src/mjlab/rl/moe_model.py)                                           | MoEActorModel 구현                                  |
| [scripts/check_rewards.py](scripts/check_rewards.py)                                             | 학습 중 reward 진행 상황 점검                       |
