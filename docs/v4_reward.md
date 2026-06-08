# V4 Reward 정리 — KIMM 원본 setup

`Mjlab-Velocity-Flat-KIMM-V4` task. **38개 활성 reward.** KIMM 내부에서 검증된 풀 reward stack.

설계 원칙: tracking + posture + gait shaping + smoothness + safety 다 적극 활용. P1/G1 minimal보다 reward 많지만 V4 특수성 (heavy thigh, asymmetric mass) 보상.

---

## 활성 reward (38개) — 카테고리별

### Tracking (objective)

| reward                  | weight | 역할                          |
| ----------------------- | -----: | ----------------------------- |
| track_linear_velocity   |  +2.0  | x/y velocity 명령 추종        |
| track_angular_velocity  |  +5.0  | yaw rate 명령 추종 (강하게)   |

### Posture / balance

| reward         | weight | 역할                                  |
| -------------- | -----: | ------------------------------------- |
| upright        |  +1.0  | torso 꼿꼿 (xy=(1, 5), pitch 10°/roll 1°) |
| base_height    |  +1.0  | torso z 0.84 m 유지                   |
| pose           |  +1.0  | per-joint std, regime-aware           |
| waist_yaw_fixed |  +1.0  | waist 회전 고정                        |
| body_ang_vel   | -0.05  | torso wobble 페널티                   |
| angular_momentum | -0.05 | whole-body 회전 페널티                 |

**`pose` 상세:**
- weight_standing=5.0, weight_walking=1.0
- std_standing: 모든 관절 0.05 (정지 시 엄격)
- std_walking (관절별):
  - hip_pitch: 0.30 / **hip_roll: 0.15** / hip_yaw: 0.15
  - knee: 0.35 / ankle_pitch: 0.25 / **ankle_roll: 0.10**
  - waist_yaw: 0.20

**`upright` 상세:**
- weight_xy = (1.0, 5.0): roll 페널티 5×
- std_xy = (sin 10°, sin 1°): pitch 5°까지 78%, roll 1° 넘으면 페널티

### Gait (보행 패턴)

| reward                       | weight | 역할                                |
| ---------------------------- | -----: | ----------------------------------- |
| biped_air_time               |  +1.0  | single-support phase 보상 (target 0.5초) |
| biped_first_swing_foot       |  +2.0  | standing→moving 시 첫 swing 방향    |
| biped_double_support_time    |  +1.0  | 양발 contact 시간 (standing 시)     |
| biped_standing_stability     |  +1.0  | 정지 시 양발 contact 강제           |
| knee_height                  |  +1.0  | 무릎 높이 (V4 specific?)             |
| thigh_swing_pattern          |  +1.0  | 허벅지 swing 패턴                   |
| shank_swing_pattern          |  +1.0  | 정강이 swing 패턴                   |

### Heel-strike / toe-off pattern (10개)

| reward                            | weight |
| --------------------------------- | -----: |
| heelstrike_foot_pitch_pattern     | +0.25  |
| toeoff_foot_pitch_pattern         | +0.25  |
| heelstrike_knee_pattern           | +0.25  |
| toeoff_knee_pattern               | +0.25  |
| heelstrike_hip_pitch_pattern      | +0.25  |
| toeoff_hip_pitch_pattern          | +0.25  |
| heelstrike_hip_roll_pattern       | +0.25  |
| toeoff_hip_roll_pattern           | +0.25  |
| heelstrike_hip_yaw_pattern        | +0.25  |
| toeoff_hip_yaw_pattern            | +0.25  |

각 phase마다 ideal joint 각도 강제 → 인간 같은 heel-toe gait.

### Foot 동작 / 제어

| reward                    | weight | 역할                                |
| ------------------------- | -----: | ----------------------------------- |
| foot_clearance            |  -5.0  | swing 발 target 0.15m 안 도달 시 페널티 |
| foot_swing_height         |  -1.0  | landing 시 peak < target 페널티     |
| foot_center_noncontact    |  -1.0  | 발 중심부 미접지 페널티              |
| foot_slip                 |  -0.1  | contact 중 미끄러짐                  |
| foot_force_rate           |  -0.1  | foot contact force 급변              |
| foot_lateral_distance     |  -0.1  | 발 좌우 거리                         |
| foot_landing              | -0.001 | landing 충격                         |

### Safety / smoothness

| reward             | weight | 역할                          |
| ------------------ | -----: | ----------------------------- |
| dof_pos_limits     |  -5.0  | 관절 limit 근접 페널티        |
| self_collisions    |  -2.0  | 자기충돌 페널티               |
| joint_torque_rate  |  -1.0  | torque 급변                   |
| action_rate_l2     |  -0.1  | action 급변 (smoothness)      |
| joint_effort_limit | -0.01  | torque saturation             |
| joint_power        | -1e-4  | mechanical power (energy)      |

---

## 핵심 reward 함수 상세

### 1. track_linear_velocity

`exp(-‖v_cmd - v_actual‖² / σ²)` — speed-bucket std (standing/walking/running).

### 2. track_angular_velocity (weight 5.0, V4가 강함)

`exp(-(yaw_err² + roll_pitch_err²) / σ²)`

### 3. upright

```python
projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)
gx, gy = proj[:, 0], proj[:, 1]
return weight_xy[0]·exp(-gx²/σx²) + weight_xy[1]·exp(-gy²/σy²)
```

| param | 값 |
|---|---|
| weight_xy | (1.0, 5.0) |
| std_x (pitch) | sin(10°) = 0.174 |
| std_y (roll) | sin(1°) = 0.0175 |

**roll 5×** — V4 다리 벌림 방지에 직접 효과.

### 4. base_height

`exp(-(z - 0.84)² / σ²)`. torso_link z 측정.

### 5. pose (variable_posture)

regime별 weight + 관절별 std. hip_roll/ankle_roll tight (좁음) — reward hack 방지.

### 6. biped_air_time

single-support phase (한쪽 발 air) target 0.5초. standing 시 자동 0.

### 7. heel/toe pattern (10개, weight 0.25 each = 2.5 합)

walking phase 별 ideal joint 각도 강제 — 자연스러운 보행 패턴.

### 8. knee_height (V4-specific?)

무릎 높이 reward. V4 leg 길어서 추가됐을 가능성.

### 9. foot_center_noncontact

발 중앙(`foot4`, `foot5`) 미접지 페널티. 발끝/뒤꿈치만 접지 (forefoot/heel-only) 방지.

---

## Reward 시나리오 분석 (대략)

### Standing (cmd = 0)

| reward 종류 | 기대 per-step |
| ----------- | ------------: |
| track (2개) | ~7.0          |
| posture (6개) | ~10.0        |
| biped (4개) | ~2.0          |
| heel/toe (10개) | 0 (walking phase 아님) |
| **합계 positive** | **~19** |

페널티 거의 0 (가만히 있으니).

### Walking (cmd vx=0.5, 좋은 보행)

| reward 종류 | 기대 per-step |
| ----------- | ------------: |
| track (2개) | ~6.5          |
| posture (6개) | ~5.0         |
| biped (4개) | ~3.5          |
| gait shaping (heel/toe + thigh/shank + knee_height) | ~6.0 |
| foot (-5, -1, -1, ...) | ~-2.0 (소소한 페널티) |
| safety | ~-1.0          |
| **합계 net** | **~18** |

Standing(~19) ≈ Walking(~18) — 비슷한 수준. PPO가 walking이 cmd 추종에 답이라 학습.

---

## V4 vs G1 어댑테이션

| 항목               | G1                | V4                       |
| ------------------ | ----------------- | ------------------------ |
| 팔                 | 14 DOF            | 없음 (legs 12 + waist 1) |
| 무릎 관절명        | knee_joint        | **knee_joint** (P1 스타일) |
| base_height target | 0.78 m            | **0.84 m**               |
| Floating root      | pelvis            | torso_link               |
| 발 capsule         | 7개               | 8개                      |
| 모터               | Unitree (5020, 7520) | RMD X12/X8            |
| Reward 수          | ~10               | **38** (KIMM 풀 stack)    |

---

## 갱신 정책

reward / weight 변경할 때마다 이 문서 갱신:
- weight 변경 → 해당 entry 수정
- 새 reward 추가 → 카테고리 표에 추가
- 학습 결과 → "학습 history" 표에 사례 추가 (TBD)
