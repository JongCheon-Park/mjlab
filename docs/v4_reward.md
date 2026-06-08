# V4 Reward 정리 — Minimal walking set

`Mjlab-Velocity-Flat-V4` task의 모든 reward (총 33개 등록, **9개만 활성**).

설계 원칙: 표준 보행 RL에서 공통 사용하는 최소 reward만 켬. 만약 안 걸으면
**reward가 아닌 robot/PD/scale 문제**로 좁혀짐.

---

## 활성 reward (9개) — 표준 보행 set

| #  | reward                  | weight | raw 범위    | per-step 기여 | 역할                                         |
| -- | ----------------------- | -----: | ----------- | ------------- | ----------------------------11              |
| 1  | track_linear_velocity   |   5.0  | [0, 1]      | [0, 5]        | x/y velocity 명령 추종        |
| 2  | track_angular_velocity  |   5.0  | [0, 1]      | [0, 5]        | yaw rate 명령 추종            |
| 3  | upright                 |   1.0  | [0, 2]      | [0, 2]        | torso 꼿꼿 (gravity xy → 0)   |
| 4  | base_height             |   1.0  | [0, 1]      | [0, 1]        | torso z 0.84 m 유지           |
| 5  | pose                    |   1.0  | [0, 5] std / [0, 1] walk    | regime 의존 | default 자세 retention        |
| 6  | biped_air_time          |   1.0  | [0, 2]      | [0, 2]        | 발 들기 (single-support 보상) |
| 7  | dof_pos_limits          |  -1.0  | [0, ~0.1]   | [-0.1, 0]     | 관절 limit 페널티             |
| 8  | self_collisions         |  -1.0  | [0, ~10]    | [-10, 0]      | 자기충돌 페널티               |
| 9  | action_rate_l2          | -0.02  | [0, ~50]    | [-1, 0]       | action 급변 페널티 (smoothness)|

**Per-step max 합:** ~15 점 positive, 최대 -11 점 negative.

---

## 비활성 reward (24개, weight=0)

| 카테고리       | 목록                                                    |
| -------------- | ------------------------------------------------------- |
| 균형           | body_ang_vel, angular_momentum                          |
| 토크 안전      | joint_effort_limit, joint_torque_rate                   |
| Foot 동작      | foot_clearance, foot_swing_height, foot_slip, foot_flat, soft_landing |
| Biped 보조     | biped_first_swing_foot, biped_double_support_time, biped_standing_stability |
| Swing          | thigh_swing_target                                      |
| Pattern (8개)  | heelstrike/toeoff × {foot_pitch, knee, hip_pitch, hip_roll} |
| Pattern arm (3개) | heelstrike/toeoff_shoulder_pitch, toeoff_elbow      |

→ cfg 에는 등록되어 있음 (검증/디버깅용), 학습에는 영향 없음.

---

## 핵심 9개 상세

### 1. track_linear_velocity

[`rewards.py:477`](../src/mjlab/tasks/velocity/mdp/rewards.py#L477)

```python
xy_error = sum((cmd[:, :2] - actual[:, :2])²)
z_error  = actual[:, 2]²
return exp(-(xy_error + z_error) / σ²)
```

- σ: speed-bucket — standing √0.1, walking √0.2, running √0.5
- 출력 [0, 1] × weight 5.0 = max 5
- **cmd=0 시 actual=0 이면 자동 max (정지 추종 자동 만족)**

### 2. track_angular_velocity

[`rewards.py:510`](../src/mjlab/tasks/velocity/mdp/rewards.py#L510)

```python
z_error  = (cmd[:, 2] - actual[:, 2])²
xy_error = sum(actual[:, :2]²)         # ← xy 회전도 페널티
return exp(-(z_error + xy_error) / σ²)
```

- σ: low_std √0.25, high_std √0.5
- 출력 [0, 1] × weight 5.0

### 3. upright

[`rewards.py:731`](../src/mjlab/tasks/velocity/mdp/rewards.py#L731), [`rewards.py:2882` flat_orientation_multi](../src/mjlab/tasks/velocity/mdp/rewards.py#L2882)

```python
projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)
gx, gy = projected_gravity_b[:, 0], projected_gravity_b[:, 1]
return weight_xy[0]·exp(-gx²/σx²) + weight_xy[1]·exp(-gy²/σy²)
```

- weight_xy = (1.0, 1.0), **std_xy = (sin 10°, sin 1°)**
- 출력 [0, 2] × weight 1.0
- **앞뒤(pitch) 5°까지 reward 78% 유지** (sin 10° Gaussian width)
- 좌우(roll) 1°만 넘어도 reward 37% 급락 (엄격)

Pitch tolerance curve:

|  pitch | reward |
| -----: | -----: |
|   0°   | 100%   |
|   3°   |  91%   |
| **5°** | **78%**|
|  10°   |  37%   |
|  15°   |  11%   |

### 4. base_height

[`rewards.py:2913`](../src/mjlab/tasks/velocity/mdp/rewards.py#L2913)

```python
height_error = (root_link_pos_w[:, 2] - target_height)²
return exp(-height_error / σ²)
```

- **target_height = 0.84** (torso_link standing z)
- σ: std 0.03, std_walking 0.05, std_running 0.07
- 출력 [0, 1] × weight 1.0

### 5. pose (variable_posture) — ★ per-joint std

[`rewards.py:2768`](../src/mjlab/tasks/velocity/mdp/rewards.py#L2768)

```python
err² = (q - q_default)²
return weight_regime · exp(-mean(err² / std²))
```

**weight (regime별):**

| regime    | weight |
| --------- | -----: |
| standing  |  5.0   |
| walking   |  1.0   |
| running   |  1.0   |
| turning   |  1.0   |

**std (관절별, walking regime) — ★ 핵심 변화:**

| 관절          | std    | 의미                            |
| ------------- | -----: | ------------------------------- |
| hip_pitch     | 0.40   | stride 자유                     |
| **hip_roll**  | **0.15** | **다리 벌림 방지 (좁음)** ★    |
| hip_yaw       | 0.20   | 발 회전 제한                    |
| knee          | 0.70   | 굽힘 자유                       |
| ankle_pitch   | 0.30   | 발끝 자유                       |
| **ankle_roll**| **0.10** | **발목 옆 흔들림 방지 (좁음)** ★ |
| waist_yaw     | 0.20   | 허리 회전 제한                  |

**왜 hip_roll/ankle_roll 좁게?**

v4_t14 결과 (uniform std=0.5): 정책이 hip_roll 자유를 악용해 **다리를 쫙 벌려서 무게중심 낮춤 + 발 발발거리며 이동하는 reward hack** 발견.

| hip_roll 벌림 | uniform std=0.5 | **per-joint std=0.15** |
| ------------: | --------------: | ---------------------: |
|   5°          |          0.99   |              0.61      |
|  10°          |          0.91   |              0.14      |
|  20°          |          0.61   |          **0.005** ⬇️  |

→ per-joint std로 20° 벌림 페널티가 60×. 정상 보행 폭만 허용.

기타 regime std: standing(0.05~0.20), turning(hip_yaw 0.6 풀어줌), running(0.15~0.80).

- 출력 standing [0, 5], walking/running/turning [0, 1]

### 6. biped_air_time

[`rewards.py:908`](../src/mjlab/tasks/velocity/mdp/rewards.py#L908) — `class biped_air_time`

```python
# Single-support phase 보상. target_air_time=0.5 sec.
foot_reward = exp(-(t_air - 0.5)² / σ²)
# Standing 시 자동 0 (command mask 내장).
# Walking: 발 1개만 공중 = single_support → reward 부여
return where(standing, 0, where(walking, single_support_reward, 0))
```

- 출력 [0, 2] × weight 1.0
- **이게 walking emerge 핵심 신호** — 발 1개씩 들기 강제

### 7. dof_pos_limits

[`envs/mdp/rewards.py:81`](../src/mjlab/envs/mdp/rewards.py#L81)

```python
out_of_limits = clip(q - upper, min=0) + clip(lower - q, min=0)
return sum(out_of_limits, dim=1)
```

- 출력 [0, ~0.1] × weight -1.0

### 8. self_collisions

[`rewards.py:826`](../src/mjlab/tasks/velocity/mdp/rewards.py#L826)

```python
force_mag = norm(data.force_history, dim=-1)
hit = (force_mag > 10.0).any(dim=1)      # 10 N 임계
return hit.sum(dim=-1).float()
```

- 출력 [0, ~10] × weight -1.0

### 9. action_rate_l2

[`envs/mdp/rewards.py:58`](../src/mjlab/envs/mdp/rewards.py#L58)

```python
return sum((action - prev_action)², dim=1)
```

- 출력 [0, ~50] × weight -0.02

---

## Reward 시나리오 분석

### Standing (cmd = 0)

| reward              | 기대값 | 비고                                |
| ------------------- | -----: | ----------------------------------- |
| track_linear        |   5.0  | cmd=0, actual≈0 → max               |
| track_angular       |   5.0  | cmd=0, yaw_rate≈0 → max             |
| upright             |   2.0  | torso 꼿꼿                          |
| base_height         |   1.0  | z=0.84 유지                         |
| pose (standing)     |   5.0  | weight_standing=5 × 1.0             |
| biped_air_time      |   0    | standing_mask → 0                   |
| 합계 (positive)     | **18** |                                     |
| 페널티              |   ~0   | 가만히 있으면 거의 0                |

**STANDING reward ~18** (가만히 있으면 거저 받는 점수)

### Walking 성공 (cmd vx=0.5)

| reward              | 기대값  | 비고                                |
| ------------------- | ------: | ----------------------------------- |
| track_linear        |   ~4.5  | partial→good 추종                   |
| track_angular       |   ~4.5  | yaw=0 유지                          |
| upright             |   ~1.8  | dynamic 약간 흔들림                 |
| base_height         |   ~0.9  | 살짝 흔들림                         |
| pose (walking)      |   ~0.7  | weight 1.0, std 0.5                 |
| biped_air_time      |   ~1.5  | single-support active               |
| 합계 (positive)     | **~14** |                                     |
| action_rate         |  ~-0.3  | 적당한 변화                         |
| 합계 net            | **~13** |                                     |

**WALKING ~13 vs STANDING 18.** Standing이 더 높음 → walking emerge 어려움 가능.

### Walking 실패 (cmd vx=0.5, action=0)

| reward              | 기대값 | 비고                                |
| ------------------- | -----: | ----------------------------------- |
| track_linear        |  ~0.4  | actual=0 vs cmd=0.5, big error      |
| track_angular       |   5.0  | yaw=0                               |
| upright             |   2.0  | 가만히 → 꼿꼿                       |
| base_height         |   1.0  |                                     |
| pose (walking)      |   1.0  | default 자세 유지                   |
| biped_air_time      |   0    |                                     |
| 합계                | **~9** |                                     |

→ 안 가면 9점, 가면 13점, 가만히 standing 명령일 땐 18점.

---

## 학습 history (reward 측면)

| run         | reward 변경                                  | 결과                              |
| ----------- | -------------------------------------------- | --------------------------------- |
| v4_t14      | 9개 minimal, pose uniform std=0.5            | reward 196, **다리 벌리고 shuffle (reward hack)** |
| v4_t16      | + foot penalty + bilateral_symmetry + body_ang_vel (P1 weight) | 너무 강해서 초기 fall 많음, 학습 막힘 |
| **v4_t18**  | **v4_t14 + pose per-joint std (단일 변화)**  | **다리 벌림 사라질지 평가 중**     |

## 진단 가치

| 학습 결과                          | 해석                                       |
| ---------------------------------- | ------------------------------------------ |
| walking emerge, peak_height > 30mm | 9개 reward 충분, V4 fundamentally OK       |
| walking 안 됨, 발만 까딱           | track_linear weight 키우거나 biped_air_time curriculum |
| standing 계속 못 잡음              | PD 문제 (X8 Kp 재조정)                     |
| 전혀 안 움직임                     | action scale 키우거나 entropy_coef 키움    |
| 다리 벌리고 shuffle                | pose hip_roll std 더 좁힘                  |

---

## V4 vs G1 어댑테이션

| 항목                  | G1                | V4                       |
| --------------------- | ----------------- | ------------------------ |
| 팔                    | 14 DOF            | 없음 (legs 12 + waist 1) |
| 무릎 관절명           | knee_joint        | knee_pitch_joint         |
| base_height target    | 0.78 m            | **0.84 m**               |
| Floating root         | pelvis            | torso_link               |
| 발 capsule            | 7개               | 8개                      |
| 발 마찰               | DR 0.4–1.0        | 고정 0.6 (Phase 1)       |

---

## 갱신 정책

reward / weight 변경할 때마다 이 문서 갱신. 특히:
- 새 reward 켤 때 → 표에 추가
- weight 변경 → 해당 entry 수정
- 학습 결과 → "진단 가치" 표에 사례 추가
