# KIMM V4 학습 설정 — Design Doc

`Mjlab-Velocity-Flat-V4` task. **9개 reward 최소 set + per-joint pose std + 5-network MoE.**

## 0. 파일 구조

```
src/mjlab/asset_zoo/robots/kimm_v4/
├── kimm_v4_constants.py    Robot spec (actuator, keyframe, scale, collision)
├── __init__.py
└── xmls/kimm_v4.xml        MJCF

src/mjlab/tasks/velocity/config/kimm_v4/
├── env_cfgs.py             환경 (reward, sensor, action)
├── rl_cfg.py               PPO + MoE actor
└── __init__.py             Task 등록
```

모든 학습 로그: `logs/rsl_rl/v4_velocity_wholebody_moe/<timestamp>_<run_name>/`

---

## 1. Robot — kimm_v4_constants.py

### 1.1 Actuator (PD)

| 모터 | 관절                       | Kp  | Kd  | armature | effort | ωₙ_motor | ζ    |
| ---- | -------------------------- | --: | --: | -------: | -----: | -------: | ---: |
| X12  | hip_pitch, knee            | 509 | 65  | 0.516    | 170 Nm | 5.0 Hz   | 2.00 |
| X8   | hip_roll/yaw, ankle, waist | 250 | 16  | 0.0577   | 86 Nm  | 10.5 Hz  | 2.13 |

**X8 Kp=250 결정 근거:**
- ωₙ=5Hz × armature 식 적용 시 Kp=57 → ankle이 robot 전체 mass 못 잡음
- 검증: Kp=57 시 5초 standing → 8/8 falls
- **Kp=250으로 키움 → 0/8 falls** (10초 std=0.00015)
- ankle effective ωₙ at standing ≈ 1 Hz (P1 수준)

### 1.2 Action scale (G1 formula)

| 그룹 | 관절             | scale (rad) | 도(°) |
| ---- | ---------------- | ----------: | ----: |
| X12  | hip_pitch, knee  | 0.0835      | 4.78  |
| X8   | 나머지           | 0.086       | 4.93  |

`0.25 × effort / Kp`. 정책 mean μ가 큰 값 학습해서 stride 만듦 (P1 검증 방식).

### 1.3 KNEES_BENT 초기 자세

| 관절          | 각도   |
| ------------- | ------ |
| hip_pitch     | ±5°    |
| hip_roll      | 0      |
| hip_yaw       | 0      |
| knee_pitch    | +10°   |
| ankle_pitch   | -5°    |
| ankle_roll    | 0      |
| waist_yaw     | 0      |
| torso z spawn | 0.84 m |

검증: action=0 으로 10초 standing → falls 0/64, 평균 z=0.838 (std 0.00015).

### 1.4 Foot collision

```python
condim   = 3       # 발 capsule만, 마찰 평면
priority = 1
friction = 0.6
geom     = (left|right)_foot[1-8]_collision   # V4는 8 capsule
```

---

## 2. 환경 — env_cfgs.py

### 2.1 관절 패턴 (팔 없음)

```python
LEG_JOINT_PATTERNS = (
  r".*_hip_.*_joint",
  r".*_knee_pitch_joint",   # V4 (G1은 knee_joint)
  r".*_ankle_.*_joint",
)
WAIST_JOINT_PATTERNS = (r"waist_yaw_joint",)
```

→ wholebody action: **13 DOF** (legs 12 + waist 1).

### 2.2 DR (Domain Randomization) — 모두 disable

| Event                | 상태   |
| -------------------- | ------ |
| push_robot           | OFF    |
| foot_friction        | OFF    |
| encoder_bias         | OFF    |
| torso_pseudo_inertia | OFF    |
| joint_armature       | OFF    |
| joint_friction       | OFF    |
| joint_damping        | OFF    |
| actuator_rfi         | OFF    |
| reset_base           | ON ✓   |
| reset_robot_joints   | ON ✓   |

### 2.3 Sensor

| Sensor              | 역할                                |
| ------------------- | ----------------------------------- |
| feet_ground_contact | 발 ↔ 지면 contact + air time 추적   |
| self_collision      | robot 내부 충돌 (torso_link subtree) |
| foot_height_scan    | 발 site 주변 6 raycast (반경 3cm)   |

### 2.4 base_height 측정 기준

`base_height_l2` reward 가 쓰는 z = `root_link_pos_w[:, 2]` = **torso_link** world z.

| 기준 body  | 측정 z 위치 (standing 시) |
| ---------- | ------------------------- |
| torso_link | **0.838 m** ← target 기준 |
| base_link  | 0.905 m (torso + 0.067)   |
| IMU site   | 1.428 m (torso + 0.590)   |

→ `target_height = 0.84` 는 torso_link 기준. IMU 기준 아님.

### 2.5 Reward — 9개 활성

§3 참조. `docs/v4_reward.md`에 상세.

### 2.6 Curriculum

`command_vel` (default — velocity_env_cfg.py 정의):

| step        | lin_vel_x      | lin_vel_y      | ang_vel_z    |
| ----------- | -------------- | -------------- | ------------ |
| 0           | (-0.5, 0.5)    | (-0.5, 0.5)    | (-0.5, 0.5)  |
| 60K (2.5K iter) | (-1.0, 1.0) | (-1.0, 1.0)   | (-1.0, 1.0)  |
| 120K        | (-1.5, 2.0)    | (-1.0, 1.0)    | (-1.5, 1.5)  |
| 240K        | (-1.5, 3.0)    | (-1.0, 1.0)    | (-2.0, 2.0)  |
| 480K        | (-1.5, 3.5)    | (-1.0, 1.0)    | (-2.0, 2.0)  |

---

## 3. RL / Actor — rl_cfg.py

### 3.1 MoE 구조 — 5 networks

```
                    obs (240 dim)
                         │
       ┌─────────────────┼──────────────────────────────────┐
       ▼                 ▼          ▼          ▼            ▼
   SHARED MLP        expert[0]   expert[1]  expert[2]   expert[3]
   (256→128)         (vx)         (vy)       (yaw)       (STANDING)
   always on         └─────── 4-way routing ─────────────────┘
                              1 selected
                         │
              shared(128) + expert(128) → concat (256)
                         │
                    head MLP (128)
                         │
                  action μ (13 dim) → Gaussian sample
```

활성 path per step = **shared + 1 routed expert = 2** (총 5 networks 등록).

### 3.2 Routing 규칙

| 조건                              | 선택 expert  |
| --------------------------------- | ------------ |
| \|cmd\|_max < 0.05 m/s            | 3 (STANDING) |
| argmax(\|cmd[0]\|) = 0 (vx 우세)  | 0 (vx)       |
| argmax(\|cmd[1]\|) = 1 (vy 우세)  | 1 (vy)       |
| argmax(\|cmd[2]\|) = 2 (yaw 우세) | 2 (yaw)      |

### 3.3 Network dimensions

| 모듈         | layers          | output dim |
| ------------ | --------------- | ---------- |
| shared MLP   | (256, 128)      | 128        |
| 각 expert    | (192, 128)      | 128        |
| head MLP     | (128,)          | 13         |
| critic MLP   | (512, 256, 128) | 1          |

### 3.4 PPO hyperparams

| param                | 값                          |
| -------------------- | --------------------------- |
| clip_param           | 0.2                         |
| entropy_coef         | 0.01                        |
| num_learning_epochs  | 5                           |
| num_mini_batches     | 4                           |
| learning_rate        | 1e-3 (adaptive, target KL 0.01) |
| gamma                | 0.99                        |
| lam                  | 0.95                        |
| max_grad_norm        | 1.0                         |
| num_steps_per_env    | 24                          |
| max_iterations       | 30,000                      |

### 3.5 Distribution

Gaussian, `init_std=1.0`, std_type=scalar. **No tanh squashing, no action clipping** — policy 출력 unbounded.

---

## 4. Task 등록

```python
register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-V4",
  env_cfg=v4_flat_env_cfg(observe_waist=True, control_waist=True),
  rl_cfg=v4_wholebody_moe_ppo_runner_cfg(
    experiment_name="v4_velocity_wholebody_moe",
    num_experts=4,
    standing_threshold=0.05,
  ),
  runner_cls=VelocityOnPolicyRunner,
)
```

---

## 5. 실행 명령

### 학습
```sh
CUDA_VISIBLE_DEVICES=0 uv run train Mjlab-Velocity-Flat-V4 \
  --env.scene.num-envs 4096 \
  --agent.run-name v4_t18
```

### Play
```sh
uv run play Mjlab-Velocity-Flat-V4 \
  --checkpoint-file logs/rsl_rl/v4_velocity_wholebody_moe/<ts>_v4_t18/model_5000.pt \
  --viewer viser
```

### W&B
```sh
CUDA_VISIBLE_DEVICES=0 uv run train Mjlab-Velocity-Flat-V4 \
  --env.scene.num-envs 4096 --agent.run-name v4_t18 \
  --agent.wandb-project mjlab_v4 \
  --agent.wandb-tags '("v4","t18","poseStdPerJoint")'
```

---

## 6. 학습 기대치 / milestones

| iter  | 기대                       | 핵심 metric                              |
| ----: | -------------------------- | ---------------------------------------- |
| 300   | mean reward 양수           | `Reward/total`                           |
| 1500  | tracking 추종 시작         | `Reward/track_linear_velocity` raw > 0.3 |
| 3000  | 첫 stride emerge           | `Metrics/peak_height_mean` > 30mm        |
| 5000  | 보행 시작                  | `Metrics/twist/error_vel_xy` < 0.5       |
| 10000 | 안정 보행                  | mean reward plateau                      |
| 30000 | full curriculum 통과       | robust gait                              |

**Walking emerge 핵심 신호:** `peak_height_mean` ≥ 30mm + `biped_air_time` raw ≥ 0.3.

---

## 7. 검증된 fact

| 테스트                              | 결과                              |
| ----------------------------------- | --------------------------------- |
| Standing (action=0, 10sec, 64 envs) | falls 0/64, z=0.838 std 0.00015   |
| env.step 4 regime (stand/walk/turn/run) | 모두 pass                     |
| Reward 9개 fire 검증                | standing/walking 다 정상          |
| Lint + format                       | 통과                              |

---

## 8. 학습 history (lessons learned)

| run         | 설정                                          | 결과                                  |
| ----------- | --------------------------------------------- | ------------------------------------- |
| v4_t1 (옛)  | P1 강한 shaping, Kp X8=57                     | 30K iter peak 25mm, marching          |
| v4_t8-11    | G1 minimal, Kp X8=57                          | 30K iter peak 4mm, **standing 못 잡음** |
| v4_t14      | Kp X8=250, G1 minimal, pose uniform std       | 30K iter reward 196, **다리 벌려서 shuffle (reward hack)** |
| v4_t16      | + foot penalty + P1 weight (너무 강함)        | 초기 자주 falls, 학습 막힘            |
| **v4_t18**  | **v4_t14 + pose per-joint std (단일 변화)**   | **다리 벌림 사라질지 평가 중**         |

**진단 핵심:**
1. **X8 Kp=57 → 250** (standing 가능) — 모든 학습의 baseline
2. **pose uniform std 0.5 → per-joint** (hip_roll 0.15 tight) — reward hack 방지
3. 한 번에 너무 많이 바꾸면 학습 무너짐 (v4_t16 사례)
