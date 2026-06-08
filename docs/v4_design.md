# KIMM V4 학습 설정 — Design Doc

`Mjlab-Velocity-Flat-KIMM-V4` task. KIMM 내부 검증된 V4 보행 학습 setup.

## 0. 파일 구조

```
src/mjlab/asset_zoo/robots/kimm_v4/
├── kimm_v4_constants.py    Robot spec (actuator, keyframe, scale, collision)
├── kimm_v4_actuators.py    RMD X12/X8 motor 모듈 사양
├── __init__.py
└── xmls/kimm_v4.xml        MJCF + meshes

src/mjlab/tasks/velocity/config/v4/
├── env_cfgs.py             환경 (reward, sensor, action, curriculum)
├── rl_cfg.py               PPO + plain MLP actor (NOT MoE)
└── __init__.py             Task 등록
```

학습 로그: `logs/rsl_rl/v4_velocity/<timestamp>_<run_name>/`

## 1. 등록된 task

| Task ID                          | 환경     | 용도                  |
| -------------------------------- | -------- | --------------------- |
| `Mjlab-Velocity-Flat-KIMM-V4`    | 평지     | 일반 보행 학습 (주력) |
| `Mjlab-Velocity-Rough-KIMM-V4`   | 거친 지면 | 지형 적응 학습         |

## 2. Robot — kimm_v4_constants.py

### 2.1 Actuator (auto-computed PD)

motor 스펙(RMD X12/X8) + drivetrain efficiency 로부터 자동 계산:

```python
FN_HZ = 5.0
ZETA = 2.0
drivetrain_efficiency = 2.0
```

결과:

| 모터 | 관절                       | Kp    | Kd    | armature | effort | ωₙ_motor |
| ---- | -------------------------- | ----: | ----: | -------: | -----: | -------: |
| X12  | hip_pitch, knee            | 509.3 | 64.84 | 0.5160   | 170 Nm | 5.0 Hz   |
| X8   | hip_roll/yaw, ankle, waist | 56.9  | 7.25  | 0.0577   | 86 Nm  | 5.0 Hz   |

> 참고: X8 모터 inertia 작아서 standing 시 ankle effective ωₙ < 1 Hz.
> KIMM 검증된 reward stack 으로 학습 안정성 보장됨.

### 2.2 Action scale (G1 formula: `0.25 × effort / Kp`)

| 그룹  | 관절                                   | scale (rad) | 도(°) |
| ----- | -------------------------------------- | ----------: | ----: |
| X12   | hip_pitch, knee                         | 0.0835      | 4.78  |
| X8    | hip_roll/yaw, ankle_pitch/roll, waist  | 0.3776      | 21.63 |

정책 mean μ가 큰 값 학습해서 stride 만듦.

### 2.3 KNEES_BENT 초기 자세

| 관절          | 각도   |
| ------------- | ------ |
| hip_pitch     | ±5°    |
| hip_roll      | 0      |
| hip_yaw       | 0      |
| knee          | +10°   |
| ankle_pitch   | -5°    |
| ankle_roll    | 0      |
| waist_yaw     | 0      |
| torso z spawn | 0.84 m |

좌우 hip_pitch 부호 반대 (mirror). 다른 관절은 동일 부호.

### 2.4 Foot collision

```python
condim   = 3       # 발 capsule만, 마찰 평면
priority = 1
friction = 0.6
geom     = (left|right)_foot[1-8]_collision   # 8 capsule
```

## 3. 환경 — env_cfgs.py

### 3.1 관절 / Action

- 전체 관절 컨트롤 (13 DOF: legs 12 + waist 1)
- knee 관절명: `knee_joint` (P1 스타일)

### 3.2 Sensor

| Sensor              | 역할                                          |
| ------------------- | --------------------------------------------- |
| feet_ground_contact | 발 ↔ 지면 contact + air time 추적             |
| feet_center_contact | 발 가운데(foot4/5) 접지 여부                  |
| self_collision      | torso_link subtree 내부 충돌                  |
| foot_height_scan    | 발 site 주변 6 raycast (반경 3cm)             |
| terrain_scan        | torso_link 기준 raycast (rough env만)          |

### 3.3 base_height 측정 기준

`base_height` reward 는 `root_link_pos_w[:, 2]` 사용 = **torso_link** z.

| 기준 body  | standing 시 z |
| ---------- | ------------- |
| torso_link | 0.838 m       |
| base_link  | 0.905 m       |
| IMU site   | 1.428 m       |

`target_height = 0.84` 는 torso_link 기준.

### 3.4 Reward — 38개 활성

`docs/v4_reward.md` 참조.

### 3.5 Curriculum

`command_vel`, `track_*_std`, `biped_air_time_weight`, 기타 KIMM 내부 curriculum (env_cfgs.py 정의).

### 3.6 DR (Domain Randomization)

KIMM 코드는 일부 DR 활성: `push_robot`, `foot_friction`, `encoder_bias`, `joint_armature`, `joint_friction`, `joint_damping` 등.

**비활성 (BuiltinPdActuator 비호환 — 어쩔 수 없이 pop):**

| Event          | 사유                                                          |
| -------------- | ------------------------------------------------------------- |
| `pd_gains`     | KIMM DR code 가 `BuiltinPositionActuator` 만 지원 (Pd 미지원) |
| `actuator_rfi` | 동일 — 1-to-1 mapping 가정, BuiltinPdActuator 는 2-to-1     |

→ V4 env_cfg.py 에서 `cfg.events.pop(...)` 으로 disable.

만약 살리고 싶으면 `envs/mdp/dr/actuator.py` 의 KIMM DR 코드를 BuiltinPdActuator 지원하도록 패치 필요.

정확한 활성/비활성은 학습 시 log 의 events 표 확인.

## 4. RL / Actor — rl_cfg.py

### 4.1 Network (plain MLP, NOT MoE)

| 모듈        | layers          |
| ----------- | --------------- |
| actor MLP   | (512, 256, 128) |
| critic MLP  | (512, 256, 128) |

### 4.2 PPO hyperparams

| param                | 값                              |
| -------------------- | ------------------------------- |
| clip_param           | 0.2                             |
| entropy_coef         | 0.01                            |
| num_learning_epochs  | 5                               |
| num_mini_batches     | 4                               |
| learning_rate        | 1e-3 (adaptive, target KL 0.01) |
| gamma                | 0.99                            |
| lam                  | 0.95                            |
| max_grad_norm        | 1.0                             |
| num_steps_per_env    | 24                              |
| max_iterations       | 30,000                          |
| save_interval        | 100                             |
| experiment_name      | "v4_velocity"                   |

### 4.3 Distribution

Gaussian, `init_std=1.0`, std_type=scalar. **No tanh squashing, no action clipping** — policy 출력 unbounded.

## 5. 실행

### 학습 (평지)

```sh
CUDA_VISIBLE_DEVICES=0 uv run train Mjlab-Velocity-Flat-KIMM-V4 \
  --env.scene.num-envs 4096 \
  --agent.run-name v4_t1
```

### 학습 (rough)

```sh
CUDA_VISIBLE_DEVICES=0 uv run train Mjlab-Velocity-Rough-KIMM-V4 \
  --env.scene.num-envs 4096 \
  --agent.run-name v4_rough_t1
```

### Play

```sh
uv run play Mjlab-Velocity-Flat-KIMM-V4 \
  --checkpoint-file logs/rsl_rl/v4_velocity/<ts>_v4_t1/model_5000.pt \
  --viewer viser
```

### W&B 같이

```sh
CUDA_VISIBLE_DEVICES=0 uv run train Mjlab-Velocity-Flat-KIMM-V4 \
  --env.scene.num-envs 4096 \
  --agent.run-name v4_t1 \
  --agent.wandb-project mjlab_v4 \
  --agent.wandb-tags '("v4","t1","kimm_original")'
```

## 6. Git 워크플로우

| 명령                                     | 동작                                |
| ---------------------------------------- | ----------------------------------- |
| `git checkout v4-dev`                    | V4 작업 브랜치                      |
| `git commit -am "..."`                   | 변경 commit                         |
| `git push`                               | origin/v4-dev (너의 fork) 에 push   |
| `git checkout main && git pull`          | upstream/main (mjlab 원본) 따라잡기 |
| `git checkout v4-dev && git merge main`  | v4-dev 에 main 변경 가져옴          |

Remote 구성:
- **origin** = `https://github.com/JongCheon-Park/mjlab.git` (너의 fork — push 가능)
- **upstream** = `https://github.com/mujocolab/mjlab.git` (원본 mjlab — pull 받기만)

브랜치:
- **main** = upstream 따라가는 깨끗한 mjlab (61af2b75)
- **v4-dev** = V4 작업 + KIMM 내부 mjlab 수정 (main과 merge됨)

## 6.1 V4 코드 출처

V4 작업 코드는 `V4__20260608/` 백업 폴더에서 복원 (commit `68d7d2ee`):
- `kimm_v4_constants.py`, `kimm_v4_actuators.py`, `xmls/kimm_v4.xml` (robot)
- `config/v4/env_cfgs.py`, `config/v4/rl_cfg.py`, `config/v4/__init__.py` (task)
- `velocity_env_cfg.py`, `velocity/mdp/*.py` (KIMM 내부 mjlab 수정)
- `actuator_torque.py` (별도 제공)

원본 보존: `V4__20260608/` 폴더는 reference로 유지. src/mjlab/ 의 파일과 항상 동일해야 함.

검증:
```sh
diff src/mjlab/asset_zoo/robots/kimm_v4/kimm_v4_constants.py \
     V4__20260608/mjlab/src/mjlab/asset_zoo/robots/kimm_v4/kimm_v4_constants.py
# (출력 없음 = 동일)
```

유일한 functional 변경 (BuiltinPdActuator 호환):
- `env_cfgs.py`: `cfg.events.pop("pd_gains")`, `cfg.events.pop("actuator_rfi")` 추가

## 7. 학습 기대치 / milestones

| iter  | 기대                          | 핵심 metric                              |
| ----: | ----------------------------- | ---------------------------------------- |
| 300   | mean reward 양수, fall 줄어듬 | `Reward/total`                           |
| 1500  | tracking 추종 시작            | `Reward/track_linear_velocity` raw > 0.3 |
| 3000  | 첫 stride emerge              | `Metrics/peak_height_mean` > 30mm        |
| 5000  | 보행 시작                     | `Metrics/twist/error_vel_xy` < 0.5       |
| 10000 | 안정 보행                     | mean reward plateau                      |
| 30000 | full curriculum 통과          | robust gait                              |

## 8. 진단 체크리스트

| 증상                          | 의심                                                  |
| ----------------------------- | ----------------------------------------------------- |
| 자주 넘어짐 (iter > 1K)       | `base_height` target / `dof_pos_limits` weight        |
| 발 안 듬 (peak_height < 10mm) | `biped_air_time` curriculum / `foot_clearance` weight |
| 다리 벌리고 shuffle           | `pose` hip_roll std / heelstrike pattern weights      |
| 짞짞이 보행                   | `biped_first_swing_foot` weight / XML 좌우 비대칭     |
| forward lean 후 발 뒤         | `upright` pitch std / `thigh_swing_pattern` weight    |
| reward 음수 (안 학습)         | `joint_torque_rate` / `joint_power` 페널티 분석       |
