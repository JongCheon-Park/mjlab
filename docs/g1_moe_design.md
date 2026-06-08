# G1 Velocity MoE Actor — Design Note

mjlab의 `Mjlab-Velocity-Flat-Unitree-G1` baseline과 비교하기 위해
**Mixture-of-Experts (MoE) actor** 변형을 추가했다. 환경/critic/PPO
hyperparameter는 모두 baseline과 동일하게 두고 **policy network 구조만**
바꿔서 ablation을 깔끔하게 한다.

## 1. 새 task

| 항목              | Baseline                         | MoE 변형                             |
| ----------------- | -------------------------------- | ------------------------------------ |
| Task ID           | `Mjlab-Velocity-Flat-Unitree-G1` | `Mjlab-Velocity-Flat-Unitree-G1-MoE` |
| Env config        | `unitree_g1_flat_env_cfg()`      | 동일                                 |
| Critic            | MLP (512, 256, 128)              | 동일                                 |
| Actor             | MLP (512, 256, 128)              | **MoE (shared + 3 experts + head)**  |
| `experiment_name` | `g1_velocity`                    | `g1_velocity_moe`                    |
| Runner            | `VelocityOnPolicyRunner`         | 동일                                 |

`experiment_name`이 다르므로 로컬 로그 디렉토리는 자동 분리:

```
logs/rsl_rl/g1_velocity/<timestamp>_<run>/
logs/rsl_rl/g1_velocity_moe/<timestamp>_<run>/
```

W&B project는 둘 다 `mjlab` 기본값이라 같은 dashboard에 들어가지만 run name이
다르므로 비교에는 더 편하다.

## 2. MoE actor 구조

```
                  ┌──→ shared MLP ─────┐
obs (99) ─ norm ──┤                       cat ──→ head MLP ──→ μ (29)
                  └──→ expert_k MLP ────┘                          │
                       (k = arg max |cmd|)                         └─→ Gaussian(σ scalar) → action
                       k ∈ {x, y, yaw}
```

- shared와 expert는 **병렬**로 같은 normalized obs를 받는다.
- 매 forward마다 **rule-based gate**가 cmd 3개 중 절댓값이 가장 큰 축의
  expert 한 개만 선택한다. 학습 가능한 gating network가 아니다.
- 선택되지 않은 expert의 출력은 one-hot mask로 0이 되어 head에 영향이 없고,
  autograd 경로도 끊겨서 **그 step에선 gradient가 흐르지 않는다.**
- shared와 선택된 expert의 출력을 concat한 뒤 head MLP가 action 평균을 낸다.
- σ는 baseline과 동일한 scalar Gaussian (학습 파라미터 29개).

### Routing rule

```python
cmd = obs_norm[..., -3:]                  # (lin_vel_x, lin_vel_y, ang_vel_z)
expert_idx = cmd.abs().argmax(dim=-1)     # 0=x, 1=y, 2=yaw
```

obs는 actor obs group의 마지막 3 dim이 velocity command다 (`generated_commands`로
주입). normalized 값에 argmax를 걸어서 cmd 축마다 range가 달라도 공정하게
비교된다. cmd가 모두 0인 standing 상태에서는 argmax가 0(x-expert)로 떨어진다.

## 3. 차원 / 파라미터 (런타임 측정)

obs 차원은 mjlab env에서 직접 측정.

| 입출력               | Baseline | MoE |
| -------------------- | -------: | --: |
| Actor input dim      |       99 |  99 |
| Critic input dim     |      111 | 111 |
| Output (num_actions) |       29 |  29 |

### Actor 파라미터 분해

| 블록                   | 구조                                    |    파라미터 |
| ---------------------- | --------------------------------------- | ----------: |
| **Baseline** Actor MLP | 99 → 512 → 256 → 128 → 29 + log_std(29) | **219,194** |
| **MoE** shared         | 99 → 256 → 128                          |      58,496 |
| **MoE** expert × 3     | 각 99 → 192 → 128                       |     131,712 |
| **MoE** head           | (128+128) → 128 → 29                    |      36,637 |
| **MoE** log_std        | scalar Gaussian, dim 29                 |          29 |
| **MoE Actor 합계**     |                                         | **226,874** |

### 합계 비교

| 비교          |    Baseline |         MoE |      차이 |
| ------------- | ----------: | ----------: | --------: |
| Actor params  |     219,194 |     226,874 | **+3.5%** |
| Critic params |     221,697 |     221,697 |      동일 |
| **합계**      | **440,891** | **448,571** | **+1.7%** |

baseline ±5% 이내라서 "네트워크 구조 차이만 보는 ablation"에 적합한 수준.

## 4. 추가/수정된 파일

| 파일                                             | 역할                                                         |
| ------------------------------------------------ | ------------------------------------------------------------ |
| `src/mjlab/rl/moe_model.py`                      | `MoEActorModel`, `MoEActorModelCfg`, ONNX/JIT export wrapper |
| `src/mjlab/tasks/velocity/config/g1/rl_cfg.py`   | `unitree_g1_moe_ppo_runner_cfg()` 추가                       |
| `src/mjlab/tasks/velocity/config/g1/__init__.py` | `Mjlab-Velocity-Flat-Unitree-G1-MoE` task 등록               |

`MoEActorModel`은 rsl-rl `MLPModel`이 노출하는 인터페이스 (forward,
get_latent, distribution properties, update_normalization, as_jit, as_onnx,
recurrent stub) 전체를 동일하게 구현해서 PPO/Runner와 그대로 호환된다.
ONNX export는 routing 로직 (`one_hot * stack(experts) → sum`)이 정적
연산으로만 구성되어 있어 `dynamo=False` 경로로 정상 export된다.

## 5. 실행 방법

### 5.1 학습 (`train`)

```sh
# Baseline
uv run train Mjlab-Velocity-Flat-Unitree-G1     --env.scene.num-envs 4096

# MoE 변형
uv run train Mjlab-Velocity-Flat-Unitree-G1-MoE --env.scene.num-envs 4096

# 특정 GPU(예: 1번)에서 학습
uv run train Mjlab-Velocity-Flat-Unitree-G1-MoE --env.scene.num-envs 4096 --gpu-ids 1
```

W&B에서 더 명확히 분리하고 싶을 때:

```sh
uv run train Mjlab-Velocity-Flat-Unitree-G1-MoE \
  --env.scene.num-envs 4096 \
  --agent.wandb-project mjlab_moe \
  --agent.wandb-tags '("moe","g1","flat")'
```

### 5.2 학습된 정책 시각화 (`play`)

학습 결과 체크포인트는 `logs/rsl_rl/g1_velocity_moe/<timestamp>/model_*.pt`에
저장된다. 기본은 가장 최근 체크포인트를 자동으로 로드한다.

```sh
# 가장 최근 학습 결과 그대로 재생 (체크포인트는 자동 탐색)
uv run play Mjlab-Velocity-Flat-Unitree-G1-MoE

# env 수 줄여서 시각화 (viewer가 가벼워짐)
uv run play Mjlab-Velocity-Flat-Unitree-G1-MoE --num-envs 16

# 특정 로컬 체크포인트 지정
uv run play Mjlab-Velocity-Flat-Unitree-G1-MoE \
  --checkpoint-file logs/rsl_rl/g1_velocity_moe/2026-05-08_12-00-00/model_2000.pt

# W&B에 업로드된 체크포인트로 재생
uv run play Mjlab-Velocity-Flat-Unitree-G1-MoE \
  --wandb-run-path <entity>/<project>/<run_id> \
  --wandb-checkpoint-name model_2000.pt

# 학습 안 된 dummy agent로 환경만 보기 (action=0 또는 random)
uv run play Mjlab-Velocity-Flat-Unitree-G1-MoE --agent zero
uv run play Mjlab-Velocity-Flat-Unitree-G1-MoE --agent random

# 비디오 녹화 (logs/.../videos/play/ 에 저장)
uv run play Mjlab-Velocity-Flat-Unitree-G1-MoE --video --video-length 500

# Termination 비활성 — 넘어져도 계속 굴러가게 (모션 디버깅용)
uv run play Mjlab-Velocity-Flat-Unitree-G1-MoE --no-terminations

# Baseline과 MoE 동시 시각 비교 (별도 터미널에서 각각 실행)
uv run play Mjlab-Velocity-Flat-Unitree-G1     --num-envs 4
uv run play Mjlab-Velocity-Flat-Unitree-G1-MoE --num-envs 4
```

자주 쓰는 옵션 정리:

| 옵션                          | 설명                                                 |
| ----------------------------- | ---------------------------------------------------- |
| --agent {trained,zero,random} | 정책 종류. 기본 `trained`                            |
| --checkpoint-file <path>      | 로컬 `.pt` 체크포인트 경로                           |
| --wandb-run-path <path>       | `<entity>/<project>/<run_id>` 형식. W&B에서 다운로드 |
| --wandb-checkpoint-name       | 위 run 안에서 특정 체크포인트 (예: `model_2000.pt`)  |
| --num-envs <int>              | env 수 override. viewer 부담 줄일 때 작게            |
| --device <str>                | 예: `cuda:1`. 미지정 시 자동 선택                    |
| --video                       | mp4 녹화 활성                                        |
| --video-length <int>          | 녹화 frame 수 (기본 200)                             |
| --no-terminations             | termination 모두 끔                                  |
| --viewer {auto,native,viser}  | 뷰어 종류                                            |

체크포인트 자동 탐색 규칙: `--checkpoint-file`도 `--wandb-run-path`도 안 주면
`logs/rsl_rl/<experiment_name>/` 아래 가장 최근 timestamp 디렉토리의
가장 큰 iteration 체크포인트를 자동으로 사용한다 ([train.py:114](src/mjlab/scripts/train.py#L114),
[play.py 체크포인트 로직](src/mjlab/scripts/play.py)).

## 6. 향후 확장 메모

- **soft routing** 비교: `merge="sum"` + softmax gate (학습형)으로 바꿔
  hard 룰과 비교.
- **standing 분기**: cmd 3개 모두 작을 때(예: norm < 0.1) 별도 stand expert로
  보내는 4-expert 셋업.
- **hysteresis**: 대각선 보행 (v_x ≈ v_y) 경계에서 스위칭 떨림이 보이면
  `_select_expert`에 직전 expert 우선 마진 추가.
- **expert에 cmd 명시 입력**: 현재는 obs 안에 cmd가 포함되어 expert가 간접적으로
  보지만, expert에 cmd 스칼라를 추가 concat해서 학습 신호를 강화 가능.

## 7. 참고

robotics 제어 MoE에서 **별도 input encoder는 일반적이지 않음** — proprioceptive
obs가 차원이 낮을 때 (이 case는 99) raw obs를 직접 shared/expert에 넣는다.
대표 예시:

| 연구                    | 입력       | encoder 유무              |
| ----------------------- | ---------- | ------------------------- |
| MELA (Yang et al. 2020) | proprio    | ✗                         |
| MCP (Peng et al. 2019)  | state      | ✗                         |
| RT-2, OpenVLA           | image+text | ✓ (vision encoder는 필수) |
| RSL legged baseline     | proprio    | ✗                         |

본 설계는 위 패턴을 따라 raw obs를 그대로 두 갈래에 입력한다.
