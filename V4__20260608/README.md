# KIMM V4 without fourbar

이 폴더는 MJLab의 기존 `v4` 파일 이름과 패키지 경로를 유지한 공유용 코드입니다.
별도 wrapper import 방식이 아니라, 아래 두 폴더를 대상 MJLab checkout의 같은 위치에 복사해서 사용합니다.

```text
mjlab/src/mjlab/asset_zoo/robots/kimm_v4/
mjlab/src/mjlab/tasks/velocity/config/v4/
```

## 변경점

- `kimm_v4_constants.py`
  - `FourbarPdActuatorCfg` 제거
  - ankle pitch/roll을 `BuiltinPdActuatorCfg` 또는 `IdealPdActuatorCfg` 기반 일반 PD actuator에 포함
  - `V4_ACTION_SCALE` 이름 유지
  - `get_v4_robot_cfg()` 이름 유지
- `velocity/config/v4/env_cfgs.py`
  - `fourbar_constraint_barrier` reward 제거
  - `fourbar_constraint_boundary` termination 제거
  - 나머지 V4 reward/curriculum 구성은 기존 V4 설정과 공유
- `velocity/config/v4/rl_cfg.py`
  - 파일명과 함수명 유지

## 사용 방법

대상 MJLab repository root에서 이 폴더의 `mjlab/src/...` 내용을 같은 경로로 덮어씁니다.
그 후 기존과 동일하게 V4 task/config를 사용하면 됩니다.

```python
from mjlab.tasks.velocity.config.v4.env_cfgs import kimm_v4_flat_env_cfg

cfg = kimm_v4_flat_env_cfg()
```
