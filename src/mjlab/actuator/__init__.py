"""Actuator implementations for mjlab."""

from mjlab.actuator.actuator import Actuator as Actuator
from mjlab.actuator.actuator import ActuatorCfg as ActuatorCfg
from mjlab.actuator.actuator import ActuatorCmd as ActuatorCmd
from mjlab.actuator.actuator import CommandField as CommandField
from mjlab.actuator.builtin_actuator import (
  BuiltinMotorActuator as BuiltinMotorActuator,
)
from mjlab.actuator.builtin_actuator import (
  BuiltinMotorActuatorCfg as BuiltinMotorActuatorCfg,
)
from mjlab.actuator.builtin_actuator import (
  BuiltinMuscleActuator as BuiltinMuscleActuator,
)
from mjlab.actuator.builtin_actuator import (
  BuiltinMuscleActuatorCfg as BuiltinMuscleActuatorCfg,
)
from mjlab.actuator.builtin_actuator import (
  BuiltinPositionActuator as BuiltinPositionActuator,
)
from mjlab.actuator.builtin_actuator import (
  BuiltinPositionActuatorCfg as BuiltinPositionActuatorCfg,
)

# Aliases for KIMM-internal naming (BuiltinPd* ↔ BuiltinPosition*)
BuiltinPdActuator = BuiltinPositionActuator
BuiltinPdActuatorCfg = BuiltinPositionActuatorCfg
from mjlab.actuator.builtin_actuator import (
  BuiltinVelocityActuator as BuiltinVelocityActuator,
)
from mjlab.actuator.builtin_actuator import (
  BuiltinVelocityActuatorCfg as BuiltinVelocityActuatorCfg,
)
from mjlab.actuator.builtin_group import BuiltinActuatorGroup as BuiltinActuatorGroup
from mjlab.actuator.dc_actuator import DcMotorActuator as DcMotorActuator
from mjlab.actuator.dc_actuator import DcMotorActuatorCfg as DcMotorActuatorCfg
from mjlab.actuator.learned_actuator import LearnedMlpActuator as LearnedMlpActuator
from mjlab.actuator.learned_actuator import (
  LearnedMlpActuatorCfg as LearnedMlpActuatorCfg,
)
from mjlab.actuator.pd_actuator import IdealPdActuator as IdealPdActuator
from mjlab.actuator.pd_actuator import IdealPdActuatorCfg as IdealPdActuatorCfg
from mjlab.actuator.xml_actuator import XmlActuator as XmlActuator
from mjlab.actuator.xml_actuator import XmlActuatorCfg as XmlActuatorCfg


# Compat stubs for KIMM-internal Fourbar actuator (not present in mjlab core).
# These satisfy import/isinstance checks from beom DR modules; the actuator
# group is never instantiated through the public mjlab API, so isinstance
# checks against these stubs are never True and the dependent code paths
# stay dormant. Attributes are declared so type-checkers stay quiet.
class FourbarPdActuator:  # type: ignore[no-redef]
  motor_armature: object
  default_motor_armature: object
  force_limit: object
  last_motor_torque: object

  def set_motor_armature(self, *args: object, **kwargs: object) -> None: ...


class FourbarPdActuatorCfg:  # type: ignore[no-redef]
  pass


class FourbarAnkleGroupCfg:  # type: ignore[no-redef]
  pass
