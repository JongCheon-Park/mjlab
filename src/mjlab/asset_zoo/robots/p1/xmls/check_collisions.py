import os

import mujoco

# XML Path
xml_path = (
  "/ws/Jaebeom/pixi-mjlab/mjlab/src/mjlab/asset_zoo/robots/kimm_p1/xmls/kimm_p1.xml"
)

if not os.path.exists(xml_path):
  print(f"Error: XML not found at {xml_path}")
  exit(1)

# Load model
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Forward kinematics (compute collisions)
mujoco.mj_forward(model, data)

# Check contacts
print(f"Total contacts: {data.ncon}")
penetration_found = False
for i in range(data.ncon):
  contact = data.contact[i]
  # Negative distance means penetration
  if contact.dist < -1e-5:
    penetration_found = True
    geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
    geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
    body1_name = mujoco.mj_id2name(
      model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[contact.geom1]
    )
    body2_name = mujoco.mj_id2name(
      model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[contact.geom2]
    )

    print(f"[COLLISION] Dist: {contact.dist:.6f}")
    print(f"  Geom1: {geom1_name} (Body: {body1_name})")
    print(f"  Geom2: {geom2_name} (Body: {body2_name})")
    print("-" * 30)

if not penetration_found:
  print("No penetrating collisions found in default pose.")
