"""Valida la cinemática inversa (cpg/kinematics.py) de dos formas:

1. Round-trip analítico: para ángulos aleatorios (q1,q2,q3) dentro de rangos
   razonables, calcula el target con FK, le aplica IK, y compara la posición
   resultante (FK(IK(target))) contra el target original.
2. Contra MuJoCo: toma esos mismos targets, resuelve IK, aplica los ángulos
   resultantes en el modelo real del Go2, y compara la posición del pie que
   da MuJoCo contra el target deseado (esto es lo que realmente importa para
   el CPG: "le pido este target, ¿el robot pone el pie ahí?").

Uso:
    python3 validate_ik.py
"""
import os
import sys

import mujoco
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "cpg"))

from kinematics import (  # noqa: E402
    forward_kinematics_leg,
    inverse_kinematics_leg,
    HIP_OFFSET_Y,
)

SCENE_PATH = os.path.join(
    REPO_ROOT, "third_party/unitree_mujoco/unitree_robots/go2/scene.xml"
)

LEGS = ["FR", "FL", "RR", "RL"]

# Rangos "seguros" de ángulos para generar targets alcanzables y físicamente
# razonables (evita configuraciones degeneradas cerca de los límites del joint).
Q1_RANGE = (-0.3, 0.3)
Q2_RANGE = (0.2, 1.3)
Q3_RANGE = (-2.5, -0.5)

N_SAMPLES = 200
SEED = 0


def foot_position_in_base_frame(mj_model, mj_data, leg):
    base_body = mj_model.body("base_link")
    hip_body = mj_model.body(f"{leg}_hip")
    foot_body = mj_model.body(f"{leg}_foot")
    base_rot_world = mj_data.xmat[base_body.id].reshape(3, 3)
    hip_pos_world = mj_data.xpos[hip_body.id]
    foot_pos_world = mj_data.xpos[foot_body.id]
    return base_rot_world.T @ (foot_pos_world - hip_pos_world)


def set_joint_angles(mj_model, mj_data, leg, q1, q2, q3):
    for suffix, q in zip(["hip", "thigh", "calf"], [q1, q2, q3]):
        joint = mj_model.joint(f"{leg}_{suffix}_joint")
        mj_data.qpos[joint.qposadr[0]] = q


def main():
    rng = np.random.default_rng(SEED)
    mj_model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    mj_data = mujoco.MjData(mj_model)

    max_roundtrip_err = 0.0
    max_mujoco_err = 0.0

    for leg in LEGS:
        for _ in range(N_SAMPLES):
            q1 = rng.uniform(*Q1_RANGE)
            q2 = rng.uniform(*Q2_RANGE)
            q3 = rng.uniform(*Q3_RANGE)

            target = forward_kinematics_leg(leg, q1, q2, q3)

            q_ik = inverse_kinematics_leg(leg, *target)
            recovered = forward_kinematics_leg(leg, *q_ik)
            roundtrip_err = np.linalg.norm(recovered - target)
            max_roundtrip_err = max(max_roundtrip_err, roundtrip_err)

            mujoco.mj_resetData(mj_model, mj_data)
            set_joint_angles(mj_model, mj_data, leg, *q_ik)
            mujoco.mj_kinematics(mj_model, mj_data)
            foot_mujoco = foot_position_in_base_frame(mj_model, mj_data, leg)
            mujoco_err = np.linalg.norm(foot_mujoco - target)
            max_mujoco_err = max(max_mujoco_err, mujoco_err)

    print(f"Muestras por pata: {N_SAMPLES}, patas: {LEGS}")
    print(f"Error máximo round-trip (FK->IK->FK): {max_roundtrip_err * 1000:.4f} mm")
    print(f"Error máximo contra MuJoCo (FK->IK->MuJoCo): {max_mujoco_err * 1000:.4f} mm")

    ok = max_roundtrip_err < 1e-6 and max_mujoco_err < 1e-3
    print(f"\nIK validada correctamente: {ok}")


if __name__ == "__main__":
    main()
