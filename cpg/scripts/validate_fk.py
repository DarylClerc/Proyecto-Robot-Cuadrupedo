"""Valida la cinemática directa (cpg/kinematics.py) contra las posiciones
reales que calcula MuJoCo para el modelo del Go2.

Para cada pata, compara:
  - Posición del pie calculada por nuestra FK analítica (en el frame de la
    cadera, i.e. del joint de abducción).
  - Posición del pie que da MuJoCo (mj_data.xpos del body "<LEG>_foot"),
    transformada al frame de la cadera usando la orientación real del body
    "<LEG>_hip" (mj_data.xmat).

Se prueba en dos configuraciones: postura neutra (todos los ángulos en 0) y
la postura de pie (STAND_UP_JOINT_POS), para cubrir un caso trivial y uno
realista.

Uso:
    python3 validate_fk.py
"""
import os
import sys

import mujoco
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "cpg"))

from kinematics import forward_kinematics_leg  # noqa: E402

SCENE_PATH = os.path.join(
    REPO_ROOT, "third_party/unitree_mujoco/unitree_robots/go2/scene.xml"
)

LEGS = ["FR", "FL", "RR", "RL"]

STAND_UP_JOINT_POS = {
    "FR": (0.00571868, 0.608813, -1.21763),
    "FL": (-0.00571868, 0.608813, -1.21763),
    "RR": (0.00571868, 0.608813, -1.21763),
    "RL": (-0.00571868, 0.608813, -1.21763),
}
ZERO_JOINT_POS = {leg: (0.0, 0.0, 0.0) for leg in LEGS}


def foot_position_in_hip_frame(mj_model, mj_data, leg):
    """Posición real del pie (world) transformada al frame de referencia de la
    pata: origen en el punto de anclaje de la cadera, pero con la orientación
    FIJA de base_link (no la del body "<leg>_hip", que ya rota con q1 y por
    lo tanto no sirve como referencia para medir el efecto de q1)."""
    base_body = mj_model.body("base_link")
    hip_body = mj_model.body(f"{leg}_hip")
    foot_body = mj_model.body(f"{leg}_foot")

    base_rot_world = mj_data.xmat[base_body.id].reshape(3, 3)
    hip_pos_world = mj_data.xpos[hip_body.id]  # pivote de q1, no cambia con q1
    foot_pos_world = mj_data.xpos[foot_body.id]

    # local = R_base^T @ (foot_world - hip_world)
    return base_rot_world.T @ (foot_pos_world - hip_pos_world)


def set_joint_angles(mj_model, mj_data, joint_pos):
    for leg, (q1, q2, q3) in joint_pos.items():
        for suffix, q in zip(["hip", "thigh", "calf"], [q1, q2, q3]):
            joint = mj_model.joint(f"{leg}_{suffix}_joint")
            mj_data.qpos[joint.qposadr[0]] = q


def run_case(mj_model, mj_data, joint_pos, label):
    mujoco.mj_resetData(mj_model, mj_data)
    set_joint_angles(mj_model, mj_data, joint_pos)
    mujoco.mj_kinematics(mj_model, mj_data)  # recalcula xpos/xmat sin integrar el tiempo

    print(f"\n=== {label} ===")
    max_err = 0.0
    for leg in LEGS:
        q1, q2, q3 = joint_pos[leg]
        fk_analytic = forward_kinematics_leg(leg, q1, q2, q3)
        fk_mujoco = foot_position_in_hip_frame(mj_model, mj_data, leg)
        err = np.linalg.norm(fk_analytic - fk_mujoco)
        max_err = max(max_err, err)
        print(
            f"{leg}: analítica={np.round(fk_analytic, 4)}  "
            f"mujoco={np.round(fk_mujoco, 4)}  error={err * 1000:.3f} mm"
        )
    print(f"Error máximo: {max_err * 1000:.3f} mm")
    return max_err


def main():
    mj_model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    mj_data = mujoco.MjData(mj_model)

    err_zero = run_case(mj_model, mj_data, ZERO_JOINT_POS, "Postura neutra (q=0)")
    err_stand = run_case(mj_model, mj_data, STAND_UP_JOINT_POS, "Postura de pie")

    ok = err_zero < 1e-4 and err_stand < 1e-4
    print(f"\nFK validada correctamente: {ok}")


if __name__ == "__main__":
    main()
