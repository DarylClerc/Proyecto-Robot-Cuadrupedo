"""Validación final por DDS: parado -> caminata con los parámetros YA
OPTIMIZADOS por CEM (cpg/cem_best_params.npy), usando la IK para convertir
la trayectoria del pie en comandos articulares.

CEM (cem.py) ajusta los parámetros evaluando con control directo sobre
mj_data.ctrl (rollout.py), sin pasar por DDS -- necesario para que miles de
rollouts sean rápidos. Este script corre el resultado YA optimizado, pero
esta vez sí a través de la interfaz DDS real (LowCmd_/LowState_ vía
UnitreeSdk2Bridge), la misma que usaría el robot físico. El objetivo es
confirmar que el comportamiento es el mismo por ambas vías antes de dar por
cerrada la parte de CPG.

Uso:
    MUJOCO_GL=egl python3 walk_cpg_open_loop.py
"""
import os
import sys
import time

import mujoco
import numpy as np
import imageio.v2 as imageio

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "cpg"))
sys.path.insert(0, os.path.join(REPO_ROOT, "third_party/unitree_mujoco/simulate_python"))

from kinematics import inverse_kinematics_leg, HIP_OFFSET_Y  # noqa: E402
from oscillator import CpgParams, foot_target  # noqa: E402

from unitree_sdk2py_bridge import UnitreeSdk2Bridge  # noqa: E402
from unitree_sdk2py.core.channel import (  # noqa: E402
    ChannelFactoryInitialize,
    ChannelPublisher,
)
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_  # noqa: E402
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_  # noqa: E402
from unitree_sdk2py.utils.crc import CRC  # noqa: E402

SCENE_PATH = os.path.join(
    REPO_ROOT, "third_party/unitree_mujoco/unitree_robots/go2/scene.xml"
)
OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "walk_cpg_dds_validation.mp4"
)
PARAMS_PATH = os.path.join(REPO_ROOT, "cpg", "cem_best_params.npy")

DOMAIN_ID = 1
INTERFACE = "lo"

TIMESTEP = 0.002
STAND_UP_SECONDS = 2.0
WALK_SECONDS = 8.0
FPS = 30

LEGS = ["FR", "FL", "RR", "RL"]
# Índice del primer motor (hip) de cada pata, en el orden del actuator list.
MOTOR_BASE_INDEX = {"FR": 0, "FL": 3, "RR": 6, "RL": 9}

STAND_UP_JOINT_POS = np.array(
    [
        0.00571868, 0.608813, -1.21763, -0.00571868, 0.608813, -1.21763,
        0.00571868, 0.608813, -1.21763, -0.00571868, 0.608813, -1.21763,
    ],
    dtype=float,
)
STAND_DOWN_JOINT_POS = np.array(
    [
        0.0473455, 1.22187, -2.44375, -0.0473455, 1.22187, -2.44375,
        0.0473455, 1.22187, -2.44375, -0.0473455, 1.22187, -2.44375,
    ],
    dtype=float,
)


def build_lowcmd():
    cmd = unitree_go_msg_dds__LowCmd_()
    cmd.head[0] = 0xFE
    cmd.head[1] = 0xEF
    cmd.level_flag = 0xFF
    cmd.gpio = 0
    for i in range(20):
        cmd.motor_cmd[i].mode = 0x01
        cmd.motor_cmd[i].q = 0.0
        cmd.motor_cmd[i].kp = 0.0
        cmd.motor_cmd[i].dq = 0.0
        cmd.motor_cmd[i].kd = 0.0
        cmd.motor_cmd[i].tau = 0.0
    return cmd


def base_height_and_tilt(mj_data):
    height = mj_data.qpos[2]
    w, x, y, z = mj_data.qpos[3:7]
    body_z_axis_world_z = 1 - 2 * (x * x + y * y)
    tilt = np.arccos(np.clip(body_z_axis_world_z, -1.0, 1.0))
    return height, tilt


def main():
    ChannelFactoryInitialize(DOMAIN_ID, INTERFACE)

    mj_model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    mj_model.opt.timestep = TIMESTEP
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, mj_data)
    mujoco.mj_forward(mj_model, mj_data)

    bridge = UnitreeSdk2Bridge(mj_model, mj_data)

    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()
    cmd = build_lowcmd()
    crc = CRC()

    params_array = np.load(PARAMS_PATH)
    params = CpgParams.from_array(params_array)
    print("Parámetros óptimos (CEM), validando por DDS:")
    for name, value in zip(
        ["frequency", "duty_factor", "stride_length", "ground_clearance",
         "stance_height", "kp_walk", "kd_walk"],
        params_array,
    ):
        print(f"  {name}: {value:.4f}")

    renderer = mujoco.Renderer(mj_model, height=480, width=640)
    frames = []
    steps_per_frame = max(1, round((1.0 / FPS) / TIMESTEP))

    total_seconds = STAND_UP_SECONDS + WALK_SECONDS
    n_steps = int(total_seconds / TIMESTEP)

    x_start = None
    running_time = 0.0
    for step in range(n_steps):
        running_time += TIMESTEP

        if running_time < STAND_UP_SECONDS:
            # Fase de pararse (igual que stand_go2_headless.py).
            phase = np.tanh(running_time / 1.2)
            target = phase * STAND_UP_JOINT_POS + (1 - phase) * STAND_DOWN_JOINT_POS
            kp = phase * 50.0 + (1 - phase) * 20.0
            for i in range(12):
                cmd.motor_cmd[i].q = target[i]
                cmd.motor_cmd[i].kp = kp
                cmd.motor_cmd[i].dq = 0.0
                cmd.motor_cmd[i].kd = 3.5
                cmd.motor_cmd[i].tau = 0.0
        else:
            # Fase de caminata: CPG + IK por pata.
            t_walk = running_time - STAND_UP_SECONDS
            if x_start is None:
                x_start = mj_data.qpos[0]
            for leg in LEGS:
                foot_pos = foot_target(t_walk, leg, HIP_OFFSET_Y[leg], params)
                q1, q2, q3 = inverse_kinematics_leg(leg, *foot_pos)
                base_idx = MOTOR_BASE_INDEX[leg]
                for offset, q in enumerate([q1, q2, q3]):
                    cmd.motor_cmd[base_idx + offset].q = q
                    cmd.motor_cmd[base_idx + offset].kp = params.kp_walk
                    cmd.motor_cmd[base_idx + offset].dq = 0.0
                    cmd.motor_cmd[base_idx + offset].kd = params.kd_walk
                    cmd.motor_cmd[base_idx + offset].tau = 0.0

        cmd.crc = crc.Crc(cmd)
        pub.Write(cmd)

        time.sleep(0)
        mujoco.mj_step(mj_model, mj_data)

        if step % steps_per_frame == 0:
            renderer.update_scene(mj_data)
            frames.append(renderer.render())

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    imageio.mimwrite(OUTPUT_PATH, frames, fps=FPS)

    height, tilt = base_height_and_tilt(mj_data)
    fell = height < 0.15 or tilt > np.deg2rad(45)
    forward_distance = mj_data.qpos[0] - x_start

    print(f"Video guardado en: {OUTPUT_PATH}")
    print(f"Distancia avanzada (x) durante la caminata: {forward_distance:.3f} m")
    print(f"Altura final del torso: {height:.3f} m")
    print(f"Inclinación final: {np.rad2deg(tilt):.2f} deg")
    print(f"Robot did not fall: {not fell}")


if __name__ == "__main__":
    main()
