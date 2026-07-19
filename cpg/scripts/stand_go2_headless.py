"""El Go2 se para y se mantiene estable, en modo headless.

Adaptación de third_party/unitree_mujoco/example/python/stand_go2.py:
  - Sin ventana interactiva (mujoco.viewer.launch_passive) -> renderiza a video (EGL).
  - Simulador (bridge DDS) y controlador corren en el mismo proceso/loop, sin
    depender de tiempo real de wall-clock.

Usa la interfaz DDS (LowCmd_/LowState_ vía UnitreeSdk2Bridge) que expone el
robot real.

Uso:
    MUJOCO_GL=egl python3 stand_go2_headless.py
"""
import os
import sys
import time

import mujoco
import numpy as np
import imageio.v2 as imageio

# third_party/unitree_mujoco/simulate_python contiene unitree_sdk2py_bridge.py
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SIMULATE_PYTHON_DIR = os.path.join(
    REPO_ROOT, "third_party/unitree_mujoco/simulate_python"
)
sys.path.insert(0, SIMULATE_PYTHON_DIR)

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
    os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "stand_go2_headless.mp4"
)

DOMAIN_ID = 1  # dominio de simulación (0 se reserva para el robot real)
INTERFACE = "lo"  # loopback, todo corre en el mismo proceso

TIMESTEP = 0.002  # paso de física, igual al dt de control del ejemplo original
STAND_UP_SECONDS = 3.0  # tiempo de transición de acostado a parado
HOLD_SECONDS = 3.0  # tiempo manteniéndose de pie, para evaluar estabilidad
FPS = 30

# Posturas articulares (12 valores, 3 por pata) tomadas del ejemplo oficial.
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
    """Altura del torso (z) e inclinación (0 = perfectamente vertical)."""
    height = mj_data.qpos[2]
    quat = mj_data.qpos[3:7]  # (w, x, y, z)
    # Eje z del cuerpo en coordenadas globales, a partir del quaternion.
    w, x, y, z = quat
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

    bridge = UnitreeSdk2Bridge(mj_model, mj_data)  # publica LowState, aplica LowCmd

    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()
    cmd = build_lowcmd()
    crc = CRC()

    renderer = mujoco.Renderer(mj_model, height=480, width=640)
    frames = []
    steps_per_frame = max(1, round((1.0 / FPS) / TIMESTEP))

    total_seconds = STAND_UP_SECONDS + HOLD_SECONDS
    n_steps = int(total_seconds / TIMESTEP)

    running_time = 0.0
    for step in range(n_steps):
        running_time += TIMESTEP

        if running_time < STAND_UP_SECONDS:
            phase = np.tanh(running_time / 1.2)
            target = phase * STAND_UP_JOINT_POS + (1 - phase) * STAND_DOWN_JOINT_POS
            kp = phase * 50.0 + (1 - phase) * 20.0
        else:
            target = STAND_UP_JOINT_POS
            kp = 50.0

        for i in range(12):
            cmd.motor_cmd[i].q = target[i]
            cmd.motor_cmd[i].kp = kp
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].kd = 3.5
            cmd.motor_cmd[i].tau = 0.0
        cmd.crc = crc.Crc(cmd)
        pub.Write(cmd)

        # Pequeña espera para que el subscriptor DDS (mismo proceso) alcance a
        # procesar el mensaje antes de aplicar el siguiente paso de física.
        time.sleep(0)
        mujoco.mj_step(mj_model, mj_data)

        if step % steps_per_frame == 0:
            renderer.update_scene(mj_data)
            frames.append(renderer.render())

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    imageio.mimwrite(OUTPUT_PATH, frames, fps=FPS)

    height, tilt = base_height_and_tilt(mj_data)
    fell = height < 0.15 or tilt > np.deg2rad(45)

    print(f"Video guardado en: {OUTPUT_PATH}")
    print(f"Final base height: {height:.3f} m")
    print(f"Final base tilt: {np.rad2deg(tilt):.2f} deg")
    print(f"Robot did not fall: {not fell}")


if __name__ == "__main__":
    main()
