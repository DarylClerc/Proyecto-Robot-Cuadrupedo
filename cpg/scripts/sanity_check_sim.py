"""Sanity check: carga el modelo Go2 en MuJoCo, simula sin control (solo gravedad)
y guarda un video, para verificar que el pipeline headless (EGL) funciona en el servidor.

Uso:
    MUJOCO_GL=egl python3 sanity_check_sim.py
"""
import os

import mujoco
import numpy as np
import imageio.v2 as imageio

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCENE_PATH = os.path.join(
    REPO_ROOT, "third_party/unitree_mujoco/unitree_robots/go2/scene.xml"
)
OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "sanity_check.mp4"
)

SIM_SECONDS = 3.0
FPS = 30


def main():
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)

    mujoco.mj_resetData(model, data)
    # Levantar un poco al robot para ver cómo cae y se asienta.
    data.qpos[2] += 0.15
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=480, width=640)

    frames = []
    steps_per_frame = max(1, int(1.0 / FPS / model.opt.timestep))
    n_frames = int(SIM_SECONDS * FPS)

    for _ in range(n_frames):
        for _ in range(steps_per_frame):
            mujoco.mj_step(model, data)
        renderer.update_scene(data)
        frames.append(renderer.render())

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    imageio.mimwrite(OUTPUT_PATH, frames, fps=FPS)
    print(f"OK: video guardado en {OUTPUT_PATH}")
    print(f"Altura final del torso (z): {data.qpos[2]:.3f} m")


if __name__ == "__main__":
    main()
