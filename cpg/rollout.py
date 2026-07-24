"""Simulador rápido para evaluar un set de parámetros del CPG.

A diferencia de stand_go2_headless.py / walk_cpg_open_loop.py (que pasan por
la interfaz DDS real, usada para validar sim-to-real), este módulo aplica el
control directamente sobre mj_data.ctrl. Es lo que usa CEM, que corre miles
de simulaciones y no puede pagar el overhead de DDS por cada una.
"""
import os

import mujoco
import numpy as np

from kinematics import inverse_kinematics_leg, HIP_OFFSET_Y
from oscillator import CpgParams, foot_target

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# scene.xml trae obstáculos tipo escalera (terreno) a partir de x=1.1m, que
# no se usan para la caminata 2D; se apartan del área de trabajo en
# _move_terrain_obstacles_away() en vez de mantener un XML paralelo.
SCENE_PATH = os.path.join(
    REPO_ROOT, "third_party/unitree_mujoco/unitree_robots/go2/scene.xml"
)

LEGS = ["FR", "FL", "RR", "RL"]

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

# Orden de las 12 articulaciones, igual al orden de <actuator> en go2.xml.
JOINT_ORDER = []
for leg in ["FR", "FL", "RR", "RL"]:
    for suffix in ["hip", "thigh", "calf"]:
        JOINT_ORDER.append(f"{leg}_{suffix}_joint")

TIMESTEP = 0.002
STAND_UP_SECONDS = 1.5
KP_STAND_END = 50.0
KD_STAND = 3.5
# kp/kd de la caminata vienen de CpgParams (params.kp_walk/kd_walk), son
# parte de lo que ajusta CEM.

HEIGHT_FALL_THRESHOLD = 0.15
TILT_FALL_THRESHOLD = np.deg2rad(45)


def _move_terrain_obstacles_away(model):
    """scene.xml trae geoms de escalera/terreno (sin nombre, hijos directos
    del worldbody) pensados para otras pruebas, empezando en x=1.1m. Para la
    caminata 2D del CPG no queremos ese terreno -- se los aparta lejos
    (colisión y visualmente) en vez de mantener un scene.xml paralelo
    (evita problemas de resolución de meshdir al mover el archivo de
    carpeta)."""
    for i in range(model.ngeom):
        geom = model.geom(i)
        if geom.name == "" and model.geom_bodyid[i] == 0:
            model.geom_pos[i, 0] += 1000.0


class Go2Sim:
    """Wrapper delgado sobre MjModel/MjData con acceso indexado a las 12
    articulaciones en un orden fijo y conocido (independiente del orden
    interno de qpos/qvel de MuJoCo)."""

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(SCENE_PATH)
        self.model.opt.timestep = TIMESTEP
        _move_terrain_obstacles_away(self.model)
        self.data = mujoco.MjData(self.model)

        self.qpos_adr = np.array(
            [self.model.joint(name).qposadr[0] for name in JOINT_ORDER]
        )
        self.dof_adr = np.array(
            [self.model.joint(name).dofadr[0] for name in JOINT_ORDER]
        )

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def joint_q(self):
        return self.data.qpos[self.qpos_adr]

    def joint_dq(self):
        return self.data.qvel[self.dof_adr]

    def apply_pd(self, q_desired, kp, kd):
        q = self.joint_q()
        dq = self.joint_dq()
        self.data.ctrl[:] = kp * (q_desired - q) + kd * (0.0 - dq)

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def base_height_and_tilt(self):
        height = self.data.qpos[2]
        w, x, y, z = self.data.qpos[3:7]
        cos_tilt = 1 - 2 * (x * x + y * y)
        tilt = np.arccos(np.clip(cos_tilt, -1.0, 1.0))
        return height, tilt


def rollout(
    params: CpgParams,
    walk_seconds=4.0,
    record_video=False,
    video_fps=30,
    camera_azimuth=120,
    camera_elevation=-15,
    camera_distance=2.0,
):
    """Corre stand-up + caminata CPG para los params dados.

    Devuelve un dict con métricas de la caminata (usadas por la función de
    costo de CEM) y, si record_video=True, también los frames renderizados.
    """
    sim = Go2Sim()
    sim.reset()

    renderer = None
    frames = None
    cam = None
    steps_per_frame = 1
    if record_video:
        renderer = mujoco.Renderer(sim.model, height=480, width=640)
        frames = []
        steps_per_frame = max(1, round((1.0 / video_fps) / TIMESTEP))
        # Cámara que sigue al robot (si no, se sale de cuadro apenas camina
        # unos pasos y el video muestra una escena vacía).
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = sim.model.body("base_link").id
        cam.distance = camera_distance
        cam.azimuth = camera_azimuth
        cam.elevation = camera_elevation

    n_stand_steps = int(STAND_UP_SECONDS / TIMESTEP)
    n_walk_steps = int(walk_seconds / TIMESTEP)

    fell = False
    fell_at_step = None

    # --- Fase de pararse ---
    running_time = 0.0
    for step in range(n_stand_steps):
        running_time += TIMESTEP
        phase = np.tanh(running_time / 1.2)
        q_desired = phase * STAND_UP_JOINT_POS + (1 - phase) * STAND_DOWN_JOINT_POS
        kp = phase * KP_STAND_END + (1 - phase) * 20.0
        sim.apply_pd(q_desired, kp, KD_STAND)
        sim.step()

        if record_video and step % steps_per_frame == 0:
            renderer.update_scene(sim.data, camera=cam)
            frames.append(renderer.render())

        height, tilt = sim.base_height_and_tilt()
        if height < HEIGHT_FALL_THRESHOLD or tilt > TILT_FALL_THRESHOLD:
            fell = True
            fell_at_step = step
            break

    # --- Fase de caminata (CPG + IK) ---
    x_start = sim.data.qpos[0]
    y_start = sim.data.qpos[1]
    heights = []
    tilts = []
    # Muestras de x cada 0.5s, para medir velocidad en ventanas finas a lo
    # largo del rollout (usado por la función de costo).
    sample_every_steps = max(1, round(0.5 / TIMESTEP))
    x_samples = [x_start]

    if not fell:
        t_walk = 0.0
        for step in range(n_walk_steps):
            if step % sample_every_steps == 0:
                x_samples.append(sim.data.qpos[0])
            q_desired = np.empty(12)
            for i, leg in enumerate(LEGS):
                foot_pos = foot_target(t_walk, leg, HIP_OFFSET_Y[leg], params)
                try:
                    q1, q2, q3 = inverse_kinematics_leg(leg, *foot_pos)
                except ValueError:
                    # Target inalcanzable (parámetros absurdos propuestos por CEM):
                    # cortamos el rollout y penalizamos como caída.
                    fell = True
                    break
                q_desired[3 * i : 3 * i + 3] = [q1, q2, q3]
            if fell:
                fell_at_step = n_stand_steps + step
                break

            sim.apply_pd(q_desired, params.kp_walk, params.kd_walk)
            sim.step()
            t_walk += TIMESTEP

            if record_video and step % steps_per_frame == 0:
                renderer.update_scene(sim.data, camera=cam)
                frames.append(renderer.render())

            height, tilt = sim.base_height_and_tilt()
            heights.append(height)
            tilts.append(tilt)

            if height < HEIGHT_FALL_THRESHOLD or tilt > TILT_FALL_THRESHOLD:
                fell = True
                fell_at_step = n_stand_steps + step
                break

    x_end = sim.data.qpos[0]
    y_end = sim.data.qpos[1]
    x_samples.append(x_end)

    sample_dt = sample_every_steps * TIMESTEP
    segment_velocities = [
        (x_samples[i + 1] - x_samples[i]) / sample_dt
        for i in range(len(x_samples) - 1)
    ]

    # Velocidad sostenida: promedio de los últimos ~3s del rollout (usada
    # como métrica principal de avance en vez de la distancia neta total).
    n_sustained_segments = max(1, round(3.0 / sample_dt)) if not fell else 0
    sustained_segments = (
        segment_velocities[-n_sustained_segments:] if n_sustained_segments else []
    )
    sustained_velocity = float(np.mean(sustained_segments)) if sustained_segments else 0.0

    metrics = {
        "fell": fell,
        "fell_at_step": fell_at_step,
        "forward_distance": x_end - x_start,
        "sustained_velocity": sustained_velocity,
        "segment_velocities": segment_velocities,
        "lateral_drift": abs(y_end - y_start),
        "mean_height": float(np.mean(heights)) if heights else 0.0,
        "mean_tilt": float(np.mean(tilts)) if tilts else float(TILT_FALL_THRESHOLD),
        "n_walk_steps_completed": len(heights),
        "n_walk_steps_target": n_walk_steps,
    }

    if record_video:
        metrics["frames"] = frames

    return metrics
