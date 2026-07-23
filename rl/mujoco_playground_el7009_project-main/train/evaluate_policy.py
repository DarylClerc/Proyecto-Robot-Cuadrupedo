import os
import time
import argparse
from pathlib import Path

import numpy as np
import jax
import mujoco
import imageio.v2 as imageio
from brax.training.agents.ppo import checkpoint

from mujoco_playground._src.locomotion.go2 import go2_constants
from mujoco_playground._src.locomotion.go2.base import get_assets
# from mujoco_playground.experimental.sim2sim.gamepad_reader import Gamepad

xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags


class PPOController:

    def __init__(self, path_to_checkpoint: str,
                 default_angles: np.ndarray,
                 n_substeps: int,
                 action_scale: float = 0.5,
                 vel_scale_x: float = 1.5,
                 vel_scale_y: float = 0.8,
                 vel_scale_rot: float = 2 * np.pi,):

        self.policy = jax.jit(checkpoint.load_policy(path_to_checkpoint))
        self._output_names = ["continuous_actions"]

        self._action_scale = action_scale
        self._default_angles = default_angles
        self._last_action = np.zeros_like(default_angles, dtype=np.float32)

        self._counter = 0
        self._n_substeps = n_substeps

        self._optimizer_key = jax.random.key(0)

        # Comando fijo (x_vel, y_vel, yaw_vel), ver get_joy_cmd().
        self._joy_cmd = np.array([1.0, 0.0, 0.0])

    def set_joy_cmd(self, x_vel: float, y_vel: float, yaw_vel: float):
        self._joy_cmd = np.array([x_vel, y_vel, yaw_vel])

    def get_joy_cmd(self):  # La idea es que jueguen con esta función para enviar cmd al robot
        return self._joy_cmd

    def get_observation(self, model, data):

        linvel = data.sensor("local_linvel").data
        gyro = data.sensor("gyro").data
        imu_xmat = data.site_xmat[model.site("imu").id].reshape(3, 3)
        gravity = imu_xmat.T @ np.array([0, 0, -1])
        joint_angles = data.qpos[7:] - self._default_angles
        joint_velocities = data.qvel[6:]
        obs = np.hstack([
            linvel,
            gyro,
            gravity,
            joint_angles,
            joint_velocities,
            self._last_action,
            self.get_joy_cmd()])

        return {"state": obs.astype(np.float32)}

    def get_cmd(self, model, data):

        key, rng = jax.random.split(self._optimizer_key)

        self._counter += 1
        if self._counter % self._n_substeps == 0:

            obs = self.get_observation(model, data)
            pred_cmd = self.policy(obs, rng)[0]
            self._last_action = pred_cmd
            data.ctrl[:] = pred_cmd * self._action_scale + self._default_angles


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evalúa una política PPO entrenada, en modo headless."
    )
    parser.add_argument(
        "checkpoint_path", type=str,
        help="Ruta al checkpoint (carpeta numerada dentro de go2_train_logs/<exp>/)."
    )
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--x-vel", type=float, default=1.0)
    parser.add_argument("--y-vel", type=float, default=0.0)
    parser.add_argument("--yaw-vel", type=float, default=0.0)
    parser.add_argument(
        "--output", type=str, default=None,
        help="Ruta del video de salida (default: <checkpoint>/eval_<cmd>.mp4)."
    )
    parser.add_argument(
        "--camera", choices=["fixed", "tracking"], default="fixed",
        help="'fixed': camara fija, se ve el desplazamiento real (recomendado "
             "para verificar avance). 'tracking': sigue al robot, mejor para "
             "ver detalle del paso pero el fondo (grilla repetida) hace que "
             "el desplazamiento no se perciba a simple vista."
    )
    parser.add_argument(
        "--sequence", action="store_true",
        help="Ignora --x-vel/--y-vel/--yaw-vel/--seconds y en su lugar corre "
             "una secuencia de comandos que cambian cada pocos segundos "
             "(adelante, lateral, atras, rotar), para demostrar en un solo "
             "video que la MISMA politica responde a distintos comandos."
    )
    return parser.parse_args()


# Secuencia de demostracion: (x_vel, y_vel, yaw_vel, duracion_segundos).
DEMO_SEQUENCE = [
    (1.0, 0.0, 0.0, 4.0),   # adelante
    (0.0, 0.7, 0.0, 4.0),   # lateral
    (-1.0, 0.0, 0.0, 4.0),  # atras
    (0.0, 0.0, 1.0, 4.0),   # rotacion en el lugar
]


def main():
    args = parse_args()

    # Si al cargar la política ven un error de una llave con valor None,
    # eliminen en el json del checkpoint la llave "mean_kernel_init_fn"
    # (probablemente falta un parche de brax que salió a posteriori).

    ctrl_dt = 0.02
    sim_dt = 0.004
    n_substeps = int(round(ctrl_dt / sim_dt))
    checkpoint_path = Path(args.checkpoint_path).resolve()  # orbax exige ruta absoluta

    m = mujoco.MjModel.from_xml_path(
        go2_constants.FEET_ONLY_FLAT_TERRAIN_XML.as_posix(),
        assets=get_assets())
    m.opt.timestep = sim_dt
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)
    d.qpos[:] = m.keyframe("home").qpos
    mujoco.mj_forward(m, d)

    policy = PPOController(
        path_to_checkpoint=checkpoint_path.as_posix(),
        default_angles=np.array(m.keyframe("home").qpos[7:]),
        n_substeps=n_substeps,
        action_scale=0.5,
    )
    renderer = mujoco.Renderer(m, height=480, width=640)
    fps = 30
    steps_per_frame = max(1, round((1.0 / fps) / sim_dt))

    cam = mujoco.MjvCamera()
    if args.camera == "tracking":
        # Sigue al robot: bueno para ver detalle del paso, pero como el
        # fondo (grilla) es periodico, el desplazamiento neto NO se
        # percibe a simple vista (la camara se mueve exactamente con el
        # robot, como un camarografo trotando al lado).
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = m.body("trunk").id
        cam.distance = 1.4
        cam.azimuth = 120
        cam.elevation = -15

    frames = []

    if args.sequence:
        output_path = args.output or str(checkpoint_path / "eval_sequence.mp4")
        # Camara fija amplia, centrada en el punto de partida: la trayectoria
        # combinada (adelante+lateral+atras+rotar) puede volver cerca del
        # origen, asi que conviene un plano general fijo en vez de intentar
        # centrarlo en un punto medio calculado.
        if args.camera == "fixed":
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat = np.array(m.keyframe("home").qpos[:3])
            cam.distance = 9.0
            cam.azimuth = 130
            cam.elevation = -35
        for x_vel, y_vel, yaw_vel, seconds in DEMO_SEQUENCE:
            policy.set_joy_cmd(x_vel, y_vel, yaw_vel)
            print(f"  comando: x_vel={x_vel} y_vel={y_vel} yaw_vel={yaw_vel} ({seconds}s)")
            n_steps = int(seconds / sim_dt)
            for step in range(n_steps):
                policy.get_cmd(m, d)
                mujoco.mj_step(m, d)
                if step % steps_per_frame == 0:
                    renderer.update_scene(d, camera=cam)
                    frames.append(renderer.render())
    else:
        policy.set_joy_cmd(args.x_vel, args.y_vel, args.yaw_vel)
        output_path = args.output or str(
            checkpoint_path / f"eval_x{args.x_vel}_y{args.y_vel}_yaw{args.yaw_vel}.mp4"
        )
        if args.camera == "fixed":
            # Camara fija de plano general: se ve el robot desplazarse contra
            # un fondo estatico. Apuntada a la mitad del recorrido esperado
            # segun el comando, con suficiente distancia para no perderlo de
            # cuadro.
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            expected_travel = np.array(
                [args.x_vel, args.y_vel, 0.0]
            ) * min(args.seconds, 8.0)
            cam.lookat = np.array(m.keyframe("home").qpos[:3]) + expected_travel / 2
            cam.distance = max(10.0, np.linalg.norm(expected_travel) + 6.0)
            cam.azimuth = 130
            cam.elevation = -35
        n_steps = int(args.seconds / sim_dt)
        for step in range(n_steps):
            policy.get_cmd(m, d)
            mujoco.mj_step(m, d)
            if step % steps_per_frame == 0:
                renderer.update_scene(d, camera=cam)
                frames.append(renderer.render())

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    imageio.mimwrite(output_path, frames, fps=fps)
    print(f"Video guardado en: {output_path}")


if __name__ == "__main__":
    main()
