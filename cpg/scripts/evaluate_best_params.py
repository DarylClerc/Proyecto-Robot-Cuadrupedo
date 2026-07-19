"""Corre un rollout con los parámetros óptimos encontrados por CEM
(cpg/cem_best_params.npy) y guarda un video más largo para inspección visual.

Uso:
    MUJOCO_GL=egl python3 evaluate_best_params.py
"""
import os
import sys

import numpy as np
import imageio.v2 as imageio

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "cpg"))

from oscillator import CpgParams  # noqa: E402
from rollout import rollout  # noqa: E402
from cost import compute_cost  # noqa: E402

PARAMS_PATH = os.path.join(REPO_ROOT, "cpg", "cem_best_params.npy")
OUTPUT_PATH = os.path.join(REPO_ROOT, "cpg", "outputs", "walk_cpg_best_params.mp4")

WALK_SECONDS = 8.0
FPS = 30


def main():
    params_array = np.load(PARAMS_PATH)
    params = CpgParams.from_array(params_array)
    print("Parámetros óptimos (CEM):")
    for name, value in zip(
        ["frequency", "duty_factor", "stride_length", "ground_clearance", "stance_height"],
        params_array,
    ):
        print(f"  {name}: {value:.4f}")

    metrics = rollout(params, walk_seconds=WALK_SECONDS, record_video=True, video_fps=FPS)
    frames = metrics.pop("frames")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    imageio.mimwrite(OUTPUT_PATH, frames, fps=FPS)

    print(f"\nVideo guardado en: {OUTPUT_PATH}")
    print(f"Se cayó: {metrics['fell']}")
    print(f"Distancia avanzada: {metrics['forward_distance']:.3f} m")
    print(f"Deriva lateral: {metrics['lateral_drift']:.3f} m")
    print(f"Altura media del torso: {metrics['mean_height']:.3f} m")
    print(f"Inclinación media: {np.rad2deg(metrics['mean_tilt']):.2f} deg")
    print(f"Costo: {compute_cost(metrics):.4f}")


if __name__ == "__main__":
    main()
