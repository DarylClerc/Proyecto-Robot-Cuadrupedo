"""Función de costo para CEM.

Premia la velocidad sostenida (últimos ~3s del rollout, no la distancia neta
total: una ráfaga inicial seguida de detenerse puede rendir más distancia
neta que una marcha lenta pero sostenida). Penaliza deriva lateral,
inclinación, desviación de altura, frenazos y caídas.
"""
import numpy as np

TARGET_HEIGHT = 0.30  # m, altura nominal deseada del torso durante la caminata

W_LATERAL = 3.0
W_TILT = 2.0
W_HEIGHT = 3.0
W_DECEL = 4.0
FALL_PENALTY = 5.0

# Caídas más tempranas se penalizan más (indican inestabilidad casi inmediata).
EARLY_FALL_WEIGHT = 2.0


def compute_cost(metrics):
    sustained = metrics["sustained_velocity"]
    lateral = metrics["lateral_drift"]
    height = metrics["mean_height"]
    tilt = metrics["mean_tilt"]

    cost = -sustained
    cost += W_LATERAL * lateral
    cost += W_TILT * tilt
    cost += W_HEIGHT * abs(height - TARGET_HEIGHT)

    # Primer segmento (0-0.5s) excluido: transición desde la postura de pie,
    # naturalmente más lenta.
    seg_vel = metrics["segment_velocities"][1:]
    if seg_vel:
        typical_vel = float(np.median(seg_vel))
        min_vel = float(np.min(seg_vel))
        decel = typical_vel - min_vel
        cost += W_DECEL * max(0.0, decel)

    if metrics["fell"]:
        cost += FALL_PENALTY
        completion = metrics["n_walk_steps_completed"] / max(
            1, metrics["n_walk_steps_target"]
        )
        cost += EARLY_FALL_WEIGHT * (1.0 - completion)

    return cost
