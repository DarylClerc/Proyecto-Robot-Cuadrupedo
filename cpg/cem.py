"""Cross-Entropy Method (CEM) para ajustar los parámetros del CPG.

Referencia: de Boer et al. (2005), "A Tutorial on the Cross-Entropy Method";
Pinneri et al. (2020), iCEM ("Sample-efficient Cross-Entropy Method for
Real-time Planning") - de ahí el piso mínimo de desviación estándar, para no
colapsar la búsqueda prematuramente.

Vector de parámetros optimizado: [frequency, duty_factor, stride_length,
ground_clearance, stance_height, kp_walk, kd_walk] (ver
cpg/oscillator.py::CpgParams). Las ganancias PD de la caminata se incluyen
en la búsqueda en vez de dejarlas fijas.

Cada rollout es independiente, así que la evaluación de la población se
paraleliza con multiprocessing.
"""
import time
from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np

from oscillator import CpgParams
from rollout import rollout
from cost import compute_cost

PARAM_NAMES = [
    "frequency", "duty_factor", "stride_length", "ground_clearance",
    "stance_height", "kp_walk", "kd_walk",
]

# (min, max) de cada parámetro. stride_length permite valores negativos,
# para que CEM pueda corregir el signo de la trayectoria si hiciera falta.
BOUNDS_LOW = np.array([0.5, 0.3, -0.20, 0.01, -0.40, 20.0, 1.0])
BOUNDS_HIGH = np.array([3.0, 0.8, 0.20, 0.10, -0.15, 250.0, 15.0])

INIT_MEAN = np.array([1.5, 0.5, 0.05, 0.05, -0.30, 80.0, 5.0])
INIT_STD = np.array([0.5, 0.1, 0.08, 0.03, 0.06, 40.0, 3.0])
STD_FLOOR = np.array([0.05, 0.02, 0.01, 0.005, 0.01, 5.0, 0.5])

# Con margen sobre la duración que se quiere garantizar en la evaluación
# final: el rollout de entrenamiento no puede penalizar fallas posteriores
# a walk_seconds.
WALK_SECONDS = 15.0


def _evaluate(params_array):
    params = CpgParams.from_array(params_array)
    metrics = rollout(params, walk_seconds=WALK_SECONDS, record_video=False)
    cost = compute_cost(metrics)
    return cost, metrics


@dataclass
class CemResult:
    best_params: np.ndarray
    best_cost: float
    history: list  # lista de (iteration, mean_cost, best_cost_so_far)


def run_cem(
    population_size=40,
    elite_frac=0.2,
    n_iterations=15,
    n_workers=16,
    seed=0,
    verbose=True,
):
    rng = np.random.default_rng(seed)
    mean = INIT_MEAN.copy()
    std = INIT_STD.copy()
    n_elite = max(2, int(population_size * elite_frac))

    best_params = mean.copy()
    best_cost = np.inf
    history = []

    with Pool(n_workers) as pool:
        for it in range(n_iterations):
            t0 = time.time()

            samples = rng.normal(mean, std, size=(population_size, len(mean)))
            samples = np.clip(samples, BOUNDS_LOW, BOUNDS_HIGH)

            results = pool.map(_evaluate, samples)
            costs = np.array([c for c, _ in results])

            elite_idx = np.argsort(costs)[:n_elite]
            elite_samples = samples[elite_idx]

            mean = elite_samples.mean(axis=0)
            std = elite_samples.std(axis=0)
            std = np.maximum(std, STD_FLOOR)

            iter_best_idx = np.argmin(costs)
            if costs[iter_best_idx] < best_cost:
                best_cost = costs[iter_best_idx]
                best_params = samples[iter_best_idx].copy()

            history.append((it, float(costs.mean()), float(best_cost)))

            if verbose:
                dt = time.time() - t0
                print(
                    f"iter {it:2d}  mean_cost={costs.mean():+.3f}  "
                    f"best_cost={best_cost:+.3f}  ({dt:.1f}s)"
                )
                print(f"           mean_params={np.round(mean, 4)}")

    return CemResult(best_params=best_params, best_cost=best_cost, history=history)


if __name__ == "__main__":
    result = run_cem()
    best = CpgParams.from_array(result.best_params)
    print("\n=== Mejor resultado ===")
    for name, value in zip(PARAM_NAMES, result.best_params):
        print(f"{name}: {value:.4f}")
    print(f"costo: {result.best_cost:.4f}")

    np.save("cem_best_params.npy", result.best_params)
    print("\nParámetros guardados en cem_best_params.npy")
