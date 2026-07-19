# CPG + CEM — Caminata 2D del Go2

## Entorno

Este servidor no tiene permisos de administrador (sudo), por lo que en vez de Docker
se usa un entorno Conda con dependencias instaladas vía pip (todas con wheels
prebuilt, sin compilar nada en C++).

```bash
conda create -n robot-cpg python=3.10
conda activate robot-cpg
pip install -r cpg/requirements.txt
cd third_party/unitree_sdk2_python && pip install -e . && cd -
```

## Renderizado headless (sin GUI)

El servidor no tiene servidor gráfico (X11), por lo que no se puede usar
`mujoco.viewer.launch_passive` (la ventana interactiva que usa `unitree_mujoco`
por defecto). En su lugar, se usa renderizado offscreen vía EGL:

```bash
export MUJOCO_GL=egl
```

Los scripts en `cpg/scripts/` renderizan a video (`.mp4`) en vez de abrir una
ventana. Esto además es lo que conviene para CEM, que corre miles de
simulaciones y no necesita visualización en vivo.

## Estructura

- `kinematics.py`: cinemática directa e inversa por pata.
- `oscillator.py`: generador de trayectoria del pie (CPG, marcha de trote).
- `rollout.py`: simulación rápida (control directo sobre MuJoCo) usada por CEM.
- `cost.py`: función de costo para CEM.
- `cem.py`: optimizador Cross-Entropy Method.
- `cem_best_params.npy`: mejores parámetros encontrados por la última corrida de CEM.
- `scripts/sanity_check_sim.py`: sanity check del pipeline MuJoCo + EGL (caída libre).
- `scripts/stand_go2_headless.py`: postura de pie, vía interfaz DDS.
- `scripts/validate_fk.py`, `scripts/validate_ik.py`: validación de cinemática contra MuJoCo.
- `scripts/evaluate_best_params.py`: evalúa `cem_best_params.npy` con control directo.
- `scripts/walk_cpg_open_loop.py`: evalúa `cem_best_params.npy` vía interfaz DDS.
- `outputs/`: videos generados (ignorados por git).
