# RL — Locomoción omnidireccional del Go2 (MuJoCo Playground)

## Entorno

```bash
conda create -n robot-rl python=3.11
conda activate robot-rl
cd mujoco_playground_el7009_project-main
pip install -e ".[cuda]"
pip install "jax[cuda12]<0.10" flax optax orbax-checkpoint --upgrade
pip install imageio imageio-ffmpeg
```

`mujoco_menagerie` (assets del Go2) se descarga automáticamente la primera
vez que se carga el entorno (`mujoco_playground/external_deps/`, ~2GB,
ignorado por git).

## Renderizado headless

Igual que en `cpg/`: sin GUI en el servidor, se usa EGL.

```bash
export MUJOCO_GL=egl
```

## Estructura

- `mujoco_playground_el7009_project-main/`: código base de la tutora (fork
  de MuJoCo Playground con el Go2 incorporado).
  - `mujoco_playground/_src/locomotion/go2/joystick.py`: entorno y función
    de recompensa (archivo modificado — ver `RESULTADOS.md` sección 3).
  - `train/train_go2.py`: entrenamiento (PPO). Acepta `--formulation A|B`.
  - `train/evaluate_policy.py`: evaluación headless de una política
    entrenada, con comando fijo (`--x-vel/--y-vel/--yaw-vel`) o secuencia
    de comandos (`--sequence`).
  - `go2_train_logs/`: checkpoints y videos generados (ignorado por git).
- `RESULTADOS.md`: proceso completo, métricas y análisis comparativo de
  las formulaciones de recompensa.

## Uso rápido

```bash
cd mujoco_playground_el7009_project-main
MUJOCO_GL=egl python3 train/train_go2.py --formulation B
MUJOCO_GL=egl python3 train/evaluate_policy.py go2_train_logs/<exp>/<checkpoint> --sequence
```
