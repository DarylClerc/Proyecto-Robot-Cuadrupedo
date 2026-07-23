# RL — Locomoción omnidireccional del Go2 (MuJoCo Playground)

Entrenamiento de una política de locomoción omnidireccional (adelante,
atrás, lateral, rotación) mediante PPO, con función de recompensa propia
diseñada sobre el código base de MuJoCo Playground. Resultados y análisis
en [RESULTADOS.md](RESULTADOS.md).

## Instalación

```bash
conda create -n robot-rl python=3.11
conda activate robot-rl
cd mujoco_playground_el7009_project-main
pip install -e ".[cuda]"
pip install "jax[cuda12]<0.10" flax optax orbax-checkpoint --upgrade
pip install imageio imageio-ffmpeg
```

`mujoco_menagerie` (assets del Go2) se descarga automáticamente la primera
vez que se carga el entorno.

Para renderizar sin ventana (necesario en servidores sin entorno gráfico):

```bash
export MUJOCO_GL=egl
```

## Cómo correr cada script

Todos se ejecutan desde `mujoco_playground_el7009_project-main/`, con el
entorno `robot-rl` activado.

**1. Entrenar una política**:
```bash
python3 train/train_go2.py --formulation A   # tracking + regularización
python3 train/train_go2.py --formulation B   # A + gait shaping (default)
```
Cada corrida crea una carpeta en `go2_train_logs/exp_<timestamp>/` con
checkpoints numerados (uno cada ~10% del entrenamiento), un gráfico de la
recompensa (`reward_curve.png`) y un video de evaluación automático
(`policy_test.mp4`).

**2. Evaluar una política entrenada** — corre el checkpoint bajo un
comando fijo o una secuencia de comandos, y guarda el video:
```bash
# comando fijo: avanzar
python3 train/evaluate_policy.py go2_train_logs/<exp>/<checkpoint> \
  --x-vel 1.0 --y-vel 0.0 --yaw-vel 0.0 --seconds 8

# secuencia de comandos que cambian cada 4s (adelante, lateral, atrás, rotar)
python3 train/evaluate_policy.py go2_train_logs/<exp>/<checkpoint> --sequence
```

Flags relevantes de `evaluate_policy.py`:
- `--camera fixed` (default): cámara fija, para verificar desplazamiento real.
- `--camera tracking`: cámara que sigue al robot, para ver detalle del paso.
- `--output <ruta>`: dónde guardar el video (default: dentro de la carpeta del checkpoint).

**Nota sobre checkpoints**: si al cargar uno aparece
`KeyError: None` en `mean_kernel_init_fn`, hay que eliminar esa llave del
`ppo_network_config.json` del checkpoint (bug conocido de compatibilidad
de versión de `brax`).

## Estructura

- `mujoco_playground_el7009_project-main/`: código base de la tutora (fork
  de MuJoCo Playground con el Go2 incorporado).
  - `mujoco_playground/_src/locomotion/go2/joystick.py`: entorno y función
    de recompensa (archivo modificado — ver `RESULTADOS.md` sección 3).
  - `train/train_go2.py`, `train/evaluate_policy.py`: scripts descritos arriba.
  - `go2_train_logs/`: checkpoints y videos generados (ignorado por git).
- `RESULTADOS.md`: proceso completo, métricas y análisis comparativo de
  las formulaciones de recompensa.
