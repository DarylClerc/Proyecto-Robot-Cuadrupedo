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

- `scripts/sanity_check_sim.py`: carga el Go2, simula unos segundos sin control
  (solo gravedad) y guarda un video. Sirve para validar que el pipeline
  MuJoCo + EGL funciona.
- `outputs/`: videos generados (ignorados por git).
