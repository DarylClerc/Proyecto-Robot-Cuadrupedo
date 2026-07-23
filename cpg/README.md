# CPG + CEM — Caminata 2D del Go2

Controlador de marcha basado en Central Pattern Generation (CPG) en task
space, con los parámetros del oscilador ajustados mediante Cross-Entropy
Method (CEM). Resultados y análisis en [RESULTADOS.md](RESULTADOS.md).

## Instalación

```bash
conda create -n robot-cpg python=3.10
conda activate robot-cpg
pip install -r cpg/requirements.txt
cd third_party/unitree_sdk2_python && pip install -e . && cd -
```

Para renderizar sin ventana (necesario en servidores sin entorno gráfico):

```bash
export MUJOCO_GL=egl
```

## Cómo correr cada script

Todos se ejecutan desde `cpg/` o `cpg/scripts/`, con el entorno `robot-cpg`
activado.

**1. Sanity check del simulador** — carga el Go2 y lo deja caer por
gravedad, para verificar que MuJoCo + renderizado funcionan:
```bash
python3 scripts/sanity_check_sim.py
```

**2. Postura de pie** — el robot se para y se mantiene estable, usando la
interfaz DDS real (la misma que usa el robot físico):
```bash
python3 scripts/stand_go2_headless.py
```

**3. Validar cinemática** — compara la cinemática directa e inversa contra
MuJoCo:
```bash
python3 scripts/validate_fk.py
python3 scripts/validate_ik.py
```

**4. Optimizar el CPG con CEM** — ajusta los parámetros del oscilador
(frecuencia, largo de paso, altura de swing, ganancias PD, etc.) y guarda
el resultado en `cem_best_params.npy`:
```bash
python3 cem.py
```

**5. Evaluar el resultado de CEM**:
```bash
# control directo (rápido, el mismo mecanismo que usa CEM)
python3 scripts/evaluate_best_params.py

# validación cruzada vía interfaz DDS real
python3 scripts/walk_cpg_open_loop.py
```

Todos los scripts que simulan guardan un video `.mp4` en `outputs/`.

## Estructura

- `kinematics.py`: cinemática directa e inversa por pata.
- `oscillator.py`: generador de trayectoria del pie (CPG, marcha de trote).
- `rollout.py`: simulación rápida (control directo sobre MuJoCo) usada por CEM.
- `cost.py`: función de costo para CEM.
- `cem.py`: optimizador Cross-Entropy Method.
- `cem_best_params.npy`: mejores parámetros encontrados por la última corrida de CEM.
- `scripts/`: los scripts ejecutables descritos arriba.
- `outputs/`: videos generados (ignorados por git).
