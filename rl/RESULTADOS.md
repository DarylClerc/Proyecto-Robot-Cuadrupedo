# Resultados experimentales - RL (Go2, MuJoCo Playground)

Este documento reune el proceso y los resultados de la Parte 2 (RL) del
proyecto: adaptacion del codigo base, diseno de la funcion de recompensa,
y comparacion de dos formulaciones.

## 1. Adaptacion del entorno

El codigo base de la tutora (fork de MuJoCo Playground con el Go2) requirio
tres correcciones antes de poder entrenar:

1. `impl="warp"` (config por defecto) es incompatible con la version
   instalada de `mujoco-mjx`/`warp-lang` (`AttributeError` en
   `mjx.put_model`). Se cambio a `impl="jax"` (backend MJX clasico).
2. `brax==0.14.2` (ultima version en PyPI) usa `jax.device_put_replicated`,
   eliminada en versiones recientes de JAX. Se fijo `jax==0.9.2` (todavia
   la tiene, con warning de deprecacion), dejando que `pip` resolviera
   versiones compatibles de `flax`/`optax`/`orbax-checkpoint`.
3. `train_go2.py`/`evaluate_policy.py` usaban GUI (`matplotlib` interactivo,
   `mujoco.viewer.launch_passive`) - adaptados a headless (EGL, export a
   video), mismo patron que en la parte CPG.

Entorno: conda `robot-rl` (Python 3.11), GPU detectada (2x RTX 4090).

## 2. Smoke test (recompensa de ejemplo)

Antes de disenar la recompensa propia, se entreno con la recompensa base
del codigo (`tracking_lin_vel`, `tracking_ang_vel` unicamente) para validar
el pipeline completo (entrenar -> checkpoint -> evaluar -> video).

- 200M steps, ~10 min (2:20 compilar + 7:29 entrenar)
- Recompensa: 7.4 -> 24.9

Hallazgo relevante: evaluando bajo distintos comandos (dentro del rango
de entrenamiento, `command_config.a=[1.5, 0.8, 1.2]`):

| Comando | Resultado (8s) | Evaluacion |
|---|---|---|
| vx=1.0 | 9.15 m, deriva minima | Limpio |
| vx=-1.0 | -8.65 m, deriva minima | Limpio |
| vy=0.7 | 6.42 m lateral, 4.4 grados deriva angular | Bien |
| yaw=1.0 rad/s | Solo giro 167 grados en 8s (aprox 0.37 rad/s real) | Debil |
| yaw=0.5 rad/s | Giro -92.7 grados (direccion CONTRARIA al comando) | Falla |

Caminar es solido, pero el tracking de velocidad angular es notoriamente
debil con la recompensa de ejemplo - motivo el diseno de la seccion 3.

## 3. Diseno de la funcion de recompensa

Se implementaron 7 terminos nuevos en `joystick.py`, agrupados en dos
formulaciones comparables activando/desactivando terminos via
`reward_config.scales`:

Formulacion A - tracking + regularizacion estandar (Rudin et al. 2022,
"Learning to Walk in Minutes Using Massively Parallel Deep RL", CoRL):
- `tracking_lin_vel`, `tracking_ang_vel` (existentes; se separo el
  `tracking_sigma` angular del lineal, para que el error de yaw pese mas
  en la recompensa - ver hallazgo de la seccion 2)
- `ang_vel_xy`: penaliza velocidad angular de roll/pitch (evita bamboleo)
- `lin_vel_z`: penaliza velocidad vertical (evita rebote)
- `orientation`: penaliza inclinacion del torso (gravedad proyectada)
- `torques`: penaliza suma de torques al cuadrado (eficiencia energetica)
- `action_rate`: penaliza cambios bruscos entre acciones consecutivas

Formulacion B - A + gait shaping (Rudin et al. 2022; Margolis & Agrawal
2023, "Walk These Ways", CoRL):
- `feet_air_time`: premia tiempo de swing razonable al aterrizar (evita
  arrastrar los pies en vez de dar pasos)
- `feet_slip`: penaliza velocidad horizontal de la pata en contacto
  (evita deslizamiento)

### Bug encontrado y corregido: signos invertidos

Primer intento: las funciones de penalizacion (`orientation`, `lin_vel_z`,
`torques`, `action_rate`, `ang_vel_xy`, `feet_slip`) se escribieron
devolviendo directamente el costo con signo negativo, pero ademas se les
asigno un `scale` negativo en el config -- negativo por negativo da
positivo, invirtiendo el incentivo (premiaba inclinarse, rebotar,
deslizar). Se detecto porque ambas formulaciones entrenadas colapsaban de
forma consistente (caidas, giros erraticos de hasta 165 grados, alturas
anomalas) a pesar de que la metrica de recompensa reportada durante el
entrenamiento subia -- la recompensa "alta" reflejaba el incentivo
invertido, no una buena politica. Fix: los `scales` de esos terminos
deben ser positivos (el signo del efecto ya esta en la funcion).
Confirmado con un chequeo directo del breakdown de terminos antes de
reentrenar.

## 4. Comparacion A vs B (tras el fix)

Ambas formulaciones: 200M steps, ~10 min cada una (entrenadas en paralelo,
una por GPU).

| | Recompensa final (entrenamiento) | Desviacion estandar |
|---|---|---|
| A | 31.9 | +/-6.9 |
| B | 31.0 | +/-5.8 |

Evaluacion bajo comandos (8s cada uno, dentro del rango de entrenamiento):

| Comando | Formulacion A | Formulacion B |
|---|---|---|
| Adelante (vx=1.0) | dx=+9.66m, deriva y=-0.10m, yaw=+3.5 grados | dx=+9.91m, deriva y=-1.23m, yaw=-17.4 grados |
| Atras (vx=-1.0) | dx=-9.25m, deriva y=+1.35m | dx=-9.58m, deriva y=+1.19m |
| Lateral (vy=0.7) | dx=+0.81m, dy=+6.02m | dx=+0.01m, dy=+5.89m |
| Rotacion (yaw=1.0) | giro 120 grados en 8s | giro 115.6 grados en 8s |

Deslizamiento de patas (comando adelante, velocidad media de la pata
mientras esta en contacto con el piso):

| Formulacion | Deslizamiento medio |
|---|---|
| A | 0.262 m/s |
| B | 0.182 m/s (-30%) |

Torque cuadratico medio (energia, comando adelante): A=908, B=1142 -
B consume mas energia que A, consistente con dar pasos mas "activos"
(despegar mejor el pie) en vez de un paso mas plano/economico.

### Analisis

- Lateral: B es notablemente mas limpio (dx cercano a 0, movimiento
  practicamente puro en y) mientras que A deriva ~0.8m en x durante el
  mismo comando -- el termino `feet_slip` parece ayudar a mantener un
  apoyo mas preciso incluso en direcciones no ensayadas explicitamente.
- Deslizamiento: confirma cuantitativamente que `feet_slip` cumple su
  funcion (-30% de deslizamiento medio).
- Costo energetico: B paga ese beneficio con mas torque - un trade-off
  esperable y coherente con la literatura (gait shaping explicito
  generalmente sacrifica algo de eficiencia energetica por mejor calidad
  de paso).
- Rotacion: similar en ambas (120 vs 115.6 grados sobre 8s, muy por
  debajo del ideal ~458 grados) -- el ajuste de `tracking_sigma_ang`
  mejoro el comportamiento respecto a la recompensa base (ya no gira en
  direccion contraria), pero el tracking angular sigue siendo el punto
  mas debil de ambas formulaciones. Queda como trabajo futuro (podria
  necesitar mayor peso relativo, mayor rango de comando durante
  entrenamiento, o curriculum de dificultad).

## 5. Videos de referencia

Las tres carpetas (`baseline_recompensa_ejemplo/`, `formulacionA_final/`,
`formulacionB_final/`), dentro de `000206438400/`, tienen el mismo set de
6 videos, generados con los mismos comandos y misma cámara para poder
compararlas directamente:

- `01_adelante.mp4` — adelante
- `02_atras.mp4` — atrás
- `03_lateral.mp4` — lateral
- `04_rotacion.mp4` — rotación
- `05_secuencia_angulo1.mp4` / `06_secuencia_angulo2.mp4` — secuencia de
  20s (adelante → lateral → atrás → rotar, 5s cada comando) mostrando en
  un solo video que es una única política omnidireccional, no un modelo
  por dirección; grabada desde dos ángulos de cámara distintos.

Curvas de recompensa: `<carpeta>/reward_curve.png`.

**Nota**: el video de rotación de la formulación base usa `yaw=1.0`
(dentro del rango de entrenamiento), a diferencia del primer intento
reportado en la sección 2 (`yaw=1.5`, fuera de rango), para que la
comparación entre las tres sea justa.

## 6. Pendiente

- Mejorar tracking angular (ver analisis arriba).
- Documento de reporte breve (2 paginas) para la entrega intermedia.
- Decidir `.gitignore` de `rl/` (codigo de terceros y checkpoints/videos
  no deberian subirse completos a git).
