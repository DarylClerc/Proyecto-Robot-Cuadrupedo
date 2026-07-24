# Resultados experimentales — CPG + CEM (Go2)

Este documento reúne las pruebas realizadas para validar cada componente del
sistema, de forma aislada y en conjunto, con métricas cuantitativas y
evidencia en video. Corresponde a la Parte 1 (CPG) del proyecto.

## 1. Pipeline de simulación (aislado)

**Prueba**: `scripts/sanity_check_sim.py` — carga el modelo del Go2, lo deja
caer solo por gravedad (sin control) durante 3s, en modo headless (EGL).

**Objetivo**: validar que la cadena MuJoCo + renderizado offscreen + export
a video funciona en el servidor (sin GUI, sin Docker/sudo).

**Resultado**: altura final del torso 0.077m (el robot colapsa, esperado sin
actuación). Video: `outputs/01_sanity_check.mp4`.

## 2. Postura de pie, vía interfaz DDS (aislado)

**Prueba**: `scripts/stand_go2_headless.py` — adaptación headless del
ejemplo oficial `stand_go2.py`, usando la interfaz DDS real
(`LowCmd_`/`LowState_` vía `UnitreeSdk2Bridge`), la misma que expone el
robot físico.

**Métricas**:

| Métrica | Valor |
|---|---|
| Altura final del torso | 0.354 m |
| Inclinación final | 1.51° |
| ¿Se cayó? | No |

Video: `outputs/02_stand_go2_headless.mp4`.

## 3. Cinemática directa (FK), aislada

**Prueba**: `scripts/validate_fk.py` — compara la FK analítica
(`kinematics.py`) contra la posición real que calcula MuJoCo
(`mj_kinematics`), para las 4 patas, en 2 posturas (neutra y de pie).

**Resultado**:

| Postura | Error máximo (FK analítica vs MuJoCo) |
|---|---|
| Neutra (q=0) | 0.000 mm |
| De pie | 0.000 mm |

## 4. Cinemática inversa (IK), aislada

**Prueba**: `scripts/validate_ik.py` — 200 ángulos aleatorios × 4 patas
(800 muestras), dentro de rangos operativos razonables. Dos chequeos por
muestra:
1. Round-trip analítico: `FK(IK(FK(q))) == FK(q)`.
2. Contra MuJoCo: se aplican los ángulos que da la IK y se compara la
   posición real del pie contra el target pedido.

**Resultado**:

| Chequeo | Error máximo (800 muestras) |
|---|---|
| Round-trip analítico | 0.0000 mm |
| Contra MuJoCo | 0.0000 mm |

## 5. Optimización CEM

**Configuración**: población 40, elite 20% (8 individuos), 15 iteraciones,
15s de caminata simulada por evaluación, paralelizado en 16 núcleos
(~2 minutos por corrida completa).

**Parámetros optimizados**: frecuencia, duty factor, largo de paso,
altura de despegue (swing), altura nominal de stance, y las ganancias PD
de la caminata (`kp_walk`, `kd_walk`).

**Convergencia** (costo promedio de la población, por iteración):

| Iteración | Costo promedio | Mejor costo hasta esa iteración |
|---|---|---|
| 0 | +1.514 | +0.134 |
| 2 | +0.152 | +0.047 |
| 4 | +0.083 | +0.004 |
| 6 | +0.039 | −0.018 |
| 8 | +0.024 | −0.040 |
| 10 | +0.029 | −0.040 |
| 12 | +0.016 | −0.044 |
| 14 | +0.013 | **−0.067** |

**Parámetros finales encontrados**:

| Parámetro | Valor |
|---|---|
| frequency | 1.170 Hz |
| duty_factor | 0.671 |
| stride_length | 0.104 m |
| ground_clearance | 0.030 m |
| stance_height | −0.288 m |
| kp_walk | 150.1 |
| kd_walk | 4.28 |

## 6. Caminata final — control directo

**Prueba**: `scripts/evaluate_best_params.py`, 8s de caminata con los
parámetros de la sección 5, control directo sobre `mj_data.ctrl` (el mismo
mecanismo que usa CEM para evaluar).

| Métrica | Valor |
|---|---|
| Distancia avanzada | 1.196 m |
| Velocidad media | ≈0.15 m/s |
| Deriva lateral | 0.002 m |
| Altura media del torso | 0.299 m |
| Inclinación media | 0.73° |
| ¿Se cayó? | No |

Video: `outputs/05_walk_cpg_best_params.mp4`.

## 7. Caminata final — validación cruzada por DDS

**Prueba**: `scripts/walk_cpg_open_loop.py`, mismos parámetros que la
sección 6, pero a través de la interfaz DDS real (no control directo).

| Métrica | Control directo (§6) | Vía DDS |
|---|---|---|
| Distancia avanzada (8s) | 1.196 m | 1.193 m |
| Inclinación final | 0.73° (media) | 1.26° |
| ¿Se cayó? | No | No |

**Observación**: ambas vías producen prácticamente el mismo resultado
(diferencia de 3mm). En una versión anterior de `walk_cpg_open_loop.py`
había una discrepancia de ~23% (0.924 m vs 1.196 m) que se dejó como
pendiente de investigar; la causa real era que ese script cargaba
`scene.xml` completo, el cual incluye una serie de obstáculos tipo
escalera (pensados para pruebas de terreno) sin apartarlos del área de
trabajo -- a diferencia de `rollout.py`, que sí los aparta. Al aplicar el
mismo tratamiento en `walk_cpg_open_loop.py`, la distancia recorrida por
DDS pasó de 0.924 m a 1.193 m, cerrando la discrepancia casi por
completo. Esto también se confirmó con una corrida de 20s: 2.993 m por
DDS vs 2.997 m con control directo (antes, con los obstáculos presentes,
la misma prueba de 20s solo alcanzaba 0.930 m -- parecía que la caminata
"se frenaba", cuando en realidad estaba siendo estorbada por geometría
que ni siquiera debía estar en la escena).

Videos: `outputs/06_walk_cpg_dds_validation.mp4` (8s, cámara original),
`outputs/09_walk_cpg_dds_validation_largo.mp4` (20s, confirma que se
sostiene en el tiempo), `outputs/10_walk_cpg_dds_validation_lateral.mp4`
(vista de perfil).

## 8. Iteración documentada: intento previo con deriva angular

**Contexto**: una corrida anterior de CEM (sin `kp_walk`/`kd_walk` en la
búsqueda) encontró parámetros con mayor `stride_length`/frecuencia, que
avanzaban más rápido pero con ganancias PD insuficientes para seguir la
trayectoria del oscilador con precisión, generando una asimetría de
tracking entre patas izquierda/derecha que curva la trayectoria en vez de
mantenerla recta.

Video: `outputs/04_walk_cpg_fast_diagonal.mp4`.

## 9. Problemas encontrados durante el desarrollo (resumen)

1. **Servidor sin sudo**: Docker no fue viable; se usó Conda con
   dependencias instaladas vía pip (wheels precompilados).
2. **Sin GUI**: se usó renderizado headless vía EGL en vez de la ventana
   interactiva de `mujoco.viewer`.
3. **Bug en la validación de FK**: la primera versión del script de
   validación medía la posición del pie en el frame rotado de la propia
   cadera, anulando el efecto de la abducción que se quería medir.
4. **Discontinuidad de velocidad en la trayectoria del pie**: la transición
   stance→swing tenía un salto de velocidad (stance lineal empalmando con
   swing coseno), que generaba una oscilación de cabeceo creciente en el
   tiempo. Se corrigió usando un perfil coseno (velocidad nula en los
   extremos) en ambas fases.
5. **Función de costo sobreajustada al horizonte de evaluación**: al premiar
   la distancia neta total, CEM encontraba marchas que avanzaban rápido y
   luego se detenían. Se cambió la métrica principal a "velocidad sostenida"
   (promedio de los últimos ~3s del rollout).
6. **Ganancias PD insuficientes**: con `kp`/`kd` fijos a mano, el control no
   lograba seguir la trayectoria del swing a tiempo. Se agregaron como
   parámetros ajustables por CEM.
