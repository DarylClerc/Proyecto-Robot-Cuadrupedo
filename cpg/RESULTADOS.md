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
actuación). Video: `outputs/sanity_check.mp4`.

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

Video: `outputs/stand_go2_headless.mp4`.

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

Video: `outputs/walk_cpg_best_params.mp4`.

## 7. Caminata final — validación cruzada por DDS

**Prueba**: `scripts/walk_cpg_open_loop.py`, mismos parámetros que la
sección 6, pero a través de la interfaz DDS real (no control directo).

| Métrica | Control directo (§6) | Vía DDS |
|---|---|---|
| Distancia avanzada (8s) | 1.196 m | 0.924 m |
| Inclinación final | 0.73° (media) | 2.18° |
| ¿Se cayó? | No | No |

**Observación**: ambas vías producen una caminata estable y sin caídas,
pero con una diferencia de magnitud (~23%) no despreciable entre el
resultado por control directo y por DDS. En una corrida anterior (con otros
parámetros) la diferencia había sido de ~1mm; esta vez es mayor. Posibles
causas a investigar: diferencias menores en la duración de la fase de
pararse entre ambos scripts (1.5s vs 2.0s), o en cómo cada camino lee el
estado articular (`mj_data.qpos` directo vs `mj_data.sensordata` a través
del bridge DDS). No invalida el resultado (ambas vías caminan establemente)
pero es una discrepancia cuantitativa a documentar y resolver antes de
considerar el sistema completamente cerrado.

Video: `outputs/walk_cpg_dds_validation.mp4`.

## 8. Iteración documentada: intento previo con deriva angular

**Contexto**: una corrida anterior de CEM (sin `kp_walk`/`kd_walk` en la
búsqueda) encontró parámetros con mayor `stride_length`/frecuencia, que
avanzaban más rápido pero con ganancias PD insuficientes para seguir la
trayectoria del oscilador con precisión, generando una asimetría de
tracking entre patas izquierda/derecha que curva la trayectoria en vez de
mantenerla recta.

Video: `outputs/walk_cpg_fast_diagonal.mp4`.

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
