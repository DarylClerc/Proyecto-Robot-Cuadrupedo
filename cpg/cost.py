"""Función de costo para CEM.

Diseño (ver enunciado): "recompensar el avance del robot y el mantenimiento
de una posición estable de su centro de masa, penalizando fuertemente las
caídas".

cost = -sustained_velocity                       (avanzar SOSTENIDO = costo más bajo)
       + w_lateral * lateral_drift                (penaliza desviarse de línea recta)
       + w_tilt * mean_tilt                       (penaliza inclinación del torso)
       + w_height * |mean_height - target_height| (penaliza agacharse/pararse de más)
       + w_decel * max(0, decel)                  (penaliza cualquier frenazo, no solo al final)
       + FALL_PENALTY si se cae                   (dominante, para evitar caídas)

Historia de este diseño (vale la pena documentarlo, es parte real del
proceso de ajuste):

Primera versión: cost = -forward_distance (distancia neta total del rollout).
CEM encontraba sistemáticamente parámetros que avanzaban muy bien pero se
detenían por completo bastante antes de terminar la ventana de evaluación
(caminaba rápido los primeros segundos, después quedaba "pedaleando" en el
mismo lugar). La razón: una ráfaga inicial agresiva seguida de quedarse
plantado todavía puede rendir más distancia NETA acumulada que una marcha
lenta pero sostenida durante toda la ventana -- la distancia total no
distingue "avancé todo el tiempo" de "avancé rápido un rato y después nada".

Intentos intermedios de arreglarlo con penalizaciones de "frenazo" (mitades,
cuartos, mínimo de ventanas de 0.5s) todos relativos al resto del MISMO
rollout no bastaron por la misma razón de fondo: si la mayor parte de la
ventana ya está estancada, el "típico"/mediana de esa ventana también es
bajo, y la brecha contra el mínimo se ve pequeña aunque el robot esté
literalmente detenido la mitad del tiempo.

Fix real: cambiar la métrica principal. En vez de premiar la distancia
NETA total, se premia `sustained_velocity`: el promedio de velocidad
durante los ÚLTIMOS ~3 segundos del rollout (ver rollout.py). Una marcha
que se detiene en cualquier punto simplemente no puede tener buen puntaje
en este término, sin importar qué tan bien arrancó -- el criterio ya no es
"cuánto avanzaste en total" sino "qué tan rápido seguís yendo al final".
El término w_decel se mantiene como penalización adicional, para desalentar
cualquier frenazo aunque no llegue a manifestarse en los últimos 3s.

Nota: esta es una formulación razonable, no la única posible. Parte del
análisis pedido en el enunciado es comparar esta función con al menos una
alternativa (ver cpg/scripts/compare_cost_functions.py más adelante).
"""
import numpy as np

TARGET_HEIGHT = 0.30  # m, altura nominal deseada del torso durante la caminata

W_LATERAL = 3.0
W_TILT = 2.0
W_HEIGHT = 3.0
W_DECEL = 4.0
FALL_PENALTY = 5.0

# Penalización adicional si se cae temprano: caídas más tempranas son peores
# (indican inestabilidad casi inmediata), así CEM no trata todas las caídas
# como equivalentes.
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

    # Se descarta el primer segmento (0-0.5s de caminata): es la transición
    # desde la postura de pie, naturalmente más lenta, no un frenazo real.
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
        # completion en [0,1); mientras más temprano cae, más se penaliza.
        cost += EARLY_FALL_WEIGHT * (1.0 - completion)

    return cost
