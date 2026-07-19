"""Cinemática directa (FK) de una pata del Go2.

Geometría extraída de third_party/unitree_mujoco/unitree_robots/go2/go2.xml:
  - Cada pata tiene 3 joints en serie: abducción (eje x), hip pitch (eje y),
    rodilla/knee (eje y).
  - hip_offset_y (d): distancia entre el joint de abducción y el de hip pitch,
    +0.0955 m en patas izquierdas, -0.0955 m en patas derechas.
  - thigh_len (l1) = calf_len (l2) = 0.213 m.

Convención: la posición del pie se expresa en el frame de la cadera (el joint
de abducción), con los mismos ejes que el body base_link (x adelante,
y izquierda, z arriba), en q1=q2=q3=0.
"""
import numpy as np

THIGH_LEN = 0.213
CALF_LEN = 0.213

HIP_OFFSET_Y = {
    "FL": 0.0955,
    "FR": -0.0955,
    "RL": 0.0955,
    "RR": -0.0955,
}


def forward_kinematics(q1, q2, q3, hip_offset_y):
    """Posición (x, y, z) del pie relativa al joint de abducción de la pata.

    q1: ángulo de abducción/adducción (eje x)
    q2: ángulo de hip pitch (eje y)
    q3: ángulo de rodilla (eje y), típicamente negativo (Go2 dobla hacia atrás)
    hip_offset_y: HIP_OFFSET_Y[leg], con signo según el lado de la pata
    """
    d = hip_offset_y
    l1 = THIGH_LEN
    l2 = CALF_LEN

    s1, c1 = np.sin(q1), np.cos(q1)
    s2, c2 = np.sin(q2), np.cos(q2)
    s23, c23 = np.sin(q2 + q3), np.cos(q2 + q3)

    reach = l1 * c2 + l2 * c23  # proyección horizontal (en el plano sagital) de la pata

    x = -l1 * s2 - l2 * s23
    y = d * c1 + s1 * reach
    z = d * s1 - c1 * reach

    return np.array([x, y, z])


def forward_kinematics_leg(leg, q1, q2, q3):
    """Igual que forward_kinematics, pero recibiendo el nombre de la pata
    ("FL", "FR", "RL", "RR") en vez del offset directamente."""
    return forward_kinematics(q1, q2, q3, HIP_OFFSET_Y[leg])


def inverse_kinematics(x, y, z, hip_offset_y):
    """Cinemática inversa: dado un target (x, y, z) del pie relativo al joint
    de abducción, devuelve (q1, q2, q3).

    Se resuelve en dos partes:
      1. q1 (abducción) a partir de (y, z, d): estas dos ecuaciones no
         dependen de q2/q3, solo de "reach" = l1*cos(q2) + l2*cos(q2+q3),
         que se despeja primero como sqrt(y^2 + z^2 - d^2).
      2. q2, q3 a partir de (x, reach): es exactamente un brazo planar de 2
         eslabones (ley de cosenos), ya que l1 = l2 = 0.213 m.

    Se asume la rodilla dobla en la dirección "natural" del Go2 (q3 <= 0,
    ver rango del joint en go2.xml: [-2.7227, -0.83776]), y reach > 0
    (pata apuntando hacia abajo, no invertida sobre la cadera).
    """
    d = hip_offset_y
    l1 = THIGH_LEN
    l2 = CALF_LEN

    # --- Paso 1: reach y q1, a partir de (y, z) ---
    reach_sq = y**2 + z**2 - d**2
    if reach_sq < 0:
        raise ValueError(
            f"Target inalcanzable: y={y:.4f}, z={z:.4f}, d={d:.4f} "
            f"(y^2+z^2 < d^2, la pata no llega ni estirando la abducción)."
        )
    reach = np.sqrt(reach_sq)

    denom_yz = y**2 + z**2  # == d^2 + reach^2
    c1 = (d * y - reach * z) / denom_yz
    s1 = (reach * y + d * z) / denom_yz
    q1 = np.arctan2(s1, c1)

    # --- Paso 2: q2, q3 a partir de (x, reach), brazo planar 2R ---
    r2 = x**2 + reach**2
    cos_q3 = (r2 - l1**2 - l2**2) / (2 * l1 * l2)
    cos_q3 = np.clip(cos_q3, -1.0, 1.0)  # por seguridad ante ruido numérico
    q3 = -np.arccos(cos_q3)  # raíz negativa: coincide con la convención del Go2

    A = l1 + l2 * np.cos(q3)
    B = l2 * np.sin(q3)
    u = -x
    v = reach
    sin_q2 = (A * u - B * v) / r2
    cos_q2 = (B * u + A * v) / r2
    q2 = np.arctan2(sin_q2, cos_q2)

    return np.array([q1, q2, q3])


def inverse_kinematics_leg(leg, x, y, z):
    """Igual que inverse_kinematics, pero recibiendo el nombre de la pata."""
    return inverse_kinematics(x, y, z, HIP_OFFSET_Y[leg])
