# Caminata de un Robot Cuadrúpedo Unitree Go2

## Descripción general

Este repositorio contiene el desarrollo del proyecto "Caminata de un robot cuadrúpedo", cuyo objetivo es implementar y evaluar estrategias de locomoción para el robot Unitree Go2 utilizando el simulador MuJoCo.

El proyecto considera dos enfoques independientes para abordar el problema de locomoción:

1. **Central Pattern Generation (CPG) + Cross-Entropy Method (CEM)**
   Enfoque basado en osciladores para generar patrones periódicos de movimiento en las patas del robot. En esta parte se busca obtener una caminata 2D, principalmente hacia adelante.

2. **Reinforcement Learning (RL)**
   Enfoque basado en aprendizaje por refuerzo. En esta parte se busca diseñar una función de recompensa que permita entrenar una política de locomoción omnidireccional, es decir, capaz de desplazarse hacia adelante, hacia atrás, lateralmente y realizar rotaciones sobre su eje.

El trabajo se desarrolla en simulación, utilizando MuJoCo como entorno principal. De forma opcional, si se obtiene una locomoción suficientemente estable, el controlador o política podría ser desplegado en la plataforma real Unitree Go2.

## Objetivo del proyecto

El objetivo principal es desarrollar controladores o políticas que permitan al robot Unitree Go2 desplazarse de manera estable en simulación.

De forma más específica, se busca:

- Levantar correctamente el entorno de simulación en MuJoCo.
- Utilizar los repositorios oficiales de Unitree para simular y controlar el robot Go2.
- Implementar un controlador basado en CPG para generar una caminata 2D.
- Optimizar los parámetros del CPG utilizando Cross-Entropy Method.
- Diseñar y comparar funciones de recompensa para locomoción mediante Reinforcement Learning.
- Evaluar el desempeño de los enfoques implementados considerando estabilidad, avance, suavidad del movimiento y capacidad de seguimiento de comandos.

## Plataforma utilizada

El robot considerado en este proyecto es el Unitree Go2, un robot cuadrúpedo que posee 12 grados de libertad principales, distribuidos en 3 articulaciones por cada pata.

La interfaz de control permite enviar comandos de posición articular, por lo que para el enfoque basado en CPG será necesario convertir posiciones cartesianas deseadas de las patas a posiciones articulares mediante cinemática inversa.

## Entorno de trabajo

Se recomienda inicialmente usar Docker sobre Ubuntu 22.04 para evitar problemas de compatibilidad. En el servidor de trabajo de este proyecto no hay permisos de administrador (sudo), por lo que Docker no es viable ahí; en su lugar se usa un entorno **Conda** (`robot-cpg`, Python 3.10) con dependencias instaladas vía `pip` (wheels precompilados, sin compilar nada en C++). Detalles de instalación en [cpg/README.md](cpg/README.md).

Componentes:
- MuJoCo (renderizado headless vía EGL, el servidor no tiene entorno gráfico)
- MuJoCo Playground (para la parte de RL)
- Python
- CUDA, para el entrenamiento de RL (2x RTX 4090 disponibles)
- Repositorios oficiales de Unitree, incluidos como submódulos en `third_party/`:
  - [unitree_mujoco](third_party/unitree_mujoco)
  - [unitree_sdk2_python](third_party/unitree_sdk2_python) (versión Python del SDK, no la de C++, para evitar dependencias que requieren sudo)

## Referencias principales

Las referencias consideradas inicialmente para el desarrollo del proyecto son:

- de Boer, P. T., Kroese, D. P., Mannor, S., & Rubinstein, R. Y. *A Tutorial on the Cross-Entropy Method*. Annals of Operations Research, 134, 19–67, 2005.
- Pinneri, C., Sawant, S., Blaes, S., Achterhold, J., Stueckler, J., Rolinek, M., & Martius, G. *Sample-efficient Cross-Entropy Method for Real-time Planning*. Conference on Robot Learning, 2020.
- Suzuki, S., Matayoshi, K., Hayashibe, M., & Owaki, D. *Foot trajectory as a key factor for diverse gait patterns in quadruped robot locomotion*. Scientific Reports, 15, 2025.
- Zakka, K., Tabanpour, B., Liao, Q., Haiderbhai, M., Holt, S., Luo, J., Allshire, A., Frey, E., Sreenath, K., Kahrs, L. A., Sferrazza, C., Tassa, Y., & Abbeel, P. *MuJoCo Playground*. arXiv:2502.08844, 2025.

## Estado del proyecto

**Estado actual:** Parte 1 (CPG + CEM) implementada y validada. Código en [cpg/](cpg/):

- Cinemática directa e inversa por pata, derivadas analíticamente y validadas contra MuJoCo (error 0.000mm).
- Postura de pie validada usando la interfaz DDS real (`LowCmd_`/`LowState_`, la misma que expone el robot físico).
- Oscilador CPG en task space (trayectoria swing/stance por pata, marcha de trote).
- Optimización de los parámetros del oscilador con Cross-Entropy Method.
- Caminata 2D hacia adelante resultante: estable, sin caídas, deriva lateral e inclinación mínimas, validada tanto con control directo como por la interfaz DDS.

Parte 2 (RL): pendiente de iniciar.
