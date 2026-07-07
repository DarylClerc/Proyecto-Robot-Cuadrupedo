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

Para asegurar reproducibilidad y evitar problemas de compatibilidad, se recomienda utilizar un entorno basado en:

- Docker
- Ubuntu 22.04
- MuJoCo
- MuJoCo Playground
- Python
- CUDA, en caso de utilizar GPU para entrenamiento
- Repositorios oficiales de Unitree:
  - unitree_mujoco
  - unitree_sdk

El uso de Docker permite mantener un entorno controlado y facilita la instalación de dependencias necesarias para la simulación y entrenamiento.

## Referencias principales

Las referencias consideradas inicialmente para el desarrollo del proyecto son:

- de Boer, P. T., Kroese, D. P., Mannor, S., & Rubinstein, R. Y. *A Tutorial on the Cross-Entropy Method*. Annals of Operations Research, 134, 19–67, 2005.
- Pinneri, C., Sawant, S., Blaes, S., Achterhold, J., Stueckler, J., Rolinek, M., & Martius, G. *Sample-efficient Cross-Entropy Method for Real-time Planning*. Conference on Robot Learning, 2020.
- Suzuki, S., Matayoshi, K., Hayashibe, M., & Owaki, D. *Foot trajectory as a key factor for diverse gait patterns in quadruped robot locomotion*. Scientific Reports, 15, 2025.
- Zakka, K., Tabanpour, B., Liao, Q., Haiderbhai, M., Holt, S., Luo, J., Allshire, A., Frey, E., Sreenath, K., Kahrs, L. A., Sferrazza, C., Tassa, Y., & Abbeel, P. *MuJoCo Playground*. arXiv:2502.08844, 2025.

## Estado del proyecto

**Estado actual:** en etapa de planificación. Aún no se ha implementado código; este documento describe el problema, los enfoques propuestos y la metodología planeada.
