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

## Estructura del repositorio

La estructura inicial propuesta para el proyecto es la siguiente:

```
cuadruped-go2/
├── README.md
├── docs/
│   ├── reporte_intermedio.pdf
│   └── referencias.md
├── videos/
│   ├── simulacion_inicial.mp4
│   └── pruebas_fallidas/
├── setup/
│   ├── docker/
│   └── instrucciones_instalacion.md
├── cpg/
│   ├── oscillators/
│   ├── inverse_kinematics/
│   ├── cem/
│   └── experiments/
├── rl/
│   ├── rewards/
│   ├── configs/
│   ├── checkpoints/
│   └── experiments/
└── results/
    ├── plots/
    ├── logs/
    └── evaluations/
```

Esta estructura puede modificarse durante el desarrollo según las necesidades del proyecto.

## Avance actual

Hasta la fecha, se han realizado las siguientes tareas iniciales:

- Revisión del enunciado del proyecto.
- Separación del trabajo en dos enfoques principales: CPG + CEM y RL.
- Identificación de las herramientas principales del proyecto.
- Revisión preliminar de MuJoCo Playground.
- Revisión de los repositorios oficiales de Unitree.
- Identificación del script `stand_go2.py` como punto de partida para comprender el envío de comandos articulares al robot.
- Identificación de `joystick.py` como archivo relevante para la modificación de funciones de recompensa en RL.
- Definición preliminar del pipeline de trabajo para CPG.
- Definición preliminar del pipeline de trabajo para RL.
- Preparación del repositorio para documentar avances, errores, videos y resultados.

## Enfoque 1: CPG + CEM

### Descripción

El primer enfoque consiste en implementar una caminata 2D hacia adelante mediante Central Pattern Generation.

Un CPG permite generar señales periódicas similares a patrones biológicos de locomoción. En este proyecto, se utilizará un oscilador por cada pata del robot. Cada oscilador generará una trayectoria deseada del pie en el espacio cartesiano.

El robot no recibe directamente posiciones cartesianas de los pies, sino posiciones articulares. Por esta razón, será necesario implementar la cinemática inversa de cada pierna para transformar una posición deseada del pie en coordenadas (x, y, z) a los ángulos articulares correspondientes.

### Pipeline propuesto

El flujo de trabajo para la parte CPG será:

```
Oscilador por pata
→ trayectoria cartesiana del pie en XYZ
→ cinemática inversa
→ posiciones articulares
→ envío de comandos al robot
→ simulación en MuJoCo
→ evaluación de desempeño
→ optimización con CEM
```

### Parámetros preliminares del CPG

Algunos parámetros que podrían ser ajustados mediante CEM son:

- Frecuencia del oscilador.
- Amplitud del paso.
- Altura máxima del pie durante la fase de swing.
- Largo del paso.
- Posición base del pie.
- Desfase entre patas.
- Duración relativa de la fase de apoyo y fase de swing.

### Función de costo preliminar

Para optimizar la caminata mediante CEM, se propone una función de costo que considere avance, estabilidad y penalización de caídas.

Una formulación preliminar es:

```
Costo = - avance hacia adelante
        + penalización por caída
        + penalización por inclinación excesiva del torso
        + penalización por altura inestable del centro de masa
        + penalización por movimientos bruscos
        + penalización por comandos articulares extremos
```

El objetivo de esta función es favorecer trayectorias que permitan al robot avanzar de forma estable, evitando caídas o movimientos poco naturales.

## Enfoque 2: Reinforcement Learning

### Descripción

El segundo enfoque consiste en utilizar Reinforcement Learning para entrenar una política de locomoción omnidireccional.

A diferencia del enfoque CPG, en esta parte no se debe formular completamente el problema de RL desde cero. El proyecto entrega un código base basado en MuJoCo Playground, por lo que el foco principal estará en el diseño y análisis de la función de recompensa.

La política entrenada deberá permitir desplazamientos:

- Hacia adelante.
- Hacia atrás.
- Lateralmente.
- Con rotaciones sobre el eje vertical del robot.

### Archivo principal a modificar

El archivo principal asociado al diseño de recompensa es:

```
mujoco_playground/_src/locomotion/go2/joystick.py
```

La idea es agregar nuevas funciones de recompensa manteniendo la estructura general del código base.

### Pipeline propuesto

El flujo de trabajo para la parte RL será:

```
Revisión del código base
→ diseño de función de recompensa
→ entrenamiento con train_go2.py
→ revisión de curvas de recompensa
→ selección de checkpoints
→ evaluación con evaluate_policy.py
→ comparación entre funciones de recompensa
```

### Recompensa preliminar

Una primera formulación de recompensa podría considerar:

```
Reward = seguimiento de velocidad deseada
         - penalización por caída
         - penalización por inclinación del torso
         - penalización por alto uso de torque
         - penalización por cambios bruscos en las acciones
```

Esta recompensa busca inducir una caminata estable, suave y capaz de seguir comandos de velocidad.

### Alternativa de recompensa

También se propone evaluar una segunda función de recompensa más completa:

```
Reward = seguimiento de velocidad lineal
         + seguimiento de velocidad angular
         + estabilidad de altura del torso
         + estabilidad de orientación
         - penalización por energía utilizada
         - penalización por deslizamiento de pies
         - penalización por contactos no deseados
         - penalización por acciones bruscas
```

La comparación entre ambas formulaciones permitirá analizar qué términos favorecen una locomoción más estable y eficiente.

## Problemas identificados

Durante la etapa inicial del proyecto se identificaron algunos desafíos técnicos relevantes:

- Instalación correcta de MuJoCo Playground desde source.
- Compatibilidad entre librerías, CUDA y drivers de GPU.
- Configuración de Docker con soporte para GPU.
- Comprensión de la interfaz de control utilizada por Unitree.
- Identificación del orden correcto de las articulaciones del Go2.
- Implementación de cinemática inversa para cada pierna.
- Definición adecuada de la trayectoria cartesiana de los pies en CPG.
- Diseño de una función de costo útil para CEM.
- Diseño de recompensas que no generen comportamientos indeseados en RL.
- Evaluación objetiva de estabilidad y desempeño de la caminata.

Estos problemas serán abordados progresivamente durante el desarrollo del proyecto.

## Próximos pasos

Los próximos pasos del proyecto son:

- Finalizar la instalación y validación del entorno de simulación.
- Ejecutar ejemplos básicos en MuJoCo y Unitree MuJoCo.
- Probar el script `stand_go2.py` para lograr que el robot se mantenga de pie.
- Implementar una primera versión de cinemática inversa para una pierna.
- Extender la cinemática inversa a las cuatro patas.
- Diseñar una trayectoria simple de pie para caminata hacia adelante.
- Implementar un primer CPG con parámetros manuales.
- Definir la función de costo para CEM.
- Implementar CEM para ajustar los parámetros del CPG.
- Revisar el código base de RL y el archivo `joystick.py`.
- Diseñar al menos dos funciones de recompensa distintas.
- Entrenar políticas con `train_go2.py`.
- Evaluar políticas usando `evaluate_policy.py`.
- Comparar resultados entre formulaciones de recompensa.
- Documentar resultados con gráficos, videos y análisis.

## Evidencia y resultados

Los videos de simulación, capturas y resultados experimentales serán almacenados en la carpeta:

```
videos/
```

Los gráficos, logs y evaluaciones serán almacenados en:

```
results/
```

Se recomienda guardar evidencia incluso cuando los resultados sean desfavorables, ya que las pruebas fallidas permiten identificar problemas de simulación, control, estabilidad o diseño de recompensa.

## Criterios de evaluación considerados

Para analizar el desempeño de los controladores o políticas, se considerarán criterios como:

- Distancia recorrida hacia adelante.
- Estabilidad del torso.
- Ausencia de caídas.
- Altura del centro de masa.
- Suavidad de los movimientos.
- Consumo energético aproximado.
- Seguimiento de comandos de velocidad.
- Capacidad de movimiento omnidireccional en el caso de RL.
- Robustez ante diferentes condiciones iniciales.

## Referencias principales

Las referencias consideradas inicialmente para el desarrollo del proyecto son:

- de Boer, P. T., Kroese, D. P., Mannor, S., & Rubinstein, R. Y. *A Tutorial on the Cross-Entropy Method*. Annals of Operations Research, 134, 19–67, 2005.
- Pinneri, C., Sawant, S., Blaes, S., Achterhold, J., Stueckler, J., Rolinek, M., & Martius, G. *Sample-efficient Cross-Entropy Method for Real-time Planning*. Conference on Robot Learning, 2020.
- Suzuki, S., Matayoshi, K., Hayashibe, M., & Owaki, D. *Foot trajectory as a key factor for diverse gait patterns in quadruped robot locomotion*. Scientific Reports, 15, 2025.
- Zakka, K., Tabanpour, B., Liao, Q., Haiderbhai, M., Holt, S., Luo, J., Allshire, A., Frey, E., Sreenath, K., Kahrs, L. A., Sferrazza, C., Tassa, Y., & Abbeel, P. *MuJoCo Playground*. arXiv:2502.08844, 2025.

## Estado del proyecto

**Estado actual:** en desarrollo inicial.

La primera etapa está enfocada en levantar el entorno de simulación, comprender la interfaz de control del Unitree Go2 y definir la metodología para ambos enfoques de locomoción.

La entrega intermedia resume el avance inicial, los problemas encontrados y el plan de trabajo para continuar con la implementación.
