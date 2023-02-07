---
aliases:
- /fastchai/fastai/2021/01/09/fastchai-rbracco-audio-y-estudio-propio
categories:
- fastchai
- fastai
date: '2021-01-09'
description: Audio y estudio autodidacta
layout: post
title: Audio y auto estudio interview
toc: true

---

{% twitter https://twitter.com/bhutanisanyam1/status/1168218396552351744 %}

# Parte 1
https://youtu.be/k-gZAyg5ib8?t=0 Introducción

* https://twitter.com/madeupmasters
* https://forums.fast.ai/t/things-jeremy-says-to-do/36682

"Para lograr ser autodidacta se requiere ser consistente y un desafío balanceado".

Y hay que ver en teoría en una escuela "quieren que construyas un compilador antes antes de hacer una pieza útil de software"

"La AI puede ser más flexible para detectar cambios en un audio que la programación tradicional" lo cual podría ser complicado de programar de manera imperativa o funcional.

tomó los cursos
*  https://cs50.harvard.edu/college/2021/spring/ ó https://online-learning.harvard.edu/course/cs50-introduction-computer-science para principiantes en computación y amplio
* https://www.coursera.org/learn/algorithms-part1
* https://www.coursera.org/learn/algorithms-part2 y probablemente sirva de algo el código de este libro https://algs4.cs.princeton.edu/home/
* https://www.coursera.org/learn/machine-learning by Andrew Ng

"focus on building not on theory" "top-down approach", en otras palabras es más practico hacer cosas que entender el trasfondo que puede ser más complejo y requerir de más análisis.

# parte 2
https://youtu.be/k-gZAyg5ib8?t=986 brincar a una competencia sin saber nada te da la libertad de experimentar lo que se te ocurra. Un punto interesante es que si tienes suficientes bases puedes explorar más libremente que sabiendo las herramientas default usadas o los modelos usados por defecto "un periodo de creatividad libre", una de las estrategias que tomó en ese tiempo fue pasar la onda de audio directo al modelo y posteriormente se dió cuenta que tendría que conocer más acerca del procesamiento de señales para poder sacar más información del audio.

"It is easy to get addicted to online classes" se puede entrar en un "ciclo infinito" de aprender que puede bloquear el de aplicar o hacer algo.

https://hackernoon.com/how-not-to-do-fast-ai-or-any-ml-mooc-3d34a7e0ab8c

"What should I don next? is to see what you have in your head and see what is stopping you, then learn that", enforcarse en aprender sólo las cosas que te faltan por aprender es mejor, es como el eslabon más débil de la cadena siempre es el que se puede romper, hay que aprender a hacer objetivos basados en los "bloqueos" que nos encontramos en el camino.

Kaggle te puede aislar hacia resolver el problema a la mano en vez de toparte con lo que es construir tu propio proyecto porque ya tienes de alguna forma los datos procesados y un objetivo, pero si no lo tienes todo de inicio puede resultar en otra forma de aprendizaje (¿talvez no es posible?, ¿los datos afectan?). Es mejor si se pueden hacer las dos cosas Kaggle+proyecto(s) propio(s).

http://christinemcleavey.com/musenet/

# parte 3

https://youtu.be/k-gZAyg5ib8?t=1356

El objetivo de fastai_audio es contruir un modulo que sea compatible con fastai con la misma usabilidad.

Resnets y densenets parecen funcionar bien con FFTs y audio. Y se puede usar transfer learning desde imagenette aunque no tenga nada que ver con audio, probablemente porque las primeras capas reconocen líneas, direcciones y otras características básicas.

Y a base de prueba y error los defaults en fastai_audio han sido puestos en la librería.

Y en cuando a hacer cosas, es mejor hacerlas aunque tengan errores o no sean perfectas (o incluso aunque no las entiendas), talvez alguien más pueda ver el error y corregirlo. Si se espera a ser experto en el area antes de empezar algo, pues por lo menos falta llegar a ese punto primero sin hacer nada práctico de antemano. Y es bueno tener a alguién o algun recurso en quien confiar ya sea una persona, un foro en general alguién a quien poder preguntar aunque tal vez no tengan todas las respuestas.

(SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition][https://arxiv.org/abs/1904.08779] y https://ai.googleblog.com/2019/04/specaugment-new-data-augmentation.html

# parte 4

https://youtu.be/k-gZAyg5ib8?t=2730

"consistency matters above all" la consistencia importa más que cualquier otra cosa, poner esfuerzo constante con días de descanso es mejor que hacer mucho en este momento, dejarlo, volver y que sobre esforzarse al punto de "quemarte". "Intellectual work is exhausting" el trabajo intelectual es agotador, "I'm working from 8:00 a.m. to 2:00 p.m." trabajo de las 8:00 a.m. a las 2:00 p.m. Con un tiempo más límitado tienes que enfocarte en hacer ciertas cosas y tomar decisiones en ese tiempo sin tener todo el tiempo del día para cosas "fribolas".