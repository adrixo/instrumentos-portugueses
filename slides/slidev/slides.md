---
theme: default
title: "Visual Information Retrieval for Traditional Portuguese Instruments"
info: |
  Academic presentation on visual information retrieval over a labelled corpus of
  traditional Portuguese instrument frames.
class: text-left
highlighter: shiki
drawings:
  persist: false
transition: fade
mdc: true
# hash router avoids double-base URLs when served under a GitHub Pages subpath
routerMode: hash
---

# Visual Information Retrieval for Traditional Portuguese Instruments

<p class="subtitle">
Evaluating dense, late-interaction, multimodal and agentic retrieval systems on a labelled visual corpus.
</p>

<p class="meta">
Adrian Valera Roman
</p>

---

## Índice de contenidos

<div class="toc-large">

1. Motivación: recuperación de información en archivos audiovisuales
2. Corpus anotado y formulación del caso de estudio
3. Sistemas evaluados
4. Resultados de recuperación y reranking
5. Coste temporal por consulta
6. Conclusiones y futuras líneas de trabajo

</div>

---

## Motivación

La recuperación de información permite explorar grandes corpus culturales sin revisar manualmente cada vídeo, imagen o documento. En archivos audiovisuales, una consulta útil no suele ser un identificador técnico, sino una necesidad semántica:

<div class="callout">
Encontrar fragmentos visuales donde aparece un instrumento tradicional concreto, aunque el vídeo no esté etiquetado con ese instrumento.
</div>

El mapa de <span class="emph">A Música Portuguesa a Gostar Dela Própria</span> reúne numerosos vídeos de música tradicional portuguesa. En ese tipo de archivo, la pregunta de IR sería: dado un instrumento, ¿qué frames o vídeos deberían aparecer primero?

---

## Del archivo al corpus evaluable

<div class="two-col wide-left">

<div>

El punto de partida es un archivo audiovisual amplio: actuaciones, entrevistas, bailes, grabaciones de campo y vídeos con instrumentos en contextos muy variables.

Para evaluar sistemas de recuperación no basta con tener vídeos: se necesita ground truth. Por eso se etiquetó un dataset visual y se publicó como:

<div class="citation">
Comprehensive dataset of Portuguese folk instruments for computer vision and heritage research<br>
Data in Brief 61, 2025. DOI 10.1016/j.dib.2025.111739<br>
Dataset: Mendeley DOI 10.17632/pk7txkgt4v.2
</div>

</div>

<div class="paper-stack">
  <img src="./assets/paper/dataset_collection.jpg" />
  <img src="./assets/paper/dataset_annotations.jpg" />
</div>

</div>

---

## Dataset

<div class="two-col">

<div>

<div class="metric-grid">
  <div class="metric">
    <div class="label">Train</div>
    <div class="value">3,954</div>
    <div class="note">imágenes</div>
  </div>
  <div class="metric">
    <div class="label">Valid</div>
    <div class="value">1,351</div>
    <div class="note">imágenes</div>
  </div>
  <div class="metric">
    <div class="label">Test</div>
    <div class="value">1,317</div>
    <div class="note">imágenes</div>
  </div>
  <div class="metric">
    <div class="label">Clases</div>
    <div class="value">22</div>
    <div class="note">instrumentos</div>
  </div>
</div>

<p class="body-text">
Cada imagen puede contener varios instrumentos. Esto convierte el problema en recuperación multi-etiqueta: una imagen es relevante para una consulta si el instrumento aparece visualmente en el frame.
</p>

</div>

<div class="paper-frame tall">
  <img src="./assets/paper/dataset_classes.jpg" />
</div>

</div>

---

## Caso de estudio: formulación IR

<div class="two-col wide-left">

<div>

El caso de estudio se plantea como una tarea clásica de recuperación:

- <span class="emph">Consulta</span>: nombre textual de un instrumento, en portugués, español o inglés.
- <span class="emph">Documento</span>: una imagen, normalmente un frame extraído de un vídeo.
- <span class="emph">Relevancia</span>: el instrumento consultado aparece en la imagen.
- <span class="emph">Salida</span>: ranking de imágenes ordenadas por probabilidad de relevancia.

Esto permite comparar sistemas con métricas IR estándar: Recall@K, nDCG@K, mAP y MRR.

</div>

<img class="hero-figure" src="./assets/ir_case_study.png" />

</div>

---

## Relevancia y control de información

<div class="two-col">

<div>

La entrada disponible para los modelos es solo visual:

<div class="callout compact">
Consulta textual + imagen del frame.
</div>

Los nombres de archivo, identificadores de vídeo y etiquetas del dataset no se exponen durante la inferencia. Solo se usan después, para construir qrels y calcular métricas.

</div>

<div class="flow-card">
  <div class="flow-step">Consulta<br><strong>“adufe”</strong></div>
  <div class="arrow">→</div>
  <div class="flow-step">Frame<br><strong>imagen</strong></div>
  <div class="arrow">→</div>
  <div class="flow-step">Modelo<br><strong>score</strong></div>
  <div class="arrow">→</div>
  <div class="flow-step">Ranking<br><strong>image_id</strong></div>
</div>

</div>

---

## Sistemas evaluados

<img class="full-bleed-figure" src="./assets/systems_overview.png" />

<p class="small center">
Cuatro familias de aproximaciones: embeddings globales, interacción tardía, reranking multimodal y búsqueda agéntica.
</p>

---

## Sistema 1: recuperación densa

<div class="two-col wide-left">

<div>

Los modelos densos proyectan la consulta y cada imagen a un espacio vectorial común. El ranking se obtiene por similitud entre vectores.

- Un embedding por consulta.
- Un embedding por imagen.
- Muy eficiente para indexar y recuperar a gran escala.
- Limitación: puede perder detalles pequeños o instrumentos visualmente parecidos.

Sistemas evaluados: OpenCLIP ViT-B/32, OpenCLIP ViT-L/14 y JinaCLIP.

</div>

<div class="system-crop dense"></div>

</div>

---

## Sistema 2: interacción tardía

<div class="two-col wide-left">

<div>

ColQwen representa la imagen y la consulta mediante múltiples vectores. En lugar de comparar un único embedding global, calcula coincidencias entre tokens visuales y textuales.

- Mejor sensibilidad a partes locales de la imagen.
- Útil cuando el instrumento ocupa una zona pequeña.
- Más costoso que un índice denso global.

Este enfoque es especialmente interesante para instrumentos que aparecen parcialmente o entre otros objetos visuales.

</div>

<div class="system-crop late"></div>

</div>

---

## Sistema 3: reranking multimodal

<div class="two-col wide-left">

<div>

El reranking multimodal parte de una lista candidata generada por recuperación densa. Después, un VLM examina cada imagen candidata y decide si el instrumento está presente.

- El VLM no busca en todo el corpus: solo reordena candidatos.
- Produce una decisión y una confianza.
- Puede incorporar evidencia visual explícita.

La calidad final depende del techo impuesto por los candidatos recuperados inicialmente.

</div>

<div class="system-crop vlm"></div>

</div>

---

## Sistema 4: búsqueda agéntica

<div class="two-col wide-left">

<div>

La búsqueda agéntica añade una estrategia de inspección visual sobre el reranking multimodal.

- Primero pregunta por la imagen completa.
- Si hay incertidumbre, puede generar recortes deterministas.
- Puede producir una breve descripción visual.
- Fusiona evidencias para decidir el score final.

El objetivo no es solo “mirar más”, sino mirar de forma controlada cuando la imagen completa no basta.

</div>

<div class="system-crop agentic"></div>

</div>

---

## Diseño experimental

<div class="two-col">

<div>

La evaluación compara sistemas sobre las mismas consultas y el mismo split de test:

- 22 instrumentos.
- 3 idiomas por instrumento.
- 66 consultas.
- Métricas macro por consulta/instrumento.

</div>

<div>

El protocolo separa dos fases:

- <span class="emph">Recuperación inicial</span>: ranking directo sobre el corpus.
- <span class="emph">Reranking</span>: reordenación de candidatos ya recuperados.

Esto permite distinguir entre capacidad de encontrar candidatos y capacidad de ordenar correctamente los candidatos encontrados.

</div>

</div>

---

## Resultados: Recall@K

<img class="chart recall" src="./assets/metrics_recall_at_k.png" />

<p class="small center">
JinaCLIP mantiene el mejor Recall@100 entre los sistemas de recuperación directa. Qwen3.6 top-100 recupera el techo del candidato denso y mejora la ordenación temprana dentro de ese conjunto.
</p>

---

## Resultados: calidad del ranking

<img class="chart" src="./assets/metrics_quality_bars.png" />

<p class="small center">
Qwen3.6 top-100 obtiene el mejor nDCG@10, mAP y MRR de la comparación. La búsqueda agéntica mantiene una ligera ventaja de cobertura frente al candidato denso usado por Qwen.
</p>

---

## Lectura de los resultados

<div class="compact-table">

| Enfoque | Lectura principal |
|---|---|
| Dense retrieval | Muy competitivo y barato; JinaCLIP lidera Recall@100 y mAP. |
| Late interaction | No domina en promedio, pero ayuda en clases con señales locales. |
| VLM reranking | Mejora la ordenación de candidatos, especialmente nDCG/MRR. |
| Qwen3.6-27B VLM | Top-100 mejora las primeras posiciones y mAP; no añade cobertura fuera del candidato denso. |
| Agentic reranking | Añade valor cuando la inspección completa no basta, pero aumenta el coste. |

</div>

<div class="callout">
El cuello de botella no es solo el razonamiento visual: también importa que la primera etapa recupere suficientes candidatos relevantes.
</div>

---

## Coste temporal por consulta

<div class="two-col wide-left cost-slide">

<div>

<img class="chart inset" src="./assets/latency_boxplot.png" />

</div>

<div>

La comparación temporal debe leerse como parte del diseño del sistema:

- OpenCLIP L/14: 0.019 s/consulta; JinaCLIP: 0.104 s/consulta.
- ColQwen: 0.479 s/consulta con embeddings de corpus cacheados.
- B4 VLM: 30.1 s/consulta; B5 agéntico: 44.9 s/consulta.
- Qwen3.6-27B: 129.3 s/consulta incremental para candidatos 51-100; top-100 desde cero se estima en 284.2 s/consulta.
- B5 tiene una cola más larga por recortes, captions y llamadas adicionales.

<p class="small">
Box-and-whisker en escala logarítmica. Dense/ColQwen se midieron con benchmark por consulta; B4/B5/Qwen3.6 se estiman desde marcas temporales de traces.
</p>

</div>

</div>

---

## Conclusiones

1. La tarea es útil para explorar archivos audiovisuales de patrimonio musical cuando el usuario busca por instrumentos, no por metadatos técnicos.
2. Un dataset anotado convierte el corpus en un banco de pruebas cuantitativo para sistemas de IR visual.
3. Los modelos densos son una base sólida y eficiente.
4. Los VLMs y la búsqueda agéntica aportan mejoras de ordenación; Qwen3.6 lo confirma en top-100 con el mejor nDCG@10, mAP y MRR, aunque con mayor coste temporal.
5. El coste temporal debe considerarse junto a la métrica: el mejor ranking no siempre es el sistema más operativo.

---

## Futuras líneas de trabajo

- Escalar Qwen3.6-27B a top-200 y bajo el mismo entorno GPU para separar calidad del modelo, red y coste de serving.
- Medir latencia completa por query para todos los enfoques bajo cachés y hardware controlados.
- Llevar la evaluación de frames a recuperación de vídeos completos.
- Integrar señales temporales: múltiples frames, audio y contexto de actuación.
- Añadir aprendizaje específico para instrumentos visualmente cercanos.
- Diseñar una interfaz de búsqueda para exploración del archivo por investigadores y público general.

---

## Bibliografía

<div class="refs">

<p>[1] A Música Portuguesa a Gostar Dela Própria, “Mapa,” accessed Jun. 30, 2026. [Online]. Available: https://amusicaportuguesaagostardelapropria.org/map</p>

<p>[2] N. Zendron <em>et al.</em>, “Comprehensive dataset of Portuguese folk instruments for computer vision and heritage research,” <em>Data in Brief</em>, vol. 61, Art. no. 111739, 2025, doi: 10.1016/j.dib.2025.111739.</p>

<p>[3] N. Zendron <em>et al.</em>, “Portuguese folk instruments dataset,” Mendeley Data, V2, 2025, doi: 10.17632/pk7txkgt4v.2.</p>

<p>[4] A. Radford <em>et al.</em>, “Learning transferable visual models from natural language supervision,” in <em>Proc. ICML</em>, 2021.</p>

<p>[5] G. Ilharco, M. Wortsman, R. Wightman, C. Gordon, N. Carlini, R. Taori, A. Dave, V. Shankar, H. Namkoong, J. Miller, H. Hajishirzi, A. Farhadi, and L. Schmidt, “OpenCLIP,” Zenodo, 2021, doi: 10.5281/zenodo.5143773.</p>

<p>[6] Jina AI, “Jina CLIP: Your CLIP model is also your text retriever,” arXiv:2405.20204, 2024.</p>

<p>[7] M. Faysse <em>et al.</em>, “ColPali: Efficient document retrieval with vision language models,” arXiv:2407.01449, 2024.</p>

<p>[8] P. Wang <em>et al.</em>, “Qwen2-VL: Enhancing vision-language model's perception of the world at any resolution,” arXiv:2409.12191, 2024.</p>

<p>[9] Qwen Team, “Qwen2.5-VL technical report,” arXiv:2502.13923, 2025.</p>

<p>[10] T. Yao <em>et al.</em>, “ReAct: Synergizing reasoning and acting in language models,” in <em>Proc. ICLR</em>, 2023.</p>

<p>[11] G. Gerganov, “llama.cpp,” GitHub repository, 2023. [Online]. Available: https://github.com/ggml-org/llama.cpp</p>

<p>[12] Unsloth, “Qwen3.6-27B-GGUF,” Hugging Face model card, accessed Jun. 30, 2026. [Online]. Available: https://huggingface.co/unsloth/Qwen3.6-27B-GGUF</p>

</div>
