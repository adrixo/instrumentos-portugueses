---
theme: default
title: "Multimodal Visual Retrieval of Traditional Portuguese Musical Instruments: Dense, VLM, and Agentic Reranking"
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

# Multimodal Visual Retrieval of Traditional Portuguese Musical Instruments

<p class="subtitle">
Dense, VLM, and Agentic Reranking — a reproducible comparison on a labelled visual corpus.
</p>

<p class="meta">
Adrian Valera Roman · Álvaro Lozano Murciego<br>
<span class="small">adrianvalrom.usal@usal.es · loza@usal.es</span>
</p>

---

## Índice de contenidos

<div class="toc-large">

1. Motivación y formulación como recuperación visual
2. Corpus anotado y control de fugas de información
3. Sistemas evaluados e hipótesis de investigación
4. Diseño experimental y reproducibilidad
5. Resultados: recall, calidad y **significancia estadística**
6. Techo de candidatos y ablaciones
7. Análisis por instrumento, coste y limitaciones
8. Conclusiones y futuras líneas de trabajo

</div>

<div class="review">
Estructura revisada: se añaden significancia estadística, techo de candidatos, ablaciones, análisis por instrumento y limitaciones.
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

<div class="review">
Escala del test: 22 instrumentos × 3 idiomas = <span class="review-inline">66 consultas</span>. n reducido → los intervalos de confianza serán anchos (lo verificamos en Resultados).
</div>

</div>

<img class="hero-figure" src="./assets/ir_case_study.png" />

</div>

---

## Control de fuga de información

<div class="two-col">

<div>

La entrada disponible para los modelos es <span class="emph">solo visual</span>: consulta textual + imagen del frame. Los nombres de archivo, IDs de vídeo (Vimeo) y etiquetas del dataset no se exponen durante la inferencia; solo se usan después, para construir qrels y métricas.

<div class="review">

Garantías metodológicas (verificadas con tests automáticos):

- `image_id` anónimo; mapping privado nunca usado en inferencia.
- Prompts y trazas saneados de filename / ruta / clase ground truth.
- Modo offline: sin internet ni búsqueda web durante el reranking.

</div>

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

## Setup experimental y reproducibilidad

<div class="review">
Slide nuevo: condiciones exactas del experimento para reproducibilidad.
</div>

<div class="kv-grid">
  <div class="kv"><div class="k">Split de evaluación</div><div class="v">test — 1,317 imágenes, 66 consultas</div></div>
  <div class="kv"><div class="k">Dense base (candidatos)</div><div class="v">OpenCLIP ViT-L/14</div></div>
  <div class="kv"><div class="k">Determinismo</div><div class="v">seed = 42 · temperature = 0.0</div></div>
  <div class="kv"><div class="k">Profundidad de candidatos</div><div class="v">top-200 (B4/B5) · top-100 (Qwen3.6)</div></div>
  <div class="kv"><div class="k">Serving VLM</div><div class="v">vLLM (Qwen2.5-VL-3B) · llama.cpp (Qwen3.6-27B)</div></div>
  <div class="kv"><div class="k">Reproducibilidad</div><div class="v">Docker GPU · git commit · DVC · MLflow</div></div>
</div>

<div class="callout compact">
Modo desarrollo (valid) para elegir modelo, prompts y umbrales; modo final (test) solo para métricas. No se ajusta nada tras mirar test.
</div>

---

## Sistemas evaluados

<img class="full-bleed-figure" src="./assets/systems_overview.png" />

<p class="small center">
Cuatro familias: <span class="review-inline">Dense</span> (embeddings globales), <span class="review-inline">Late-interaction</span>, <span class="review-inline">VLM-rerank</span> multimodal y búsqueda <span class="review-inline">Agéntica</span>.
<span class="review-inline">Códigos internos: B1 / B3 / B4 / B5</span> (B2 = baseline textual BM25, descartado por diseño: el corpus se trata como puramente visual).
</p>

---

## Dense retrieval <span class="review-inline" style="font-size:0.6em">(B1)</span>

<div class="two-col wide-left">

<div>

Los modelos densos proyectan la consulta y cada imagen a un espacio vectorial común. El ranking se obtiene por similitud entre vectores.

- Un embedding por consulta, un embedding por imagen.
- Muy eficiente para indexar y recuperar a gran escala.
- Limitación: puede perder detalles pequeños o instrumentos visualmente parecidos.

Sistemas evaluados: OpenCLIP ViT-B/32, OpenCLIP ViT-L/14 y JinaCLIP.

</div>

<div class="system-crop dense"></div>

</div>

---

## Late-interaction <span class="review-inline" style="font-size:0.6em">(B3)</span>

<div class="two-col wide-left">

<div>

ColQwen representa la imagen y la consulta mediante múltiples vectores. En lugar de comparar un único embedding global, calcula coincidencias entre tokens visuales y textuales (late interaction).

- Mejor sensibilidad a partes locales de la imagen.
- Útil cuando el instrumento ocupa una zona pequeña.
- Más costoso que un índice denso global.

Especialmente interesante para instrumentos que aparecen parcialmente o entre otros objetos.

</div>

<div class="system-crop late"></div>

</div>

---

## VLM-rerank <span class="review-inline" style="font-size:0.6em">(B4)</span>

<div class="two-col wide-left">

<div>

El reranking multimodal parte de una lista candidata generada por recuperación densa. Un VLM examina cada imagen candidata y decide si el instrumento está presente.

- El VLM no busca en todo el corpus: solo reordena candidatos.
- Produce una decisión y una confianza (JSON cerrado, `temperature=0`).
- Modelo base: <span class="review-inline">Qwen2.5-VL-3B</span>; se compara además con <span class="review-inline">Qwen3.6-27B</span>.

<div class="callout compact">
La calidad final está acotada por el techo de los candidatos recuperados inicialmente.
</div>

</div>

<div class="system-crop vlm"></div>

</div>

---

## Búsqueda agéntica <span class="review-inline" style="font-size:0.6em">(B5)</span>

<div class="two-col wide-left">

<div>

Añade una estrategia de inspección visual sobre el reranking, como <span class="review-inline">grafo determinista propio</span> (no un agente libre):

- Primero pregunta por la imagen completa.
- Si hay incertidumbre, genera recortes deterministas.
- Puede producir una breve descripción (caption).
- Fusiona evidencias para el score final.

El objetivo es mirar de forma controlada cuando la imagen completa no basta.

</div>

<div class="system-crop agentic"></div>

</div>

---

## Hipótesis de investigación

<div class="review">
Slide nuevo: enmarca las hipótesis del ADR §3 y adelanta el veredicto (detalle en Resultados).
</div>

<div class="hyp-grid">
  <div class="hyp">
    <span class="verdict no">no confirmada</span><br>
    <strong>H1 · Dense global</strong> recupera sin entrenamiento, pero falla en instrumentos pequeños o parecidos.
  </div>
  <div class="hyp">
    <span class="verdict no">refutada</span><br>
    <strong>H2 · Late-interaction</strong> mejora sobre dense global. → No en promedio (Δ no significativo).
  </div>
  <div class="hyp">
    <span class="verdict partial">matizada</span><br>
    <strong>H3 · VLM-rerank</strong> mejora el ranking. → Solo con un VLM <em>grande</em>, y solo la ordenación.
  </div>
  <div class="hyp">
    <span class="verdict no">refutada</span><br>
    <strong>H4 · Agéntico</strong> mejora sobre una sola llamada VLM. → Captions inútiles; crops solo un margen no significativo.
  </div>
</div>

---

## Diseño experimental

<div class="two-col">

<div>

La evaluación compara sistemas sobre las mismas consultas y el mismo split de test:

- 22 instrumentos · 3 idiomas · <span class="review-inline">66 consultas</span>.
- Mismo dense base (OpenCLIP L/14) para todos los rerankers.
- Métricas macro por consulta/instrumento.

</div>

<div>

El protocolo separa dos fases:

- <span class="emph">Recuperación inicial</span>: ranking directo sobre el corpus.
- <span class="emph">Reranking</span>: reordenación de candidatos ya recuperados.

Esto distingue entre capacidad de <em>encontrar</em> candidatos y capacidad de <em>ordenar</em> los encontrados.

</div>

</div>

---

## Resultados: Recall@K

<img class="chart recall" src="./assets/metrics_recall_at_k.png" />

<div class="review">
JinaCLIP lidera Recall@100 (0.194) pero <span class="review-inline">sin diferencia significativa</span> (IC 95% solapados, n=66). Qwen3.6 no añade recall: topa en el techo del denso (0.181).
</div>

---

## Resultados: tabla macro (test)

<div class="review">
Slide nuevo: los números detrás de las figuras. Mejor por columna en negrita.
</div>

<div class="data-table">

| Sistema (código) | R@20 | R@50 | R@100 | nDCG@10 | nDCG@100 | mAP | MRR |
|---|---|---|---|---|---|---|---|
| OpenCLIP B/32 (B1) | 0.024 | 0.078 | 0.148 | 0.156 | 0.201 | 0.051 | 0.225 |
| OpenCLIP L/14 (B1) | 0.053 | 0.105 | 0.181 | 0.250 | 0.246 | 0.076 | 0.344 |
| JinaCLIP (B1) | 0.047 | **0.114** | **0.194** | 0.252 | 0.264 | 0.084 | 0.328 |
| ColQwen (B3) | 0.033 | 0.087 | 0.162 | 0.197 | 0.227 | 0.070 | 0.304 |
| VLM-rerank 3B (B4) | 0.042 | 0.103 | 0.190 | 0.260 | 0.255 | 0.076 | 0.378 |
| Agéntico (B5) | 0.046 | 0.109 | 0.193 | 0.262 | 0.264 | 0.080 | 0.423 |
| **Qwen3.6-27B (B4·27B)** | **0.063** | 0.111 | 0.181 | **0.389** | **0.272** | **0.088** | **0.495** |

</div>

<p class="small center">Diferencias pequeñas (~0.01–0.02) sobre 66 consultas: requieren test de significancia (siguiente slide).</p>

---

## Resultados: calidad del ranking

<img class="chart" src="./assets/metrics_quality_bars.png" />

<div class="review">
Barras = IC 95%. Solo <span class="review-inline">Qwen3.6-27B</span> (rojo) se separa en nDCG@10 y MRR; el resto solapan entre sí ⇒ no significativo.
</div>

---

## Significancia estadística

<div class="review">
Slide nuevo (ADR §6.4/§18.3): bootstrap 95% CI + test de permutación pareado, corrección Holm. n=66.
</div>

<div class="data-table">

| Comparación (delta) | nDCG@10 | mAP | MRR | Recall@100 |
|---|---|---|---|---|
| JinaCLIP vs OpenCLIP L/14 | +0.002 | +0.008 | −0.016 | +0.013 |
| ColQwen vs OpenCLIP L/14 | −0.053 | −0.006 | −0.039 | −0.019 |
| VLM-rerank 3B vs OpenCLIP L/14 | +0.010 | +0.000 | +0.034 | +0.010 |
| Agéntico vs VLM-rerank 3B | +0.002 | +0.003 | +0.045 | +0.002 |
| **Qwen3.6-27B vs OpenCLIP L/14** | **+0.138 ✅** | **+0.012 ✅** | **+0.151 ✅** | +0.000 |
| **Qwen3.6-27B vs VLM-rerank 3B** | **+0.128 ✅** | **+0.012 ✅** | **+0.118 ✅** | −0.010 |

</div>

<div class="callout">
✅ = significativo (Holm dentro de cada métrica, m=6, p&lt;0.05). <strong>Solo 7 de 30 comparaciones lo son, y todas son del VLM grande.</strong> Entre dense, late-interaction, VLM pequeño y agéntico: ninguna diferencia es significativa.
</div>

<p class="small center"><span class="review-inline">Nota</span>: la fila 27B vs 3B mezcla tamaño y profundidad de candidatos (top-100 vs top-200); la comparación limpia es 27B vs dense base, también significativa.</p>

---

## Techo de candidatos

<img class="chart" src="./assets/candidate_ceiling.png" />

<div class="review">
Solo el <span class="review-inline">29%</span> de los relevantes entra en el top-200 (oracle 27.8%). El reranker no recupera lo que la etapa 1 no trajo: <span class="review-inline">el cuello de botella es la recuperación</span>, no el VLM.
</div>

---

## Ablaciones de la búsqueda agéntica

<div class="review">
Slide nuevo: ablación por componente. Aislamos crops y captions quitándolos uno a uno.
</div>

<div class="data-table">

| Variante | Recall@100 | nDCG@100 | mAP | MRR |
|---|---|---|---|---|
| VLM pointwise (sin agente) | 0.190 | 0.255 | 0.076 | 0.378 |
| Agéntico completo (crops + caption) | 0.193 | 0.264 | 0.080 | 0.423 |
| Agéntico **sin captions** | 0.193 | 0.264 | 0.080 | 0.423 |
| Agéntico **sin crops** | 0.190 | 0.255 | 0.076 | 0.378 |

</div>

<div class="callout">
Quitar los <strong>captions</strong> no cambia nada (idéntico al completo) → <span class="review-inline">los captions son inútiles</span>. Quitar los <strong>crops</strong> colapsa el sistema al VLM pointwise → los crops son <span class="review-inline">el único componente con efecto</span>, pero marginal y no significativo (MRR +0.045, recall +0.002; p&gt;0.05).
</div>

---

## Lectura de los resultados

<div class="compact-table">

| Enfoque | Lectura principal (corregida) |
|---|---|
| Dense retrieval | Base sólida y barata; lidera Recall@100/mAP en punto estimado, <span class="review-inline">sin ventaja significativa</span>. |
| Late interaction | No mejora sobre dense en promedio (Δ negativo, no significativo). |
| VLM-rerank 3B | Sube MRR/nDCG en punto estimado, pero <span class="review-inline">ninguna mejora es significativa</span>. |
| Qwen3.6-27B | <span class="review-inline">Único con mejoras significativas</span> en nDCG@10, mAP y MRR; no añade recall. |
| Agéntico | Captions inútiles; crops solo un margen no significativo; más coste. |

</div>

<div class="callout">
El cuello de botella no es el razonamiento visual, sino que la primera etapa recupere suficientes candidatos relevantes.
</div>

---

## Análisis por instrumento

<div class="review">
Slide nuevo: la media macro esconde una varianza enorme entre clases.
</div>

<div class="two-col tight">

<div>

<strong>Recall@100 por clase (test):</strong>

- Fáciles: viola-beiroa 0.76, concertina 0.49, caixa-tamboril 0.38.
- Difíciles: violão 0.08, viola-braguesa 0.13.
- <span class="review-inline">Fallo total (0.00 en todos los sistemas)</span>: matracas, palheta, sarronca, reque-reque.

Ningún sistema domina: el mejor por clase se reparte entre dense, late-interaction, VLM-rerank y agéntico.

</div>

<div>

<strong>Error analysis (ej. adufe):</strong>

- 8 falsos positivos en top-100.
- 150 relevantes no recuperados (fuera del top-100).

Coherente con el techo de candidatos: la mayoría de errores son <span class="review-inline">de cobertura</span>, no de ordenación.

</div>

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
- ColQwen: 0.479 s/consulta con embeddings cacheados.
- VLM-rerank 3B: 30.1 s/consulta; agéntico: 44.9 s/consulta.
- Qwen3.6-27B: 129.3 s/consulta incremental (candidatos 51-100).

<div class="review">
El reranking cuesta ~1,600–2,500× más que el dense para una mejora de ordenación acotada por el techo. Latencias de Qwen3.6 <span class="review-inline">estimadas</span> desde timestamps y en serving distinto (llama.cpp): no comparables 1:1.
</div>

</div>

</div>

---

## Limitaciones y amenazas a la validez

<div class="review">
Slide nuevo (ADR §22).
</div>

- <strong>Potencia estadística baja</strong>: 66 consultas → IC 95% anchos; casi ninguna diferencia alcanza significancia.
- <strong>Conocimiento previo</strong>: los modelos fundacionales pueden haber visto instrumentos comunes; se prioriza open-weight y ejecución offline para acotarlo.
- <strong>Techo de la primera etapa</strong>: los rerankers no recuperan positivos fuera del top-N inicial.
- <strong>Confound Qwen3.6</strong>: distinto tamaño (27B vs 3B), profundidad (top-100 vs top-200) y serving; su ventaja mezcla esos factores.
- <strong>Coste del agéntico</strong>: no compensa cuando el VLM-rerank simple ya agota el margen disponible.

---

## Conclusiones

<div class="review">
Conclusiones reescritas para reflejar la significancia estadística.
</div>

1. Con n=66, <span class="review-inline">solo un VLM grande (Qwen3.6-27B) produce mejoras de ordenación estadísticamente significativas</span> (nDCG@10 +0.14, mAP +0.012, MRR +0.15; Holm p&lt;0.05).
2. <span class="review-inline">Ningún sistema mejora el recall</span>: todos topan en el techo del candidato denso (~29% en top-200). El cuello de botella es la recuperación inicial.
3. Entre dense, late-interaction, VLM pequeño y agéntico, las diferencias <span class="review-inline">no son significativas</span>: el dense, barato, es una elección defendible.
4. Los captions no cambian la métrica; los crops solo un margen no significativo (ablación).
5. El coste crece hasta ~2,500× para una mejora acotada por el techo: el mejor ranking no es el más operativo.

---

## Futuras líneas de trabajo

- Mejorar la <span class="review-inline">primera etapa</span> (el verdadero cuello de botella): dense fine-tuning o fusión de retrievers para subir el techo de candidatos.
- Escalar Qwen3.6-27B a top-200 y bajo el mismo entorno GPU/serving para separar calidad del modelo, profundidad y coste.
- Medir latencia completa por query para todos los enfoques con cachés y hardware controlados.
- Llevar la evaluación de frames a recuperación de vídeos completos.
- Integrar señales temporales: múltiples frames, audio y contexto de actuación.
- Aumentar el número de consultas para ganar potencia estadística.

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
