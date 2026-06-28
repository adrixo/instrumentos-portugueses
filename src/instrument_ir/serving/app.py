"""Buscador web en Gradio (ADR §12). gradio es opcional (extra [extras]).

Navega los runfiles generados y muestra la galería de resultados con enlace al vídeo de Vimeo.
La búsqueda por texto libre en vivo requeriría cargar el retriever (GPU); aquí se navega lo precomputado.
"""

from __future__ import annotations

from pathlib import Path

from .search import SearchService


def build_app(service: SearchService | None = None):
    import gradio as gr

    service = service or SearchService()
    runs = service.list_runs()

    def _queries(run_name):
        return gr.update(choices=service.list_queries(run_name) if run_name else [])

    def _search(run_name, query_id, top_k):
        if not run_name or not query_id:
            return [], "Selecciona sistema y query."
        results = service.search(run_name, query_id, int(top_k))
        gallery = []
        for r in results:
            mark = "✓" if r["relevant"] else "·"
            caption = f"#{r['rank']} {mark} score={r['score']} · vimeo {r['vimeo_id']}"
            if r["image_path"] and Path(r["image_path"]).exists():
                gallery.append((r["image_path"], caption))
        hits = sum(1 for r in results if r["relevant"])
        summary = f"{hits}/{len(results)} relevantes en top-{len(results)}"
        return gallery, summary

    with gr.Blocks(title="Instrument Retrieval Lab") as demo:
        gr.Markdown("# Instrument Retrieval Lab\nBúsqueda visual de instrumentos portugueses.")
        with gr.Row():
            run_dd = gr.Dropdown(runs, label="Sistema (runfile)")
            query_dd = gr.Dropdown([], label="Query (instrumento/idioma)")
            topk = gr.Slider(5, 100, value=20, step=5, label="Top-K")
        btn = gr.Button("Buscar")
        summary = gr.Markdown()
        gallery = gr.Gallery(label="Resultados", columns=5, height=600)

        run_dd.change(_queries, inputs=run_dd, outputs=query_dd)
        btn.click(_search, inputs=[run_dd, query_dd, topk], outputs=[gallery, summary])
    return demo


def launch(server_name: str = "0.0.0.0", server_port: int = 7860):
    build_app().launch(server_name=server_name, server_port=server_port)
