#!/usr/bin/env python3
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEMO_ROOT = Path(__file__).resolve().parent
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))

import gradio as gr
from demo_v2.cache_manager import cache

cache.load()

with gr.Blocks(title="Min Test") as app:
    gr.Markdown("# Min Test")
    with gr.Tabs():
        with gr.Tab("Tab A"):
            gr.Markdown("This is Tab A")
            gr.Textbox(label="Input")
        with gr.Tab("Tab B"):
            gr.Markdown("This is Tab B")
            gr.Button("Click me")
        with gr.Tab("Tab C"):
            gr.Markdown("This is Tab C")
            gr.Image(label="Image")

app.launch(server_name="0.0.0.0", server_port=7992)
