"""Tab 2: Data Browser + Global Embedding Mosaic."""
from __future__ import annotations

import gradio as gr
import numpy as np
from PIL import Image

from demo_v2.cache_manager import cache
from demo_v2.utils.constants import SOURCE_DISPLAY_NAMES
from demo_v2.utils.map_utils import render_folium_map, render_time_source_matrix
from demo_v2.utils.harbin_annotations_v2 import get_annotated_patches, load_harbin_annotations



_DATASET_INFO_MD = """
### 数据集概览

| 数据源 | 全称 | 分辨率 | 说明 |
|--------|------|--------|------|
| **Sentinel-2** | Copernicus Sentinel-2 MSI | 10m | 主要光学数据源，用于土地覆盖、植被和城市特征 |
| **Sentinel-1** | Copernicus Sentinel-1 SAR | 10m | 全天候 SAR，结构特征提取 |
| **Landsat** | Landsat 8/9 OLI | 30m | 长时序光学补充 |
| **高分光学 (S2-HR)** | 高分系列光学影像 | 2m | 高分辨率光学补充，精细地物识别 |
| **高分雷达 (S1-HR)** | 高分系列 SAR 影像 | 3m | 高分辨率 SAR，建筑物与工地区域监测 |
| **DEM** | Copernicus DEM GLO-30 | 30m | 地形高程上下文 |
| **WorldCover** | ESA WorldCover | 10m | 土地覆盖分类标签 |
| **Dynamic World** | Google Dynamic World | 10m | 动态土地利用 |
| **JRC Water** | JRC Global Surface Water | 30m | 水体覆盖 |

所有数据均配准到 128×128 像素网格（UTM 投影，10m 分辨率）。
"""


def _format_patch_info(patch_id: str) -> str:
    meta = cache.get_meta(patch_id)
    if meta is None:
        return f"**{patch_id}** — 未找到元数据"
    info = f"**{patch_id}**\n\n"
    b = meta.bounds
    info += f"Bounds: [{b[0]:.0f}, {b[1]:.0f}] → [{b[2]:.0f}, {b[3]:.0f}]  \n"
    info += f"CRS: {meta.crs}  \n"
    info += f"**{len(meta.sources)} sources:**\n\n"
    for src, n in sorted(meta.sources.items()):
        display_name = SOURCE_DISPLAY_NAMES.get(src, src)
        info += f"- {display_name}: **{n}** frames\n"
    return info


def _preview_patch(patch_id: str, show_annot: bool):
    if not patch_id or patch_id not in cache.patch_ids:
        return "*Invalid Patch ID*", None
    info = _format_patch_info(patch_id)
    if show_annot:
        annotations = load_harbin_annotations()
        recs = annotations.get(patch_id, [])
        if recs:
            info += "\n### 🏷️ 变化标注\n\n"
            for r in recs[:10]:
                info += f"- **{r['category']}** ({r['source']}) | {r['period']} | {r.get('remark', '')}\n"
            if len(recs) > 10:
                info += f"\n*共 {len(recs)} 条标注，仅显示前 10 条。*\n"
    patch_dir = cache.get_patch_dir(patch_id)
    time_src_img = render_time_source_matrix(patch_dir, patch_id)
    return info, time_src_img


def _embedding_map_to_rgb(emb_map: np.ndarray, version: str, use_global_norm: bool = True) -> np.ndarray:
    """将单个 embedding map [D, H, W] 转为 RGB [H, W, 3]."""
    pca, vmin, vmax = cache.get_global_pca(version)
    D, H, W = emb_map.shape
    flat = emb_map.reshape(D, -1).T
    rgb = pca.transform(flat).reshape(H, W, 3)
    if use_global_norm:
        rgb = (rgb - vmin) / (vmax - vmin + 1e-8)
    else:
        rgb_min = rgb.min(axis=(0, 1), keepdims=True)
        rgb_max = rgb.max(axis=(0, 1), keepdims=True)
        rgb = (rgb - rgb_min) / (rgb_max - rgb_min + 1e-8)
    rgb = np.clip(rgb, 0, 1)
    return rgb


def _render_global_mosaic(version: str) -> Image.Image | None:
    """渲染全域 PCA-RGB 拼接图."""
    if version not in cache.embedding_maps:
        return None

    maps = cache.embedding_maps[version]
    ids = cache.embedding_map_patch_ids.get(version, cache.patch_ids)
    _, _, H, W = maps.shape

    records = []
    for i, pid in enumerate(ids):
        meta = cache.get_meta(pid)
        if meta is None:
            continue
        records.append((meta.bounds, i))

    if not records:
        return None

    all_x = sorted({round(b[0]) for b, _ in records})
    all_y = sorted({round(b[1]) for b, _ in records}, reverse=True)
    x_to_col = {x: c for c, x in enumerate(all_x)}
    y_to_row = {y: r for r, y in enumerate(all_y)}
    nrows = len(all_y)
    ncols = len(all_x)

    canvas = np.ones((nrows * H, ncols * W, 3), dtype=np.float32)

    for bounds, idx in records:
        col = x_to_col.get(round(bounds[0]))
        row = y_to_row.get(round(bounds[1]))
        if col is None or row is None:
            continue
        emb_map = maps[idx]
        rgb = _embedding_map_to_rgb(emb_map, version, use_global_norm=True)
        r0, c0 = row * H, col * W
        canvas[r0:r0 + H, c0:c0 + W] = rgb

    canvas_u8 = (np.clip(canvas, 0, 1) * 255).astype(np.uint8)
    pil_img = Image.fromarray(canvas_u8)
    scale = 3
    pil_img = pil_img.resize((pil_img.width * scale, pil_img.height * scale), Image.NEAREST)
    return pil_img


def build_data_browser_tab():
    """Build Tab 2: Data Browser + Global Mosaic."""
    gr.Markdown("## Data & Embedding Field")

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=250):
            gr.Markdown("### Patch Selector")
            patch_id_box = gr.Textbox(
                label="Patch ID",
                placeholder="Click a patch on the map, then click Sync below...",
                interactive=True,
                elem_id="patch_id_box",
            )
            btn_sync = gr.Button("📍 同步地图选中的 Patch", variant="secondary")
            show_annot_checkbox = gr.Checkbox(
                label="高亮有标注的 Patch",
                value=False,
            )
            btn_preview = gr.Button("Preview", variant="primary", size="lg")
            patch_info = gr.Markdown("")

        with gr.Column(scale=4, min_width=600):
            gr.Markdown("### Interactive Map  \n*Click a patch rectangle to select it*")
            folium_html = gr.HTML(value="")

    gr.Markdown("### Time x Source Matrix")
    time_source_image = gr.Image(
        label="Time x Source Matrix",
        interactive=False,
        elem_id="time_source_matrix_img",
    )

    gr.Markdown("### Global Embedding PCA-RGB Mosaic")
    available_versions = list(cache.embedding_maps.keys())
    default_version = available_versions[0] if available_versions else None
    with gr.Row():
        mosaic_version = gr.Dropdown(
            choices=available_versions,
            value=default_version,
            label="模型版本",
        )
        btn_mosaic = gr.Button("Generate Global Mosaic", variant="secondary")
    img_mosaic = gr.Image(label="Global Embedding Field", height=600)

    gr.Markdown(_DATASET_INFO_MD)

    def _load_map(highlight_annot: bool):
        annotated = set(get_annotated_patches()) if highlight_annot else set()
        return render_folium_map(cache.patch_metas, annotated_ids=annotated)

    # Timer: poll pending patch from JS bridge (client-side only)
    _patch_timer = gr.Timer(0.5, active=True)
    _patch_timer.tick(
        fn=None,
        inputs=[patch_id_box],
        outputs=[patch_id_box],
        js="(x) => { var p = window._aef_pending_patch; if (p) { window._aef_pending_patch = null; return p; } return x; }",
    )

    btn_preview.click(
        fn=_preview_patch,
        inputs=[patch_id_box, show_annot_checkbox],
        outputs=[patch_info, time_source_image],
    )
    show_annot_checkbox.change(
        fn=_load_map,
        inputs=[show_annot_checkbox],
        outputs=[folium_html],
    )
    btn_mosaic.click(
        fn=_render_global_mosaic,
        inputs=[mosaic_version],
        outputs=[img_mosaic],
    )

    return folium_html, lambda: _load_map(False)
