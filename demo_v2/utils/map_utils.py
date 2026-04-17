"""地图渲染工具模块。

- Folium 交互式地图 (UTM → WGS84) with click → postMessage bridge
- Matplotlib 静态地图作为备选
- 时间×数据源矩阵可视化（按年份垂直堆叠）
"""
from __future__ import annotations

import io
from datetime import datetime
from typing import Optional

import folium
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from branca.element import MacroElement
from jinja2 import Template as JTemplate
from PIL import Image
from pyproj import Transformer

matplotlib.use("Agg")

# ── UTM (EPSG:32652) 范围常量 ──
UTM_X_MIN = 291438.0
UTM_X_MAX = 324718.0
UTM_Y_MIN = 5068572.0
UTM_Y_MAX = 5099292.0

# 图像尺寸 (像素)
IMG_WIDTH = 1200
IMG_HEIGHT = 1100

# 边距
MARGIN_LEFT = 80
MARGIN_RIGHT = 40
MARGIN_TOP = 50
MARGIN_BOTTOM = 50

# UTM→WGS84 转换器 (EPSG:32652 → EPSG:4326)
_transformer = Transformer.from_crs("EPSG:32652", "EPSG:4326", always_xy=True)


def _utm_to_lonlat(x: float, y: float) -> tuple[float, float]:
    """UTM 坐标 → (lon, lat)。"""
    lon, lat = _transformer.transform(x, y)
    return lon, lat


def _utm_to_pixel(x: float, y: float) -> tuple[int, int]:
    plot_w = IMG_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    plot_h = IMG_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM
    px = MARGIN_LEFT + (x - UTM_X_MIN) / (UTM_X_MAX - UTM_X_MIN) * plot_w
    py = MARGIN_TOP + (1 - (y - UTM_Y_MIN) / (UTM_Y_MAX - UTM_Y_MIN)) * plot_h
    return int(px), int(py)


def pixel_to_utm(px: int, py: int) -> tuple[float, float] | None:
    plot_w = IMG_WIDTH - MARGIN_LEFT - MARGIN_RIGHT
    plot_h = IMG_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM

    x_in_plot = px - MARGIN_LEFT
    y_in_plot = py - MARGIN_TOP

    if x_in_plot < 0 or x_in_plot > plot_w or y_in_plot < 0 or y_in_plot > plot_h:
        return None

    x_frac = x_in_plot / plot_w
    y_frac = 1.0 - y_in_plot / plot_h

    utm_x = UTM_X_MIN + x_frac * (UTM_X_MAX - UTM_X_MIN)
    utm_y = UTM_Y_MIN + y_frac * (UTM_Y_MAX - UTM_Y_MIN)
    return utm_x, utm_y


def find_nearest_patch(
    utm_x: float, utm_y: float, patch_metas: list,
) -> Optional[str]:
    best_id: str | None = None
    best_dist = float("inf")
    for m in patch_metas:
        xmin, ymin, xmax, ymax = m.bounds
        if xmin <= utm_x <= xmax and ymin <= utm_y <= ymax:
            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2
            dist = (utm_x - cx) ** 2 + (utm_y - cy) ** 2
            if dist < best_dist:
                best_dist = dist
                best_id = m.patch_id
    if best_id is None:
        for m in patch_metas:
            dist = (utm_x - m.center_x) ** 2 + (utm_y - m.center_y) ** 2
            if dist < best_dist:
                best_dist = dist
                best_id = m.patch_id
    return best_id


# ── Folium 交互式地图 ──


def render_folium_map(
    patch_metas: list,
    highlight_id: str | None = None,
    annotated_ids: set[str] | None = None,
) -> str:
    """渲染 Folium 交互式地图，返回 HTML 字符串。

    点击 patch 时通过 postMessage 发送 patch_id 到父页面。
    """
    cx = (UTM_X_MIN + UTM_X_MAX) / 2
    cy = (UTM_Y_MIN + UTM_Y_MAX) / 2
    center_lon, center_lat = _utm_to_lonlat(cx, cy)

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles="OpenStreetMap",
    )

    annotated_ids = annotated_ids or set()

    # Build GeoJSON FeatureCollection for all patches
    features = []
    for meta in patch_metas:
        xmin, ymin, xmax, ymax = meta.bounds
        sw_lon, sw_lat = _utm_to_lonlat(xmin, ymin)
        ne_lon, ne_lat = _utm_to_lonlat(xmax, ymax)
        is_hl = (meta.patch_id == highlight_id)
        is_annot = (meta.patch_id in annotated_ids)
        sources_str = ", ".join(f"{s}({n})" for s, n in sorted(meta.sources.items()))
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[sw_lon, sw_lat], [ne_lon, sw_lat],
                                 [ne_lon, ne_lat], [sw_lon, ne_lat],
                                 [sw_lon, sw_lat]]],
            },
            "properties": {
                "patch_id": meta.patch_id,
                "is_hl": 1 if is_hl else 0,
                "is_annot": 1 if is_annot else 0,
                "n_sources": len(meta.sources),
                "sources_str": sources_str,
            },
        })

    geojson_data = {"type": "FeatureCollection", "features": features}

    def _style_function(f):
        is_hl = f["properties"]["is_hl"]
        is_annot = f["properties"]["is_annot"]
        if is_hl:
            return {
                "color": "#e03131",
                "weight": 3,
                "fillColor": "#e03131",
                "fillOpacity": 0.4,
            }
        if is_annot:
            return {
                "color": "#f59f00",
                "weight": 2,
                "fillColor": "#f59f00",
                "fillOpacity": 0.35,
            }
        return {
            "color": "#4dabf7",
            "weight": 1,
            "fillColor": "#4dabf7",
            "fillOpacity": 0.15,
        }

    folium.GeoJson(
        data=geojson_data,
        style_function=_style_function,
        highlight_function=lambda f: {"weight": 3, "fillOpacity": 0.4},
        tooltip=folium.GeoJsonTooltip(
            fields=["patch_id", "n_sources"],
            aliases=["Patch", "Sources"],
        ),
    ).add_to(m)

    # JS click handler: store patch_id in parent window variable
    click_macro = MacroElement()
    click_macro._template = JTemplate("""
    {% macro script(this, kwargs) %}
        {{ this._parent.get_name() }}.eachLayer(function(layer) {
            if (layer.eachLayer) {
                layer.eachLayer(function(sublayer) {
                    if (sublayer.feature && sublayer.feature.properties &&
                        sublayer.feature.properties.patch_id) {
                        sublayer.on('click', function(e) {
                            try {
                                window.parent._aef_pending_patch = sublayer.feature.properties.patch_id;
                            } catch(err) {
                                window._aef_pending_patch = sublayer.feature.properties.patch_id;
                            }
                        });
                    }
                });
            }
        });
    {% endmacro %}
    """)
    m.add_child(click_macro)

    return m._repr_html_()


# ── 时间×数据源矩阵 ──


def render_time_source_matrix(patch_dir, patch_id: str) -> Image.Image:
    """渲染 时间×数据源 矩阵（单张横向图，配合前端横向滚动显示）."""
    from pathlib import Path
    import re
    import rasterio
    from demo_v2.utils.constants import SOURCE_DISPLAY_NAMES, RAW_DIR
    from demo_v2.utils.visualization import colorize_worldcover

    patch_dir = Path(patch_dir)
    patch_id_name = patch_dir.name  # e.g. "patch_000000"

    # 部分 patch 的月度场景数据不完整，回退到季度聚合数据目录
    FALLBACK_RAW_DIR = Path("/workspace/raw/harbin")

    # 分类：时序 vs 静态
    TEMPORAL_ORDER = ["s2", "s1", "landsat", "s2_hr", "s1_hr", "highres", "modis_ndvi", "modis_lst", "era5"]
    STATIC_ORDER = ["dem", "dem_derived", "worldcover", "io_lulc",
                    "osm_buildings", "osm_landuse", "osm_roads",
                    "osm_waterways", "osm_railway"]

    def _src_dir(src: str) -> Path:
        """适配本机数据布局: RAW_DIR/{source}/{patch_id}/"""
        return RAW_DIR / src / patch_id_name

    def _fallback_src_dir(src: str) -> Path:
        """季度聚合数据回退目录"""
        return FALLBACK_RAW_DIR / src / patch_id_name

    def _extract_date(fname: str) -> str | None:
        # 支持 8 位日期 (20230101) 和季度 (2023Q1) 格式
        m = re.search(r'(\d{8})', fname)
        if m:
            return m.group(1)
        m = re.search(r'(\d{4}Q\d)', fname)
        if m:
            return m.group(1)
        return None

    def _month_label(date_str: str) -> str:
        if 'Q' in date_str:
            # 季度数据映射为代表月份，统一时间轴
            year = date_str[:4]
            quarter = int(date_str[-1])
            month_map = {1: '02', 2: '05', 3: '08', 4: '11'}
            return f"{year}-{month_map[quarter]}"
        return f"{date_str[:4]}-{date_str[4:6]}"

    def _render_thumb(ax, tif_path, src_name):
        """渲染单个缩略图到 axes。"""
        try:
            with rasterio.open(str(tif_path)) as ds:
                data = ds.read()
            if src_name in ("s2", "landsat", "s2_hr", "highres") and data.shape[0] >= 3:
                if src_name in ("s2", "s2_hr") and data.shape[0] >= 4:
                    rgb = data[[2, 1, 0]].astype(np.float32)
                else:
                    rgb = data[:3].astype(np.float32)
                valid = rgb[rgb > 0]
                if len(valid) > 0:
                    p2, p98 = np.percentile(valid, [2, 98])
                    if p98 > p2:
                        rgb = (rgb - p2) / (p98 - p2)
                rgb = np.clip(rgb, 0, 1).transpose(1, 2, 0)
                ax.imshow(rgb)
            elif src_name == "worldcover":
                ax.imshow(colorize_worldcover(data[0]))
            elif src_name in ("dem", "dem_derived"):
                ax.imshow(data[0], cmap="terrain")
            elif src_name == "modis_ndvi":
                ax.imshow(data[0], cmap="YlGn")
            elif src_name == "modis_lst":
                ax.imshow(data[0], cmap="RdYlBu_r")
            elif src_name in ("s1", "s1_hr"):
                if data.shape[0] >= 2 and data[1].max() > 0:
                    vv = data[0].astype(np.float32)
                    vh = data[1].astype(np.float32)
                    vv_n = np.clip((vv + 25) / 35, 0, 1)
                    vh_n = np.clip((vh + 30) / 35, 0, 1)
                    rgb = np.stack([vv_n, vh_n, vv_n / (vh_n + 1e-6) * 0.3], axis=-1)
                    rgb = np.clip(rgb, 0, 1)
                    ax.imshow(rgb)
                else:
                    vv = data[0].astype(np.float32)
                    vv_n = np.clip((vv + 25) / 35, 0, 1)
                    ax.imshow(vv_n, cmap="gray")
            else:
                ax.imshow(data[0], cmap="viridis")
        except Exception:
            ax.set_facecolor("white")
            for spine in ax.spines.values():
                spine.set_visible(False)

    # ── 收集时序数据 ──
    temporal_available: list[tuple[str, dict[str, list[Path]]]] = []
    all_months: set[str] = set()

    for src in TEMPORAL_ORDER:
        month_groups: dict[str, list[Path]] = {}
        # 1) 优先读取月度场景数据
        src_dir = _src_dir(src)
        if src_dir.exists():
            for f in sorted(src_dir.glob("*.tif")):
                date = _extract_date(f.stem)
                if date:
                    ml = _month_label(date)
                    month_groups.setdefault(ml, []).append(f)
                    all_months.add(ml)
        # 2) 回退读取季度聚合数据（仅补充缺失月份）
        fb_dir = _fallback_src_dir(src)
        if fb_dir.exists():
            for f in sorted(fb_dir.glob("*.tif")):
                date = _extract_date(f.stem)
                if date:
                    ml = _month_label(date)
                    if ml not in month_groups:
                        month_groups[ml] = [f]
                        all_months.add(ml)
        if month_groups:
            temporal_available.append((src, month_groups))

    # ── 收集静态数据 ──
    static_available: list[tuple[str, Path]] = []
    for src in STATIC_ORDER:
        src_dir = _src_dir(src)
        if src_dir.exists():
            files = sorted(src_dir.glob("*.tif"))
            if files:
                static_available.append((src, files[0]))
                continue
        # 回退目录
        fb_dir = _fallback_src_dir(src)
        if fb_dir.exists():
            files = sorted(fb_dir.glob("*.tif"))
            if files:
                static_available.append((src, files[0]))

    if not temporal_available and not static_available:
        fig, ax = plt.subplots(figsize=(8, 2), dpi=100)
        ax.text(0.5, 0.5, "No data sources available",
                ha="center", va="center", fontsize=14)
        ax.axis("off")
        img = _fig_to_pil(fig)
        plt.close(fig)
        return img

    # 按时间顺序排序月份，显示所有月份（前端通过 CSS 横向滚动查看）
    sorted_months = sorted(all_months)

    n_temporal = len(temporal_available)
    n_months = len(sorted_months)
    n_static = len(static_available)
    has_static = n_static > 0

    n_cols = max(n_months, n_static, 1)
    n_rows = n_temporal + (1 if has_static else 0)

    cell_w, cell_h = 2.0, 1.8
    fig_w = 2.5 + cell_w * n_cols + 0.5
    fig_h = 1.5 + cell_h * n_rows + 0.5
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(max(fig_w, 12), max(fig_h, 5)),
        dpi=120, squeeze=False,
    )

    # ── 渲染时序行 ──
    for row, (src_name, month_groups) in enumerate(temporal_available):
        for col in range(n_cols):
            ax = axes[row, col]
            ax.set_xticks([])
            ax.set_yticks([])
            if col < n_months:
                month = sorted_months[col]
                files = month_groups.get(month, [])
                if files:
                    _render_thumb(ax, files[0], src_name)
                    if len(files) > 1:
                        ax.text(
                            0.95, 0.05, f"x{len(files)}",
                            transform=ax.transAxes, fontsize=7,
                            color="white", ha="right", va="bottom",
                            bbox=dict(boxstyle="round,pad=0.1",
                                      fc="black", alpha=0.6),
                        )
                else:
                    ax.set_facecolor("white")
                    for spine in ax.spines.values():
                        spine.set_visible(False)
            else:
                ax.axis("off")

    # 时序列标题（月份）
    for col in range(min(n_months, n_cols)):
        axes[0, col].set_title(sorted_months[col], fontsize=10,
                               fontweight="bold", pad=8)

    # 时序行标签
    for row, (src_name, _) in enumerate(temporal_available):
        display = SOURCE_DISPLAY_NAMES.get(src_name, src_name)
        axes[row, 0].set_ylabel(display, fontsize=11, rotation=0,
                                labelpad=100, va="center", ha="right")

    # ── 渲染静态行（合并一行）──
    if has_static:
        static_row = n_rows - 1
        for col in range(n_cols):
            ax = axes[static_row, col]
            ax.set_xticks([])
            ax.set_yticks([])
            if col < n_static:
                src_name, tif_path = static_available[col]
                _render_thumb(ax, tif_path, src_name)
                display = SOURCE_DISPLAY_NAMES.get(src_name, src_name)
                ax.set_xlabel(display, fontsize=8, labelpad=4)
            else:
                ax.axis("off")
        axes[static_row, 0].set_ylabel(
            "Static", fontsize=11, rotation=0,
            labelpad=100, va="center", ha="right", fontweight="bold",
        )

    fig.suptitle(f"Time x Source Matrix : {patch_id}",
                 fontsize=15, fontweight="bold", y=0.98)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.12, right=0.98, top=0.90, bottom=0.06,
                        wspace=0.08, hspace=0.20)
    img = _fig_to_pil(fig)
    plt.close(fig)
    return img

def render_overview_map(
    patch_metas: list,
    highlight_id: str | None = None,
    color_by: str = "none",
) -> Image.Image:
    """渲染 AOI 概览地图 (Matplotlib)。"""
    dpi = 100
    fig, ax = plt.subplots(
        figsize=(IMG_WIDTH / dpi, IMG_HEIGHT / dpi), dpi=dpi,
    )

    pad_x = (UTM_X_MAX - UTM_X_MIN) * 0.05
    pad_y = (UTM_Y_MAX - UTM_Y_MIN) * 0.05
    ax.set_xlim(UTM_X_MIN - pad_x, UTM_X_MAX + pad_x)
    ax.set_ylim(UTM_Y_MIN - pad_y, UTM_Y_MAX + pad_y)
    ax.set_aspect("equal")
    ax.set_facecolor("#f0f4f8")

    if color_by == "s2_count":
        vals = [m.sources.get("s2", 0) for m in patch_metas]
        vmin, vmax = min(vals) if vals else 0, max(vals) if vals else 1
        cmap = plt.cm.YlGn
    elif color_by == "source_count":
        vals = [len(m.sources) for m in patch_metas]
        vmin, vmax = min(vals) if vals else 0, max(vals) if vals else 1
        cmap = plt.cm.Blues
    else:
        vals = [0] * len(patch_metas)
        vmin, vmax = 0, 1
        cmap = None

    for i, m in enumerate(patch_metas):
        xmin, ymin, xmax, ymax = m.bounds
        width = xmax - xmin
        height = ymax - ymin
        is_highlight = (m.patch_id == highlight_id)

        if cmap is not None and vmax > vmin:
            frac = (vals[i] - vmin) / (vmax - vmin)
            fc = cmap(frac, alpha=0.6)
        else:
            fc = "#4dabf7" if not is_highlight else "#ff6b6b"

        ec = "#e03131" if is_highlight else "#333333"
        lw = 2.5 if is_highlight else 0.5

        rect = mpatches.FancyBboxPatch(
            (xmin, ymin), width, height,
            boxstyle="square,pad=0",
            facecolor=fc if not is_highlight else "#ff6b6b80",
            edgecolor=ec,
            linewidth=lw,
            alpha=0.7 if not is_highlight else 0.9,
        )
        ax.add_patch(rect)

    ax.set_xlabel("UTM Easting (m)", fontsize=10)
    ax.set_ylabel("UTM Northing (m)", fontsize=10)
    n_total = len(patch_metas)
    title = f"Harbin AEF Patch Distribution ({n_total} patches)"
    if highlight_id:
        title += f" | Selected: {highlight_id}"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()
    img = _fig_to_pil(fig)
    plt.close(fig)
    return img


def render_patch_sources(
    patch_dir,
    patch_id: str,
    sources_to_show: list[str] | None = None,
) -> Image.Image:
    """渲染单个 patch 的多数据源可视化。"""
    from pathlib import Path
    import rasterio
    from demo_v2.utils.constants import RAW_DIR

    patch_dir = Path(patch_dir)
    patch_id_name = patch_dir.name
    if sources_to_show is None:
        sources_to_show = ["s2", "s1", "dem", "worldcover", "landsat", "modis_ndvi"]

    def _src_dir2(src: str) -> Path:
        return RAW_DIR / src / patch_id_name

    available = [s for s in sources_to_show if _src_dir2(s).exists()]
    n = max(len(available), 1)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), dpi=100)
    if nrows == 1 and ncols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, src_name in enumerate(available):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        src_dir = _src_dir2(src_name)
        files = sorted(src_dir.glob("*.tif"))
        if not files:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=12)
            ax.set_title(src_name)
            ax.axis("off")
            continue

        with rasterio.open(str(files[0])) as ds:
            data = ds.read()

        if src_name == "s2" and data.shape[0] >= 3:
            rgb = data[[2, 1, 0]].astype(np.float32)
            valid = rgb[rgb > 0]
            if len(valid) > 0:
                p2, p98 = np.percentile(valid, [2, 98])
                if p98 > p2:
                    rgb = (rgb - p2) / (p98 - p2)
            rgb = np.clip(rgb, 0, 1).transpose(1, 2, 0)
            ax.imshow(rgb)
            ax.set_title(f"Sentinel-2 RGB ({files[0].stem})")
        elif src_name == "s1" and data.shape[0] >= 2:
            vv = data[0].astype(np.float32)
            vh = data[1].astype(np.float32)
            vv_n = np.clip((vv + 25) / 35, 0, 1)
            vh_n = np.clip((vh + 30) / 35, 0, 1)
            rgb = np.stack([vv_n, vh_n, vv_n / (vh_n + 1e-6) * 0.3], axis=-1)
            rgb = np.clip(rgb, 0, 1)
            ax.imshow(rgb)
            ax.set_title(f"Sentinel-1 VV/VH ({files[0].stem})")
        elif src_name == "dem":
            ax.imshow(data[0], cmap="terrain")
            ax.set_title("DEM Elevation (m)")
        elif src_name == "worldcover":
            from demo_v2.utils.visualization import colorize_worldcover
            ax.imshow(colorize_worldcover(data[0]))
            ax.set_title("WorldCover")
        elif src_name == "landsat" and data.shape[0] >= 3:
            rgb = data[:3].astype(np.float32)
            valid = rgb[rgb > 0]
            if len(valid) > 0:
                p2, p98 = np.percentile(valid, [2, 98])
                if p98 > p2:
                    rgb = (rgb - p2) / (p98 - p2)
            rgb = np.clip(rgb, 0, 1).transpose(1, 2, 0)
            ax.imshow(rgb)
            ax.set_title(f"Landsat RGB ({files[0].stem})")
        else:
            ax.imshow(data[0], cmap="viridis")
            ax.set_title(f"{src_name} ({files[0].stem})")
        ax.axis("off")

    for idx in range(len(available), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].axis("off")

    fig.suptitle(f"Patch: {patch_id}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    img = _fig_to_pil(fig)
    plt.close(fig)
    return img


def _fig_to_pil(fig: plt.Figure) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    buf.seek(0)
    return Image.open(buf).copy()
