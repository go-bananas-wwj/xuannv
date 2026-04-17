# xuannv_embdding Demo V2 — 统一可视化平台

基于 Gradio 的交互式 Web 界面，整合模型概览、数据浏览、Embedding 可视化、变化检测、模型对比和下游任务评估。

## 快速启动

```bash
cd /workspace/xuannv
python demo_v2/app.py --port 7870
```

公网分享：
```bash
python demo_v2/app.py --port 7870 --share
```

## 五个核心标签页

| Tab | 功能 | 说明 |
|-----|------|------|
| 🏠 模型概览 | 项目介绍、版本演进、反坍缩机制可视化 | 展示 V1→V2→V3 的改进故事 |
| 🗺️ 数据与 Embedding 场 | Folium 交互地图、Time×Source 矩阵、PCA-RGB 可视化 | 支持单 patch 和全域 mosaic 查看 |
| 🔥 交互式变化检测 | 选择 Patch + 时间窗口 → 变化热力图/二值化图/叠加图 | 体现"任意窗口查询"核心能力 |
| ⚖️ 模型能力对比 | V1 vs V2 vs V3 同屏对比 | 统一实时推理，公平对比 |
| 📊 下游任务 | 湿地监测、少样本 AUC 曲线 | 展示通用表征能力 |

## 预计算数据

- **V1**: 已有预计算 `embedding_maps.npy`，PCA-RGB 和快速查看无需实时推理
- **V2/V3**: 仅有 checkpoint，变化检测和对比功能会自动走实时推理（约 3-5s/patch）

如需为 V2/V3 生成预计算 embedding：
```bash
python scripts/export_embeddings.py \
    --config configs/qwen_v2_temporal.yaml \
    --checkpoint /workspace/outputs/xuannv_embdding_v2/best.pt \
    --output-dir /workspace/outputs/xuannv_embdding_v2/embeddings

python scripts/export_embeddings.py \
    --config configs/qwen_v3_temporal.yaml \
    --checkpoint /workspace/outputs/xuannv_embdding_v3/epoch_599.pt \
    --output-dir /workspace/outputs/xuannv_embdding_v3/embeddings
```

## 目录结构

```
demo_v2/
├── app.py                    # Gradio 主入口
├── config.py                 # 模型注册与状态查询
├── cache_manager.py          # 统一缓存（embeddings / S2 RGB / Grid）
├── engines/
│   ├── model_engine.py       # 模型加载与实时推理
│   └── change_detection.py   # 变化检测引擎
├── components/
│   ├── project_intro.py      # Tab 1: 模型概览
│   ├── data_browser.py       # Tab 2: 数据浏览器
│   ├── embedding_viz.py      # Tab 2: Embedding PCA-RGB
│   ├── change_detection_tab.py  # Tab 3: 变化检测
│   ├── comparison_tab.py     # Tab 4: 模型对比
│   └── downstream_tab.py     # Tab 5: 下游任务
└── utils/
    ├── constants.py          # 路径、颜色表、时间窗口预设
    ├── visualization.py      # 公共绘图函数
    └── map_utils.py          # Folium 地图 + Time×Source 矩阵
```
