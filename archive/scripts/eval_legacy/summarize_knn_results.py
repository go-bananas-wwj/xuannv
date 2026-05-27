#!/usr/bin/env python3
import json
from pathlib import Path

EXPS = ["aef_baseline", "aef_high_consist", "aef_no_static", "aef_skip_l2",
        "aef_diff_recon", "aef_high_kappa", "aef_cyclic_unif", "aef_no_uniform"]

tasks = ["worldcover", "jrc_water", "dynamic_world"]
print("# Round 1 下游 KNN 评估汇总\n")
print("| 实验 | WorldCover Acc | WorldCover mIoU | JRC Water Acc | Dynamic World Acc |")
print("|------|---------------|-----------------|---------------|-------------------|")
for exp in EXPS:
    path = Path(f"/workspace/outputs/xuannv_round1/{exp}/downstream_knn/results.json")
    if not path.exists():
        continue
    with open(path) as f:
        d = json.load(f)
    vals = []
    for task in tasks:
        k5 = d.get(task, {}).get("k5", {})
        vals.append(f"{k5.get('accuracy',0)*100:.2f}%")
        vals.append(f"{k5.get('mean_iou',0)*100:.2f}%")
    print(f"| {exp} | {vals[0]} | {vals[1]} | {vals[2]} | {vals[4]} |")

print("\n## 结论")
print("- 所有实验的 backbone 表征质量都极差，远低于可用水平")
print("- 自编码器 Way A 只学到空间重建，没有语义/时序表征")
print("- 必须切换到 Way B（跨时相重建）才能学到有用的时序敏感特征")
