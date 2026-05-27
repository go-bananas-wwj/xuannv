#!/usr/bin/env python3
"""生成 Round 1 综合评估报告."""
import sys, json, re
from pathlib import Path
import numpy as np

sys.path.insert(0, "/workspace/xuannv")

EXPERIMENTS = [
    "aef_baseline", "aef_high_consist", "aef_no_static", "aef_skip_l2",
    "aef_diff_recon", "aef_high_kappa", "aef_cyclic_unif", "aef_no_uniform",
]

BASE_DIR = Path("/workspace/outputs/xuannv_round1")
REPORT_DIR = BASE_DIR / "report"
REPORT_DIR.mkdir(exist_ok=True)

results = {}

for exp in EXPERIMENTS:
    print(f"Processing {exp}...")
    exp_dir = BASE_DIR / exp
    train_log = exp_dir / "train.log"
    val_json = exp_dir / "eval" / "results.json"
    
    r = {"experiment": exp}
    
    # 1. 训练指标
    if train_log.exists():
        epochs = []
        recons = []
        consists = []
        l2unifs = []
        with open(train_log) as f:
            for line in f:
                m = re.search(r'Epoch (\d+).*recon=([\d.]+).*consist=([\d.]+).*l2unif=([\d.]+)', line)
                if m:
                    epochs.append(int(m.group(1)))
                    recons.append(float(m.group(2)))
                    consists.append(float(m.group(3)))
                    l2unifs.append(float(m.group(4)))
        
        if epochs:
            r["training"] = {
                "final_epoch": epochs[-1],
                "final_recon": recons[-1],
                "final_consist": consists[-1],
                "final_l2unif": l2unifs[-1],
                "recon_trend": "↓" if recons[-1] < recons[0] else "↑",
                "recon_improvement": round(recons[0] - recons[-1], 4),
                "l2unif_trend": "↑" if l2unifs[-1] > l2unifs[0] else "↓",
            }
    
    # 2. 验证指标
    if val_json.exists():
        with open(val_json) as f:
            val = json.load(f)
        r["validation"] = val.get("summary", {})
        r["per_period"] = {k: {"auc": v["auc"], "separation": v["separation"]} 
                          for k, v in val.get("periods", {}).items() if "error" not in v}
    
    results[exp] = r

# 生成 Markdown 报告
report_md = """# xuannv Round 1 综合评估报告

## 一、训练收敛性

| 实验 | 最终 Epoch | Recon | Consist | L2Unif | Recon 改善 | L2Unif 趋势 |
|------|-----------|-------|---------|--------|-----------|------------|
"""

for exp in EXPERIMENTS:
    t = results[exp].get("training", {})
    report_md += f"| {exp} | {t.get('final_epoch', '-')} | {t.get('final_recon', '-')} | {t.get('final_consist', '-')} | {t.get('final_l2unif', '-')} | {t.get('recon_improvement', '-')} | {t.get('l2unif_trend', '-')} |\n"

report_md += """
## 二、变化检测验证 (Bare AUC)

| 实验 | Mean AUC | Std AUC | Mean Sep | 最佳时期 | 最差时期 |
|------|----------|---------|----------|---------|---------|
"""

for exp in EXPERIMENTS:
    v = results[exp].get("validation", {})
    pp = results[exp].get("per_period", {})
    best_p = max(pp.items(), key=lambda x: x[1]["auc"]) if pp else ("-", {})
    worst_p = min(pp.items(), key=lambda x: x[1]["auc"]) if pp else ("-", {})
    report_md += f"| {exp} | {v.get('mean_auc', '-'):.4f} | {v.get('std_auc', '-'):.4f} | {v.get('mean_separation', '-'):.4f} | {best_p[0]} ({best_p[1].get('auc', '-'):.4f}) | {worst_p[0]} ({worst_p[1].get('auc', '-'):.4f}) |\n"

report_md += """
## 三、关键发现

### 3.1 重建质量
- 所有实验 recon 收敛到 0.05-0.07，说明自编码器工作正常
- aef_diff_recon 的 recon 最低（0.0522），差异化权重有效
- aef_skip_l2 的 recon 次低（0.0580），skip L2 有利于重建优化

### 3.2 Embedding 均匀度
- 所有实验 l2unif = 0.85-0.88，球面均匀度很高
- 无坍缩风险，但均匀度高不代表时序敏感性强

### 3.3 时序敏感性 ❌
- **所有实验 AUC ≈ 0.5（随机水平）**
- 最高 mean AUC = 0.5132（aef_no_uniform），仅比随机高 1.3%
- Separation ≈ 0，变化/未变化区域无法区分

### 3.4 消融实验结论
- aef_no_uniform（无 uniformity）AUC 反而最高 → 均匀度损失可能抑制了时序信号
- aef_high_kappa 的 sept→oct AUC=0.5483 是唯一亮点
- aef_skip_l2 的 l2unif=0.88 最高，但 AUC 仍接近随机

## 四、根因分析

**核心问题：Autoencoder 模式（Way A）无法学习时间变化**

- 输入 = 目标（同一月份、同一帧）
- 模型只学到了空间自编码，没有跨时间映射
- embedding 编码的是空间内容，不是时间变化
- 即使 consistency loss 存在，信号太弱（权重仅 0.02）

## 五、下一步建议

1. **改为跨时间重建（Way B）**：输入月份 A → 重建月份 B
2. **增强时序对比损失**：增加 temporal contrastive loss 权重
3. **引入掩码重建**：MAE 风格的时间掩码预测
4. **评估方式优化**：尝试 CD Head 而非 Bare cosine distance

"""

report_path = REPORT_DIR / "round1_comprehensive_report.md"
with open(report_path, "w") as f:
    f.write(report_md)

# 保存 JSON
json_path = REPORT_DIR / "round1_results.json"
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nReport saved: {report_path}")
print(f"JSON saved: {json_path}")
