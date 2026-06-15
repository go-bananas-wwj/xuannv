# xuannv v1 Haidian 生产包

本目录封装在海淀标注下游任务上表现最好的 checkpoint `exp_multires_v1_0612_backup/epoch_80.pt`，
提供三类推理能力：

1. 海淀 6 个标注任务（单时相 / 双时相，Linear / MLP）
2. WorldCover kNN 语义分割
3. 哈尔滨双时相变化检测（cosine distance）

> 注：下方目录结构为最终完整形态。执行 Task 1 后会生成 `model/` 与 `scripts/copy_model.sh`；
> `xuannv_v1/`、`tests/` 以及 `run_*.sh` 脚本会在后续任务中生成。

## 目录结构

```
production/v1_haidian/
├── README.md
├── model/
│   ├── config_multires_v1.yaml
│   ├── epoch_80.pt          # 由 scripts/copy_model.sh 生成，不入 git
│   └── CHECKPOINT_SOURCE    # 由 scripts/copy_model.sh 生成
├── xuannv_v1/
│   ├── __init__.py
│   ├── backbone.py
│   ├── haidian_tasks.py
│   ├── worldcover_knn.py
│   └── changedetection.py
├── scripts/
│   ├── copy_model.sh
│   ├── run_haidian.sh
│   ├── run_worldcover.sh
│   └── run_changedetection.sh
├── outputs/                  # 推理结果（gitignored）
└── tests/
    ├── test_backbone.py
    ├── test_haidian_tasks.py
    └── test_changedetection.py
```

## 快速开始

1. 复制权重：
   ```bash
   bash production/v1_haidian/scripts/copy_model.sh
   ```

2. 运行海淀 6 任务：
   ```bash
   bash production/v1_haidian/scripts/run_haidian.sh
   ```

3. 运行 WorldCover kNN：
   ```bash
   bash production/v1_haidian/scripts/run_worldcover.sh
   ```

4. 运行哈尔滨变化检测：
   ```bash
   bash production/v1_haidian/scripts/run_changedetection.sh
   ```

## 说明

- `model/epoch_80.pt` 约 1.4GB，默认不入 git。
- 脚本默认设备为 `npu:0`；如无 NPU，请在脚本中改为 `--device cpu`。
- 所有输出写入 `production/v1_haidian/outputs/`。
