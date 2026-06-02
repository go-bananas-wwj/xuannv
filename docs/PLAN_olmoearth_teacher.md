# 用 OlmoEarth 作为教师模型提升玄女（xuannv）效果 — 详细计划

> 撰写日期：2026-06-02
> 目标任务：通用遥感 embedding / 表征学习（兼顾变化检测 AUC 与 KNN 下游）
> 数据：哈尔滨新区（harbin）+ 海淀（haidian）
> 决策：**蒸馏方案（A）与微调方案（B）并行做对比实验，择优**

---

## 0. TL;DR（结论先行）

| 维度 | 方案 A：知识蒸馏（OlmoEarth→玄女） | 方案 B：直接微调 OlmoEarth |
|------|-----------------------------------|----------------------------|
| 产物 | 仍是玄女 128D 模型（轻量、上线不变） | 89M 的 OlmoEarth-Base 模型 |
| 改动量 | 中（加一个 teacher 分支 + 投影头 + 蒸馏 loss） | 大（换推理栈、换 NPU 适配、换下游头） |
| NPU 友好度 | 高（teacher 仅前向、可离线/可冻结） | 中（OlmoEarth 89M 全量微调，显存/算子风险） |
| 解决"数据少"短板 | ★★★（把 OlmoEarth 全球先验蒸给玄女） | ★★（在小数据上微调易过拟合） |
| 上线成本 | 不变（玄女架构不变） | 高（embedding 768D，下游全部重训） |
| **推荐** | **首选**（主线） | 作为对照上界（baseline 参考） |

**核心判断**：玄女的最大短板是**训练数据规模太小（424 patch）**，而非架构。OlmoEarth 携带 28.5 万全球样本的先验，最适合作为"软标签/特征教师"把先验**蒸馏**进玄女，既保留玄女的反坍缩与双窗口时序优势，又补上数据短板。因此**以方案 A 为主线，方案 B 作为效果上界对照**。

---

## 1. 两个核心疑问的技术解答

### Q1：输入不一样 —— OlmoEarth 对不同数据源处理不同，怎么对齐？

**OlmoEarth 的输入机制（已核实代码）**：
- 每个模态**独立 tokenize**：`olmoearth_pretrain/nn/tokenization.py` + `flexi_patch_embed.py`，每个模态有自己的 patch-embed 分支（按 BandSet 分组），再拼成统一 token 序列进 FlexiViT 编码器。
- 输入张量布局 **BHWTC**（Batch, H, W, Time, Channel），配合 `*_mask`（BHWTS，S=band set 数）和 `timestamps`（day, month0-11, year）。
- **天生支持缺失模态**：`MaskedOlmoEarthSample` 只需传入你拥有的模态字段即可，其余模态不传 = 缺失。代码库有 `missing_modalities_training` 佐证。
- band order 必须严格对齐（`Modality.*.band_order`）：
  - **S2 L2A**（12 波段，3 个 BandSet）：`B02 B03 B04 B08`(10m) + `B05 B06 B07 B8A B11 B12`(20m) + `B01 B09`(40m)
  - **S1**（2 波段）：`vv vh`
  - **Landsat**（11 波段，2 个 BandSet）：`B8`(15m) + `B1 B2 B3 B4 B5 B6 B7 B9 B10 B11`(30m)

**玄女当前输入（已核实 configs/config.yaml + src/data/transforms.py）**：
- `INPUT_SOURCES = ["s2", "s1", "landsat"]`，`source_channels: {s2:6, s1:2, landsat:6}`
- 即玄女 S2 只用 6 波段、Landsat 只用 6 波段，而 OlmoEarth 要 12/11 波段。

**对齐策略（关键工程点）**：
1. **教师侧用全波段**：给 OlmoEarth 喂"完整波段"以发挥其能力——这要求在数据预处理阶段为 teacher 单独准备 12 波段 S2 + 2 波段 S1 +（可选）Landsat。哈尔滨/海淀原始数据目录已含 `s2 s1 landsat`（见 `/workspace/raw/...`），需确认能否补齐到 OlmoEarth 的波段集。
2. **波段缺失就降级**：若只能拿到玄女的 6 波段 S2，则只填入对应 BandSet 的可用波段，其余波段位置用 mask 标记缺失（OlmoEarth 支持 band-set 级缺失）。**优先只用 S2（最稳）作为教师输入**，S1/Landsat 作为增强。
3. **空间/patch 对齐**：玄女 `image_size=128`、stem stride=2 → embedding_map 约 64×64；OlmoEarth 推理 `patch_size=4` 在 128×128 上得 32×32 token。蒸馏时需把两者 resize/插值到同一空间网格（建议统一到玄女的 embedding_map 分辨率，对 teacher 特征做双线性插值）。
4. **结论**：输入"处理不一样"**不需要改玄女的输入**——teacher 与 student 各用各的输入管线，只要求两者覆盖**同一时空 patch**，在**输出特征层**对齐即可（特征蒸馏天然解耦输入差异）。

### Q2："价格不一样" —— 维度/规模/算力不一样，怎么解决？

理解为三层不匹配，逐一解决：

| 不匹配 | OlmoEarth-Base | 玄女 | 解决方案 |
|--------|---------------|------|----------|
| **embedding 维度** | 768D | 128D | 加 **投影头 projector**：把玄女 128D 经 MLP 升到 768D 去对齐 teacher（或把 teacher 768D 降到 128D）。蒸馏在投影后的空间算 loss，推理时丢弃 projector，玄女仍输出 128D。 |
| **空间粒度** | patch=4（32×32 token） | stem/2（~64×64） | teacher 特征双线性插值到 student 网格；或 student 下采样到 teacher 网格。 |
| **算力/参数** | 89M（推理 ~GB 级显存） | 轻量 | teacher **只前向、可冻结、可离线预提取**特征缓存到磁盘，训练时直接读取，几乎不增加 NPU 训练开销。 |

**"价格"的另一种可能含义（算力成本）**：OlmoEarth-Base 89M 在 910B4 上做**纯前向推理**完全可行（config 里 `use_flash_attn=false`，无 CUDA 专属算子，纯 PyTorch，可走 torch_npu）。建议**离线一次性提取 teacher 特征**，避免训练时重复跑 teacher。

---

## 2. 现状盘点（已核实）

**玄女 (xuannv)**
- 架构：`AEFModel`（src/models/model.py）= SensorEncoderBank（多源独立 stem）+ STPEncoder（Space/Time/Precision，8 blocks/8 heads）+ VMFBottleneck。
- 输出：`AEFOutput.embedding_map [B,128,H,W]` / `embedding [B,128]` / `pre_norm_*`。
- 训练：`DDPv13Trainer`（src/training/trainer.py），自监督（重建 + 5 类反坍缩 + 双窗口时序对比）。
- **已内置 EMA teacher-student**（config: `teacher_momentum=0.996`）—— 自蒸馏框架已在，**接外部 teacher 改造成本低**。
- 硬件：8×910B4，hccl，torch2.1+torch_npu，conda `xuannv`。
- 现成 checkpoint：`/workspace/outputs/haidian_train/exp_v25_haidian_loss_opt_0530/epoch_best_epoch69.pt` 等可作 student 初始化。

**OlmoEarth**
- 本地权重：`/workspace/.container_root/.cache/huggingface/hub/models--allenai--OlmoEarth-v1-Base/snapshots/.../weights.pth`（830MB，Base，768D）。
- 加载：`from olmoearth_pretrain.model_loader import ModelID, load_model_from_id; load_model_from_id(ModelID.OLMOEARTH_V1_BASE)`。
- 取特征：`model.encoder(sample, fast_pass=True, patch_size=4)["tokens_and_masks"].sentinel2_l2a`，再 `.mean(dim=[3,4])` 得 `BHWC` 特征图。

---

## 3. 方案 A（主线）：OlmoEarth → 玄女 知识蒸馏

### A.1 蒸馏范式选择
采用**特征蒸馏（feature/representation distillation）**，而非 logit 蒸馏（无分类 logit 可言）：
- 让玄女 student 的（投影后）embedding_map 去**回归/对齐** OlmoEarth teacher 的 patch 特征图。
- loss 候选：① cosine 对齐（`1 - cos`）；② smooth-L1；③ 关系蒸馏（RKD，对齐 patch-patch 相似度矩阵，避免维度尺度问题，强烈推荐与 cosine 组合）。

### A.2 架构改造（最小侵入）
新增文件 / 改动（均在 `/workspace/xuannv/src/` 内）：
1. `src/distill/olmoearth_teacher.py`（**新建**）
   - 封装 OlmoEarth 加载（torch_npu, eval, requires_grad_(False)）。
   - `extract_teacher_feature(s2, s1, landsat, timestamps) -> [B, 768, h, w]`。
   - 处理 band 对齐、归一化（用 `Normalizer(Strategy.COMPUTED)`）、缺失模态 mask。
2. `src/models/projector.py`（**新建**）
   - `DistillProjector(128 -> 768)`：2 层 MLP + GELU + LN。仅训练期使用。
3. `src/training/losses.py`（**追加**）
   - `feature_distill_loss(student_proj, teacher_feat)`：cosine + RKD 组合。
4. `src/training/trainer.py`（**改动**）
   - 在 `DDPv13Trainer` 加入 teacher 前向（或读离线缓存）+ projector + `distill_weight`。
5. `configs/config_distill.yaml`（**新建**，继承 config.yaml）
   - 新增 `distill:` 段：`enabled, teacher_ckpt, teacher_patch_size, distill_weight, distill_type(cosine|rkd|both), offline_cache_dir, projector_dims`。

### A.3 离线特征缓存（推荐，降 NPU 压力）
- 新脚本 `scripts/distill/precompute_teacher.py`：
  - 遍历哈尔滨/海淀所有 patch×月份，跑 OlmoEarth 前向，存 `teacher_feat/{patch}/{YYYYMM}.npy`（fp16）。
  - 与玄女 dataset 的 `monthly_samples` 索引对齐（patch_id, year, month）。
- 训练时 dataset 额外返回对应 teacher 特征张量；trainer 直接算蒸馏 loss。
- 预估存储：768D × 32×32 × fp16 ≈ 1.5MB/样本 × (patch×月份数)。哈尔滨 424 patch × ~12 月 ≈ 7–8GB，可接受。

### A.4 训练流程
1. 数据对齐校验（patch/月份/时空范围一致）。
2. 离线提取 teacher 特征（一次性，7 卡并行，复用 `launch_eval.sh` 分片思路）。
3. 用现成 checkpoint 软启动 student（`--soft-restart epoch_best_epoch69.pt`）。
4. 联合训练：`总 loss = 原玄女自监督 loss + distill_weight × 蒸馏 loss`。
   - `distill_weight` 建议从 0.5 起，warmup 后调到 1.0；监控 `raw_unif` 不被蒸馏带崩（>-0.5 报警）。
5. 在哈尔滨先验证管线，再迁移海淀（`--soft-restart` 跨域）。

### A.5 关键超参（初值建议）
- `distill_weight: 0.5 → 1.0`（cosine_warmup 同步）
- `distill_type: both`（cosine 0.7 + RKD 0.3）
- teacher `patch_size: 4`，teacher 输入优先 **S2 全 12 波段**（S1/Landsat 作消融）
- projector dims: `[128, 384, 768]`
- 其余沿用 Round 9 基线（lr 1e-4, 100ep×200step, bs4）

---

## 4. 方案 B（对照上界）：直接微调 OlmoEarth

### B.1 思路
把 OlmoEarth-Base 当 backbone，在哈尔滨/海淀上做**轻量微调 + 线性探针**，得到该数据上的"能力天花板"，用于判断方案 A 蒸馏到了几成。

### B.2 步骤
1. NPU 环境适配 OlmoEarth 推理栈（torch_npu；确认无 flash_attn 依赖，已核实可关闭）。
2. 冻结编码器，仅训练下游头（线性探针）→ 得无微调上界。
3. 解冻最后 N 层 + LoRA/Adapter 微调（小数据防过拟合）。
4. 评估同一套指标（AUC + KNN）。

### B.3 风险
- 89M 全量微调在 424 patch 上**极易过拟合**，故只做**线性探针 + 末层/LoRA 微调**。
- embedding 768D，下游头需重训；不替换线上玄女，仅作 benchmark。

---

## 5. 统一评估协议（A、B 同口径对比）

复用玄女现成评估（`scripts/eval/`）：
| 指标 | 脚本 | 目标 |
|------|------|------|
| 变化检测 AUC | `auc_eval.py` | >0.7 及格, >0.8 良好, >0.85 优秀 |
| KNN 下游（WorldCover / JRC Water / Dynamic World） | `knn_eval.py` | 越高越好 |
| embedding 健康度 | trainer 指标 | `raw_unif∈[-4,-1]`, `erank>32`, `active_dims=64/64` |

**对照组设计**：
1. **C0 基线**：玄女 Round 9（无蒸馏）。
2. **C1 蒸馏(A)**：玄女 + OlmoEarth 蒸馏（主线）。
3. **C2 上界(B)**：OlmoEarth 线性探针 / LoRA。
4. 消融：teacher 输入模态（仅S2 vs S2+S1+Landsat）、distill_type（cosine vs RKD vs both）、distill_weight。

**判定**：若 C1 在 AUC/KNN 上显著超过 C0，且接近 C2，则蒸馏成功；否则回退检查特征对齐/权重。

---

## 6. 里程碑

| 阶段 | 交付物 | 验收 |
|------|--------|------|
| M1 调研对齐 | 本计划 + band/时空对齐校验脚本 | 确认 teacher/student patch 一一对应 |
| M2 Teacher 离线特征 | `precompute_teacher.py` + 缓存 | 哈尔滨全量特征落盘，抽查可视化合理 |
| M3 蒸馏管线打通 | teacher 分支 + projector + loss + config | 冒烟测试通过，loss 正常下降，`raw_unif` 不崩 |
| M4 哈尔滨蒸馏实验 | C0/C1 对比 | C1 AUC/KNN ≥ C0 |
| M5 上界对照 | C2（OlmoEarth 探针） | 给出能力天花板 |
| M6 海淀迁移 + 消融 | 跨域结果 + 消融表 | 跨域不退化，选出最优配置 |

---

## 7. 风险与对策

| 风险 | 对策 |
|------|------|
| teacher 与 student 时空 patch 对不齐 | M1 写校验脚本，按 (patch_id, year, month) 严格索引 |
| 蒸馏把玄女反坍缩特性带崩（uniform 退化） | distill_weight 小步上升 + 监控 `raw_unif`/`erank`，必要时只蒸 RKD（关系蒸馏不破坏分布） |
| OlmoEarth 波段在哈尔滨数据缺失 | 优先只用 S2；缺失波段走 band-set mask |
| OlmoEarth 在 NPU 算子不兼容 | 已确认关 flash_attn；若仍有问题，teacher 改在 CPU/单卡离线提取 |
| 维度尺度差异导致 L2 蒸馏不稳 | 用 cosine + RKD（尺度无关），不用裸 MSE |
| 数据太少过拟合 | 蒸馏本身即正则；方案 B 只做探针/LoRA |

---

## 8. 立即可执行的第一步（M1）

1. 校验哈尔滨数据能否补齐 OlmoEarth 所需 S2 波段：检查 `/workspace/raw/phase1_harbin/harbin_scenes_cloud_filtered/s2` 的波段数与命名。
2. 写 `scripts/distill/check_alignment.py`：列出玄女 `monthly_samples` 与原始影像的时空对应，确认能为每个样本生成 teacher 输入。
3. 在 1 个 patch 上跑通 OlmoEarth 前向（torch_npu），确认输出 `[B,768,h,w]` 形状与数值正常。

> 经确认后，即进入 M2 离线特征提取与 M3 蒸馏管线开发。
