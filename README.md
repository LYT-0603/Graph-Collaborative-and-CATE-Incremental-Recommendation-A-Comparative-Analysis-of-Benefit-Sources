# 图协同推荐与 CATE 增量推荐的比较及收益来源分析

**Graph Collaborative and CATE Incremental Recommendation A Comparative Analysis of Benefit Sources**

本项目研究：图协同表示如何影响购买预测与个性化推荐 CATE 估计，以及按购买概率排序与按 CATE 排序分别带来什么收益。数据基础是淘宝 UserBehavior 的真实历史图；推荐处理和购买结果由明确声明的半合成机制生成。

## 材料入口

- [新版中文论文大纲](docs/图协同推荐与CATE增量推荐_中文论文大纲_摘要重写版.md)；同目录提供 Word 版。
- [课程汇报 PPT](outputs/图协同推荐与CATE增量推荐_淘宝数据完整实验_文献扩展_10分钟版_v3.pptx) 与 [十分钟讲稿](outputs/淘宝购物推荐项目_10分钟汇报讲稿.md)。PPT/讲稿保留既有课程版本；训练限制以本 README 和新版大纲为准。
- [逐种子结果](outputs/taobao_cate_graph_experiment/metrics_by_seed.csv)、[汇总及配对区间](outputs/taobao_cate_graph_experiment/key_results.json)。
- [数据来源与使用范围](DATA_SOURCES.md)、[方法与复现边界](docs/METHOD_NOTES.md)、[文献及原作者实现](docs/RELATED_WORK.md)。

## 已实现的方法

历史图先用无图 BPR 矩阵分解学习 32 维表示，再将该参数作为两层 LightGCN 的初始化继续训练。对每种冻结表示，拟合相同结构的共享 64 单元隐藏层及两个事实结果头，估计不推荐与推荐时的购买概率。两者之差作为估计 CATE。

由此构成四组等候选、等名单容量的策略：无图概率排序、无图 CATE 排序、LightGCN 概率排序、LightGCN CATE 排序。每用户固定 100 个候选，选择 Top-10。

**重要限制：** LightGCN 使用 MF 热启动和额外微调，表示比较包含训练方案差异。图亲和度生成器也使用预训练 LightGCN 分数，可能有利于同类方法。本项目没有声称隔离出纯架构优势，也没有完成交叉拟合 DR-Learner、真实曝光因果识别或线上 A/B 实验。

## 快速复现

建议 Python 3.12。主要实验只需 NumPy 和 pandas，无需 GPU 或 PyTorch。已包含固定历史样本、特征、候选、划分及参考输出，可直接运行：

```bash
python -m pip install -r requirements.txt
python work/taobao_project_presentation/run_full_experiment.py --output-dir artifacts/reproduction
python work/taobao_project_presentation/summarize_experiment.py --input-dir artifacts/reproduction
python tools/verify_reproduction.py --actual artifacts/reproduction
```

复现输出写入 `artifacts/reproduction`，原始参考结果位于 `outputs/taobao_cate_graph_experiment`。运行时间取决于 CPU 与 BLAS 配置，通常为数分钟。跨平台浮点差异可能影响早停；校验脚本报告数值差异，不承诺所有平台逐字节相同。

数据预处理和图可视化另需安装 `requirements-visualization.txt`：

```bash
python -m pip install -r requirements-visualization.txt
python work/taobao_project_presentation/preprocess_project.py --graph-dir outputs/taobao_project_preprocessing/full_sample --output artifacts/preprocessed
```

该命令从已附抽样历史图重新生成 core、特征、用户划分及候选。若要从完整原始 ZIP 开始，请从[天池数据页](https://tianchi.aliyun.com/dataset/649)自行下载，再运行：

```bash
python outputs/taobao_project_preprocessing/build_taobao_graph.py --input /path/to/UserBehavior.csv.zip --output artifacts/historical_graph
python work/taobao_project_presentation/preprocess_project.py --graph-dir artifacts/historical_graph --output artifacts/preprocessed --source-zip /path/to/UserBehavior.csv.zip
```

主训练入口固定读取已附 `outputs/taobao_project_preprocessing/core2`，便于与论文参考结果一致。若改用新抽样图，需调整代码中 `DATA/CORE` 并重新记录配置，不能将不同数据生成的输出混为同一次复现。

## 实验设置与结果

2-core 历史图：1,801 位用户、13,060 件商品、35,144 条配对边。训练/验证/测试用户为 1,081/360/360；两个环境、五个种子、两种表示共 20 次结果模型拟合。

|半合成机制|表示|购买概率 RMSE|CATE RMSE|CATE 秩相关|概率排序增量|CATE 排序增量|
|---|---|---:|---:|---:|---:|---:|
|feature_effect|无图 MF|0.0244|0.0178|0.7631|0.7238|0.7149|
|feature_effect|LightGCN|0.0227|0.0171|0.7744|0.7198|0.7152|
|graph_effect|无图 MF|0.0438|0.0420|0.7875|2.4021|2.4111|
|graph_effect|LightGCN|0.0299|0.0273|0.9234|2.5362|2.5431|

数值为五种子均值；增量单位为每位测试用户 Top-10 名单的期望新增购买次数。图效应环境中，固定 LightGCN 结果头时 CATE 排序相对概率排序的增量差约 +0.0069，配对种子 95% t 区间约 [0.0029, 0.0110]；名单期望购买差的区间仍跨零。feature_effect 中增量差约 −0.0046，区间跨零。

这些结果来自当前自定义机制，不构成真实淘宝销量提升或一般性方法优越性的证明。

## 指标含义

- **购买概率 RMSE**：预测推荐后购买概率与模拟真实概率的均方根误差，不是以观测二元标签计算的 Brier score。
- **CATE RMSE / MAE**：估计效应与真实效应的均方根/绝对误差，衡量效应数值恢复能力。
- **Spearman 相关**：估计效应与真效应的整体排序一致性；并不等于逐用户 Top-10 价值。
- **自然购买 B**：所选名单内不推荐时的期望购买次数，先按用户求和再取均值。
- **增量购买 ΔV**：所选名单内真实 CATE 之和，再对用户平均。
- **名单期望购买 L**：所选名单内推荐后的期望购买，等于 B 加 ΔV。它不是完整候选全集的总购买指标。
- **Oracle / regret**：真实效应排序最优增量、真实推荐后概率排序最优名单购买；regret 为对应目标上 Oracle 与估计策略的差距。Oracle 使用不可部署真值。
- **95% 区间**：五个共同模拟种子的配对差 t 区间（自由度 4），不覆盖重新抽样图、用户总体或不同机制的不确定性。

## 目录

```text
work/taobao_project_presentation/  预处理、训练、指标汇总源码
tools/                           数据与复现校验
outputs/taobao_project_preprocessing/  固定图、core、特征、候选、划分、可视化
outputs/taobao_cate_graph_experiment/   半合成标签、预测、训练记录和参考结果
outputs/                         既有 PPT 和讲稿
docs/                            新版中英文题名、中文摘要大纲、方法与文献
SHA256SUMS.json                   文件校验清单
```

这里的模型代码是本项目的 NumPy 独立实现，参考成熟论文的方法思想，不是原作者官方代码的完整移植。数据与相关论文的权利归其提供者；公开仓库不等于对第三方数据授予额外许可。
