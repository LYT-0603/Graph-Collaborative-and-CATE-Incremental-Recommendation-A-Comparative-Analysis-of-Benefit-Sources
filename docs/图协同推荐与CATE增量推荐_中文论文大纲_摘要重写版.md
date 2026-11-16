# 图协同推荐与 CATE 增量推荐的比较及收益来源分析

作者：李禹潼　预计初稿完成时间：12.20

英文题名：Graph Collaborative and CATE Incremental Recommendation A Comparative Analysis of Benefit Sources

## 摘要

学习用户与商品之间的协同关系，是现代推荐系统提升购买预测能力的重要途径。以 LightGCN 为代表的图推荐方法通过历史交互传播用户与商品表示，但更准确地识别高购买概率商品，并不意味着推荐本身促成了更多购买。条件平均处理效应（CATE）关注推荐相对于不推荐的购买增量，为区分既有需求与可被推荐改变的行为提供了另一种决策依据。因此，有必要在统一评价条件下，分别考察图协同信息与增量排序目标的作用。

本文基于淘宝多行为日志构建用户—商品历史二部图，在保留真实交互结构与处理前特征的基础上，建立具有已知反事实真值的半合成环境。我们结合无图矩阵分解和 LightGCN 两类表示，通过双结果头估计推荐与不推荐时的购买概率，形成购买概率排序与 CATE 排序的交叉比较框架。该框架同时考察购买预测、效应估计与推荐策略，并将列表期望购买分解为自然购买和推荐增量，以解释不同方法的收益来源。

在两种预设机制及五个随机种子的实验中，当 LightGCN 协同信号被纳入效应生成机制时，图表示对应的 CATE 均方根误差由约 0.0420 降至 0.0273；基于 CATE 的排序获得了较小但稳定的增量优势，而列表期望购买差异的区间仍跨零。另一机制下的增量优势则缺乏明确证据。这些结果表明，购买预测、效应恢复与策略增量需要分别评价。本文提供了一套可复现的收益分解框架；相关发现仅适用于当前半合成设定，仍需独立生成机制与更多数据集检验。

**关键词：** LightGCN；条件平均处理效应；图协同推荐；半合成数据；策略增量价值

## 引言

购物推荐通常根据用户偏好预测商品被购买的可能性。矩阵分解和图协同过滤为这一任务提供了成熟方法，NGCF 与 LightGCN 进一步利用用户—商品连接传播协同信息[1–4]。然而，购买预测混合了用户的既有需求与推荐可能带来的变化。若将所有推荐后购买都视为推荐贡献，容易高估对原本就会购买的用户或商品的干预价值。

因果推荐与 uplift 研究将推荐视为处理，关注有推荐与无推荐的结果差异[5–9]。LightGCN 提供关系表示，CATE 定义推荐增量，两者可以组合使用。本文据此将“采用何种历史表示”与“按何种分数排序”拆开比较，避免把模型架构与决策目标混为一谈。研究贡献定位于比较设计和收益解释，已有模型构成方法基础。

本文回答三个问题：第一，当前 LightGCN 训练方案相较无图矩阵分解能否改善推荐后购买概率预测？第二，这种表示能否同时改善 CATE 数值估计与效应排序？第三，在相同候选与推荐数量下，按估计 CATE 排序是否带来更高的增量价值，以及它与列表期望购买之间有什么差异？淘宝日志缺少推荐曝光，因此本文采用半合成评价，结论围绕声明的生成机制展开。

## 1 理论基础与相关研究

### 1.1 用户商品图与协同表示

说明用户节点、商品节点和交互边的含义，介绍矩阵分解、NGCF、GraphSAGE、PinSage 与 LightGCN 的联系。重点解释邻居传播如何把共同兴趣编码到表示中，以及协同图为何不能直接解释为变量间的因果关系图[1–4]。

### 1.2 自然购买与推荐增量

定义同一用户—商品对在推荐与不推荐下的潜在购买结果。CATE 是给定处理前信息后的平均购买概率差，不是可同时观察的个体结果差。用“本来就很可能购买”和“因推荐而明显增加购买可能性”的候选说明两种排序目标的区别。

### 1.3 异质性效应与结果模型

介绍元学习、反事实表示学习及半合成效应评估[10–14]。本文采用共享表示的双结果头，分别估计两个处理状态的结果，再相减得到 CATE。说明这一实现与独立模型式 T-Learner 的联系，并与尚未实现的交叉拟合 DR-Learner 区分。

### 1.4 策略评价与相关方法

梳理因果推荐、uplift 排序、日志策略评价与双重稳健估计[5–9,15–18]。解释随机曝光、倾向得分与离线反事实评价各自解决的问题；已知模拟真值的策略评价不等于在真实日志上完成了双重稳健估计。

### 1.5 本文的研究位置

现有工作构成图推荐与因果推荐的基础。本文将表示、排序和价值指标放入同一实验框架，区分收益来自更好的购买预测、效应恢复，还是推荐名单的目标变化。其价值是提供透明的比较证据与适用边界，而非提出新的图网络架构。

## 2 淘宝历史图与半合成因果环境

### 2.1 数据来源与时间隔离

数据来自天池淘宝 UserBehavior，包含用户、商品、类别、行为与时间五列，行为包括浏览、收藏、加购和购买[19]。采用 2017 年 11 月 25 日至 12 月 1 日构建历史图；后两天记录单独保留。原始数据没有推荐曝光、评分、价格和评论，不能将浏览直接当作随机推荐。

### 2.2 清洗建图与可视化

说明校验、异常时间过滤、重复事件及用户抽样规则。抽样历史图包含 2,005 位用户、89,924 件商品与 112,128 条去重配对边；2-core 筛选后保留 1,801 位用户、13,060 件商品和 35,144 条边。展示行为比例、时间分布、度分布与局部二部图，讨论筛选对覆盖率的影响。

### 2.3 节点特征与固定候选

提取历史交互次数、活跃程度、近期性、类别偏好及商品流行度等处理前信息。边表保留多行为计数与时间属性，主 LightGCN 使用二值邻接。每位用户从图内历史已知且本人未交互的商品中固定抽取 100 个候选，候选没有交互不意味着真实负反馈。

### 2.4 半合成环境的生成过程

保留真实用户、商品、图和时间结构，依据活跃度、流行度、类别及图亲和度生成推荐概率，再随机分配是否推荐。分别指定不推荐和推荐时的购买概率，按实际处理采样二元购买结果。真实 CATE 取两种最终购买概率之差；推荐概率与购买概率保持在合法范围。

### 2.5 两种机制及设计限制

feature_effect 让图亲和度影响自然购买，效应主要随活跃度和类别亲和变化；graph_effect 则让图亲和度进入效应。两者共享历史图、候选和用户划分，但自然购买机制也有变化，并非只改变一个效应变量。图亲和度来自预训练 LightGCN 分数，属于对同类方法有利的设定，须单独讨论。

### 2.6 训练信息与因果假设

结果模型仅接触合法处理前输入、实际推荐状态及事实购买结果，模拟真值只用于评价。说明一致性、条件随机分配、重叠和候选间无干扰假设；此处条件无混杂由模拟器保证，不代表淘宝真实日志满足相同条件。当前效应均为非负，尚未考察负效应与商品替代。

## 3 模型策略与实验设计

### 3.1 历史表示与训练预算

无图基线采用 32 维矩阵分解表示，以 BPR 目标训练 50 轮。LightGCN 从该矩阵分解参数初始化，再通过两层归一化传播微调 50 轮，并平均零至二层表示。两者均只读取历史边，随后冻结。需报告学习率及训练时间，额外微调预算限制了纯架构因果解释。

### 3.2 共享结构的双结果头

将用户、商品表示与处理前数值特征输入共享 64 单元隐藏层，输出推荐和不推荐时的购买概率。按实际处理分支计算事实二元交叉熵，以验证用户的事实对数损失早停。两种表示采用相同头结构，分别拟合；每个已拟合模型同时生成概率分数与 CATE 分数。

### 3.3 四组策略与公平控制

比较无图概率排序、无图 CATE 排序、LightGCN 概率排序和 LightGCN CATE 排序。每位测试用户在同一 100 个候选内推荐 10 件商品，同环境同种子共享推荐处理与事实结果。将固定表示下的排序比较作为直接策略对照，将表示差异解释为当前训练方案的综合差异。

### 3.4 预测误差与效应误差

购买概率 RMSE 将预测与真实推荐后购买概率逐对比较，经平方、平均和开方得到；CATE RMSE 对估计效应与真实效应执行相同计算，越小越好。补充效应平均绝对误差及 Spearman 秩相关。若用 PEHE 表示效应均方误差，其平方根即 CATE RMSE，二者不视为独立证据。

### 3.5 策略价值与 Oracle 对照

对每位用户的推荐名单，分别汇总商品的自然购买概率和真实增量，再对测试用户取平均，得到自然购买、增量购买和列表期望购买。Oracle 使用不可部署的真实效应或真实购买概率生成相应目标的最优名单；regret 为其目标值与模型策略值之差。完整候选集的总策略价值与名单购买指标另行区分。

### 3.6 划分与统计不确定性

训练、验证、测试用户分别为 1,081、360、360 人；历史表示学习可使用全部处理前图，测试用户的处理结果标签不参与拟合。两个环境、五个种子和两种表示共完成 20 次结果模型训练。报告均值、标准差及五个配对种子差值的 t 区间；这不是用户 bootstrap 或多图重抽样区间。

## 4 实验结果研究贡献与局限

### 4.1 购买预测是否改善

报告完整均值与标准差。feature_effect 中购买概率 RMSE 约从 0.0244 降至 0.0227，graph_effect 中约从 0.0438 降至 0.0299。分别讨论改进幅度与生成机制的关系，明确预测更准确本身不能证明推荐增量更高。

### 4.2 CATE 恢复是否改善

feature_effect 中 CATE RMSE 约从 0.0178 降至 0.0171，配对差区间跨零；graph_effect 中约从 0.0420 降至 0.0273，秩相关约从 0.7875 升至 0.9234。结合图信号生成来源解释差异，不据此声称所有真实推荐场景均有同样结果。

### 4.3 两类排序获得什么收益

固定 LightGCN 结果模型时，feature_effect 中 CATE 排序相对概率排序的增量差约为 −0.0046，区间跨零；graph_effect 中约为 +0.0069，95% 区间约为 0.0029 至 0.0110。单位为每位用户 Top-10 名单的期望新增购买次数，不是百分比、真实销量或销售额。

### 4.4 收益分解与误差传递

图效应环境中两类排序的列表期望购买差约为 +0.0011，其区间跨零。结合自然购买分量、增量分量、Oracle 与 regret，说明增量改进为何不必表现为更高的名单购买量；讨论两个结果头的估计误差如何影响相减后的效应排序。

### 4.5 三项贡献及证据范围

比较贡献是将历史表示与排序目标拆开；分析贡献是利用真值环境检查图信号与效应恢复的关系；评价贡献是将自然购买、推荐增量、名单购买与目标对应的 Oracle 分别报告。本文给出可复现的机制内证据，不将既有方法的组合或同类生成器优势表述为新算法突破。

### 4.6 局限与复现材料

主要限制包括单数据集抽样、2-core 覆盖损失、LightGCN 热启动和额外预算、生成机制与图模型同源、五种子不确定性及无干扰假设。当前未实现 DR-Learner、图重连、线上实验和真实 GMV 评价。提供图构建、模拟、训练、指标、结果及材料版本，后续以独立机制和多数据集补充验证。

## 结论

本文围绕协同表示与推荐增量目标建立交叉比较，将模型是否预测得更准、是否恢复了效应异质性，以及策略是否获得更高增量作为三个不同问题回答。当前实验在图信号进入效应生成的环境中观察到较明显的 CATE 误差改善，但不同排序目标的收益差异仍需结合自然购买、增量购买与估计误差理解。

后续将在 Amazon、MovieLens 和 KuaiRec 等不同结构的公开数据上重复半合成实验，并采用与参赛模型独立的邻域统计生成机制、统一优化预算及更多随机重复。引入交叉拟合 DR-Learner、负效应和商品替代，检验结论的稳健性。具备已知日志策略概率的随机化数据可用于进一步评价；MIND 等仅含曝光的数据仍需另行论证识别假设，不能仅凭曝光记录认定真实因果效应已可识别。

## 参考文献

[1] He X, Deng K, Wang X, Li Y, Zhang Y, Wang M. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation. SIGIR, 2020. https://doi.org/10.1145/3397271.3401063

[2] Wang X, He X, Wang M, et al. Neural Graph Collaborative Filtering. SIGIR, 2019. https://arxiv.org/abs/1905.08108

[3] Hamilton W L, Ying R, Leskovec J. Inductive Representation Learning on Large Graphs. NeurIPS, 2017. https://arxiv.org/abs/1706.02216

[4] Ying R, He R, Chen K, et al. Graph Convolutional Neural Networks for Web-Scale Recommender Systems. KDD, 2018. https://arxiv.org/abs/1806.01973

[5] Schnabel T, Swaminathan A, Singh A, Chandak N, Joachims T. Recommendations as Treatments: Debiasing Learning and Evaluation. ICML, 2016. https://proceedings.mlr.press/v48/schnabel16.html

[6] Sato M, Singh J, Takemori S, Sonoda T, Zhang Q, Ohkuma T. Uplift-based evaluation and optimization of recommenders. RecSys, 2019. https://doi.org/10.1145/3298689.3347018

[7] Sato M, Takemori S, Singh J, Ohkuma T. Unbiased Learning for the Causal Effect of Recommendation. RecSys, 2020. https://doi.org/10.1145/3383313.3412261

[8] Xie X, Liu Z, Wu S, et al. CausCF: Causal Collaborative Filtering for Recommendation Effect Estimation. CIKM, 2021. https://arxiv.org/abs/2105.13881

[9] Sato M. Online Evaluation Methods for the Causal Effect of Recommendations. RecSys, 2021. https://doi.org/10.1145/3460231.3474235

[10] Künzel S R, Sekhon J S, Bickel P J, Yu B. Meta-learners for estimating heterogeneous treatment effects using machine learning. PNAS, 2019, 116(10): 4156–4165. https://doi.org/10.1073/pnas.1804597116

[11] Johansson F, Shalit U, Sontag D. Learning Representations for Counterfactual Inference. ICML, 2016. https://proceedings.mlr.press/v48/johansson16.html

[12] Shalit U, Johansson F D, Sontag D. Estimating Individual Treatment Effect: Generalization Bounds and Algorithms. ICML, 2017. https://proceedings.mlr.press/v70/shalit17a.html

[13] Louizos C, Shalit U, Mooij J, et al. Causal Effect Inference with Deep Latent-Variable Models. NeurIPS, 2017. https://arxiv.org/abs/1705.08821

[14] Hill J. Bayesian Nonparametric Modeling for Causal Inference. Journal of Computational and Graphical Statistics, 2011, 20(1): 217–240. https://doi.org/10.1198/jcgs.2010.08162

[15] Swaminathan A, Joachims T. Counterfactual Risk Minimization: Learning from Logged Bandit Feedback. ICML, 2015. https://proceedings.mlr.press/v37/swaminathan15.html

[16] Dudík M, Langford J, Li L. Doubly Robust Policy Evaluation and Learning. ICML, 2011. https://arxiv.org/abs/1103.4601

[17] Kennedy E H. Towards optimal doubly robust estimation of heterogeneous causal effects. Electronic Journal of Statistics, 2023, 17(2): 3008–3049. https://doi.org/10.1214/23-EJS2157

[18] Chernozhukov V, Chetverikov D, Demirer M, et al. Double/debiased machine learning for treatment and structural parameters. The Econometrics Journal, 2018, 21(1): C1–C68. https://doi.org/10.1111/ectj.12097

[19] Alibaba Tianchi. User Behavior Data from Taobao for Recommendation [Dataset]. Accessed 2026-11-14. https://tianchi.aliyun.com/dataset/649
