# 文献与原作者实现

本项目借鉴下列方法的思想并独立实现 NumPy 版实验。原作者代码可作为扩展参考：

- [LightGCN PyTorch](https://github.com/gusye1234/LightGCN-PyTorch)
- [NGCF](https://github.com/xiangwang1223/neural_graph_collaborative_filtering)
- [CFRNet 反事实表示学习](https://github.com/clinicalml/cfrnet)
- [因果推荐策略在线比较](https://github.com/masatoh73/causal-interleaving)
- [EconML](https://github.com/py-why/EconML)，用于后续 DR-Learner 扩展，当前实验未使用。

Hill/IHDP 等文献支持保留真实协变量并模拟潜在结果的半合成研究思路。本项目采用淘宝图及自定义购买概率机制，不是对 IHDP 或文献中实验数值的复现。

## 论文


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
