# 数据来源与使用范围

来源：[Alibaba Tianchi User Behavior Data from Taobao for Recommendation](https://tianchi.aliyun.com/dataset/649)。五列为匿名用户 ID、商品 ID、类别 ID、行为类型及时间戳。原始 ZIP 的记录 MD5 为 `2c67e9b4e3908fd30a3f43ec95a7de8c`，完整原始压缩包未随仓库发布。

`outputs/taobao_project_preprocessing/full_sample` 为固定规则抽样及聚合的研究材料，含原始公开匿名编号、抽样行为、历史节点与边；core2/core3 为处理前图筛选及候选生成的衍生结果。后两天记录单独保留，仅用于描述统计，不用于生成真实推荐处理标签。

主实验保留真实用户、商品、历史交互图、类别、行为统计和时间结构。推荐处理 A、二元结果 Y、倾向概率 e、潜在购买概率 p0/p1 和真实 CATE 均由本项目代码生成。不得把这些模拟结果称为淘宝真实曝光、真实销量提升或线上因果效应。

数据没有评分、价格、用户姓名或商品文本，也没有完整推荐曝光。不得将浏览当作随机推荐，或将未交互候选当作真实负例。数据的获取和使用应遵循天池来源页面及其适用条款；本仓库不为第三方数据重新授权。
