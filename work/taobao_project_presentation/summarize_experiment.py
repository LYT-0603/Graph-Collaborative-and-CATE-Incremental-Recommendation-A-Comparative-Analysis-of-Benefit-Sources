from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
import argparse
parser = argparse.ArgumentParser(description='Summarize per-seed experiment metrics')
parser.add_argument('--input-dir', type=Path, default=ROOT/'outputs'/'taobao_cate_graph_experiment')
OUT=parser.parse_args().input_dir.resolve()
df=pd.read_csv(OUT/'metrics_by_seed.csv')
metrics=['p1_rmse','p1_mse','p1_rank_correlation','cate_rmse','cate_mae','cate_rank_correlation',
         'probability_policy_baseline_purchase','probability_policy_incremental_purchase','probability_policy_total_purchase',
         'cate_policy_baseline_purchase','cate_policy_incremental_purchase','cate_policy_total_purchase',
         'oracle_probability_incremental_purchase','oracle_probability_total_purchase',
         'oracle_cate_incremental_purchase','oracle_cate_total_purchase',
         'probability_policy_regret_to_oracle_probability','cate_policy_regret_to_oracle_cate']

def ci(v):
    v=np.asarray(v,float); n=len(v); mean=float(v.mean()); sd=float(v.std(ddof=1)) if n>1 else 0
    # Two-sided 95% t critical value, df=4 for the five paired random seeds.
    critical={1:0,2:12.706,3:4.303,4:3.182,5:2.776}.get(n,1.96)
    half=critical*sd/np.sqrt(n) if n>1 else 0
    return {'mean':mean,'sd':sd,'ci95_low':mean-half,'ci95_high':mean+half,'n_seeds':n}

groups={}
for (env,rep),part in df.groupby(['environment','representation']):
    groups.setdefault(env,{})[rep]={metric:ci(part[metric]) for metric in metrics}

contrasts={}
for env,part in df.groupby('environment'):
    pivot=part.pivot(index='seed',columns='representation')
    contrasts[env]={}
    for metric in ['p1_rmse','p1_mse','p1_rank_correlation','cate_rmse','cate_mae','cate_rank_correlation',
                   'probability_policy_incremental_purchase','probability_policy_total_purchase',
                   'cate_policy_incremental_purchase','cate_policy_total_purchase']:
        delta=pivot[metric]['lightgcn']-pivot[metric]['no_graph_mf']
        contrasts[env]['LightGCN_minus_no_graph_'+metric]=ci(delta)
    for rep in ['no_graph_mf','lightgcn']:
        sub=part.loc[part.representation.eq(rep)].set_index('seed')
        contrasts[env][rep]={}
        for metric in ['delta_incremental_cate_minus_probability','delta_total_cate_minus_probability',
                       'probability_policy_regret_to_oracle_probability','cate_policy_regret_to_oracle_cate']:
            contrasts[env][rep][metric]=ci(sub[metric])

config=json.loads((OUT/'experiment_config.json').read_text(encoding='utf-8'))
result={'experiment':config,'group_metrics':groups,'paired_contrasts':contrasts}
(OUT/'key_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')

lines=['# 实验结果与价值判断','',
       '所有预测指标均在未参与结果模型训练的 360 位测试用户、每人 100 个固定候选上计算。收益单位为每位用户 Top-10 推荐列表的期望购买次数。报告 5 个共同随机种子的均值和标准差；配对差的 95% t 区间基于 5 个种子，区间较宽时不作显著性定论。', '',
       '## 环境定义','',
       '- feature_effect：图偏好影响自然购买倾向，类别历史亲和度与用户活跃度影响 CATE。该设置让自然购买信号与增量信号部分分离。',
       '- graph_effect：LightGCN 历史协同亲和度直接影响 CATE，用于检查图表示是否能恢复图驱动的效应异质性。',
       '- 两种环境均为预先声明的半合成机制，不能作为淘宝线上因果效应或部署收益的证据。','',
       '## 模型与指标结果','',
       '|环境|表示|p1 RMSE|CATE RMSE|CATE秩相关|概率排序增量|CATE排序增量|概率排序总购买|CATE排序总购买|',
       '|---|---|---:|---:|---:|---:|---:|---:|---:|']
for env in groups:
    for rep in ['no_graph_mf','lightgcn']:
        g=groups[env][rep]
        vals=[g[m]['mean'] for m in ['p1_rmse','cate_rmse','cate_rank_correlation',
             'probability_policy_incremental_purchase','cate_policy_incremental_purchase',
             'probability_policy_total_purchase','cate_policy_total_purchase']]
        lines.append('|'+env+'|'+rep+'|'+'|'.join(f'{x:.4f}' for x in vals)+'|')
lines+=['','## 指标计算与含义','',
        '- **p1 RMSE**：在 360×100 个测试候选配对上，计算预测的推荐后购买概率与半合成真值 p1 的均方根误差。越低表示概率预测越接近真值；它不衡量推荐造成的增量。',
        '- **p1 MSE**：同一误差的平方，即 mean((p1_hat−p1)^2)，仅作代码输出的辅助量；这里与真概率比较，不是传统用二元观测 Y 计算的 Brier score。',
        '- **CATE RMSE / MAE**：比较估计 tau_hat=p1_hat−p0_hat 与真 tau=p1−p0 的均方根/绝对误差，均对测试候选配对计算。越低表示个体效应估计越接近已知真值。',
        '- **CATE 秩相关**：对测试候选上的 tau_hat 与 tau 排名计算 Spearman 相关，越高表示高增量候选排序越接近真值排序。',
        '- **增量购买 ΔV**：每位测试用户先将策略 Top-10 中各商品的真实 tau 相加，再对 360 位用户求平均。它是半合成环境中推荐相对不推荐的期望新增购买数。',
        '- **自然购买 B 与名单期望购买 L**：分别汇总名单商品的 p0 与 p1，按用户求和后取平均；L=B+ΔV。B 表示未推荐时本来也会发生的购买，L 表示推荐后的总期望购买。',
        '- **Oracle regret**：按策略目标计算 Oracle 真值排序与模型策略值之差。概率排序用总购买作目标，CATE 排序用增量购买作目标；越低越接近已知真值下的理想排序。Oracle 不可部署。',
        '- **置信区间**：先对五个共同随机种子计算配对差，再用自由度 4 的双侧 95% t 区间；它不是按测试用户重抽样的区间。',
        '', '## 主要解释','',
        '1. 阅读 p1 RMSE 和 CATE RMSE，分别判断图协同表示是否帮助购买概率和个性化效应估计；不能用购买预测好坏替代因果效应质量。',
        '2. 阅读 Top-10 的增量购买与总购买。概率策略目标是尽可能多的总购买，CATE 策略目标是尽可能多的新增购买，因此两者可能交换自然购买与增量购买。',
        '3. Oracle 只提供机制已知时的上界。估计 CATE 与 Oracle 的距离反映效应估计误差如何限制策略价值。',
        '4. 将 LightGCN 与无图协同表示的逐种子差值和置信区间用于判断图传播的额外收益。样本只有 5 个模拟种子，结论以效应大小和跨环境方向为主。','',
        '## 复现','',
        '在项目根目录安装 Python、NumPy 与 pandas 后运行 `python work/taobao_project_presentation/run_full_experiment.py`。脚本在历史图上预训练 BPR 表示，只用训练用户的半合成 A/Y 拟合双结果头，按验证用户早停，并在固定测试用户上计算指标。随机种子、模拟真值和预测输出保存在同目录。']
(OUT/'实验结果与价值判断.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('wrote key_results.json and 实验结果与价值判断.md')
