"""Validate fixed study data and, optionally, compare regenerated metrics."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--actual',type=Path)
args=parser.parse_args()
core=ROOT/'outputs'/'taobao_project_preprocessing'/'core2'
reference=ROOT/'outputs'/'taobao_cate_graph_experiment'
users=pd.read_csv(core/'user_nodes.csv')
items=pd.read_csv(core/'item_nodes.csv')
edges=pd.read_csv(core/'user_item_edges.csv')
candidates=pd.read_csv(core/'candidate_pairs.csv')
splits=pd.read_csv(core/'user_split.csv')
assert (len(users),len(items),len(edges))==(1801,13060,35144)
assert candidates.groupby('user_index').size().eq(100).all()
assert len(candidates)==180100
assert splits.user_index.is_unique and len(splits)==len(users)
assert splits.split.value_counts().to_dict()=={'train':1081,'validation':360,'test':360}
assert candidates[['user_index','item_index']].duplicated().sum()==0
historical=set(zip(edges.source_index.astype(int),edges.target_index.astype(int)-len(users)))
assert not any((int(u),int(i)) in historical for u,i in zip(candidates.user_index,candidates.item_index))
assert edges.groupby('source_index').size().min()>=2
assert edges.groupby('target_index').size().min()>=2
report={'fixed_data':'passed','users':len(users),'items':len(items),'historical_edges':len(edges),'candidates':len(candidates)}
for p in sorted((reference/'simulated_labels').glob('*.npz')):
    with np.load(p,allow_pickle=False) as z:
        assert len(z['A'])==len(candidates)
        assert set(np.unique(z['A'])).issubset({0,1})
        assert set(np.unique(z['Y'])).issubset({0,1})
        assert np.all((z['e']>0)&(z['e']<1))
        assert np.allclose(z['tau'],z['p1']-z['p0'])
        assert np.all((z['p0']>=0)&(z['p0']<=1)&(z['p1']>=0)&(z['p1']<=1))
report['simulation_files']=len(list((reference/'simulated_labels').glob('*.npz')))
assert report['simulation_files']==10
df=pd.read_csv(reference/'metrics_by_seed.csv')
keys=['environment','representation','seed']
assert len(df)==20 and not df[keys].duplicated().any()
for prefix in ('probability_policy','cate_policy'):
    assert np.allclose(df[prefix+'_total_purchase'],df[prefix+'_baseline_purchase']+df[prefix+'_incremental_purchase'])
assert (df['probability_policy_regret_to_oracle_probability']>=-1e-7).all()
assert (df['cate_policy_regret_to_oracle_cate']>=-1e-7).all()
report['reference_metrics']='passed'
if args.actual:
    other=pd.read_csv(args.actual/'metrics_by_seed.csv')
    left=df.set_index(keys).sort_index()
    right=other.set_index(keys).sort_index()
    assert left.index.equals(right.index)
    assert list(left.columns)==list(right.columns)
    a=left.to_numpy(float);b=right.to_numpy(float)
    assert np.isfinite(b).all()
    report['reproduced_runs']=len(other)
    report['metric_max_absolute_difference']=float(np.max(np.abs(a-b)))
    report['metric_match']=bool(np.allclose(a,b,rtol=1e-4,atol=1e-5))
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if not report['metric_match']:
        raise SystemExit('Numerical mismatch: inspect platform/BLAS and training early stopping before interpreting results.')
else:
    print(json.dumps(report,ensure_ascii=False,indent=2))
