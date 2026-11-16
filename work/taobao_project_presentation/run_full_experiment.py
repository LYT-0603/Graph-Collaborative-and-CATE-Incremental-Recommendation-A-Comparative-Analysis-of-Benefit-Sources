"""Reproducible semi-synthetic evaluation of graph representations and CATE ranking.

Pipeline: historical implicit-feedback BPR representations (no graph vs LightGCN),
fixed treatment/outcome simulations, shared two-head factual outcome MLP, and
paired held-out policy evaluation. Uses only NumPy and pandas.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'outputs' / 'taobao_project_preprocessing'
CORE = DATA / 'core2'
OUT = ROOT / 'outputs' / 'taobao_cate_graph_experiment'
DIM = 32
LAYERS = 2
K_RECOMMEND = 10
SEEDS = [17, 29, 43, 71, 101]
EPOCHS_BPR = 50
EPOCHS_OUTCOME = 35
BATCH_SIZE = 8192


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def softplus(z: np.ndarray) -> np.ndarray:
    return np.logaddexp(0, z)


class Adam:
    def __init__(self, params: dict[str, np.ndarray], lr: float):
        self.lr = lr
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]):
        self.t += 1
        for key, param in params.items():
            grad = grads[key]
            self.m[key] = .9 * self.m[key] + .1 * grad
            self.v[key] = .999 * self.v[key] + .001 * grad * grad
            mh = self.m[key] / (1 - .9 ** self.t)
            vh = self.v[key] / (1 - .999 ** self.t)
            param -= self.lr * mh / (np.sqrt(vh) + 1e-8)


def adjacency_from_edges(user_idx: np.ndarray, item_idx: np.ndarray, n_users: int, n_items: int):
    u = user_idx.astype(np.int32)
    i = (n_users + item_idx).astype(np.int32)
    src = np.concatenate([u, i])
    dst = np.concatenate([i, u])
    deg = np.bincount(src, minlength=n_users+n_items).astype(np.float32)
    weight = 1 / np.sqrt(np.maximum(deg[src] * deg[dst], 1))
    return src, dst, weight.astype(np.float32), deg


def propagate(x: np.ndarray, src: np.ndarray, dst: np.ndarray, weight: np.ndarray, layers: int):
    states = [x]
    cur = x
    for _ in range(layers):
        nxt = np.zeros_like(cur)
        np.add.at(nxt, src, cur[dst] * weight[:, None])
        states.append(nxt)
        cur = nxt
    out = sum(states) / len(states)
    return out, states


def backprop_propagation(grad_out: np.ndarray, src: np.ndarray, dst: np.ndarray,
                         weight: np.ndarray, layers: int):
    # Adjacency is symmetric; reverse message passing uses the same COO arrays.
    grad = np.zeros_like(grad_out)
    cur = grad_out.copy()
    for depth in range(layers, -1, -1):
        grad += cur / (layers + 1)
        if depth:
            prev = np.zeros_like(cur)
            np.add.at(prev, dst, cur[src] * weight[:, None])
            cur = prev
    return grad


def pairwise_negative_sample(candidates: pd.DataFrame, users: np.ndarray,
                             rng: np.random.Generator) -> np.ndarray:
    item_by_user = [g.item_index.to_numpy(np.int32) for _, g in candidates.groupby('user_index', sort=True)]
    negatives = np.empty(len(users), dtype=np.int32)
    for idx, user in enumerate(users):
        pool = item_by_user[int(user)]
        negatives[idx] = pool[rng.integers(0, len(pool))]
    return negatives


def pretrain_bpr(n_users: int, n_items: int, pos_u: np.ndarray, pos_i: np.ndarray,
                 candidate_df: pd.DataFrame, src: np.ndarray, dst: np.ndarray,
                 norm: np.ndarray, graph: bool, seed: int,
                 initial_base: np.ndarray | None = None, learning_rate: float = .025):
    rng = np.random.default_rng(seed)
    n_nodes = n_users+n_items
    base = (rng.normal(0, .08, (n_nodes, DIM))).astype(np.float32) if initial_base is None else initial_base.copy()
    params = {'base': base}
    optimizer = Adam(params, learning_rate)
    n = len(pos_u)
    losses = []
    fixed_neg = pairwise_negative_sample(candidate_df, pos_u, np.random.default_rng(seed+5000))
    def fixed_loss(matrix):
        fitted = propagate(matrix, src, dst, norm, LAYERS)[0] if graph else matrix
        d = np.sum(fitted[pos_u]*(fitted[n_users+pos_i]-fitted[n_users+fixed_neg]),axis=1)
        return float(np.mean(softplus(-d)))
    best_base = base.copy()
    best_loss = fixed_loss(base)
    best_epoch = 0
    wait = 0
    for epoch in range(EPOCHS_BPR):
        neg_i = pairwise_negative_sample(candidate_df, pos_u, rng)
        final, _ = propagate(base, src, dst, norm, LAYERS) if graph else (base, None)
        pu = final[pos_u]
        pi = final[n_users+pos_i]
        ni = final[n_users+neg_i]
        diff = np.sum(pu*(pi-ni), axis=1)
        coeff = (sigmoid(-diff)/n).astype(np.float32)
        grad_final = np.zeros_like(final)
        np.add.at(grad_final, pos_u, coeff[:, None]*(ni-pi))
        np.add.at(grad_final, n_users+pos_i, -coeff[:, None]*pu)
        np.add.at(grad_final, n_users+neg_i, coeff[:, None]*pu)
        grad_base = backprop_propagation(grad_final, src, dst, norm, LAYERS) if graph else grad_final
        # Small weight decay stabilizes rare nodes while preserving their graph signal.
        grads = {'base': grad_base + 1e-4*base}
        optimizer.step(params, grads)
        vloss = fixed_loss(base)
        losses.append({'epoch': epoch+1, 'sampled_train_bpr_loss': float(np.mean(softplus(-diff))),
                       'fixed_negative_bpr_loss': vloss})
        if vloss < best_loss-1e-5:
            best_loss = vloss; best_base = base.copy(); best_epoch = epoch+1; wait = 0
        else:
            wait += 1
            if wait >= 9: break
    final, _ = propagate(best_base, src, dst, norm, LAYERS) if graph else (best_base, None)
    fit = {'best_epoch': best_epoch, 'best_fixed_negative_bpr_loss': best_loss,
           'epochs_run': len(losses), 'loss_trace': losses}
    return final[:n_users].copy(), final[n_users:].copy(), fit, best_base


def load_project():
    graph = np.load(CORE/'bipartite_graph.npz', allow_pickle=False)
    features = np.load(CORE/'scaled_node_features.npz', allow_pickle=False)
    users = pd.read_csv(CORE/'user_nodes.csv')
    items = pd.read_csv(CORE/'item_nodes.csv')
    edges = pd.read_csv(CORE/'user_item_edges.csv')
    candidates = pd.read_csv(CORE/'candidate_pairs.csv')
    split = pd.read_csv(CORE/'user_split.csv')
    n_users, n_items = len(users), len(items)
    src, dst, norm, degrees = adjacency_from_edges(
        edges.source_index.to_numpy(), edges.target_index.to_numpy()-n_users, n_users, n_items)
    ufeat = features['user_features'].astype(np.float32)
    ifeat = features['item_features'].astype(np.float32)
    pair_u = candidates.user_index.to_numpy(np.int32)
    pair_i = candidates.item_index.to_numpy(np.int32)
    # Candidate category IDs are categorical values, never treated as continuous numbers.
    item_category = items.category_id.to_numpy()
    user_category = users.preferred_category_id.to_numpy()
    match = (user_category[pair_u] == item_category[pair_i]).astype(np.float32)
    pref_share = users.preferred_category_share.to_numpy(np.float32)[pair_u]
    activity = users.interaction_count.to_numpy(np.float32)
    activity = np.log1p(activity)
    activity = np.clip((activity - np.quantile(activity,.05))/(np.quantile(activity,.95)-np.quantile(activity,.05)+1e-8),0,1)[pair_u]
    pop = items.global_historical_interaction_count.to_numpy(np.float32)
    pop = np.log1p(pop)
    pop = np.clip((pop - np.quantile(pop,.05))/(np.quantile(pop,.95)-np.quantile(pop,.05)+1e-8),0,1)[pair_i]
    # Candidate-specific category affinity uses only the sampled users' pre-cutoff history.
    event_path = DATA/'full_sample'/'sample_events.csv'
    events = pd.read_csv(event_path, usecols=['user_id','category_id','timestamp'])
    cutoff = int(pd.Timestamp('2017-12-02 00:00:00', tz='Asia/Shanghai').timestamp())
    events = events.loc[events.timestamp < cutoff]
    events = events.loc[events.user_id.isin(users.user_id)]
    cat_counts = events.groupby(['user_id','category_id']).size().rename('category_events').reset_index()
    user_counts = events.groupby('user_id').size().rename('all_events')
    cat_counts = cat_counts.join(user_counts,on='user_id')
    cat_counts['category_affinity'] = cat_counts.category_events / cat_counts.all_events
    candidate_keys = candidates[['user_id','category_id']].copy()
    cat_affinity = candidate_keys.merge(cat_counts[['user_id','category_id','category_affinity']],
                                         on=['user_id','category_id'],how='left').category_affinity.fillna(0).to_numpy(np.float32)
    pair_covars = np.column_stack([match, cat_affinity, pref_share, activity, pop]).astype(np.float32)
    train_users = split.loc[split.split.eq('train'),'user_index'].to_numpy(np.int32)
    val_users = split.loc[split.split.eq('validation'),'user_index'].to_numpy(np.int32)
    test_users = split.loc[split.split.eq('test'),'user_index'].to_numpy(np.int32)
    observed_u = edges.source_index.to_numpy(np.int32)
    observed_i = (edges.target_index.to_numpy()-n_users).astype(np.int32)
    return (n_users,n_items,ufeat,ifeat,pair_u,pair_i,pair_covars,candidates,
            train_users,val_users,test_users,observed_u,observed_i,src,dst,norm,degrees,users,items)


def simulate(cov: np.ndarray, graph_affinity: np.ndarray, seed: int, env: str):
    match, category_affinity, pref_share, activity, pop = cov.T
    g = np.clip(graph_affinity,0,1)
    # Assignment is associated with observable activity, popularity, match and graph affinity.
    e = .15 + .70*sigmoid(-.65 + .7*activity + .5*pop + .35*match + .35*g + 2.0*category_affinity)
    if env == 'feature_effect':
        # Baseline purchase propensity uses graph preference/popularity, while treatment
        # effect is concentrated on category-relevant items. This separates p1 from CATE.
        p0 = sigmoid(-3.65 + .35*activity + .42*pop + .95*g + .1*pref_share)
        tau = .02 + .06*activity + .34*category_affinity
    elif env == 'graph_effect':
        # In this environment the graph score itself modifies the treatment effect.
        p0 = sigmoid(-3.65 + .35*activity + .42*pop + .08*category_affinity + .1*pref_share)
        tau = .02 + .04*activity + .24*g
    else:
        raise ValueError(env)
    p1 = np.minimum(p0+tau,.90)
    rng = np.random.default_rng(seed)
    a = rng.binomial(1,e).astype(np.int8)
    observed_p = np.where(a==1,p1,p0)
    y = rng.binomial(1,observed_p).astype(np.int8)
    return {'e':e.astype(np.float32),'p0':p0.astype(np.float32),'p1':p1.astype(np.float32),
            'tau':(p1-p0).astype(np.float32),'A':a,'Y':y}


def make_inputs(uemb, iemb, ufeat, ifeat, pair_u, pair_i, pair_covars, pair_affinity):
    return np.concatenate([uemb[pair_u],iemb[pair_i],ufeat[pair_u],ifeat[pair_i],pair_covars,
                           pair_affinity[:,None]],axis=1).astype(np.float32)


def train_two_head(x: np.ndarray, a: np.ndarray, y: np.ndarray, train_mask: np.ndarray,
                   val_mask: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    d = x.shape[1]
    params = {
        'w1': rng.normal(0,np.sqrt(2/d),(d,64)).astype(np.float32),
        'b1': np.zeros(64,np.float32),
        'w2': rng.normal(0,.03,(64,2)).astype(np.float32),
        'b2': np.zeros(2,np.float32),
    }
    optimizer=Adam(params,.002)
    train_idx=np.flatnonzero(train_mask)
    val_idx=np.flatnonzero(val_mask)
    best=None; best_loss=float('inf'); best_epoch=0; patience=7; wait=0
    history=[]
    for epoch in range(EPOCHS_OUTCOME):
        order=rng.permutation(train_idx)
        for start in range(0,len(order),BATCH_SIZE):
            ix=order[start:start+BATCH_SIZE]
            xb=x[ix]
            z1=xb@params['w1']+params['b1']
            h=np.maximum(z1,0)
            logits=h@params['w2']+params['b2']
            aa=a[ix].astype(np.int64)
            yy=y[ix].astype(np.float32)
            pred=sigmoid(logits[np.arange(len(ix)),aa])
            dz2=np.zeros_like(logits)
            dz2[np.arange(len(ix)),aa]=(pred-yy)/len(ix)
            gw2=h.T@dz2; gb2=dz2.sum(axis=0)
            dh=dz2@params['w2'].T
            dz1=dh*(z1>0)
            gw1=xb.T@dz1; gb1=dz1.sum(axis=0)
            optimizer.step(params,{'w1':gw1,'b1':gb1,'w2':gw2,'b2':gb2})
        val_loss=loss_on(params,x,a,y,val_idx)
        history.append({'epoch':epoch+1,'validation_factual_logloss':val_loss})
        if val_loss < best_loss-1e-5:
            best_loss=val_loss; best_epoch=epoch+1; wait=0
            best={k:v.copy() for k,v in params.items()}
        else:
            wait+=1
            if wait>=patience: break
    params=best if best is not None else params
    z=np.maximum(x@params['w1']+params['b1'],0)
    logits=z@params['w2']+params['b2']
    p=sigmoid(logits).astype(np.float32)
    return p[:,0],p[:,1],{'best_epoch':best_epoch,'best_validation_factual_logloss':best_loss,
                          'epochs_run':len(history),'validation_history':history}


def loss_on(params,x,a,y,idx):
    logits=np.maximum(x[idx]@params['w1']+params['b1'],0)@params['w2']+params['b2']
    aa=a[idx].astype(np.int64); yy=y[idx].astype(np.float32)
    chosen=logits[np.arange(len(idx)),aa]
    return float(np.mean(softplus(chosen)-yy*chosen))


def rank_corr(x: np.ndarray, y: np.ndarray):
    # Spearman via average ranks for tied finite-precision predictions.
    rx=pd.Series(x).rank(method='average').to_numpy(np.float64)
    ry=pd.Series(y).rank(method='average').to_numpy(np.float64)
    if rx.std()==0 or ry.std()==0: return float('nan')
    return float(np.corrcoef(rx,ry)[0,1])


def policy_metrics(candidates: pd.DataFrame, test_user_ids: np.ndarray, p0hat, p1hat, tauhat,
                   truth: dict):
    users=candidates.user_index.to_numpy(np.int32)
    testmask=np.isin(users,test_user_ids)
    users_test=users[testmask]
    p0hat=p0hat[testmask]; p1hat=p1hat[testmask]; tauhat=tauhat[testmask]
    p0=truth['p0'][testmask]; p1=truth['p1'][testmask]; tau=truth['tau'][testmask]
    unique=np.unique(users_test)
    top_p=[]; top_t=[]; oracle_p=[]; oracle_t=[]
    for u in unique:
        ii=np.flatnonzero(users_test==u)
        def collect(score):
            take=ii[np.argpartition(score[ii],-K_RECOMMEND)[-K_RECOMMEND:]]
            return (float(p0[take].sum()),float(tau[take].sum()),float(p1[take].sum()))
        # Index within filtered test arrays.
        local=np.flatnonzero(users_test==u)
        def collect_local(score):
            take=np.argpartition(score[local],-K_RECOMMEND)[-K_RECOMMEND:]
            chosen=local[take]
            return (float(p0[chosen].sum()),float(tau[chosen].sum()),float(p1[chosen].sum()))
        top_p.append(collect_local(p1hat))
        top_t.append(collect_local(tauhat))
        oracle_p.append(collect_local(p1))
        oracle_t.append(collect_local(tau))
    arr=lambda xs: np.asarray(xs,np.float64)
    result={}
    for name, values in [('probability_policy',top_p),('cate_policy',top_t),
                         ('oracle_probability',oracle_p),('oracle_cate',oracle_t)]:
        val=arr(values)
        result[name+'_baseline_purchase']=float(val[:,0].mean())
        result[name+'_incremental_purchase']=float(val[:,1].mean())
        result[name+'_total_purchase']=float(val[:,2].mean())
    result['delta_incremental_cate_minus_probability']=result['cate_policy_incremental_purchase']-result['probability_policy_incremental_purchase']
    result['delta_total_cate_minus_probability']=result['cate_policy_total_purchase']-result['probability_policy_total_purchase']
    result['probability_policy_regret_to_oracle_probability']=result['oracle_probability_total_purchase']-result['probability_policy_total_purchase']
    result['cate_policy_regret_to_oracle_cate']=result['oracle_cate_incremental_purchase']-result['cate_policy_incremental_purchase']
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    started=time.time()
    (n_users,n_items,ufeat,ifeat,pair_u,pair_i,pair_covars,candidates,train_users,val_users,test_users,
     obs_u,obs_i,src,dst,norm,degrees,users,items)=load_project()
    # Train unsupervised historical representations once. The target outcomes are not read here.
    repr_info={}
    repreps={}
    u0,i0,mf_fit,mf_base=pretrain_bpr(n_users,n_items,obs_u,obs_i,candidates,src,dst,norm,
                                      False,20261114,learning_rate=.025)
    repreps={'no_graph_mf':(u0,i0)}
    repr_info={'no_graph_mf':{'dim':DIM,'lightgcn_layers':0,'bpr_epochs':EPOCHS_BPR,**mf_fit}}
    light_u,light_i,light_fit,light_base=pretrain_bpr(n_users,n_items,obs_u,obs_i,candidates,
                                                     src,dst,norm,True,20261114,
                                                     initial_base=mf_base,learning_rate=.001)
    repreps['lightgcn']=(light_u,light_i)
    repr_info['lightgcn']={'dim':DIM,'lightgcn_layers':LAYERS,'bpr_epochs':EPOCHS_BPR,
                           'initialized_from':'converged no-graph MF embeddings; then BPR fine-tuned through LightGCN propagation',
                           **light_fit}
    for name in ['no_graph_mf','lightgcn']:
        uemb,iemb=repreps[name]
        repreps[name]=(uemb, iemb)
        np.savez_compressed(OUT/f'{name}_historical_embeddings.npz',user_embeddings=uemb,item_embeddings=iemb)
        repr_info[name]['training_edges']=len(obs_u)

    # Graph-affinity oracle is a pre-treatment LightGCN score, normalized to candidate percentiles.
    gu,gi=repreps['lightgcn']
    raw_affinity=np.sum(gu[pair_u]*gi[pair_i],axis=1)
    graph_affinity=(pd.Series(raw_affinity).rank(pct=True).to_numpy(np.float32))
    affinity_by_rep={}
    for rep,(uemb,iemb) in repreps.items():
        scores=np.sum(uemb[pair_u]*iemb[pair_i],axis=1)
        affinity_by_rep[rep]=pd.Series(scores).rank(pct=True).to_numpy(np.float32)
    # Preserve one simulated data package per environment and random seed.
    split_by_user=pd.read_csv(CORE/'user_split.csv').set_index('user_index').split
    row_split=split_by_user.loc[pair_u].to_numpy()
    train_mask=row_split=='train'; val_mask=row_split=='validation'; test_mask=row_split=='test'
    sim_dir=OUT/'simulated_labels'; sim_dir.mkdir(exist_ok=True)
    all_metrics=[]; training_info=[]
    for env in ['feature_effect','graph_effect']:
        env_metrics=[]
        for seed in SEEDS:
            truth=simulate(pair_covars,graph_affinity,seed,env)
            np.savez_compressed(sim_dir/f'{env}_seed{seed}.npz',**truth)
            for repr_name,(uemb,iemb) in repreps.items():
                x=make_inputs(uemb,iemb,ufeat,ifeat,pair_u,pair_i,pair_covars,affinity_by_rep[repr_name])
                phat0,phat1,fit=train_two_head(x,truth['A'],truth['Y'],train_mask,val_mask,
                                               seed+ (0 if repr_name=='no_graph_mf' else 1000))
                catehat=phat1-phat0
                test_ix=np.flatnonzero(test_mask)
                # This is MSE against the known semi-synthetic p1 probability,
                # not the conventional binary-label Brier score.
                p1_mse=float(np.mean((phat1[test_ix]-truth['p1'][test_ix])**2))
                cate_rmse=float(np.sqrt(np.mean((catehat[test_ix]-truth['tau'][test_ix])**2)))
                cate_mae=float(np.mean(np.abs(catehat[test_ix]-truth['tau'][test_ix])))
                metrics={
                    'environment':env,'seed':seed,'representation':repr_name,
                    'p1_mse':p1_mse,
                    'p1_rmse':float(np.sqrt(p1_mse)),
                    'p1_bias':float(np.mean(phat1[test_ix]-truth['p1'][test_ix])),
                    'cate_rmse':cate_rmse,'cate_mae':cate_mae,
                    'cate_rank_correlation':rank_corr(catehat[test_ix],truth['tau'][test_ix]),
                    'p1_rank_correlation':rank_corr(phat1[test_ix],truth['p1'][test_ix]),
                    'test_rows':len(test_ix),'test_users':len(test_users),
                    'treatment_rate':float(truth['A'][train_mask].mean()),
                    'propensity_min':float(truth['e'].min()),'propensity_max':float(truth['e'].max()),
                    'propensity_mean':float(truth['e'].mean()),
                    'overlap_fraction_0.1_0.9':float(np.mean((truth['e']>=.1)&(truth['e']<=.9))),
                }
                metrics.update(policy_metrics(candidates,test_users,phat0,phat1,catehat,truth))
                all_metrics.append(metrics); env_metrics.append(metrics)
                training_info.append({'environment':env,'seed':seed,'representation':repr_name,**fit})
                if seed==SEEDS[0]:
                    np.savez_compressed(OUT/f'predictions_{env}_{repr_name}.npz',p0_hat=phat0,p1_hat=phat1,
                                        tau_hat=catehat,p0_true=truth['p0'],p1_true=truth['p1'],tau_true=truth['tau'],
                                        split=row_split)
            print(f'completed environment={env} seed={seed}',flush=True)

    pd.DataFrame(all_metrics).to_csv(OUT/'metrics_by_seed.csv',index=False,encoding='utf-8-sig')
    # Mean and across-seed standard deviation for the main four comparisons.
    metrics_df=pd.DataFrame(all_metrics)
    num_cols=[c for c in metrics_df.columns if c not in ('environment','seed','representation')]
    summary=(metrics_df.groupby(['environment','representation'])[num_cols]
             .agg(['mean','std']).reset_index())
    summary.columns=['_'.join(str(x) for x in col if x!='').rstrip('_') if isinstance(col,tuple) else col for col in summary.columns]
    summary.to_csv(OUT/'metrics_summary_mean_std.csv',index=False,encoding='utf-8-sig')
    (OUT/'representation_pretraining.json').write_text(json.dumps(repr_info,ensure_ascii=False,indent=2),encoding='utf-8')
    with open(OUT/'training_history.json','w',encoding='utf-8') as f:
        json.dump(training_info,f,ensure_ascii=False,indent=2)
    metadata={
        'dataset':'Taobao UserBehavior sampled 2-core historical graph',
        'n_users':n_users,'n_items':n_items,'historical_positive_edges':len(obs_u),
        'candidate_rows':len(candidates),'train_users':len(train_users),'validation_users':len(val_users),'test_users':len(test_users),
        'candidate_per_user':100,'top_k':K_RECOMMEND,'seeds':SEEDS,'representation_dim':DIM,
        'lightgcn_layers':LAYERS,'bpr_epochs':EPOCHS_BPR,'outcome_head':'shared 64-unit ReLU trunk, two factual Bernoulli potential-outcome heads',
        'label_generation':'semi-synthetic; A~Bernoulli(e(X,G)); Y~Bernoulli(p_A); true CATE=p1-p0',
        'environments':{
            'feature_effect':'graph affinity affects untreated purchase propensity; treatment effect depends on activity and candidate-specific category affinity',
            'graph_effect':'graph affinity is an explicit treatment-effect modifier',
        },
        'policy_value':'mean over test users of expected number of purchases among top-10 candidates; incremental value sums known p1-p0; baseline sums p0',
        'limitation':'No logged recommendation exposure in source data; results apply to the declared semi-synthetic mechanisms and sampled graph.'
    }
    (OUT/'experiment_config.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(OUT),'runs':len(all_metrics),'elapsed_seconds':time.time()-started,
                      'metrics':all_metrics[:2]},ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUT,
                        help='Directory for regenerated experiment outputs')
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    main()
