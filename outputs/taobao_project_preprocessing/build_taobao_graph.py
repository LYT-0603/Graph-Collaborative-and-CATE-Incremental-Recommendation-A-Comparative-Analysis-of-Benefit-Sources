"""Stream Taobao UserBehavior ZIP; make a reproducible historical bipartite graph.

Run: python build_taobao_graph.py --input UserBehavior.csv.zip --output graph_output
Requirements: numpy pandas matplotlib. Uses local vendor libraries when supplied.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import zipfile

VENDOR = Path(__file__).resolve().parent / "graph_vendor"
if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))
import numpy as np
import pandas as pd

BEHAVIORS = ["pv", "fav", "cart", "buy"]
LABELS = {"pv": "浏览", "fav": "收藏", "cart": "加购", "buy": "购买"}
ALPHA = {"pv": 1.0, "fav": 2.0, "cart": 3.0, "buy": 4.0}
TZ = "Asia/Shanghai"
COLS = ["user_id", "item_id", "category_id", "behavior", "timestamp"]


def log(message):
    print(message, flush=True)


def stamp(s):
    return int(pd.Timestamp(s, tz=TZ).timestamp())


def iso(s):
    return pd.to_datetime(s, unit="s", utc=True).dt.tz_convert(TZ).dt.strftime("%Y-%m-%d %H:%M:%S%z")


def save_csv(frame, path):
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.8g")


def scan(args, out):
    start = stamp(args.start)
    cutoff = stamp(args.cutoff)
    end = stamp(args.end)
    assert start < cutoff < end
    capacity = 1 << 23
    user_seen = np.zeros(capacity, dtype=bool)
    item_seen = np.zeros(capacity, dtype=bool)
    item_hist_count = np.zeros(capacity, dtype=np.uint32)
    item_hist_buy = np.zeros(capacity, dtype=np.uint32)
    summary = dict(source_zip=str(Path(args.input).resolve()),
                   source_columns=COLS, timezone=TZ, official_start=args.start,
                   historical_cutoff_exclusive=args.cutoff, end_exclusive=args.end,
                   total_rows=0, window_rows=0, excluded_rows=0,
                   all_behavior_counts=collections.Counter(),
                   window_behavior_counts=collections.Counter(),
                   historical_behavior_counts=collections.Counter(),
                   historical_event_count=0, raw_min_timestamp=None,
                   raw_max_timestamp=None, user_mod=args.user_mod,
                   sampling_seed=args.seed, lambda_hours=args.lambda_hours,
                   behavior_weight_assumptions=ALPHA)
    daily = collections.Counter()
    parts = []
    begin = time.time()
    with zipfile.ZipFile(args.input) as z:
        entry = next(i for i in z.infolist() if i.filename.lower().endswith(".csv"))
        summary["csv_uncompressed_bytes"] = entry.file_size
        summary["csv_member"] = entry.filename
        with z.open(entry) as f:
            reader = pd.read_csv(f, names=COLS, header=None, chunksize=args.chunk_rows,
                                 dtype={"user_id":"int32", "item_id":"int32",
                                        "category_id":"int32", "behavior":"category",
                                        "timestamp":"int64"})
            for batch, df in enumerate(reader):
                summary["total_rows"] += len(df)
                summary["all_behavior_counts"].update(df.behavior.value_counts().to_dict())
                mn, mx = int(df.timestamp.min()), int(df.timestamp.max())
                summary["raw_min_timestamp"] = mn if summary["raw_min_timestamp"] is None else min(mn,summary["raw_min_timestamp"])
                summary["raw_max_timestamp"] = mx if summary["raw_max_timestamp"] is None else max(mx,summary["raw_max_timestamp"])
                valid = df.timestamp.ge(start) & df.timestamp.lt(end) & df.behavior.isin(BEHAVIORS) & df.user_id.ge(0) & df.item_id.ge(0)
                d = df.loc[valid]
                summary["window_rows"] += len(d)
                summary["excluded_rows"] += len(df)-len(d)
                if len(d):
                    u = d.user_id.to_numpy(dtype=np.int64)
                    i = d.item_id.to_numpy(dtype=np.int64)
                    maxid = max(int(u.max()), int(i.max()))
                    if maxid >= len(item_seen):
                        size = 1 << maxid.bit_length()
                        pad = size-len(item_seen)
                        user_seen = np.pad(user_seen,(0,pad))
                        item_seen = np.pad(item_seen,(0,pad))
                        item_hist_count = np.pad(item_hist_count,(0,pad))
                        item_hist_buy = np.pad(item_hist_buy,(0,pad))
                    user_seen[u] = True
                    item_seen[i] = True
                    summary["window_behavior_counts"].update(d.behavior.value_counts().to_dict())
                    day = ((d.timestamp.to_numpy()+8*3600)//86400).astype(int)
                    counts = pd.DataFrame({"day":day,"behavior":d.behavior.to_numpy()}).value_counts()
                    daily.update({(int(k[0]),str(k[1])):int(v) for k,v in counts.items()})
                    hist = d.timestamp.to_numpy() < cutoff
                    h = d.loc[hist]
                    summary["historical_event_count"] += len(h)
                    summary["historical_behavior_counts"].update(h.behavior.value_counts().to_dict())
                    np.add.at(item_hist_count, i[hist], 1)
                    np.add.at(item_hist_buy, h.loc[h.behavior.eq("buy"),"item_id"].to_numpy(), 1)
                    # Fixed integer hash at the USER level; keep every eligible event.
                    hashed = ((u.astype(np.uint64)*np.uint64(2654435761)) ^ (u.astype(np.uint64)>>np.uint64(16)) ^ np.uint64(args.seed))
                    selected = hashed % np.uint64(args.user_mod) == 0
                    if selected.any():
                        part = d.loc[selected].copy()
                        part.behavior = part.behavior.astype(str)
                        parts.append(part)
                if batch % 5 == 0:
                    log(f"SCAN {summary['total_rows']:,} rows; selected {sum(len(p) for p in parts):,}; elapsed {time.time()-begin:.0f}s")
    sampled = pd.concat(parts, ignore_index=True)
    sampled = sampled.sort_values(["user_id","timestamp","item_id","behavior"]).reset_index(drop=True)
    summary["window_unique_users"] = int(user_seen.sum())
    summary["window_unique_items"] = int(item_seen.sum())
    summary["sample_window_events"] = len(sampled)
    summary["sample_window_users"] = int(sampled.user_id.nunique())
    summary["sample_window_items"] = int(sampled.item_id.nunique())
    summary["scan_seconds"] = round(time.time()-begin,2)
    save_csv(sampled,out/"sample_events.csv")
    rows = []
    for (day,b),count in sorted(daily.items()):
        date = pd.Timestamp(day*86400, unit="s").strftime("%Y-%m-%d")
        rows.append((date,b,count))
    save_csv(pd.DataFrame(rows,columns=["date_china","behavior","count"]), out/"full_window_daily_counts.csv")
    return sampled, summary, item_hist_count, item_hist_buy


def build(args, out, events, summary, full_item_count, full_item_buy):
    cutoff = stamp(args.cutoff)
    start = stamp(args.start)
    h = events.loc[events.timestamp.lt(cutoff)].copy()
    future = events.loc[events.timestamp.ge(cutoff)].copy()
    h["day"] = (h.timestamp+8*3600)//86400
    h["decay"] = np.exp(-(cutoff-h.timestamp)/(args.lambda_hours*3600))
    gb = h.groupby(["user_id","item_id"],observed=True,sort=True)
    edges = gb.agg(interaction_count=("timestamp","size"),first_timestamp=("timestamp","min"),last_timestamp=("timestamp","max")).reset_index()
    counts = h.groupby(["user_id","item_id","behavior"],observed=True).size().unstack(fill_value=0).reindex(columns=BEHAVIORS,fill_value=0)
    for b in BEHAVIORS:
        edges[b+"_count"] = counts[b].to_numpy(dtype=np.int32)
    rel = h.groupby(["user_id","item_id","behavior"],observed=True).agg(interaction_count=("timestamp","size"),first_timestamp=("timestamp","min"),last_timestamp=("timestamp","max"),temporal_count=("decay","sum")).reset_index()
    rel["behavior_alpha"] = rel.behavior.map(ALPHA)
    rel["weighted_temporal_count"] = rel.temporal_count*rel.behavior_alpha
    weight = rel.groupby(["user_id","item_id"]).weighted_temporal_count.sum()
    edges["edge_weight_raw"] = weight.to_numpy()
    temporal_sum = rel.groupby(["user_id","item_id"]).temporal_count.sum()
    edges["edge_weight_equal_behavior"] = temporal_sum.to_numpy()
    ud = edges.groupby("user_id").edge_weight_raw.sum()
    it = edges.groupby("item_id").edge_weight_raw.sum()
    edges["edge_weight_normalized"] = edges.edge_weight_raw/np.sqrt(edges.user_id.map(ud)*edges.item_id.map(it))
    edges["recency_hours"] = (cutoff-edges.last_timestamp)/3600
    edges["deepest_observed_behavior"] = "pv"
    for b in BEHAVIORS[1:]:
        edges.loc[edges[b+"_count"].gt(0),"deepest_observed_behavior"] = b
    ug = h.groupby("user_id",sort=True)
    users = ug.agg(interaction_count=("timestamp","size"),first_timestamp=("timestamp","min"),last_timestamp=("timestamp","max"),active_days=("day","nunique"),unique_item_count=("item_id","nunique"),unique_category_count=("category_id","nunique")).reset_index()
    ig = h.groupby("item_id",sort=True)
    items = ig.agg(sample_interaction_count=("timestamp","size"),first_timestamp=("timestamp","min"),last_timestamp=("timestamp","max"),sample_unique_user_count=("user_id","nunique")).reset_index()
    for frame,key in [(users,"user_id"),(items,"item_id")]:
        bc = h.groupby([key,"behavior"],observed=True).size().unstack(fill_value=0).reindex(columns=BEHAVIORS,fill_value=0)
        for b in BEHAVIORS:
            frame[b+"_count"] = bc[b].to_numpy(dtype=np.int32)
        frame["recency_hours"] = (cutoff-frame.last_timestamp)/3600
        frame["first_interaction_china"] = iso(frame.first_timestamp)
        frame["last_interaction_china"] = iso(frame.last_timestamp)
    users["events_per_active_day"] = users.interaction_count/users.active_days
    users["events_per_observation_day"] = users.interaction_count/((cutoff-start)/86400)
    users["purchase_event_share"] = users.buy_count/users.interaction_count
    catcount = h.groupby(["user_id","category_id"]).size().rename("n").reset_index()
    pref = catcount.sort_values(["user_id","n","category_id"],ascending=[True,False,True]).drop_duplicates("user_id").set_index("user_id")
    users["preferred_category_id"] = users.user_id.map(pref.category_id)
    users["preferred_category_share"] = users.user_id.map(pref.n)/users.interaction_count
    ic = h.groupby(["item_id","category_id"]).size().rename("n").reset_index()
    cat = ic.sort_values(["item_id","n","category_id"],ascending=[True,False,True]).drop_duplicates("item_id").set_index("item_id")
    items["category_id"] = items.item_id.map(cat.category_id)
    items["historical_category_count"] = items.item_id.map(ic.groupby("item_id").size())
    items["global_historical_interaction_count"] = full_item_count[items.item_id.to_numpy()]
    items["global_historical_buy_count"] = full_item_buy[items.item_id.to_numpy()]
    items["global_interaction_share"] = items.global_historical_interaction_count/summary["historical_event_count"]
    edges["category_id"] = edges.item_id.map(items.set_index("item_id").category_id)
    ck = catcount.set_index(["user_id","category_id"]).n
    pairidx = pd.MultiIndex.from_frame(edges[["user_id","category_id"]])
    edges["category_preference_match"] = ck.reindex(pairidx).fillna(0).to_numpy()/edges.user_id.map(users.set_index("user_id").interaction_count)
    users.insert(0,"user_index",np.arange(len(users)))
    items.insert(0,"item_index",np.arange(len(items)))
    items.insert(1,"global_node_index",np.arange(len(items))+len(users))
    umap = users.set_index("user_id").user_index
    imap = items.set_index("item_id").global_node_index
    for frame in (edges,rel):
        frame.insert(0,"source_index",frame.user_id.map(umap))
        frame.insert(1,"target_index",frame.item_id.map(imap))
    rel["relation_index"] = rel.behavior.map({b:i for i,b in enumerate(BEHAVIORS)})
    save_csv(users,out/"user_nodes.csv")
    save_csv(items,out/"item_nodes.csv")
    save_csv(edges,out/"user_item_edges.csv")
    save_csv(rel,out/"relation_edges.csv")
    future["user_in_historical_graph"] = future.user_id.isin(users.user_id)
    future["item_in_historical_graph"] = future.item_id.isin(items.item_id)
    save_csv(future,out/"future_events_holdout.csv")
    # All feature values below are calculated from pre-cutoff events only.
    unames = ["log_interaction_count","log_unique_item_count","log_pv_count","log_fav_count","log_cart_count","log_buy_count","recency_days","active_days","log_events_per_active_day","preferred_category_share"]
    ux = np.column_stack([np.log1p(users.interaction_count),np.log1p(users.unique_item_count),*[np.log1p(users[b+"_count"]) for b in BEHAVIORS],users.recency_hours/24,users.active_days,np.log1p(users.events_per_active_day),users.preferred_category_share]).astype("float32")
    inames = ["log_sample_interaction_count","log_sample_unique_user_count","log_global_interaction_count","log_global_buy_count","sample_recency_days",*["log_sample_"+b+"_count" for b in BEHAVIORS]]
    ix = np.column_stack([np.log1p(items.sample_interaction_count),np.log1p(items.sample_unique_user_count),np.log1p(items.global_historical_interaction_count),np.log1p(items.global_historical_buy_count),items.recency_hours/24,*[np.log1p(items[b+"_count"]) for b in BEHAVIORS]]).astype("float32")
    directed = edges[["source_index","target_index"]].to_numpy(dtype=np.int64).T
    rdirected = rel[["source_index","target_index"]].to_numpy(dtype=np.int64).T
    np.savez_compressed(out/"bipartite_graph.npz",edge_index=np.concatenate([directed,directed[::-1]],axis=1),edge_weight=np.tile(edges.edge_weight_normalized.to_numpy(dtype="float32"),2),edge_weight_raw=np.tile(edges.edge_weight_raw.to_numpy(dtype="float32"),2),relation_edge_index=np.concatenate([rdirected,rdirected[::-1]],axis=1),relation_type=np.tile(rel.relation_index.to_numpy(dtype="int64"),2),relation_temporal_weight=np.tile(rel.temporal_count.to_numpy(dtype="float32"),2),user_features=ux,item_features=ix,user_feature_names=np.array(unames),item_feature_names=np.array(inames),user_ids=users.user_id.to_numpy(),item_ids=items.item_id.to_numpy(),item_category_ids=items.category_id.to_numpy(),user_preferred_category_ids=users.preferred_category_id.to_numpy(),num_users=np.array(len(users)),num_items=np.array(len(items)))
    summary.update(sample_historical_events=len(h),sample_historical_users=len(users),sample_historical_items=len(items),sample_unique_user_item_edges=len(edges),sample_unique_relation_edges=len(rel),sample_future_events=len(future),historical_graph_density=len(edges)/(len(users)*len(items)),npz_directed_edges=2*len(edges),user_feature_columns=unames,item_feature_columns=inames)
    # Reconcile exported totals and bounds before plotting.
    assert edges.interaction_count.sum() == len(h) == rel.interaction_count.sum()
    assert users.interaction_count.sum() == len(h) == items.sample_interaction_count.sum()
    assert np.array_equal((edges["pv_count"]+edges["fav_count"]+edges["cart_count"]+edges["buy_count"]).to_numpy(),edges.interaction_count.to_numpy())
    assert np.isfinite(ux).all() and np.isfinite(ix).all()
    assert np.isfinite(edges.edge_weight_normalized).all() and (edges.edge_weight_normalized>0).all()
    assert h.timestamp.max()<cutoff
    assert directed[0].max()<len(users) and directed[1].min()>=len(users)
    summary["validation"] = {"event_count_reconciled":True,"feature_finite":True,"historical_cutoff_enforced":True,"bipartite_indices_valid":True}
    return h,users,items,edges,rel


def draw(args,out,h,users,items,edges,rel,summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MPath
    fonts=[Path("C:/Windows/Fonts/msyh.ttc"),Path("C:/Windows/Fonts/simhei.ttf")]
    for f in fonts:
        if f.exists():
            font_manager.fontManager.addfont(str(f)); plt.rcParams["font.family"]=font_manager.FontProperties(fname=str(f)).get_name(); break
    plt.rcParams.update({"axes.unicode_minus":False,"svg.fonttype":"none","font.size":11})
    colors={"pv":"#6B8FB5","fav":"#D99A24","cart":"#37916D","buy":"#BF4B55"}
    # Local, intentionally dense example. This is not a random sample of graph motifs.
    shared=edges.groupby("item_id").user_id.nunique().sort_values(ascending=False)
    seeds=shared.head(28).index
    score=edges.loc[edges.item_id.isin(seeds)].groupby("user_id").agg(n=("item_id","size"),w=("edge_weight_raw","sum"))
    selected_u=score.sort_values(["n","w"],ascending=False).head(12).index.tolist()
    cand=edges.loc[edges.user_id.isin(selected_u)].copy()
    isc=cand.groupby("item_id").agg(n=("user_id","nunique"),w=("edge_weight_raw","sum"),buy=("buy_count","sum"),cart=("cart_count","sum"),fav=("fav_count","sum"))
    selected_i=isc.sort_values(["n","buy","cart","fav","w"],ascending=False).head(26).index.tolist()
    sub=cand.loc[cand.item_id.isin(selected_i)].copy()
    # Ensure all four existing relations can be illustrated without invented edges.
    for b in BEHAVIORS:
        if not sub[b+"_count"].gt(0).any():
            more=cand.loc[cand[b+"_count"].gt(0)].sort_values("edge_weight_raw",ascending=False)
            if len(more) and int(more.iloc[0].item_id) not in selected_i:
                selected_i.append(int(more.iloc[0].item_id))
    sub=cand.loc[cand.item_id.isin(selected_i)].copy()
    selected_u=[u for u in selected_u if (sub.user_id==u).any()]
    ypos_u=dict(zip(selected_u,np.linspace(.91,.10,len(selected_u))))
    # Item ordering by neighbor barycenter reduces crossings.
    selected_i=sorted(selected_i,key=lambda i:(-np.mean([ypos_u[u] for u in sub.loc[sub.item_id.eq(i),"user_id"]]),i))
    ypos_i=dict(zip(selected_i,np.linspace(.94,.06,len(selected_i))))
    fig,ax=plt.subplots(figsize=(15.5,11),facecolor="white")
    ax.set(xlim=(0,1),ylim=(0,1)); ax.axis("off")
    fig.text(.075,.963,"淘宝多行为用户—商品二部图",ha="left",fontsize=23,weight="bold",color="#172A3A")
    fig.text(.075,.927,f"真实局部子图：{len(selected_u)} 位用户 · {len(selected_i)} 件商品 · {len(sub)} 个用户—商品连接",ha="left",fontsize=12,color="#415364")
    for row in sub.itertuples():
        b=row.deepest_observed_behavior
        path=MPath([(0.255,ypos_u[row.user_id]),(.44,ypos_u[row.user_id]),(.56,ypos_i[row.item_id]),(.745,ypos_i[row.item_id])],[MPath.MOVETO,MPath.CURVE4,MPath.CURVE4,MPath.CURVE4])
        ax.add_patch(PathPatch(path,facecolor="none",edgecolor=colors[b],lw=.65+1.0*np.log1p(row.edge_weight_raw),alpha=.62,zorder=1))
    um=users.set_index("user_id"); im=items.set_index("item_id")
    for u,y in ypos_u.items():
        ax.scatter([.255],[y],s=140+35*np.log1p(um.loc[u,"interaction_count"]),c="#225775",marker="o",edgecolors="white",linewidths=1.4,zorder=3)
        ax.text(.228,y,f"U {u}",ha="right",va="center",fontsize=11,color="#172A3A")
    for i,y in ypos_i.items():
        ax.scatter([.745],[y],s=80+10*np.log1p(im.loc[i,"global_historical_interaction_count"]),c="#71558A",marker="s",edgecolors="white",linewidths=1.1,zorder=3)
        ax.text(.768,y,f"I {i}",ha="left",va="center",fontsize=10,color="#172A3A")
    ax.text(.255,.995,"用户节点",ha="center",va="top",fontsize=13,weight="bold")
    ax.text(.745,.995,"商品节点",ha="center",va="top",fontsize=13,weight="bold")
    handles=[Line2D([0],[0],color=colors[b],lw=3,label=LABELS[b]) for b in BEHAVIORS]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.042),ncol=4,frameon=False,fontsize=12)
    fig.text(.5,.032,"颜色 = 该连接已观测的最高意向行为；线宽 = log(1 + 时间衰减加权次数)",ha="center",fontsize=10,color="#415364")
    fig.text(.5,.012,"仅使用 2017-11-25 至 12-01 的历史；此处边权表示交互强度，不是 CATE。局部子图偏向共享商品，不能代表总体分布。",ha="center",fontsize=9,color="#415364")
    fig.subplots_adjust(left=.035,right=.965,top=.87,bottom=.11)
    fig.savefig(out/"user_item_bipartite.png",dpi=180)
    fig.savefig(out/"user_item_bipartite.svg")
    plt.close(fig)
    save_csv(sub,out/"display_subgraph_edges.csv")
    display_rel=rel.loc[rel.user_id.isin(selected_u)&rel.item_id.isin(selected_i)]
    save_csv(display_rel,out/"display_subgraph_relations.csv")
    summary.update(display_users=len(selected_u),display_items=len(selected_i),display_pairs=len(sub),display_relations=len(display_rel))
    # Second export shows each behavior type separately rather than collapsing it.
    fig,axes=plt.subplots(1,4,figsize=(19,10.5),facecolor="white")
    for b,ax in zip(BEHAVIORS,axes):
        ax.axis("off"); ax.set(xlim=(0,1),ylim=(0,1))
        rr=display_rel.loc[display_rel.behavior.eq(b)]
        ax.set_title(f"{LABELS[b]} ({b})\n{len(rr)} 条关系",fontsize=14,color=colors[b],pad=15)
        for r in rr.itertuples():
            ax.plot([.25,.75],[ypos_u[r.user_id],ypos_i[r.item_id]],color=colors[b],alpha=.65,lw=.6+np.log1p(r.temporal_count))
        ax.scatter([.25]*len(selected_u),list(ypos_u.values()),s=20,c="#225775",zorder=3)
        ax.scatter([.75]*len(selected_i),list(ypos_i.values()),s=13,c="#71558A",marker="s",zorder=3)
        for u,y in ypos_u.items():ax.text(.23,y,str(u),ha="right",va="center",fontsize=7.5)
        for i,y in ypos_i.items():ax.text(.77,y,str(i),ha="left",va="center",fontsize=7.5)
    fig.suptitle("同一局部二部图的四种行为关系",fontsize=21,y=.98)
    fig.text(.5,.02,"四图使用相同节点位置；同一用户—商品对可以同时拥有多种关系。线宽表示该行为的时间衰减次数。",ha="center",fontsize=11)
    fig.subplots_adjust(top=.87,bottom=.07,left=.035,right=.965,wspace=.25)
    fig.savefig(out/"behavior_relations.png",dpi=180)
    plt.close(fig)
    # Dataset overview, with full vs sample scope made explicit.
    fig,axes=plt.subplots(2,2,figsize=(14,9),facecolor="white")
    ax=axes[0,0]; vals=[summary["window_behavior_counts"].get(b,0) for b in BEHAVIORS]
    bars=ax.bar([LABELS[b] for b in BEHAVIORS],vals,color=[colors[b] for b in BEHAVIORS]); ax.set_yscale("log")
    ax.set_title("全量有效窗口：四种行为次数",loc="left",weight="bold"); ax.set_ylabel("事件数（对数刻度）")
    for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,v*1.15,f"{v:,}",ha="center",fontsize=10)
    ax.margins(y=.3)
    ax=axes[0,1]
    for values,label,color in [(users.unique_item_count,"样本用户度", "#225775"),(items.sample_unique_user_count,"样本商品度","#71558A")]:
        vc=values.value_counts().sort_index(); ax.loglog(vc.index,vc.values,"o-",ms=3,lw=1,label=label,color=color,alpha=.85)
    ax.set_title("历史样本图：度分布",loc="left",weight="bold"); ax.set_xlabel("不同邻居数"); ax.set_ylabel("节点数"); ax.legend(frameon=False)
    ax=axes[1,0]; bins=np.geomspace(edges.edge_weight_raw.min(),edges.edge_weight_raw.max(),45); ax.hist(edges.edge_weight_raw,bins=bins,color="#37916D",alpha=.85); ax.set_xscale("log"); ax.set_title("历史样本图：时间衰减边权",loc="left",weight="bold"); ax.set_xlabel("加权衰减次数（对数刻度）"); ax.set_ylabel("用户—商品连接数")
    ax=axes[1,1]; daily=pd.read_csv(out/"full_window_daily_counts.csv"); daily=daily.pivot(index="date_china",columns="behavior",values="count").fillna(0).reindex(columns=BEHAVIORS)
    bot=np.zeros(len(daily)); x=np.arange(len(daily))
    for b in BEHAVIORS:
        ax.bar(x,daily[b]/1e6,bottom=bot,color=colors[b],label=LABELS[b]); bot+=daily[b].to_numpy()/1e6
    ax.set_xticks(x,[d[5:] for d in daily.index],rotation=30); ax.set_title("全量有效窗口：每日行为",loc="left",weight="bold"); ax.set_ylabel("事件数（百万）"); ax.set_xlabel("日期（北京时间）")
    ax.axvline(6.5,color="#172A3A",ls="--",lw=1); ax.legend(frameon=False,ncol=4,fontsize=8)
    fig.suptitle(f"UserBehavior 数据概况   全量 {summary['window_rows']:,} 事件 / 建图样本 {len(h):,} 历史事件",fontsize=17,y=.99)
    fig.tight_layout(rect=[0,.01,1,.96]); fig.savefig(out/"data_profile.png",dpi=160); plt.close(fig)
    # Standard GraphML for direct inspection in Gephi, limited to the plotted graph.
    import xml.etree.ElementTree as ET
    ns="http://graphml.graphdrawing.org/xmlns"; ET.register_namespace("",ns)
    root=ET.Element("{"+ns+"}graphml")
    for key,typ,target in [("node_type","string","node"),("original_id","long","node"),("behavior","string","edge"),("interaction_count","int","edge"),("temporal_count","double","edge")]:
        ET.SubElement(root,"{"+ns+"}key",id=key,**{"for":target,"attr.name":key,"attr.type":typ})
    g=ET.SubElement(root,"{"+ns+"}graph",edgedefault="undirected")
    for kind,ids in [("user",selected_u),("item",selected_i)]:
        for id_ in ids:
            node=ET.SubElement(g,"{"+ns+"}node",id=f"{kind}_{id_}")
            ET.SubElement(node,"{"+ns+"}data",key="node_type").text=kind
            ET.SubElement(node,"{"+ns+"}data",key="original_id").text=str(id_)
    for idx,r in enumerate(display_rel.itertuples()):
        el=ET.SubElement(g,"{"+ns+"}edge",id=f"e{idx}",source=f"user_{r.user_id}",target=f"item_{r.item_id}")
        for key,val in [("behavior",r.behavior),("interaction_count",r.interaction_count),("temporal_count",r.temporal_count)]:ET.SubElement(el,"{"+ns+"}data",key=key).text=str(val)
    ET.ElementTree(root).write(out/"display_subgraph.graphml",encoding="utf-8",xml_declaration=True)


def write_readme(args,out,summary):
    d={
      "user_nodes.csv":{"user_index":"0-based graph user index","interaction_count":"all raw events for selected user before cutoff","unique_item_count":"distinct historical item neighbors","active_days":"distinct dates in Asia/Shanghai","events_per_active_day":"event count / active days","events_per_observation_day":"event count / historical window days","purchase_event_share":"buy events / all events; NOT conversion probability","preferred_category_id":"most frequent historical category; smallest ID breaks ties","recency_hours":"hours since last pre-cutoff event"},
      "item_nodes.csv":{"item_index":"0-based item-local index","global_node_index":"num_users + item_index","sample_interaction_count":"events among selected users only","sample_unique_user_count":"degree in sampled graph","global_historical_interaction_count":"all users' events before cutoff, from full scan","global_historical_buy_count":"all users' buys before cutoff, from full scan","global_interaction_share":"global item events / global historical events","category_id":"mode among sampled historical records"},
      "user_item_edges.csv":{"interaction_count":"events of any behavior per pair","pv_count / fav_count / cart_count / buy_count":"per-behavior event counts","edge_weight_raw":"sum over all historical events alpha(behavior)*exp(-(T-t)/lambda)","edge_weight_equal_behavior":"same sum with every alpha=1","edge_weight_normalized":"edge_weight_raw / sqrt(sum_user_weights*sum_item_weights)","category_preference_match":"user historical fraction in item's modal category; NOT CATE or embedding cosine","deepest_observed_behavior":"visual collapse using buy > cart > fav > pv; descriptive convention, no causal interpretation"},
      "relation_edges.csv":{"one row":"one user-item-behavior relation","temporal_count":"sum of exp(-(T-t)/lambda) for this behavior","weighted_temporal_count":"behavior_alpha*temporal_count","relation_index":"pv=0, fav=1, cart=2, buy=3"}}
    (out/"data_dictionary.json").write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding="utf-8")
    md=f"""# 淘宝多行为用户—商品二部图

## 本次输出范围
输入为一份 CSV 的 ZIP 和它的 MD5 校验文件，非两个 CSV。校验值 `{summary.get('source_md5','')}`。
完整扫描 {summary['total_rows']:,} 行。有效窗口 {args.start}（含）至 {args.end}（不含），北京时间；有效 {summary['window_rows']:,} 行，排除窗口外/无效行为等 {summary['excluded_rows']:,} 行。
全量有效窗口：{summary['window_unique_users']:,} 用户、{summary['window_unique_items']:,} 商品。

由于全量图不能清晰绘制，本次导出的是**固定用户样本图**：对 user_id 做固定整数 hash 后取模 {args.user_mod} 等于 0，保留所选用户在有效窗口的全部行为，而不是截取文件前若干行。样本并不是全量图，商品的邻居度为样本内度。
建图使用 {args.start}（含）至 {args.cutoff}（不含）的历史，得到 {summary['sample_historical_users']:,} 用户、{summary['sample_historical_items']:,} 商品、{summary['sample_unique_user_item_edges']:,} 个不同用户—商品连接和 {summary['sample_unique_relation_edges']:,} 条分行为关系。
其后 {summary['sample_future_events']:,} 条样本行为单独存入 future_events_holdout.csv，**未进入节点特征、图边或边权**。未来事件只表示观测行为，未交互不能直接解释为负反馈或未曝光。

## 图与文件
- user_item_bipartite.png / .svg：可阅读的 {summary['display_users']} 用户、{summary['display_items']} 商品局部图。按共享商品选取，刻意提高可读性，不代表全图密度。
- behavior_relations.png：同一个局部图四种行为分别绘制，保留并行关系。
- data_profile.png：全量行为统计和样本图分布，标题标明统计范围。
- user_nodes.csv / item_nodes.csv：节点特征与原始 ID 映射。
- user_item_edges.csv：每个用户—商品对一行，含四种行为次数和合并边权。
- relation_edges.csv：每个用户—商品—行为组合一行，不合并行为类型。
- bipartite_graph.npz：双向 COO 边索引、边权、关系类型、用户和商品特征矩阵，ID 映射。
- display_subgraph.graphml：仅局部展示图，可导入 Gephi，保留四种行为关系。
- sample_events.csv：所选用户有效窗口的原始事件。
- full_window_daily_counts.csv / summary.json / data_dictionary.json：统计、设置和字段定义。

## 节点特征
用户有历史交互次数、不同商品数、四行为次数、首次/最近时间、recency、活跃天数、每天活跃度、偏好类别与占比。
商品有样本事件数、样本用户邻居数、类别、首次/最近时间、四行为次数；额外提供全量历史交互次数和购买次数，区分全量流行度与样本流行度。
本数据只有 user_id、item_id、category_id、behavior、timestamp。**不存在评分、评分均值/方差、评论数、价格或高评分标记；全部未伪造。收藏不是点赞，也没有可用的用户姓名或商品名称。**

## 边权的准确含义
默认观察时点 T={args.cutoff} Asia/Shanghai，lambda={args.lambda_hours} 小时，alpha(pv,fav,cart,buy)=(1,2,3,4)。
`w_ui = sum_event alpha(behavior) * exp(-(T-t_event)/(lambda_hours*3600))`。
该 alpha 是可修改的示例假设，**不是从数据学出的行为价值或因果效应**。同时输出 equal_behavior 版本，以及每一种关系独立的衰减权重。不能用不存在的评分代替行为类型。
`w_norm = w_ui / sqrt(sum_i(w_ui)*sum_u(w_ui))`，用于加权对称邻接。
本步骤不添加自环；GCN 可在模型中加自环并按框架规范处理归一化，避免对已归一化权重重复归一化。

## 后续 GNN 与 CATE
用户、商品特征维度不同，适合 HeteroData 分类型输入，或分别通过输入线性层映射到同一维度。分类 ID 单独保存，不能把类别数字直接当连续数值。
GCN 默认可以使用合并加权图。标准 GraphSAGE 不自动使用 edge_weight，使用关系型 HeteroConv、可加权聚合器或边编码器才能纳入这些权重。
节点特征是历史统计，未做全数据标准化；训练时只在训练分区拟合 scaler。
历史边权不是 CATE。后续可在候选用户—商品对上预测 tau_hat_ui；不应把真实 tau 或处理后的购买边加入输入图。
这里保存的价格缺失，不应用“价格敏感度”作为已观测真实变量。半合成研究若使用该项，必须明确其为模拟变量，且观测与识别假设一致。
候选 pair 的抽样、半合成 treatment/outcome、交叉拟合 DR、CATE 标签以及推荐策略训练**尚未执行**，本次仅完成历史图与特征。

## 使用图文件
```python
import numpy as np
g = np.load('bipartite_graph.npz', allow_pickle=False)
print(g['edge_index'].shape, g['user_features'].shape, g['item_features'].shape)
# user indices: 0..num_users-1; item global indices: num_users..num_users+num_items-1
# A repeated forward/reverse relation keeps relation_type 0..3 for undirected message passing.
# For directed heterographs, define a distinct reverse relation name in the model if needed.
```

复现：安装 numpy、pandas、matplotlib，在本目录运行：
```powershell
python build_taobao_graph.py --input "C:/path/UserBehavior.csv.zip" --output "graph_output"
```
默认为本次相同抽样。`--user-mod 1` 可保留全部用户，但聚合阶段需要显著更多内存，不能直接在普通笔记本上承诺可行。更大规模应使用分区聚合/磁盘数据库。
"""
    (out/"README_中文.md").write_text(md,encoding="utf-8")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--user-mod",type=int,default=500)
    parser.add_argument("--seed",type=int,default=20261109)
    parser.add_argument("--chunk-rows",type=int,default=1_000_000)
    parser.add_argument("--lambda-hours",type=float,default=72)
    parser.add_argument("--start",default="2017-11-25 00:00:00")
    parser.add_argument("--cutoff",default="2017-12-02 00:00:00")
    parser.add_argument("--end",default="2017-12-04 00:00:00")
    args=parser.parse_args()
    assert args.user_mod>0 and args.lambda_hours>0
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    log("Streaming source ZIP; historical reference " + args.cutoff)
    events,summary,fic,fib=scan(args,out)
    md5=hashlib.md5()
    with open(args.input,"rb") as f:
        while block:=f.read(8*1024*1024):md5.update(block)
    summary["source_md5"]=md5.hexdigest()
    companion=Path(args.input+".md5")
    if companion.exists():
        summary["expected_md5"]=companion.read_text().strip().split("= ")[-1].strip()
        summary["md5_match"]=summary["source_md5"]==summary["expected_md5"]
        assert summary["md5_match"],"MD5 mismatch"
    log("SCAN COMPLETE " + json.dumps(summary,ensure_ascii=False,default=int))
    h,users,items,edges,rel=build(args,out,events,summary,fic,fib)
    log(f"GRAPH {len(users):,} users, {len(items):,} items, {len(edges):,} pairs, {len(rel):,} relations")
    draw(args,out,h,users,items,edges,rel,summary)
    write_readme(args,out,summary)
    (out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=int),encoding="utf-8")
    log("DONE " + str(out.resolve()))


if __name__=="__main__":main()
