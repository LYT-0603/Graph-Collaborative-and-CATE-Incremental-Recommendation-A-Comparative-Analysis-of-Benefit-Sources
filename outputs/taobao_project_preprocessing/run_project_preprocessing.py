from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import zipfile
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GRAPH = ROOT / "outputs" / "taobao_project_preprocessing" / "full_sample"
DEFAULT_OUT = ROOT / "outputs" / "taobao_project_preprocessing"
EXPECTED_MD5 = "2c67e9b4e3908fd30a3f43ec95a7de8c"
BEHAVIOR_LABELS = {"pv": "浏览", "fav": "收藏", "cart": "加购", "buy": "购买"}


def sha256_or_md5(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_kcore(n_users: int, n_items: int, u: np.ndarray, i: np.ndarray, k: int) -> np.ndarray:
    """Return a mask on original user/item node indices after bipartite k-core peeling."""
    n = n_users + n_items
    degree = np.zeros(n, dtype=np.int32)
    adjacency = [[] for _ in range(n)]
    item_global = n_users + i
    for uu, ii in zip(u.tolist(), item_global.tolist()):
        adjacency[uu].append(ii)
        adjacency[ii].append(uu)
        degree[uu] += 1
        degree[ii] += 1
    alive = np.ones(n, dtype=bool)
    queue = deque(np.flatnonzero(degree < k).tolist())
    while queue:
        v = queue.popleft()
        if not alive[v] or degree[v] >= k:
            continue
        alive[v] = False
        for neighbor in adjacency[v]:
            if alive[neighbor]:
                degree[neighbor] -= 1
                if degree[neighbor] < k:
                    queue.append(neighbor)
    return alive


def degree_bins(values: np.ndarray, bins: list[tuple[str, int, int | None]], entity: str) -> list[dict]:
    rows = []
    for label, lo, hi in bins:
        mask = values >= lo if hi is None else (values >= lo) & (values <= hi)
        rows.append({"entity": entity, "degree_bin": label, "count": int(mask.sum()),
                     "share": float(mask.mean()) if len(values) else 0.0})
    return rows


def core_export(k: int, alive: np.ndarray, n_users: int, user_nodes: pd.DataFrame,
                item_nodes: pd.DataFrame, edges: pd.DataFrame, rel: pd.DataFrame,
                original_npz: dict, out: Path) -> dict:
    u_keep_old = np.flatnonzero(alive[:n_users])
    i_keep_old = np.flatnonzero(alive[n_users:])
    u_map = {int(old): new for new, old in enumerate(u_keep_old.tolist())}
    i_map = {int(old): new for new, old in enumerate(i_keep_old.tolist())}
    core = edges[edges.source_index.isin(u_map) & (edges.target_index.sub(n_users).isin(i_map))].copy()
    core["source_user_index"] = core.source_index.astype(int)
    core["source_item_index"] = core.target_index.astype(int) - n_users
    core["source_index"] = core.source_user_index.map(u_map).astype(np.int32)
    core["target_index"] = (len(u_keep_old) + core.source_item_index.map(i_map)).astype(np.int32)
    core_user_degree = np.bincount(core.source_index.to_numpy(), minlength=len(u_keep_old))
    core_item_degree = np.bincount(core.target_index.to_numpy() - len(u_keep_old), minlength=len(i_keep_old))

    users = user_nodes[user_nodes.user_index.isin(u_map)].copy().sort_values("user_index")
    users["source_user_index"] = users.user_index.astype(int)
    users["user_index"] = users.source_user_index.map(u_map).astype(np.int32)
    users = users.sort_values("user_index").reset_index(drop=True)
    users["graph_degree_in_core"] = core_user_degree

    items = item_nodes[item_nodes.item_index.isin(i_map)].copy().sort_values("item_index")
    items["source_item_index"] = items.item_index.astype(int)
    items["item_index"] = items.source_item_index.map(i_map).astype(np.int32)
    items["global_node_index"] = (len(u_keep_old) + items.item_index).astype(np.int32)
    items = items.sort_values("item_index").reset_index(drop=True)
    items["graph_degree_in_core"] = core_item_degree

    # Re-normalize weighted adjacency inside the selected core.
    user_strength = np.bincount(core.source_index.to_numpy(), weights=core.edge_weight_raw.to_numpy(), minlength=len(u_keep_old))
    item_strength = np.bincount(core.target_index.to_numpy() - len(u_keep_old), weights=core.edge_weight_raw.to_numpy(), minlength=len(i_keep_old))
    denom = np.sqrt(user_strength[core.source_index.to_numpy()] * item_strength[core.target_index.to_numpy() - len(u_keep_old)])
    core["edge_weight_normalized"] = np.divide(core.edge_weight_raw.to_numpy(), denom,
                                               out=np.zeros(len(core), dtype=np.float64), where=denom > 0)

    # Keep only behavior relations whose pair remains in the core, then reindex both directions.
    rel_core = rel[rel.source_index.isin(u_map) & (rel.target_index.sub(n_users).isin(i_map))].copy()
    rel_core["source_user_index"] = rel_core.source_index.astype(int)
    rel_core["source_item_index"] = rel_core.target_index.astype(int) - n_users
    rel_core["source_index"] = rel_core.source_user_index.map(u_map).astype(np.int32)
    rel_core["target_index"] = (len(u_keep_old) + rel_core.source_item_index.map(i_map)).astype(np.int32)

    core_dir = out / f"core{k}"
    core_dir.mkdir(parents=True, exist_ok=True)
    users.to_csv(core_dir / "user_nodes.csv", index=False, encoding="utf-8-sig")
    items.to_csv(core_dir / "item_nodes.csv", index=False, encoding="utf-8-sig")
    core.to_csv(core_dir / "user_item_edges.csv", index=False, encoding="utf-8-sig", float_format="%.8g")
    rel_core.to_csv(core_dir / "relation_edges.csv", index=False, encoding="utf-8-sig", float_format="%.8g")

    ui = core.source_index.to_numpy(dtype=np.int64)
    ii_local = core.target_index.to_numpy(dtype=np.int64) - len(u_keep_old)
    edge_index = np.vstack([np.concatenate([ui, len(u_keep_old) + ii_local]),
                            np.concatenate([len(u_keep_old) + ii_local, ui])])
    raw_w = core.edge_weight_raw.to_numpy(dtype=np.float32)
    norm_w = core.edge_weight_normalized.to_numpy(dtype=np.float32)
    rel_ui = rel_core.source_index.to_numpy(dtype=np.int64)
    rel_ii = rel_core.target_index.to_numpy(dtype=np.int64)
    rel_type = rel_core.relation_index.to_numpy(dtype=np.int64)
    rel_w = rel_core.weighted_temporal_count.to_numpy(dtype=np.float32)
    relation_edge_index = np.vstack([np.concatenate([rel_ui, rel_ii]), np.concatenate([rel_ii, rel_ui])])
    relation_type = np.concatenate([rel_type, rel_type])
    relation_weight = np.concatenate([rel_w, rel_w])
    npz_path = core_dir / "bipartite_graph.npz"
    np.savez_compressed(
        npz_path,
        edge_index=edge_index, edge_weight=np.concatenate([norm_w, norm_w]), edge_weight_raw=np.concatenate([raw_w, raw_w]),
        relation_edge_index=relation_edge_index, relation_type=relation_type,
        relation_temporal_weight=relation_weight,
        user_features=original_npz["user_features"][u_keep_old],
        item_features=original_npz["item_features"][i_keep_old],
        user_feature_names=original_npz["user_feature_names"],
        item_feature_names=original_npz["item_feature_names"],
        user_ids=original_npz["user_ids"][u_keep_old], item_ids=original_npz["item_ids"][i_keep_old],
        item_category_ids=original_npz["item_category_ids"][i_keep_old],
        user_preferred_category_ids=original_npz["user_preferred_category_ids"][u_keep_old],
        num_users=np.asarray(len(u_keep_old)), num_items=np.asarray(len(i_keep_old)),
    )

    rng = np.random.default_rng(20261114)
    perm = rng.permutation(len(users))
    n_train = int(round(len(users) * .60))
    n_val = int(round(len(users) * .20))
    split = np.full(len(users), "test", dtype=object)
    split[perm[:n_train]] = "train"
    split[perm[n_train:n_train+n_val]] = "validation"
    split_df = users[["user_index", "user_id", "source_user_index"]].copy()
    split_df["split"] = split
    split_df["seed"] = 20261114
    split_df.to_csv(core_dir / "user_split.csv", index=False, encoding="utf-8-sig")

    # Candidate pairs are fixed before any treatment/outcome sampling.
    # All candidates are unseen historical pairs, with both node IDs known from history.
    candidates = []
    all_item_indices = np.arange(len(items), dtype=np.int32)
    seen = core.groupby("source_index")["target_index"].apply(lambda s: set((s - len(users)).astype(int)))
    for user_index in range(len(users)):
        excluded = seen.get(user_index, set())
        pool = all_item_indices[~np.isin(all_item_indices, np.fromiter(excluded, dtype=np.int32))]
        chosen = rng.choice(pool, size=100, replace=False)
        candidates.append(pd.DataFrame({
            "user_index": user_index, "item_index": chosen,
            "user_id": int(users.iloc[user_index].user_id),
            "item_id": items.iloc[chosen].item_id.to_numpy(),
            "category_id": items.iloc[chosen].category_id.to_numpy(),
            "candidate_rank_in_pool": np.arange(100), "split": split[user_index],
        }))
    candidate_frame = pd.concat(candidates, ignore_index=True)
    assert candidate_frame.groupby("user_index").size().eq(100).all()
    assert not candidate_frame.duplicated(["user_index", "item_index"]).any()
    known_pairs = set(zip(core.source_index.astype(int), (core.target_index - len(users)).astype(int)))
    assert not any((int(a), int(b)) in known_pairs for a,b in zip(candidate_frame.user_index, candidate_frame.item_index))
    candidate_frame.to_csv(core_dir / "candidate_pairs.csv", index=False, encoding="utf-8-sig")

    # Fit numeric scaling on train users and items incident to a train user only.
    # Raw full-history feature meaning is retained; no A/Y or future data is read here.
    train_users = np.flatnonzero(split == "train")
    fit_items = np.unique(core.loc[core.source_index.isin(train_users), "target_index"].to_numpy() - len(users))
    raw_user_features = original_npz["user_features"][u_keep_old].astype(np.float32)
    raw_item_features = original_npz["item_features"][i_keep_old].astype(np.float32)
    user_mean = raw_user_features[train_users].mean(axis=0)
    user_std = raw_user_features[train_users].std(axis=0)
    item_mean = raw_item_features[fit_items].mean(axis=0)
    item_std = raw_item_features[fit_items].std(axis=0)
    user_std[user_std < 1e-8] = 1.0
    item_std[item_std < 1e-8] = 1.0
    np.savez_compressed(core_dir / "scaled_node_features.npz",
                        user_features=(raw_user_features-user_mean)/user_std,
                        item_features=(raw_item_features-item_mean)/item_std,
                        user_feature_names=original_npz["user_feature_names"],
                        item_feature_names=original_npz["item_feature_names"])
    scaler = {"fit_rule": "train users; items adjacent to train users in the historical core graph",
              "seed": 20261114, "user_feature_names": original_npz["user_feature_names"].tolist(),
              "item_feature_names": original_npz["item_feature_names"].tolist(),
              "user_mean": user_mean.tolist(), "user_std": user_std.tolist(),
              "item_mean": item_mean.tolist(), "item_std": item_std.tolist(),
              "num_fit_users": len(train_users), "num_fit_items": len(fit_items)}
    (core_dir / "feature_scaler.json").write_text(json.dumps(scaler, ensure_ascii=False, indent=2), encoding="utf-8")

    n_edges = len(core)
    item_deg = core_item_degree
    user_deg = core_user_degree
    assert user_deg.min() >= k and item_deg.min() >= k
    with np.load(npz_path, allow_pickle=False) as exported:
        assert exported['edge_index'].shape == (2, 2 * n_edges)
        assert exported['edge_weight'].shape == (2 * n_edges,)
        assert exported['edge_weight_raw'].shape == (2 * n_edges,)
        assert np.allclose(exported['edge_weight'][:n_edges], exported['edge_weight'][n_edges:])
        assert exported['relation_edge_index'].shape[1] == len(exported['relation_type']) == len(exported['relation_temporal_weight'])
    assert len(rel_core) == sum(int((core[x] > 0).sum()) for x in ["pv_count", "fav_count", "cart_count", "buy_count"])
    assert np.isfinite(original_npz["user_features"][u_keep_old]).all()
    assert np.isfinite(original_npz["item_features"][i_keep_old]).all()

    return {
        "k": k, "users": len(users), "items": len(items), "pair_edges": n_edges,
        "behavior_relations": len(rel_core), "density": n_edges / (len(users) * len(items)),
        "user_degree_mean": float(user_deg.mean()), "user_degree_median": float(np.median(user_deg)),
        "item_degree_mean": float(item_deg.mean()), "item_degree_median": float(np.median(item_deg)),
        "item_degree_min": int(item_deg.min()), "item_degree_max": int(item_deg.max()),
        "retain_user_share": len(users) / n_users,
        "retain_item_share": len(items) / len(item_nodes),
        "retain_edge_share": n_edges / len(edges),
        "split_counts": split_df.split.value_counts().to_dict(),
        "candidate_pairs": len(candidate_frame), "candidate_pool_per_user": 100,
        "candidate_rule": "uniform without replacement from history-known core items, excluding each user's historical neighbors",
        "npz": str(npz_path.name),
    }


def run(graph: Path, out: Path, source_zip: Path | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    with (graph / "summary.json").open(encoding="utf-8") as f:
        source_summary = json.load(f)
    with (graph / "validation.json").open(encoding="utf-8") as f:
        source_validation = json.load(f)
    if source_validation.get("status") != "passed":
        raise ValueError("The upstream graph validation did not pass")
    if not source_summary.get("md5_match"):
        raise ValueError("The upstream source MD5 did not match its recorded checksum")
    if source_zip and source_zip.exists():
        actual_md5 = sha256_or_md5(source_zip, "md5")
        if actual_md5.lower() != EXPECTED_MD5:
            raise ValueError(f"Taobao source MD5 mismatch: {actual_md5}")
    else:
        actual_md5 = source_summary.get("source_md5")

    users = pd.read_csv(graph / "user_nodes.csv")
    items = pd.read_csv(graph / "item_nodes.csv")
    edges = pd.read_csv(graph / "user_item_edges.csv")
    relations = pd.read_csv(graph / "relation_edges.csv")
    daily = pd.read_csv(graph / "full_window_daily_counts.csv")
    npz = np.load(graph / "bipartite_graph.npz", allow_pickle=False)
    raw_sample = pd.read_csv(graph / "sample_events.csv")
    cutoff_ts = int(pd.Timestamp(source_summary["historical_cutoff_exclusive"], tz="Asia/Shanghai").timestamp())
    history_sample = raw_sample[raw_sample.timestamp < cutoff_ts]
    n_users, n_items = len(users), len(items)
    assert n_users == int(source_summary["sample_historical_users"])
    assert n_items == int(source_summary["sample_historical_items"])
    assert len(edges) == int(source_summary["sample_unique_user_item_edges"])
    assert edges.interaction_count.sum() == int(source_summary["sample_historical_events"])
    assert np.isfinite(npz["user_features"]).all() and np.isfinite(npz["item_features"]).all()

    u_deg = users.unique_item_count.to_numpy(dtype=np.int64)
    i_deg = items.sample_unique_user_count.to_numpy(dtype=np.int64)
    bins = []
    bins += degree_bins(u_deg, [("1–10", 1, 10), ("11–25", 11, 25), ("26–50", 26, 50), ("51–100", 51, 100), (">100", 101, None)], "用户")
    bins += degree_bins(i_deg, [("1", 1, 1), ("2", 2, 2), ("3–5", 3, 5), ("6–10", 6, 10), (">10", 11, None)], "商品")
    pd.DataFrame(bins).to_csv(out / "degree_distribution.csv", index=False, encoding="utf-8-sig", float_format="%.8g")
    daily.to_csv(out / "daily_behavior_counts.csv", index=False, encoding="utf-8-sig")

    behavior_rows = []
    for key, count in source_summary["window_behavior_counts"].items():
        behavior_rows.append({"behavior": key, "label_zh": BEHAVIOR_LABELS.get(key, key), "event_count": int(count),
                              "share_of_all_events": int(count) / int(source_summary["window_rows"])})
    pd.DataFrame(behavior_rows).to_csv(out / "behavior_composition.csv", index=False, encoding="utf-8-sig", float_format="%.8g")

    cores = {}
    old_u = edges.source_index.to_numpy(dtype=np.int64)
    old_i = edges.target_index.to_numpy(dtype=np.int64) - n_users
    for k in (2, 3):
        alive = make_kcore(n_users, n_items, old_u, old_i, k)
        cores[f"{k}-core"] = core_export(k, alive, n_users, users, items, edges, relations, dict(npz), out)
    core_table = pd.DataFrame([{
        "graph": "完整样本图", "k": 0, "users": n_users, "items": n_items, "pair_edges": len(edges),
        "retain_user_share": 1.0, "retain_item_share": 1.0, "retain_edge_share": 1.0,
        "mean_user_degree": float(u_deg.mean()), "mean_item_degree": float(i_deg.mean()),
    }] + [{
        "graph": k, "k": c["k"], "users": c["users"], "items": c["items"], "pair_edges": c["pair_edges"],
        "retain_user_share": c["retain_user_share"], "retain_item_share": c["retain_item_share"],
        "retain_edge_share": c["retain_edge_share"], "mean_user_degree": c["user_degree_mean"], "mean_item_degree": c["item_degree_mean"],
    } for k, c in cores.items()])
    core_table.to_csv(out / "kcore_comparison.csv", index=False, encoding="utf-8-sig", float_format="%.8g")

    full_stats = {
        "source_md5": actual_md5,
        "full_valid_events": int(source_summary["window_rows"]),
        "full_user_count": int(source_summary["window_unique_users"]),
        "full_item_count": int(source_summary["window_unique_items"]),
        "history_cutoff": source_summary["historical_cutoff_exclusive"],
        "history_event_count_full": int(source_summary["historical_event_count"]),
        "sample_history_events": int(source_summary["sample_historical_events"]),
        "sample_future_events": int(source_summary["sample_future_events"]),
        "sample_users": n_users, "sample_items": n_items, "sample_pair_edges": len(edges),
        "sample_relation_edges": len(relations), "sample_density": len(edges) / (n_users * n_items),
        "sample_item_degree_one_count": int((i_deg == 1).sum()),
        "sample_item_degree_one_share": float((i_deg == 1).mean()),
        "sample_item_degree_le2_share": float((i_deg <= 2).mean()),
        "sample_user_degree_mean": float(u_deg.mean()), "sample_user_degree_median": float(np.median(u_deg)),
        "sample_item_degree_mean": float(i_deg.mean()), "sample_item_degree_median": float(np.median(i_deg)),
        "pair_mean_events": float(edges.interaction_count.mean()),
        "historical_exact_duplicate_rows": int(history_sample.duplicated().sum()),
        "historical_missing_cells": int(history_sample.isna().sum().sum()),
        "historical_min_timestamp": int(history_sample.timestamp.min()),
        "historical_max_timestamp": int(history_sample.timestamp.max()),
        "sample_behavior_relation_counts": relations.groupby("behavior").size().astype(int).to_dict(),
        "cores": cores,
        "notes": [
            "2-core/3-core are derived from pre-treatment historical edges only.",
            "User splits contain no treatment or outcome labels; outcomes must be generated only after the split is fixed.",
            "Observed future behavior is descriptive holdout data, not exposure, treatment, or a causal outcome label.",
            "No rating, price, comment count, or real CATE is present in the source data.",
        ],
    }
    (out / "eda_summary.json").write_text(json.dumps(full_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(graph / "data_dictionary.json", out / "data_dictionary.json")
    shutil.copy2(graph / "build_taobao_graph.py", out / "build_taobao_graph.py")
    shutil.copy2(graph / "requirements.txt", out / "requirements_graph_builder.txt")

    readme = f"""# 淘宝购物推荐项目阶段数据包

## 已完成

- 输入校验：来源 ZIP 的 MD5 为 `{actual_md5}`，与记录校验值一致。
- 原始数据完整时间窗内有 {int(source_summary['window_rows']):,} 条行为事件；历史图使用 2017-11-25 至 2017-12-02（不含）数据，决策后两天留出 {int(source_summary['sample_future_events']):,} 条样本事件。
- 从约 2,000 名固定哈希抽样用户构建历史二部图，再按历史边计算 2-core 与 3-core。2-core 作为协同信息较充分的实验候选，完整样本图保留作稀疏性对照。
- 2-core 图有 {cores['2-core']['users']:,} 位用户、{cores['2-core']['items']:,} 件商品和 {cores['2-core']['pair_edges']:,} 条用户—商品边。固定 seed=20261114，按用户分成训练、验证、测试集。
- 每用户固定 100 个历史未交互候选商品，均来自处理前已知商品池。2-core 共 {cores['2-core']['candidate_pairs']:,} 个候选配对，此步骤不产生处理或结果标签。
- 标准化仅用训练用户及与训练用户有历史边的商品拟合，保留 `feature_scaler.json`，数值矩阵见 `scaled_node_features.npz`。类别 ID 单独保存，不能按连续数值直接标准化。
- `core2/bipartite_graph.npz` 含双向 COO 邻接、归一化边权、关系类型和节点数值特征；节点 ID 映射在同目录 CSV 中。

## 文件

- `degree_distribution.csv`：完整样本图的用户/商品度数分组。
- `behavior_composition.csv`、`daily_behavior_counts.csv`：全量有效窗行为和逐日事件数。
- `kcore_comparison.csv` 与 `eda_summary.json`：原始抽样图、2-core、3-core 规模及统计。
- `core2/` 与 `core3/`：包括节点特征、配对边、四种行为关系、NPZ、用户划分、候选配对和训练分区拟合的标准化参数。
- `build_taobao_graph.py`、`run_project_preprocessing.py`：从原始 UserBehavior ZIP 到历史图、2-core 模型输入的处理代码。原始 ZIP 不随包重复分发。

## 重要定义与限制

- 原始行为是浏览、收藏、加购和购买。数据不含推荐曝光、评分、商品价格或评论。后续 A、Y、真实 CATE 必须在半合成生成步骤中另行定义。
- `edge_weight_raw` 是按行为权重与时间衰减求得的历史关系强度，系数只是建模假设，不能解释成 CATE。
- `future_events_holdout.csv` 不进入历史图或节点特征，且不能把未交互当作负样本或未曝光。
- 2-core 筛选改善共享商品连接，但降低商品覆盖范围。完整抽样图和不同 core 阈值均应作为敏感性检查；推论只适用于相应样本。
- 节点的原始统计特征保留全部处理前历史含义。`graph_degree_in_core` 才是筛选后图中的度数，不能把原始 `unique_item_count` 或 `sample_unique_user_count` 当作 core 图度数。
- 精确重复的历史事件有 {int(history_sample.duplicated().sum()):,} 行。时间戳只有秒级，重复可能包含真实连续行为，主处理保留这些事件用于计数，并在配对边层聚合；后续可做删除完全重复事件的敏感性分析。
- `user_split.csv` 仅提供用户级划分。模型处理与结果标签还没有生成，后续应在划分冻结后产生。

## 复现

先从已下载的原始 ZIP 运行图处理：

```powershell
python build_taobao_graph.py --input "C:/path/UserBehavior.csv.zip" --output "graph_output"
python run_project_preprocessing.py --graph-dir "graph_output" --output "project_ready"
```

依赖见 `requirements_graph_builder.txt`。建图抽样由固定整数哈希与 seed 控制，不依赖输入文件行顺序。
"""
    (out / "README_中文.md").write_text(readme, encoding="utf-8")
    (out / "run_project_preprocessing.py").write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")

    checks = {
        "source_md5_verified": actual_md5.lower() == EXPECTED_MD5,
        "upstream_graph_validation_passed": source_validation.get("status") == "passed",
        "history_and_future_are_separated": int(source_summary["sample_historical_events"]) + int(source_summary["sample_future_events"]) == int(source_summary["sample_window_events"]),
        "all_node_features_finite": bool(np.isfinite(npz["user_features"]).all() and np.isfinite(npz["item_features"]).all()),
        "core2_degrees_at_least_2": cores["2-core"]["item_degree_min"] >= 2 and min(core2_min_user_degree(out / "core2" / "user_nodes.csv")) >= 2,
        "core3_degrees_at_least_3": cores["3-core"]["item_degree_min"] >= 3 and min(core2_min_user_degree(out / "core3" / "user_nodes.csv")) >= 3,
    }
    if not all(checks.values()):
        raise AssertionError(checks)
    validation = {"status": "passed", "checks": checks}
    (out / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    zip_path = out.parent / "淘宝购物推荐项目_阶段数据预处理与复现包.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(out.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(out.parent))
    return {"summary": full_stats, "validation": validation, "output_dir": str(out), "output_zip": str(zip_path)}


def core2_min_user_degree(path: Path) -> list[int]:
    users = pd.read_csv(path)
    return [int(users.graph_degree_in_core.min())]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-dir", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--source-zip", type=Path, default=None)
    args = parser.parse_args()
    result = run(args.graph_dir.resolve(), args.output.resolve(), args.source_zip.resolve() if args.source_zip and args.source_zip.exists() else None)
    print(json.dumps(result, ensure_ascii=False, indent=2))
