"""
Centralised feature engineering pipeline for CS421 Anomaly Detection.

Pipeline:
    0. combine_labeled_data()  — optional, merge multiple labeled .npz into one
    1. load_data()             — load training .npz (combined or original) + test
    2. split_by_label()        — optional, separate anomalous/normal for EDA
    3. exclude_test_users()    — optional, remove overlapping users (trial phase)
    4. compute_item_stats()    — freeze item-level stats from training only
    5. build_features()        — engineer per-user features

Standard usage:
    XX_train, yy, XX_test = load_data("data/training_batch_with_labels.npz",
                                       test_path="data/first_batch.npz")
    item_stats  = compute_item_stats(XX_train)
    train_feats = build_features(XX_train, item_stats)
    test_feats  = build_features(XX_test, item_stats)

With combined training batches:
    combined = combine_labeled_data("data/training_batch_with_labels.npz",
                                    "data/first_batch_with_labels.npz")
    XX_train, yy, XX_test = load_data(combined, test_path="data/second_batch.npz")
    item_stats  = compute_item_stats(XX_train)
    train_feats = build_features(XX_train, item_stats)
    test_feats  = build_features(XX_test, item_stats)
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy, skew, kurtosis
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

TOTAL_ITEMS = 1000
RATING_RANGE = range(6)


# ── 0a. Generate synthetic anomalies ──────────────────────────────────────


def generate_synthetic_anomalies(
    XX_train: pd.DataFrame,
    yy: pd.DataFrame,
    n_per_type: int = 30,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate diverse synthetic anomalous users and append to training data.

    Creates 10 anomaly types so the model learns general anomalousness
    rather than memorising a single generation procedure.

    Call BEFORE compute_item_stats() so item stats reflect the augmented data.
    Returns updated (XX_train, yy) with synthetic users appended.
    """
    rng = np.random.RandomState(seed)

    # Learn normal-user statistics to make realistic fakes
    normal_users = set(yy.loc[yy["label"] == 0, "user"].values)
    XX_normal = XX_train[XX_train["user"].isin(normal_users)]
    counts_per_user = XX_normal.groupby("user").size()
    median_count = int(counts_per_user.median())
    mean_count = int(counts_per_user.mean())

    item_pop = XX_normal.groupby("item")["user"].count()
    all_items = np.arange(TOTAL_ITEMS)
    popular_items = item_pop.nlargest(100).index.values
    rare_items = item_pop.nsmallest(100).index.values
    item_avg = XX_normal.groupby("item")["rating"].mean()

    max_uid = max(XX_train["user"].max(), yy["user"].max()) + 1
    uid = max_uid

    all_rows = []
    all_labels = []

    def _add_user(items, ratings):
        nonlocal uid
        n = len(items)
        rows = np.column_stack([
            np.full(n, uid, dtype=int),
            items.astype(int),
            np.clip(np.round(ratings), 0, 5).astype(int),
        ])
        all_rows.append(rows)
        all_labels.append([uid, 1])
        uid += 1

    def _random_items(n, pool=None):
        pool = all_items if pool is None else pool
        return rng.choice(pool, size=min(n, len(pool)), replace=False)

    # Type 1: Random rater (uniform random ratings)
    for _ in range(n_per_type):
        n = rng.randint(median_count // 2, median_count * 2)
        items = _random_items(n)
        ratings = rng.randint(0, 6, size=len(items)).astype(float)
        _add_user(items, ratings)

    # Type 2: Love bomber (all high ratings)
    for _ in range(n_per_type):
        n = rng.randint(median_count // 2, median_count * 2)
        items = _random_items(n)
        ratings = rng.choice([4, 5], size=len(items)).astype(float)
        _add_user(items, ratings)

    # Type 3: Hate rater (all low ratings)
    for _ in range(n_per_type):
        n = rng.randint(median_count // 2, median_count * 2)
        items = _random_items(n)
        ratings = rng.choice([0, 1], size=len(items)).astype(float)
        _add_user(items, ratings)

    # Type 4: Bimodal (only 0s and 5s)
    for _ in range(n_per_type):
        n = rng.randint(median_count // 2, median_count * 2)
        items = _random_items(n)
        ratings = rng.choice([0, 5], size=len(items)).astype(float)
        _add_user(items, ratings)

    # Type 5: Bandwagon (only popular items, positive ratings)
    for _ in range(n_per_type):
        n = rng.randint(30, min(100, len(popular_items)))
        items = _random_items(n, popular_items)
        ratings = rng.randint(3, 6, size=len(items)).astype(float)
        _add_user(items, ratings)

    # Type 6: Niche targeter (only rare items)
    for _ in range(n_per_type):
        n = rng.randint(20, min(80, len(rare_items)))
        items = _random_items(n, rare_items)
        ratings = rng.randint(0, 6, size=len(items)).astype(float)
        _add_user(items, ratings)

    # Type 7: Average mimic (rates near item averages, random items)
    for _ in range(n_per_type):
        n = rng.randint(median_count // 2, median_count * 2)
        items = _random_items(n)
        ratings = np.array([
            item_avg.get(it, 3.0) + rng.normal(0, 0.3) for it in items
        ])
        _add_user(items, ratings)

    # Type 8: Reverse rater (opposite of item averages)
    for _ in range(n_per_type):
        n = rng.randint(median_count // 2, median_count * 2)
        items = _random_items(n)
        ratings = np.array([
            5.0 - item_avg.get(it, 2.5) + rng.normal(0, 0.5) for it in items
        ])
        _add_user(items, ratings)

    # Type 9: High volume spammer
    for _ in range(n_per_type):
        n = rng.randint(mean_count * 3, mean_count * 5)
        n = min(n, TOTAL_ITEMS)
        items = _random_items(n)
        ratings = rng.randint(0, 6, size=len(items)).astype(float)
        _add_user(items, ratings)

    # Type 10: Shifted normal (copy real user, shift ratings ±2)
    normal_uids = list(normal_users)
    for _ in range(n_per_type):
        src_uid = rng.choice(normal_uids)
        src = XX_normal[XX_normal["user"] == src_uid]
        shift = rng.choice([-2, -1, 1, 2])
        items = src["item"].values
        ratings = (src["rating"].values + shift).astype(float)
        _add_user(items, ratings)

    # Combine
    synth_interactions = np.vstack(all_rows)
    synth_XX = pd.DataFrame(synth_interactions, columns=["user", "item", "rating"])
    synth_yy = pd.DataFrame(all_labels, columns=["user", "label"])

    n_synth = len(synth_yy)
    XX_aug = pd.concat([XX_train, synth_XX], ignore_index=True)
    yy_aug = pd.concat([yy, synth_yy], ignore_index=True)

    print(f"Generated {n_synth} synthetic anomalous users (10 types x {n_per_type})")
    print(f"  Training: {len(yy)} -> {len(yy_aug)} users "
          f"({int(yy['label'].sum())} + {n_synth} = "
          f"{int(yy_aug['label'].sum())} anomalous)")

    return XX_aug, yy_aug


# ── 0b. Combine labeled data ─────────────────────────────────────────────


def combine_labeled_data(
    *npz_paths: str,
    output_path: str = "data/combined_training.npz",
) -> str:
    """Merge multiple labeled .npz files into one that load_data() can read.

    Duplicate interactions are dropped. If a user appears in more than one
    file, their label is taken from the first file that contains them.

    Returns output_path so it can be passed straight to load_data().
    """
    all_X: list[pd.DataFrame] = []
    all_y: list[pd.DataFrame] = []

    for path in npz_paths:
        data = np.load(path)
        all_X.append(pd.DataFrame(data["X"], columns=["user", "item", "rating"]))
        all_y.append(pd.DataFrame(data["y"], columns=["user", "label"]))

    XX = pd.concat(all_X, ignore_index=True).drop_duplicates()
    yy = pd.concat(all_y, ignore_index=True).drop_duplicates(
        subset="user", keep="first"
    )

    np.savez(output_path, X=XX.values, y=yy.values)

    n_anom = int(yy["label"].sum())
    print(
        f"Combined {len(npz_paths)} files → {output_path}\n"
        f"  {yy.shape[0]} users ({n_anom} anomalous, {yy.shape[0] - n_anom} normal), "
        f"{XX.shape[0]} interactions"
    )
    return output_path


# ── 1. Load data ───────────────────────────────────────────────────────────


def load_data(train_path: str, test_path: str | None = None):
    """Load raw interaction data from .npz files.

    Returns (XX_train, yy, XX_test).  XX_test is None when test_path is omitted.
    """
    data = np.load(train_path)
    XX_train = pd.DataFrame(data["X"], columns=["user", "item", "rating"])
    yy = pd.DataFrame(data["y"], columns=["user", "label"])

    XX_test = None
    if test_path is not None:
        test_data = np.load(test_path)
        XX_test = pd.DataFrame(test_data["X"], columns=["user", "item", "rating"])

    return XX_train, yy, XX_test


# ── 2. Split by label ─────────────────────────────────────────────────────


def split_by_label(
    XX: pd.DataFrame,
    yy: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split interactions into anomalous and normal user groups for EDA."""
    anom_users = set(yy.loc[yy["label"] == 1, "user"].values)
    norm_users = set(yy.loc[yy["label"] == 0, "user"].values)

    XX_anomalous = XX[XX["user"].isin(anom_users)]
    XX_normal = XX[XX["user"].isin(norm_users)]

    print(
        f"Anomalous: {len(anom_users)} users ({XX_anomalous.shape[0]} interactions) | "
        f"Normal: {len(norm_users)} users ({XX_normal.shape[0]} interactions)"
    )
    return XX_anomalous, XX_normal


# ── 3. Exclude test users (trial phase only) ──────────────────────────────


def exclude_test_users(
    XX_train: pd.DataFrame,
    yy: pd.DataFrame,
    XX_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove users that appear in XX_test from training data and labels.

    Use during the trial phase when the test set is a subset of the training batch.
    """
    test_users = XX_test["user"].unique()
    overlap = set(XX_train["user"].unique()) & set(test_users)

    XX_train_clean = XX_train[~XX_train["user"].isin(overlap)]
    yy_clean = yy[~yy["user"].isin(overlap)]

    print(
        f"Excluded {len(overlap)} overlapping users from training "
        f"({yy.shape[0]} → {yy_clean.shape[0]} users)"
    )
    return XX_train_clean, yy_clean


# ── 4. Compute item stats ─────────────────────────────────────────────────


def compute_item_stats(XX_train: pd.DataFrame, n_svd_components: int = 20) -> dict:
    """Compute item-level statistics from training interactions ONLY.

    Pass the returned dict to build_features() for both train and test
    to prevent data leakage.
    """
    item_avg = XX_train.groupby("item")["rating"].mean().rename("item_avg_rating")
    item_pop = XX_train.groupby("item")["user"].count().rename("item_popularity")

    # Global rating distribution (for JS divergence)
    rating_counts = XX_train["rating"].value_counts().reindex(RATING_RANGE, fill_value=0)
    global_rating_dist = (rating_counts / rating_counts.sum()).values.astype(float)

    # Item popularity percentiles
    pop_values = item_pop.values
    pop_25 = np.percentile(pop_values, 25)
    pop_50 = np.median(pop_values)
    pop_90 = np.percentile(pop_values, 90)

    # SVD on user-item matrix for MF residuals
    users = XX_train["user"].unique()
    items = XX_train["item"].unique()
    user_map = {u: i for i, u in enumerate(sorted(users))}
    item_map = {it: i for i, it in enumerate(sorted(items))}

    row_idx = XX_train["user"].map(user_map).values
    col_idx = XX_train["item"].map(item_map).values
    ratings = XX_train["rating"].values.astype(float)
    global_mean = ratings.mean()
    ratings_centered = ratings - global_mean

    n_users = len(user_map)
    n_items = len(item_map)
    R = csr_matrix((ratings_centered, (row_idx, col_idx)), shape=(n_users, n_items))

    n_components = min(n_svd_components, min(n_users, n_items) - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    U_reduced = svd.fit_transform(R)  # (n_users, k)
    Vt = svd.components_              # (k, n_items)

    # Average user profile (for cosine similarity)
    avg_user_profile = np.asarray(R.mean(axis=0)).flatten()  # (n_items,)

    # Item rarity threshold
    rarity_threshold = 5

    return {
        "item_avg_rating": item_avg,
        "item_popularity": item_pop,
        "global_rating_dist": global_rating_dist,
        "pop_25": pop_25,
        "pop_50": pop_50,
        "pop_90": pop_90,
        "svd_Vt": Vt,
        "svd_global_mean": global_mean,
        "svd_item_map": item_map,
        "avg_user_profile": avg_user_profile,
        "n_items_svd": n_items,
        "rarity_threshold": rarity_threshold,
    }


# ── 5. Build features ─────────────────────────────────────────────────────


def _gini(array):
    """Compute Gini coefficient of an array."""
    array = np.sort(array).astype(float)
    n = len(array)
    if n == 0 or array.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * array) - (n + 1) * np.sum(array)) / (n * np.sum(array))


def build_features(
    XX: pd.DataFrame,
    item_stats: dict,
    total_items: int = TOTAL_ITEMS,
) -> pd.DataFrame:
    """Build user-level features from raw interactions.

    Returns DataFrame with a 'user' column and all engineered features.
    """
    item_avg = item_stats["item_avg_rating"]
    item_pop = item_stats["item_popularity"]

    # ── Basic rating statistics ──────────────────────────────────────────
    stats = XX.groupby("user")["rating"].agg(
        rating_mean="mean",
        rating_std="std",
        rating_median="median",
        rating_min="min",
        rating_max="max",
        rating_count="count",
    )
    stats["rating_std"] = stats["rating_std"].fillna(0)
    stats["rating_range"] = stats["rating_max"] - stats["rating_min"]

    # Skewness and kurtosis
    stats["rating_skew"] = XX.groupby("user")["rating"].apply(
        lambda x: skew(x) if len(x) > 2 else 0.0
    )
    stats["rating_kurt"] = XX.groupby("user")["rating"].apply(
        lambda x: kurtosis(x) if len(x) > 3 else 0.0
    )

    # ── Rating proportions + entropy ─────────────────────────────────────
    rdist = XX.groupby(["user", "rating"]).size().unstack(fill_value=0)
    rdist = rdist.reindex(columns=RATING_RANGE, fill_value=0)
    rprops = rdist.div(rdist.sum(axis=1), axis=0)
    rprops.columns = [f"prop_rating_{i}" for i in RATING_RANGE]

    stats["rating_entropy"] = rprops.apply(
        lambda row: entropy(row.values[row.values > 0]), axis=1
    )
    stats = stats.join(rprops)

    # Extreme rating proportions
    stats["prop_extreme"] = rprops["prop_rating_0"] + rprops["prop_rating_5"]
    stats["prop_zero"] = rprops["prop_rating_0"]
    stats["prop_five"] = rprops["prop_rating_5"]

    # Mid-range vs extreme ratio
    stats["prop_mid"] = rprops[["prop_rating_2", "prop_rating_3"]].sum(axis=1)
    stats["extreme_mid_ratio"] = np.clip(
        stats["prop_extreme"] / (stats["prop_mid"] + 1e-9), 0, 100
    )
    stats["prop_near_extreme"] = rprops[["prop_rating_1", "prop_rating_4"]].sum(axis=1)

    # ── Item coverage ────────────────────────────────────────────────────
    stats["unique_items_rated"] = XX.groupby("user")["item"].nunique()
    stats["item_coverage_ratio"] = stats["unique_items_rated"] / total_items

    # Rating count features
    median_count = stats["rating_count"].median()
    stats["rating_count_vs_median"] = stats["rating_count"] / median_count
    stats["rating_count_log"] = np.log1p(stats["rating_count"])
    stats["repeat_rating_ratio"] = stats["rating_count"] / (stats["unique_items_rated"] + 1e-9)

    # ── Item popularity (frozen from training) ───────────────────────────
    XX_pop = XX.merge(item_pop, left_on="item", right_index=True, how="left")
    XX_pop["item_popularity"] = XX_pop["item_popularity"].fillna(0)
    pop_f = XX_pop.groupby("user")["item_popularity"].agg(
        avg_item_popularity="mean",
        std_item_popularity="std",
    )
    pop_f["std_item_popularity"] = pop_f["std_item_popularity"].fillna(0)
    stats = stats.join(pop_f)

    # Popularity percentile targeting
    pop_25 = item_stats["pop_25"]
    pop_50 = item_stats["pop_50"]
    pop_90 = item_stats["pop_90"]

    stats["prop_bottom25_pop"] = XX_pop.groupby("user")["item_popularity"].apply(
        lambda x: (x <= pop_25).mean()
    )
    stats["prop_top10_pop"] = XX_pop.groupby("user")["item_popularity"].apply(
        lambda x: (x >= pop_90).mean()
    )
    stats["pop_vs_global_median"] = stats["avg_item_popularity"] / pop_50

    # Popularity-weighted rating
    XX_pop["pop_weighted_rating"] = XX_pop["rating"] * XX_pop["item_popularity"]
    pwf = XX_pop.groupby("user")["pop_weighted_rating"].sum() / \
          XX_pop.groupby("user")["item_popularity"].sum()
    stats["pop_weighted_avg_rating"] = pwf

    stats["count_x_pop"] = stats["rating_count_log"] * stats["pop_vs_global_median"]

    # ── Deviation from item average (frozen from training) ───────────────
    XX_dev = XX.merge(item_avg, left_on="item", right_index=True, how="left")
    global_train_mean = item_avg.mean()
    XX_dev["item_avg_rating"] = XX_dev["item_avg_rating"].fillna(global_train_mean)
    XX_dev["deviation"] = XX_dev["rating"] - XX_dev["item_avg_rating"]

    dev_f = XX_dev.groupby("user")["deviation"].agg(
        mean_deviation="mean",
        std_deviation="std",
        abs_mean_deviation=lambda x: np.mean(np.abs(x)),
    )
    dev_f["std_deviation"] = dev_f["std_deviation"].fillna(0)
    stats = stats.join(dev_f)

    # Max/min deviation
    dev_extremes = XX_dev.groupby("user")["deviation"].agg(
        max_deviation="max", min_deviation="min"
    )
    stats = stats.join(dev_extremes)
    stats["deviation_range"] = stats["max_deviation"] - stats["min_deviation"]

    # Proportion above/below item average
    stats["prop_above_avg"] = XX_dev.groupby("user").apply(
        lambda df: (df["deviation"] > 0).mean()
    )
    stats["prop_below_avg"] = XX_dev.groupby("user").apply(
        lambda df: (df["deviation"] < 0).mean()
    )

    # Average quality of items targeted (frozen from training)
    iqf = XX_dev.groupby("user")["item_avg_rating"].agg(
        avg_item_avg_rating="mean",
        std_item_avg_rating="std",
    )
    iqf["std_item_avg_rating"] = iqf["std_item_avg_rating"].fillna(0)
    stats = stats.join(iqf)

    # ── NEW: Jensen-Shannon Divergence from population ───────────────────
    global_dist = item_stats["global_rating_dist"]
    user_dists = rprops.values  # already computed per-user rating distributions
    js_divs = []
    for row in user_dists:
        row_safe = np.clip(row, 1e-10, None)
        global_safe = np.clip(global_dist, 1e-10, None)
        js_divs.append(jensenshannon(row_safe, global_safe))
    stats["js_divergence"] = js_divs

    # ── NEW: Gini coefficient of item selections ─────────────────────────
    item_counts_per_user = XX.groupby("user")["item"].value_counts()
    gini_vals = {}
    for user_id in stats.index:
        if user_id in item_counts_per_user.index.get_level_values(0):
            counts = item_counts_per_user.loc[user_id].values
            gini_vals[user_id] = _gini(counts)
        else:
            gini_vals[user_id] = 0.0
    stats["item_selection_gini"] = pd.Series(gini_vals)

    # ── NEW: Item rarity features ────────────────────────────────────────
    rarity_threshold = item_stats["rarity_threshold"]
    rare_items = set(item_pop[item_pop < rarity_threshold].index)
    pop_median_val = item_stats["pop_50"]
    common_items = set(item_pop[item_pop >= pop_median_val].index)

    user_items = XX.groupby("user")["item"].apply(set)
    stats["prop_rare_items"] = user_items.apply(
        lambda items: len(items & rare_items) / max(len(items), 1)
    )
    stats["prop_common_items"] = user_items.apply(
        lambda items: len(items & common_items) / max(len(items), 1)
    )

    # ── NEW: MF residuals (SVD reconstruction error) ─────────────────────
    svd_Vt = item_stats["svd_Vt"]
    svd_global_mean = item_stats["svd_global_mean"]
    svd_item_map = item_stats["svd_item_map"]
    n_items_svd = item_stats["n_items_svd"]

    mf_rmse_vals = {}
    mf_mae_vals = {}
    for user_id, grp in XX.groupby("user"):
        items = grp["item"].values
        ratings = grp["rating"].values.astype(float)
        # Map items to SVD indices; skip unknown items
        valid_mask = np.array([it in svd_item_map for it in items])
        if valid_mask.sum() < 2:
            mf_rmse_vals[user_id] = 0.0
            mf_mae_vals[user_id] = 0.0
            continue
        item_indices = np.array([svd_item_map[it] for it in items[valid_mask]])
        r_centered = ratings[valid_mask] - svd_global_mean
        # User's latent vector: solve least squares U_u = R_u @ V^T_pinv
        V_sub = svd_Vt[:, item_indices].T  # (n_rated, k)
        u_latent, _, _, _ = np.linalg.lstsq(V_sub, r_centered, rcond=None)
        r_hat = V_sub @ u_latent
        residuals = r_centered - r_hat
        mf_rmse_vals[user_id] = np.sqrt(np.mean(residuals ** 2))
        mf_mae_vals[user_id] = np.mean(np.abs(residuals))

    stats["mf_rmse"] = pd.Series(mf_rmse_vals)
    stats["mf_mae"] = pd.Series(mf_mae_vals)
    stats["mf_rmse"] = stats["mf_rmse"].fillna(0)
    stats["mf_mae"] = stats["mf_mae"].fillna(0)

    # ── NEW: Cosine similarity to average user profile ───────────────────
    avg_profile = item_stats["avg_user_profile"]

    cos_sim_vals = {}
    for user_id, grp in XX.groupby("user"):
        items = grp["item"].values
        ratings = grp["rating"].values.astype(float)
        # Build user's rating vector in SVD item space
        user_vec = np.zeros(n_items_svd)
        for it, r in zip(items, ratings):
            if it in svd_item_map:
                idx = svd_item_map[it]
                user_vec[idx] = r - svd_global_mean
        if np.linalg.norm(user_vec) < 1e-9 or np.linalg.norm(avg_profile) < 1e-9:
            cos_sim_vals[user_id] = 0.0
        else:
            cos_sim_vals[user_id] = cosine_similarity(
                user_vec.reshape(1, -1), avg_profile.reshape(1, -1)
            )[0, 0]

    stats["cosine_sim_to_avg"] = pd.Series(cos_sim_vals)
    stats["cosine_sim_to_avg"] = stats["cosine_sim_to_avg"].fillna(0)

    return stats.reset_index()


# ── Helpers ────────────────────────────────────────────────────────────────


def get_test_labels(labeled_test_path: str, test_df: pd.DataFrame) -> np.ndarray:
    """Load ground-truth labels and align with user order in test_df."""
    data = np.load(labeled_test_path)
    yy_test = pd.DataFrame(data["y"], columns=["user", "label"])
    labels = test_df[["user"]].merge(yy_test, on="user", how="left")["label"].values
    return labels


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feature column names (excludes 'user' and 'label')."""
    return [c for c in df.columns if c not in ("user", "label")]


# Features computed purely from a user's own ratings — no dependency on
# frozen training item stats, so they are robust to distribution shift.
ROBUST_FEATURES = [
    "rating_mean", "rating_std", "rating_median", "rating_min", "rating_max",
    "rating_count", "rating_range", "rating_skew", "rating_kurt",
    "rating_entropy",
    "prop_rating_0", "prop_rating_1", "prop_rating_2",
    "prop_rating_3", "prop_rating_4", "prop_rating_5",
    "prop_extreme", "prop_zero", "prop_five",
    "prop_mid", "extreme_mid_ratio", "prop_near_extreme",
    "unique_items_rated", "item_coverage_ratio",
    "rating_count_vs_median", "rating_count_log", "repeat_rating_ratio",
    "item_selection_gini",
]


def get_robust_feature_columns() -> list[str]:
    """Return only shift-resistant feature names (no training-stat dependency)."""
    return list(ROBUST_FEATURES)
