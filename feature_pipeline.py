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
from scipy.stats import entropy

TOTAL_ITEMS = 1000
RATING_RANGE = range(6)


# ── 0. Combine labeled data ────────────────────────────────────────────────


def combine_labeled_data(
    *npz_paths: str,
    output_path: str = "data/combined_training.npz",
) -> str:
    """Merge multiple labeled .npz files into one that load_data() can read.

    If a user appears in more than one file, their label is taken 
    from the first file that contains them.

    Returns output_path so it can be passed straight to load_data().
    """
    all_X: list[pd.DataFrame] = []
    all_y: list[pd.DataFrame] = []

    for path in npz_paths:
        data = np.load(path)
        all_X.append(pd.DataFrame(data["X"], columns=["user", "item", "rating"]))
        all_y.append(pd.DataFrame(data["y"], columns=["user", "label"]))

    XX = pd.concat(all_X, ignore_index=True)
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


def load_npz(path: str) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Load a single .npz file. Returns (interactions, labels).

    Labels is None if the file has no 'y' key (unlabeled test data).

    Example::

        XX, yy = load_npz("data/training_batch_with_labels.npz")  # labeled
        XX, _  = load_npz("data/second_batch.npz")                # unlabeled
    """
    data = np.load(path)
    XX = pd.DataFrame(data["X"], columns=["user", "item", "rating"])
    yy = None
    if "y" in data:
        yy = pd.DataFrame(data["y"], columns=["user", "label"])
    return XX, yy


def load_data(train_path: str, test_path: str | None = None):
    """Load training + optional test data from .npz files.

    Returns (XX_train, yy, XX_test).  XX_test is None when test_path is omitted.
    """
    XX_train, yy = load_npz(train_path)
    XX_test = None
    if test_path is not None:
        XX_test, _ = load_npz(test_path)
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


def compute_item_stats(XX_train: pd.DataFrame) -> dict:
    """Compute item-level statistics from training interactions ONLY.

    Pass the returned dict to build_features() for both train and test
    to prevent data leakage.
    """
    item_avg = XX_train.groupby("item")["rating"].mean().rename("item_avg_rating")
    item_pop = XX_train.groupby("item")["user"].count().rename("item_popularity")
    return {"item_avg_rating": item_avg, "item_popularity": item_pop}


# ── 5. Build features ─────────────────────────────────────────────────────


def build_features(
    XX: pd.DataFrame,
    item_stats: dict,
    total_items: int=TOTAL_ITEMS,
) -> pd.DataFrame:
    """Build user-level features from raw interactions.

    Returns DataFrame with a 'user' column and all engineered features.
    """
    item_avg = item_stats["item_avg_rating"]
    item_pop = item_stats["item_popularity"]

    # Basic rating statistics
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

    # Rating proportions + entropy
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

    # Item coverage
    stats["unique_items_rated"] = XX.groupby("user")["item"].nunique()
    stats["item_coverage_ratio"] = stats["unique_items_rated"] / total_items

    # Item popularity (frozen from training)
    XX_pop = XX.merge(item_pop, left_on="item", right_index=True, how="left")
    XX_pop["item_popularity"] = XX_pop["item_popularity"].fillna(0)
    pop_f = XX_pop.groupby("user")["item_popularity"].agg(
        avg_item_popularity="mean",
        std_item_popularity="std",
    )
    pop_f["std_item_popularity"] = pop_f["std_item_popularity"].fillna(0)
    stats = stats.join(pop_f)

    # Deviation from item average (frozen from training)
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

    # Average quality of items targeted (frozen from training)
    iqf = XX_dev.groupby("user")["item_avg_rating"].agg(
        avg_item_avg_rating="mean",
        std_item_avg_rating="std",
    )
    iqf["std_item_avg_rating"] = iqf["std_item_avg_rating"].fillna(0)
    stats = stats.join(iqf)

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
