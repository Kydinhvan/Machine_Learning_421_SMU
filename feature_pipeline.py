"""
Centralised feature engineering pipeline for CS421 Anomaly Detection.

Usage:
    from feature_pipeline import load_data, compute_item_stats, build_features

    XX_train, yy, XX_test = load_data(
        train_path="training_batch_with_labels.npz",
        test_path="first_batch.npz",
    )

    item_stats  = compute_item_stats(XX_train)
    train_feats = build_features(XX_train, item_stats)
    test_feats  = build_features(XX_test, item_stats)

Usage (DURING TRIAL PHASE — subset overlaps with training):
    from feature_pipeline import (
        load_data, exclude_test_users, compute_item_stats, build_features,
    )

    XX_train, yy, XX_test = load_data(
        train_path="training_batch_with_labels.npz",
        test_path="subset_training_batch.npz",
    )
    XX_train, yy = exclude_test_users(XX_train, yy, XX_test)

    item_stats  = compute_item_stats(XX_train)
    train_feats = build_features(XX_train, item_stats)
    test_feats  = build_features(XX_test, item_stats)
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy

TOTAL_ITEMS = 1000
RATING_RANGE = range(6)  # ratings 0-5


def load_data(train_path: str, test_path: str | None = None):
    """
    Load raw interaction data from .npz files.

    Returns:
        XX_train: DataFrame[user, item, rating]
        yy:       DataFrame[user, label]  (training labels)
        XX_test:  DataFrame[user, item, rating] or None
    """
    data = np.load(train_path)
    XX_train = pd.DataFrame(data["X"], columns=["user", "item", "rating"])
    yy = pd.DataFrame(data["y"], columns=["user", "label"])

    XX_test = None
    if test_path is not None:
        test_data = np.load(test_path)
        XX_test = pd.DataFrame(test_data["X"], columns=["user", "item", "rating"])

    return XX_train, yy, XX_test


def compute_item_stats(XX_train: pd.DataFrame) -> dict:
    """
    Compute item-level statistics from training interactions ONLY.
    Call this once and pass the result to build_features() for both
    train and test data to prevent data leakage.

    Returns:
        dict with keys 'item_avg_rating' and 'item_popularity' (pd.Series).
    """
    item_avg = XX_train.groupby("item")["rating"].mean().rename("item_avg_rating")
    item_pop = XX_train.groupby("item")["user"].count().rename("item_popularity")
    return {"item_avg_rating": item_avg, "item_popularity": item_pop}


def build_features(
    XX: pd.DataFrame,
    item_stats: dict,
    total_items: int = TOTAL_ITEMS,
) -> pd.DataFrame:
    """
    Build user-level features from raw interactions.

    Args:
        XX:          DataFrame[user, item, rating] — raw interactions.
        item_stats:  dict from compute_item_stats(). Required to ensure
                     item-level features are always frozen from training data.
        total_items: catalogue size for computing coverage ratio.

    Returns:
        DataFrame indexed by user with all engineered features.
    """
    item_avg = item_stats["item_avg_rating"]
    item_pop = item_stats["item_popularity"]

    # ── 1. Basic rating statistics ────────────────────────────────────────
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

    # ── 2. Rating proportions + entropy ───────────────────────────────────
    rdist = XX.groupby(["user", "rating"]).size().unstack(fill_value=0)
    rdist = rdist.reindex(columns=RATING_RANGE, fill_value=0)
    rprops = rdist.div(rdist.sum(axis=1), axis=0)
    rprops.columns = [f"prop_rating_{i}" for i in RATING_RANGE]

    stats["rating_entropy"] = rprops.apply(
        lambda row: entropy(row.values[row.values > 0]), axis=1
    )
    stats = stats.join(rprops)

    # ── 3. Extreme rating proportions ─────────────────────────────────────
    stats["prop_extreme"] = rprops["prop_rating_0"] + rprops["prop_rating_5"]

    # ── 4. Item coverage ──────────────────────────────────────────────────
    stats["unique_items_rated"] = XX.groupby("user")["item"].nunique()
    stats["item_coverage_ratio"] = stats["unique_items_rated"] / total_items

    # ── 5. Item popularity (frozen from training) ─────────────────────────
    XX_pop = XX.merge(item_pop, left_on="item", right_index=True, how="left")
    XX_pop["item_popularity"] = XX_pop["item_popularity"].fillna(0)
    pop_f = XX_pop.groupby("user")["item_popularity"].agg(
        avg_item_popularity="mean",
        std_item_popularity="std",
    )
    pop_f["std_item_popularity"] = pop_f["std_item_popularity"].fillna(0)
    stats = stats.join(pop_f)

    # ── 6. Deviation from item average (frozen from training) ─────────────
    XX_dev = XX.merge(item_avg, left_on="item", right_index=True, how="left")
    # Unseen items fall back to global training mean
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

    # ── 7. Average quality of items targeted (frozen from training) ───────
    iqf = XX_dev.groupby("user")["item_avg_rating"].agg(
        avg_item_avg_rating="mean",
        std_item_avg_rating="std",
    )
    iqf["std_item_avg_rating"] = iqf["std_item_avg_rating"].fillna(0)
    stats = stats.join(iqf)

    return stats.reset_index()


def exclude_test_users(
    XX_train: pd.DataFrame,
    yy: pd.DataFrame,
    XX_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Remove users that appear in XX_test from the training data and labels.

    Use this during the trial phase, when the test set is a subset of the training batch.

    Args:
        XX_train: Training interactions DataFrame[user, item, rating].
        yy:       Training labels DataFrame[user, label].
        XX_test:  Test interactions DataFrame[user, item, rating].

    Returns:
        XX_train_clean: Training interactions with overlapping users removed.
        yy_clean:       Training labels with overlapping users removed.
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


def get_test_labels(labeled_test_path: str, test_df: pd.DataFrame) -> np.ndarray:
    """
    Load ground-truth labels from a labeled test file and align them
    with the user order in test_df (output of build_features).

    Args:
        labeled_test_path: path to the labeled .npz file (e.g. "data/first_batch_with_labels.npz")
        test_df:           DataFrame returned by build_features(XX_test, item_stats),
                           must contain a 'user' column.

    Returns:
        1-D numpy array of labels in the same row order as test_df / X_test.
    """
    data = np.load(labeled_test_path)
    yy_test = pd.DataFrame(data["y"], columns=["user", "label"])
    labels = test_df[["user"]].merge(yy_test, on="user", how="left")["label"].values
    return labels


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature column names (excludes 'user' and 'label')."""
    return [c for c in df.columns if c not in ("user", "label")]
