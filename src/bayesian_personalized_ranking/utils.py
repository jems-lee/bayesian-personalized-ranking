import logging
from dataclasses import dataclass
from random import sample

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_df_from_matrix(relevance_matrix):
    """Generate the data frame from the relevance matrix.

    Parameters
    ----------
    relevance_matrix
        matrix of 1s and 0s
    Returns
    -------
    df
        Columns [ user_id, item_id, relevance ]
    """

    num_users, num_items = relevance_matrix.shape
    user_ids = np.repeat(range(1, num_users + 1), num_items)
    item_ids = np.array(list(range(num_items)) * num_users)
    relevance = relevance_matrix.flatten()
    data = pd.DataFrame(
        {"user_id": user_ids, "item_id": item_ids, "relevance": relevance}
    )
    return data


def create_data_triplets(df) -> pd.DataFrame:
    """
    Generate the data triplets of user, relevant, irrelevant

    Input dataframe must have columns "user_id", "item_id", "relevance"
    """
    return (
        df.query("relevance == 1")
        .merge(df.query("relevance == 0"), on="user_id", how="inner")
        .drop(columns=["relevance_x", "relevance_y"])
        .rename(
            columns={
                "item_id_x": "relevant_item_id",
                "item_id_y": "irrelevant_item_id",
                "user_id": "user_id",
            }
        )
    )


def generate_simulated_data(k, n_users, n_items, mean=0, std=0.5, prob=1):
    """Creates simulated data with a uniform probability of missingness

    Parameters
    ----------
    k : int
    n_users : int
    n_items : int
    mean : float
    std : float
    prob_missing : float

    Returns
    -------
    df : DataFrame
        Columns [ user_id, item_id, relevance ]
    """
    userf = np.random.normal(mean, std, (n_users, k))
    itemf = np.random.normal(mean, std, (n_items, k))
    userb = np.random.normal(mean, std, n_users)
    itemb = np.random.normal(mean, std, n_items)
    global_mean = np.random.normal(mean, std, 1)
    rel_matrix = 1 * (
        np.matmul(userf, itemf.T)
        + np.tile(userb, (n_items, 1)).T
        + np.tile(itemb, (n_users, 1))
        + global_mean
        > 0
    )
    df = create_df_from_matrix(rel_matrix)
    mask = np.random.binomial(1, prob, df.shape[0])
    return df[mask == 1]


def create_data_triplets_index_only(
    df: pd.DataFrame, user_map_table: pd.DataFrame, item_map_table: pd.DataFrame
) -> np.ndarray:
    """Creates the data triplets using indices, not ids

    Parameters
    ----------
    df : DataFrame
        Columns [user_id, item_id, relevance]
    user_map_table : DataFrame
        Columns [user_id, user_idx]
    item_map_table : DataFrame
        Columns [item_id, item_idx]

    Returns
    -------
    df : ndarray
        Columns [user_idx, relevant_item_idx, irrelevant_item_idx]
    """
    df_idx = (
        df.merge(user_map_table, on="user_id", how="inner")
        .merge(item_map_table, on="item_id", how="inner")
        .drop(columns=(["user_id", "item_id"]))
    )
    return (
        df_idx.copy()
        .query("relevance == 1")
        .merge(df_idx.query("relevance == 0"), on="user_idx", how="inner")
        .drop(columns=["relevance_x", "relevance_y"])
        .rename(
            columns={
                "item_id_x": "relevant_item_idx",
                "item_id_y": "irrelevant_item_idx",
            }
        )
        .to_numpy()
    )


def create_item_map(df):
    """Maps item id to item index

    Parameters
    ----------
    df

    Returns
    -------

    """
    num_items = len(df.item_id)
    return dict(zip(df.item_id.unique(), range(num_items)))


def create_item_map_table(df):
    unique_items = df.item_id.unique()
    num_items = len(unique_items)
    return pd.DataFrame({"item_id": unique_items, "item_idx": range(num_items)})


def create_user_map(df):
    """Maps user id to user index

    Parameters
    ----------
    df

    Returns
    -------

    """
    num_users = len(df.user_id)
    return dict(zip(df.user_id.unique(), range(num_users)))


def create_user_map_table(df):
    unique_users = df.user_id.unique()
    num_users = len(unique_users)
    return pd.DataFrame({"user_id": unique_users, "user_idx": range(num_users)})


def model_recall(X: np.ndarray, y: np.ndarray, model, n: int):
    scores = model.predict(X)
    return recall_at_n(y, scores, n)


def recall_at_n(y: np.ndarray, scores: np.ndarray, n: int):
    """Calculates the recall at n of model scores.

    Parameters
    ----------
    y : ndarray
        labels of relevance 0 or 1
    scores : ndarray
        list of scores, unsorted
    n : int
        top n to consider

    Returns
    -------
    recall : float
        recall value
    """
    rank_idx = np.argsort(scores)
    top_n = y[rank_idx[-n:]]
    return top_n.mean()


def average_auc(data_triplets, user_factors, item_factors, item_bias):
    """
    Score the trained model using average AUC.
    """
    logger.info(f"data: {data_triplets.shape}")
    num_users = len(user_factors)
    # per_user = np.zeros(len(data_triplets.user_id.unique()))
    per_user = np.zeros(num_users)
    per_user_counter = np.zeros(num_users)
    for idx in range(len(data_triplets)):
        row = data_triplets[idx, :]
        u = row[0]
        i = row[1]
        j = row[2]
        pu = user_factors[u, :]
        qi = item_factors[i, :]
        qj = item_factors[j, :]
        x_ui = np.dot(pu, qi) + item_bias[i]
        x_uj = np.dot(pu, qj) + item_bias[j]
        per_user[u] += 1 * (x_ui > x_uj)
        per_user_counter[u] += 1

    average_per_user = []
    for idx in range(num_users):
        if per_user_counter[idx] > 0:
            average_per_user.append(per_user[idx] / per_user_counter[idx])

    return np.array(average_per_user)
