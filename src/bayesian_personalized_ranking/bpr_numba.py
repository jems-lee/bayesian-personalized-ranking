from typing import Tuple

import numpy as np
from numba import jit


@jit(nopython=True)
def fit_bpr(
    data_triplets: np.ndarray,
    initial_user_factors: np.ndarray,
    initial_item_factors: np.ndarray,
    initial_item_biases: np.ndarray,
    lr_bi: float = 0.01,
    lr_pu: float = 0.01,
    lr_qi: float = 0.01,
    reg_bi: float = 0.01,
    reg_pu: float = 0.01,
    reg_qi: float = 0.01,
    verbose=False,
    n_epochs=100,
    batch_size=50,
    eps=1e-5,
    decay=.01
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    m = data_triplets.shape[0]
    residuals = np.zeros(n_epochs)
    user_factors = initial_user_factors.copy()
    item_factors = initial_item_factors.copy()
    item_biases = initial_item_biases.copy()
    samples = np.random.choice(m, batch_size*n_epochs, replace=True)

    for epoch in range(n_epochs):
        old_user_factors = user_factors
        old_item_factors = item_factors
        old_item_biases = item_biases
        samples_epoch = samples[(batch_size*epoch):(batch_size*(epoch+1))]

        epoch_lr_bi = lr_bi / (1 + decay * epoch)
        epoch_lr_pu = lr_pu / (1 + decay * epoch)
        epoch_lr_qi = lr_qi / (1 + decay * epoch)

        (user_factors, item_factors, item_biases) = fit_batch(
            data_triplets=data_triplets,
            initial_user_factors=user_factors,
            initial_item_factors=item_factors,
            initial_item_biases=item_biases,
            lr_bi=epoch_lr_bi,
            lr_pu=epoch_lr_pu,
            lr_qi=epoch_lr_qi,
            reg_bi=reg_bi,
            reg_pu=reg_pu,
            reg_qi=reg_qi,
            verbose=False,
            samples=samples_epoch,
        )

        batch_norm = (
            np.linalg.norm(user_factors - old_user_factors)
            + np.linalg.norm(item_factors - old_item_factors)
            + np.linalg.norm(item_biases - old_item_biases)
        ) / batch_size
        residuals[epoch] = batch_norm
        if batch_norm < eps:
            return user_factors, item_factors, item_biases, residuals[:epoch]

    return user_factors, item_factors, item_biases, residuals


@jit(nopython=True)
def fit_batch(
    data_triplets: np.ndarray,
    initial_user_factors: np.ndarray,
    initial_item_factors: np.ndarray,
    initial_item_biases: np.ndarray,
    lr_bi: float = 0.01,
    lr_pu: float = 0.01,
    lr_qi: float = 0.01,
    reg_bi: float = 0.01,
    reg_pu: float = 0.01,
    reg_qi: float = 0.01,
    verbose=False,
    samples=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """

    Parameters
    ----------
    initial_item_biases
    samples
    data_triplets
    initial_user_factors
    initial_item_factors
    lr_bi
    lr_pu
    lr_qi
    reg_bi
    reg_pu
    reg_qi
    verbose

    Returns
    -------

    """

    user_factors = initial_user_factors.copy()
    item_factors = initial_item_factors.copy()
    item_biases = initial_item_biases.copy()

    for idx in samples:
        row = data_triplets[idx, :]
        u = row[0]
        i = row[1]
        j = row[2]

        pu = user_factors[u, :]
        qi = item_factors[i, :]
        qj = item_factors[j, :]

        x_ui = np.dot(pu, qi) + item_biases[i]
        x_uj = np.dot(pu, qj) + item_biases[j]
        x_uij = x_ui - x_uj

        coeff = -np.exp(-x_uij) / (1 + np.exp(-x_uij))

        user_factors[u, :] = pu - lr_pu * (coeff * (qi - qj) + reg_pu * pu)

        item_factors[i, :] = qi - lr_qi * (coeff * pu + reg_qi * qi)
        item_factors[j, :] = qj - lr_qi * (coeff * -pu + reg_qi * qj)

        item_biases[i] += -lr_bi * (coeff + reg_bi * item_biases[i])
        item_biases[j] += -lr_bi * (-coeff + reg_bi * item_biases[j])

    return user_factors, item_factors, item_biases


@jit(nopython=False)
def score_bpr(
    X: np.ndarray,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    user_biases: np.ndarray,
    item_biases: np.ndarray,
    global_mean: np.ndarray,
    known_users,
    known_items,
):
    """

    Parameters
    ----------
    X : ndarray
        Columns are [ user_id, item_id ]
    user_factors
    item_factors
    user_biases
    item_biases
    global_mean
    known_users : set
    known_items : set

    Returns
    -------

    """
    m = X.shape[0]
    scores = np.zeros(m)
    for i in np.arange(m):
        u = X[i, 0]
        i = X[i, 1]
        if u in known_users and i in known_items:
            scores[i] = (
                np.dot(user_factors[u, :], item_factors[i, :])
                + user_biases[u]
                + item_biases[i]
                + global_mean
            )
        elif u in known_users:
            scores[i] = user_biases[u] + global_mean
        elif i in known_items:
            scores[i] = item_biases[i] + global_mean
        else:
            scores[i] = global_mean
    return scores
