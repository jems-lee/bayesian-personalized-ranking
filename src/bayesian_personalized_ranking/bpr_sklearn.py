import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
    check_random_state,
)

from bpr_numba import fit_bpr
from utils import (
    create_user_map_table,
    create_item_map_table,
    create_user_map,
    create_item_map,
    create_data_triplets_index_only,
)


class BPR(BaseEstimator):
    def __init__(
        self,
        n_factors=10,
        n_epochs=1,
        batch_size=1,
        init_mean=0,
        init_std_dev=0.1,
        lr_all=0.005,
        reg_all=0.02,
        lr_bi=None,
        lr_pu=None,
        lr_qi=None,
        reg_bi=None,
        reg_pu=None,
        reg_qi=None,
        random_state=None,
        eps=1e-5,
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.init_mean = init_mean
        self.init_std_dev = init_std_dev
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.lr_bi = lr_bi
        self.lr_pu = lr_pu
        self.lr_qi = lr_qi
        self.reg_bi = reg_bi
        self.reg_pu = reg_pu
        self.reg_qi = reg_qi
        self.random_state = random_state
        self.user_factors = None
        self.item_factors = None
        self.item_biases = None
        self.known_users = None
        self.known_items = None
        self.user_map = None
        self.item_map = None
        self.residuals = None
        self.eps = eps

    def fit(self, X, y):
        """Fit the model using stochastic gradient descent.

        Parameters
        ----------
        X : ndarray shape ( m, 2 )
            Columns are [ user_id, item_id ]
        y : ndarray shape ( m, )
            Array of 1 : relevent and 0 if not

        Returns
        -------

        """

        X, y = check_X_y(X, y)
        n_users = len(np.unique(X[:, 0]))
        n_items = len(np.unique(X[:, 1]))

        df = pd.DataFrame({"user_id": X[:, 0], "item_id": X[:, 1], "relevance": y})
        user_map_table = create_user_map_table(df)
        item_map_table = create_item_map_table(df)
        self.user_map = create_user_map(df)
        self.item_map = create_item_map(df)
        data_triplets = create_data_triplets_index_only(
            df, user_map_table, item_map_table
        )
        print("Data triplets created")
        m = data_triplets.shape[0]

        self.is_fitted_ = True
        self.random_state_ = check_random_state(self.random_state)

        self.lr_bi = self.lr_bi if self.lr_bi is not None else self.lr_all
        self.lr_pu = self.lr_pu if self.lr_pu is not None else self.lr_all
        self.lr_qi = self.lr_qi if self.lr_qi is not None else self.lr_all
        self.reg_bi = self.reg_bi if self.reg_bi is not None else self.reg_all
        self.reg_pu = self.reg_pu if self.reg_pu is not None else self.reg_all
        self.reg_qi = self.reg_qi if self.reg_qi is not None else self.reg_all
        self.batch_size = self.batch_size if self.batch_size is not None else 1
        self.residuals = np.zeros(self.n_epochs)

        self.known_users = set(X[:, 0])
        self.known_items = set(X[:, 1])

        self.user_factors = self.random_state_.normal(
            loc=self.init_mean,
            scale=self.init_std_dev,
            size=(n_users, self.n_factors),
        )
        self.item_factors = self.random_state_.normal(
            loc=self.init_mean,
            scale=self.init_std_dev,
            size=(n_items, self.n_factors),
        )
        self.user_biases = self.random_state_.normal(
            loc=self.init_mean, scale=self.init_std_dev, size=n_users
        )
        self.item_biases = self.random_state_.normal(
            loc=self.init_mean, scale=self.init_std_dev, size=n_items
        )

        (
            self.user_factors,
            self.item_factors,
            self.item_biases,
            self.residuals,
        ) = fit_bpr(
            data_triplets=data_triplets,
            initial_user_factors=self.user_factors,
            initial_item_factors=self.item_factors,
            initial_item_biases=self.item_biases,
            lr_bi=self.lr_bi,
            lr_pu=self.lr_pu,
            lr_qi=self.lr_qi,
            reg_bi=self.reg_bi,
            reg_pu=self.reg_pu,
            reg_qi=self.reg_qi,
            verbose=False,
            n_epochs=self.n_epochs,
            batch_size=self.batch_size,
            eps=self.eps,
        )
        if len(self.residuals) < self.n_epochs:
            print(f"Converged")
        return self

    def predict(self, X: np.ndarray):
        """

        Parameters
        ----------
        X : array-like
            Columns [ user_id, item_id ]
        Returns
        -------
        scores : ndarray

        """

        check_is_fitted(self, "is_fitted_")
        X = check_array(X)
        m = X.shape[0]
        scores = np.zeros(m)
        for i in np.arange(m):
            user_id = X[i, 0]
            item_id = X[i, 1]

            if user_id in self.user_map and item_id in self.item_map:
                u_idx = self.user_map[user_id]
                i_idx = self.item_map[item_id]
                scores[i] = (
                    np.dot(self.user_factors[u_idx, :], self.item_factors[i_idx, :])
                    + self.item_biases[i_idx]
                )

            elif item_id in self.item_map:
                i_idx = self.item_map[item_id]
                scores[i] = self.item_biases[i_idx]

            else:
                # item not in training set
                scores[i] = -np.inf

        return scores
