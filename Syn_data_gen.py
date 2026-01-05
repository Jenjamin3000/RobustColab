# Generating a synthetic user-item rating matrix via matrix factorization with group structure
# The code creates:
#  - user latent vectors influenced by group centers (users in same group have similar preferences)
#  - item latent vectors
#  - ratings = global_mean + user_latent @ item_latent.T + gaussian noise
#  - a probability of missing entries (NaN)
#  - saves a CSV to /mnt/data/synthetic_ratings.csv and displays a small preview
#  - returns both dense matrix (with NaNs) and long-format DataFrame

import numpy as np
import pandas as pd
import os

def generate_synthetic_ratings(
    n_users=200,
    n_items=100,
    n_factors=5,
    n_groups=4,
    group_strength=0.5,       # how tight users in a group are around the group center (smaller -> more similar)
    user_noise=0.2,           # per-user randomness in latent factors
    item_scale=1.0,           # scale of item latent distribution
    item_bias_std=0.1,
    user_bias_std=0.1,
    global_mean=3.0,          # baseline rating mean
    noise_std=0.3,            # gaussian noise added to ratings
    rating_min=1.0,           # minimum rating after clipping (set None to skip clipping)
    rating_max=5.0,           # maximum rating after clipping (set None to skip clipping)
    random_seed=42
):
    np.random.seed(random_seed)
    rand_gen =  np.random.default_rng(random_seed)


    # Create group centers in latent space group centers drawn from normal distribution centered at 0
    group_centers = rand_gen.normal(loc=0.0, scale=1.0, size=(n_groups, n_factors))

    # Assign each user to a group (balanced by default)
    users_per_group = np.full(n_groups, n_users // n_groups, dtype=int)
    remainder = n_users - users_per_group.sum()
    users_per_group[:remainder] += 1
    user_group = np.concatenate([[g]*users_per_group[g] for g in range(n_groups)])
    rand_gen.shuffle(user_group)  # shuffle assignment

    # Create user latent vectors: group center + small per-user noise
    user_latents = np.zeros((n_users, n_factors))
    for u in range(n_users):
        g = user_group[u]
        user_latents[u] = group_centers[g] + rand_gen.normal(scale=group_strength, size=n_factors)
        # additional small per-user variation
        user_latents[u] += rand_gen.normal(scale=user_noise, size=n_factors)

    # Create item latent vectors
    item_latents = rand_gen.normal(loc=0.0, scale=item_scale, size=(n_items, n_factors))

    # Create user and item biases (here small)
    user_bias = rand_gen.normal(loc=0.0, scale=user_bias_std, size=n_users)
    item_bias = rand_gen.normal(loc=0.0, scale=item_bias_std, size=n_items)

    # Compute raw ratings matrix
    ratings = global_mean + user_latents.dot(item_latents.T)
    ratings += user_bias[:, None] + item_bias[None, :]

    # Add Gaussian noise to ratings
    ratings += rand_gen.normal(loc=0.0, scale=noise_std, size=ratings.shape)

    # Clip ratings to bounds if requested
    if rating_min is not None or rating_max is not None:
        rmin = -np.inf if rating_min is None else rating_min
        rmax = np.inf if rating_max is None else rating_max
        ratings = np.clip(ratings, rmin, rmax)

    #ratings_with_nans[:, -1] = np.nan
    #ratings_with_nans[1, -1] = 0.5

    # Prepare DataFrame outputs
    #users = [f"user_{i}" for i in range(n_users)]
    #items = [f"item_{j}" for j in range(n_items)]
    ratings_df = pd.DataFrame(ratings)#, index=users, columns=items)

    # long-format
    '''
    long_df = ratings_df.reset_index().melt(id_vars="index", var_name="item_id", value_name="rating")
    long_df = long_df.rename(columns={"index": "user_id"})
    # drop NaNs in long format if desired (here we keep NaNs to show missingness explicitly)

    # Save CSV
    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "synthetic_ratings.csv")
    ratings_df.to_csv(csv_path, index=True)

    csv_path = os.path.join(out_dir, "synthetic_ratings_long.csv")
    long_df.to_csv(csv_path, index=True)

    meta = {
        "n_users": n_users,
        "n_items": n_items,
        "n_factors": n_factors,
        "n_groups": n_groups,
        "group_strength": group_strength,
        "global_mean": global_mean,
        "noise_std": noise_std,
        "csv_path": csv_path,
        "user_group_assignments": user_group.copy(),
    }'''
    meta = np.array([])
    long_df = np.array([])

    return ratings_df, long_df, meta

def generate_random_basic_data(nb_users, nb_items, sparcity):
    ratings_df, long_df, meta = generate_synthetic_ratings(
            n_users=nb_users,
            n_items=nb_items,
            n_factors=5,
            n_groups=3,
            group_strength=0.1,
            user_noise=0.1,
            item_scale=0.1,
            user_bias_std=0.0,
            item_bias_std=0.0,
            global_mean=0.0,
            noise_std=0.05,
            rating_min=None,
            rating_max=None,
            random_seed=None
        )
    real_grade = ratings_df.iloc[0, -1]

    ratings_df = inject_nans(ratings_df, sparcity, None)

    return ratings_df, real_grade


def inject_attackers(df: pd.DataFrame, nb_attackers: int = 1, att_grade: float = 1000) -> pd.DataFrame:
    """
    Duplicate the first user's row nb_attackers times (attackers),
    and set the last item's rating for the original first user to NaN.

    Parameters
    ----------
    df : pd.DataFrame
        User-item rating matrix (users as rows, items as columns)
    nb_attackers : int
        Number of attacker users to insert

    Returns
    -------
    pd.DataFrame
        New DataFrame with attackers inserted and first user modified
    """
    # Copy DataFrame to avoid modifying original
    df_mod = df.copy()

    # Set the last item rating of the first user to NaN
    #df_mod.iloc[0, -1] = np.nan

    # Create attacker rows (copies of the first user's row)
    attacker_rows = pd.DataFrame(
        [df_mod.iloc[0].values] * nb_attackers,
        columns=df_mod.columns
    )
    attacker_rows.iloc[:, -1] = att_grade

    # Name the new attacker users
    attacker_names = [f"attacker_{i+1}" for i in range(nb_attackers)]
    attacker_rows.index = attacker_names

    # Concatenate attackers to the existing DataFrame
    if len(attacker_rows) > 0:
        df_with_attackers = pd.concat([df_mod, attacker_rows], axis=0)
    else:
        df_with_attackers = df_mod

    return df_with_attackers

def inject_nans(ratings: pd.DataFrame, missing_prob: float, random_seed) -> pd.DataFrame:
    # Introduce missing values (NaN) according to missing_prob
    rand_gen =  np.random.default_rng(random_seed)

    mask = rand_gen.random(size=ratings.shape) < missing_prob
    mask[0, -1] = 0
    ratings_with_nans = ratings.copy()
    ratings_with_nans[mask] = np.nan

    return ratings_with_nans

def inject_random_users(ratings: pd.DataFrame, nb_attackers = 1, random_seed=None):

    rand_gen =  np.random.default_rng(random_seed)

    # Copy DataFrame to avoid modifying original
    df_mod = ratings.copy()


    attacker_rows = pd.DataFrame(rand_gen.normal(loc=0.0, scale=1.0, size=(nb_attackers, ratings.shape[1])))

    # Concatenate attackers to the existing DataFrame
    if len(attacker_rows) > 0:
        df_with_attackers = pd.concat([df_mod, attacker_rows], axis=0)
    else:
        df_with_attackers = df_mod

    return df_with_attackers

def inject_random_attacker_favoritizing(df: pd.DataFrame, random_seed, nb_attackers: int = 1, att_grade: float = 1000, ) -> pd.DataFrame:
    """
    Create random users nb_attackers times (attackers) and set the last item for them as att_grade,
    and set the last item's rating for the original first user to NaN.

    Parameters
    ----------
    df : pd.DataFrame
        User-item rating matrix (users as rows, items as columns)
    random_seed : int
        random seed
    nb_attackers : int
        Number of attacker users to insert
    att_grade : float
        The grade the attackers will use for the last item

    Returns
    -------
    pd.DataFrame
        New DataFrame with attackers inserted and first user modified
    """


    rand_gen =  np.random.default_rng(random_seed)

    # Copy DataFrame to avoid modifying original
    df_mod = df.copy()

    # Create attacker rows (copies of the first user's row)
    attacker_rows = pd.DataFrame(
        rand_gen.normal(loc=0.0, scale=1.0, size=(nb_attackers, df.shape[1])),
        columns=df_mod.columns
    )
    attacker_rows.iloc[:, -1] = att_grade

    # Name the new attacker users
    attacker_names = [f"attacker_{i+1}" for i in range(nb_attackers)]
    attacker_rows.index = attacker_names

    # Concatenate attackers to the existing DataFrame
    if len(attacker_rows) > 0:
        df_with_attackers = pd.concat([df_mod, attacker_rows], axis=0)
    else:
        df_with_attackers = df_mod

    return df_with_attackers