from Algos_sims import *
from Syn_data_gen import *
import multiprocessing as mp

def compute_att_pred_for_range(ratings, range=20, reg=0.0, nb_attackers=1, algo='qrmed', lower_threshold=0.0, attackers_algo='random'):
    mal_vals_to_test = np.arange(-range, range, int(range/5))

    print(f'Start compute_att_pred_for_range with algo {algo}')

    prediction = np.array([])

    thresh = 0

    if algo == 'qrmed':
        func = pipeline_qrw_med
    elif algo == 'mean':
        func = pipeline_mean
    elif algo == 'median':
        func = pipeline_median
        thresh = lower_threshold
    elif algo == 'mf':
        func = pipeline_mf
    else:
        raise ValueError('Unknown algo {}: choose from qrmed, mean, median or mf'.format(algo))

    user_sim = np.array([])



    for mal_val in mal_vals_to_test:

        if attackers_algo == 'random':
            att_ratings = inject_random_users(ratings, nb_attackers, None)
        elif attackers_algo == 'targeted':
            att_ratings = inject_attackers(ratings, nb_attackers, mal_val)
        elif attackers_algo == 'favoritize':
            att_ratings = inject_random_attacker_favoritizing(ratings, None, nb_attackers, mal_val)
        else:
            raise ValueError('Unknown attackers_algo {} choose from random, targeted or favoritize'.format(attackers_algo))

        if algo != 'mf':
            user_sim = create_sim(att_ratings.iloc[0], att_ratings.drop(0), thresh)

        prediction = np.append(prediction, func(att_ratings, user_sim, 0, -1, reg, lower_threshold))

    return prediction


def compute_delta(ratings_df, reg=0.0, nb_attackers=1, algo='qrmed', lower_threshold=0.0, attackers_algo='random'):
    preds = compute_att_pred_for_range(ratings_df, 800, reg, nb_attackers, algo=algo, lower_threshold=lower_threshold, attackers_algo=attackers_algo)
    delta = np.max(preds) - np.min(preds)

    return delta

def compute_all_deltas(complete_ratings_df, nb_attackers=1, missing_prob=0.5, attackers_algo='random'):
    ratings_df = inject_nans(complete_ratings_df, missing_prob, None)

    return (compute_delta(ratings_df, reg=1, nb_attackers=nb_attackers, algo='qrmed', attackers_algo=attackers_algo),
            compute_delta(ratings_df, reg=5, nb_attackers=nb_attackers, algo='qrmed', attackers_algo=attackers_algo),
            compute_delta(ratings_df, lower_threshold=0, nb_attackers=nb_attackers, algo='median', attackers_algo=attackers_algo),
            compute_delta(ratings_df, lower_threshold=0.4, nb_attackers=nb_attackers, algo='median', attackers_algo=attackers_algo),
            compute_delta(ratings_df, nb_attackers=nb_attackers, algo='mean', attackers_algo=attackers_algo),
            compute_delta(ratings_df, nb_attackers=nb_attackers, algo='mf', attackers_algo=attackers_algo))

def run_multiple_epochs(n_users, n_items, missing_prob, nb_attackers, error_func, nb_epochs=5, attackers_algo='random'):
    manager = mp.Manager()

    qrwmed_reg_1 = manager.dict()
    qrwmed_reg_5 = manager.dict()
    med_thresh_0 = manager.dict()
    med_thresh_03 = manager.dict()
    mean = manager.dict()
    mf = manager.dict()

    jobs = []

    for i in range(nb_epochs):

        p = mp.Process(target=run_one_epoch, args=(i, n_users, n_items, missing_prob, error_func, qrwmed_reg_1, qrwmed_reg_5, med_thresh_0, med_thresh_03, mean, mf, nb_attackers, attackers_algo))
        jobs.append(p)
        #print(f'job {i} started')
        p.start()



    for proc in jobs:
        proc.join()

    return np.mean(qrwmed_reg_1.values()), np.mean(qrwmed_reg_5.values()), np.mean(med_thresh_0.values()), np.mean(med_thresh_03.values()), np.mean(mean.values()), np.mean(mf.values()), np.std(qrwmed_reg_1.values()), np.std(qrwmed_reg_5.values()), np.std(med_thresh_0.values()), np.std(med_thresh_03.values()), np.std(mean.values()), np.std(mf.values())

def run_one_epoch(i, n_users, n_items, missing_prob, error_func, qrwmed_reg_1, qrwmed_reg_5, med_thresh_0, med_thresh_03, mean, mf, nb_attackers, attackers_algo):

    complete_ratings_df, long_df, meta = generate_synthetic_ratings(
        n_users=n_users,
        n_items=n_items,
        n_factors=6,
        n_groups=4,
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

    #real_grade = ratings_df.iloc[0, -1]
    #ratings_att = inject_attackers(ratings_df, nb_attackers=nb_attackers, att_grade=100000)

    if error_func == 'delta':
        qrwmed_reg_1[i], qrwmed_reg_5[i], med_thresh_0[i], med_thresh_03[i], mean[i], mf[i] = compute_all_deltas(complete_ratings_df, nb_attackers=nb_attackers, missing_prob=missing_prob, attackers_algo=attackers_algo)
    elif error_func == 'error':
        qrwmed_reg_1[i], qrwmed_reg_5[i], med_thresh_0[i], med_thresh_03[i], mean[i], mf[i] = compute_all_errors(complete_ratings_df, nb_attackers=nb_attackers, missing_prob=missing_prob, attackers_algo=attackers_algo)
    else:
        raise ValueError(f'Don\'t know algo {error_func}: choose between delta and error')
    '''
    qrwmed_reg_1[i] = 1/(compute_delta(ratings_df, reg=1, nb_attackers=nb_attackers, algo='qrmed')+1)
    qrwmed_reg_5[i] = 1/(compute_delta(ratings_df, reg=5, nb_attackers=nb_attackers, algo='qrmed')+1)
    med_thresh_0[i] = 1/(compute_delta(ratings_df, lower_threshold=0, nb_attackers=nb_attackers, algo='median')+1)
    med_thresh_03[i] = 1/(compute_delta(ratings_df, lower_threshold=0.3, nb_attackers=nb_attackers, algo='median')+1)
    mean[i] = 1/(compute_delta(ratings_df, nb_attackers=nb_attackers, algo='mean')+1)
    '''

    #print(f'job {i} finished')

def compute_all_errors(complete_ratings, nb_attackers=1, missing_prob=0.5, attackers_algo='random'):

    if attackers_algo == 'random' :
        ratings_df = inject_random_users(complete_ratings, nb_attackers=nb_attackers, random_seed=None).reset_index(drop=True)
    elif attackers_algo == 'favoritize' :
        ratings_df = inject_random_attacker_favoritizing(complete_ratings, None, nb_attackers=nb_attackers).reset_index(drop=True)
    elif attackers_algo == 'targeted' :
        ratings_df = inject_attackers(complete_ratings, nb_attackers=nb_attackers).reset_index(drop=True)
    else:
        raise ValueError(f'Don\'t know algo: {attackers_algo}')



    ratings_nans = inject_nans(ratings_df, missing_prob, None)


    qrwmed_reg0_df = ratings_nans.copy()
    qrwmed_reg5_df = ratings_nans.copy()
    med_thresh0_df = ratings_nans.copy()
    med_thresh03_df = ratings_nans.copy()
    mean_df = ratings_nans.copy()

    n_users, n_items = ratings_nans.shape

    if attackers_algo != 'targeted' :
        for u in range(n_users):
            user_sim = create_sim(ratings_nans.iloc[u], ratings_nans.drop(u, axis=0))
            for i in range(n_items):
                if pd.isna(ratings_nans.iloc[u, i]):
                    qrwmed_reg0_df.iloc[u, i] = pipeline_qrw_med(ratings_nans, user_sim, u, i, reg=0)
                    qrwmed_reg5_df.iloc[u, i] = pipeline_qrw_med(ratings_nans, user_sim, u, i, reg=5)
                    med_thresh0_df.iloc[u, i] = pipeline_median(ratings_nans, user_sim, u, i, lower_threshold=0)
                    med_thresh03_df.iloc[u, i] = pipeline_median(ratings_nans, user_sim, u, i, lower_threshold=0.3)
                    mean_df.iloc[u, i] = pipeline_mean(ratings_nans, user_sim, u, i)

        mf_df = infer_mf(ratings_nans, 10)

        qrwmed_reg0_error = np.array(np.abs(ratings_df - qrwmed_reg0_df)).mean()
        qrwmed_reg5_error = np.array(np.abs(ratings_df - qrwmed_reg5_df)).mean()
        med_thresh0_error = np.array(np.abs(ratings_df - med_thresh0_df)).mean()
        med_thresh03_error = np.array(np.abs(ratings_df - med_thresh03_df)).mean()
        mean_error = np.array(np.abs(ratings_df - mean_df)).mean()
        mf_error = np.array(np.abs(ratings_df - mf_df)).mean()

    else :
        user_sim = create_sim(ratings_nans.iloc[0], ratings_nans.drop(0, axis=0))

        qrwmed_reg0_error = np.abs(pipeline_qrw_med(ratings_nans, user_sim, 0, -1, reg=0) - complete_ratings.iloc[0, -1])
        qrwmed_reg5_error = np.abs(pipeline_qrw_med(ratings_nans, user_sim, 0, -1, reg=5) - complete_ratings.iloc[0, -1])
        med_thresh0_error = np.abs(pipeline_median(ratings_nans, user_sim, 0, -1, lower_threshold=0) - complete_ratings.iloc[0, -1])
        med_thresh03_error = np.abs(pipeline_median(ratings_nans, user_sim, 0, -1, lower_threshold=0.3) - complete_ratings.iloc[0, -1])
        mean_error = np.abs(pipeline_mean(ratings_nans, user_sim, 0, -1) - complete_ratings.iloc[0, -1])
        mf_error = np.abs(infer_mf(ratings_nans, 5)[0, -1] - complete_ratings.iloc[0, -1])




    return qrwmed_reg5_error, qrwmed_reg0_error, med_thresh0_error, med_thresh03_error, mean_error, mf_error