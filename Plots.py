import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from Utils import *
import sys


def plot_line_with_CI(ax, nb_attackers_to_test, means, stds, CI, nb_epoch, color, label):
    ax.plot(nb_attackers_to_test, means, color=color, label=label)
    ax.fill_between(nb_attackers_to_test, means - CI*(stds/np.sqrt(nb_epoch)), means + CI*(stds/np.sqrt(nb_epoch)), color=(0.5, 0.5, 1), alpha=0.1)

def plot_multiple_epochs(n_users, n_items, missing_prob, nb_attackers_to_test, error_func, nb_epochs=5, attackers_algo='random'):

    qrwmed_reg_1_mean_list = np.array([])
    qrwmed_reg_5_mean_list = np.array([])
    med_thresh_0_mean_list = np.array([])
    med_thresh_03_mean_list = np.array([])
    mean_mean_list = np.array([])
    mf_mean_list = np.array([])

    qrwmed_reg_1_std_list = np.array([])
    qrwmed_reg_5_std_list = np.array([])
    med_thresh_0_std_list = np.array([])
    med_thresh_03_std_list = np.array([])
    mean_std_list = np.array([])
    mf_std_list = np.array([])

    for nb_attackers in tqdm(nb_attackers_to_test, file=sys.stdout, dynamic_ncols=True):

        qrwmed_reg_1_mean, qrwmed_reg_5_mean, med_thresh_0_mean, med_thresh_03_mean, mean_mean, mf_mean, qrwmed_reg_1_std, qrwmed_reg_5_std, med_thresh_0_std, med_thresh_03_std, mean_std, mf_std  = run_multiple_epochs(n_users, n_items, missing_prob, nb_attackers, error_func, nb_epochs=nb_epochs, attackers_algo=attackers_algo)

        qrwmed_reg_1_mean_list = np.append(qrwmed_reg_1_mean_list, qrwmed_reg_1_mean)
        qrwmed_reg_5_mean_list = np.append(qrwmed_reg_5_mean_list, qrwmed_reg_5_mean)
        med_thresh_0_mean_list = np.append(med_thresh_0_mean_list, med_thresh_0_mean)
        med_thresh_03_mean_list = np.append(med_thresh_03_mean_list, med_thresh_03_mean)
        mean_mean_list = np.append(mean_mean_list, mean_mean)
        mf_mean_list = np.append(mf_mean_list, mf_mean)

        qrwmed_reg_1_std_list = np.append(qrwmed_reg_1_std_list, qrwmed_reg_1_std)
        qrwmed_reg_5_std_list = np.append(qrwmed_reg_5_std_list, qrwmed_reg_5_std)
        med_thresh_0_std_list = np.append(med_thresh_0_std_list, med_thresh_0_std)
        med_thresh_03_std_list = np.append(med_thresh_03_std_list, med_thresh_03_std)
        mean_std_list = np.append(mean_std_list, mean_std)
        mf_std_list = np.append(mf_std_list, mf_std)

    fig, ax = plt.subplots()

    print(f'qrmed_reg_1 {qrwmed_reg_1_mean_list}')
    print(f'qrmed_reg_5 {qrwmed_reg_5_mean_list}')
    print(f'med {med_thresh_0_mean_list}')
    print(f'med thresh 03 {med_thresh_03_mean_list}')
    print(f'mean {mean_mean_list}')
    print(f'mf {mf_mean_list}')

    plot_line_with_CI(ax, nb_attackers_to_test, qrwmed_reg_1_mean_list, qrwmed_reg_1_std_list, 0.95, nb_epochs, (0.5, 0.5, 1), 'QrwMed, reg=1')
    plot_line_with_CI(ax, nb_attackers_to_test, qrwmed_reg_5_mean_list, qrwmed_reg_5_std_list, 0.95, nb_epochs, (0, 0, 1), 'QrwMed, reg=5')
    plot_line_with_CI(ax, nb_attackers_to_test, med_thresh_0_mean_list, med_thresh_0_std_list, 0.95, nb_epochs, (1, 0.5, 0.5), 'Weighted Median')
    plot_line_with_CI(ax, nb_attackers_to_test, med_thresh_03_mean_list, med_thresh_03_std_list, 0.95, nb_epochs, (1, 0, 0), 'Weighted Median, lower threshold=0.3')
    #plot_line_with_CI(ax, nb_attackers_to_test, mean_mean_list, mean_std_list, 0.95, nb_epochs, (0, 1, 0), 'Weighted Mean')
    plot_line_with_CI(ax, nb_attackers_to_test, mf_mean_list, mf_std_list, 0.95, nb_epochs, (0.5, 0.5, 0), 'Matrix Factorization')
    # Plot mean line
    '''ax.plot(nb_attackers_to_test, qrwmed_reg_1_mean_list, color=(0.5, 0.5, 1), label='QrwMed, reg=1')
    ax.plot(nb_attackers_to_test, qrwmed_reg_5_mean_list, color=(0, 0, 1), label='QrwMed, reg=5')
    ax.plot(nb_attackers_to_test, med_thresh_0_mean_list, color=(1, 0.5, 0.5), label='Weighted Median')
    ax.plot(nb_attackers_to_test, med_thresh_03_mean_list, color=(1, 0, 0), label='Weighted Median, lower threshold=0.3')
    ax.plot(nb_attackers_to_test, mean_mean_list, color=(0, 1, 0), label='Weighted Mean')

    # Add shaded error region (mean ± std)
    ax.fill_between(nb_attackers_to_test, qrwmed_reg_1_mean_list - qrwmed_reg_1_std_list, qrwmed_reg_1_mean_list + qrwmed_reg_1_std_list, color=(0.5, 0.5, 1), alpha=0.1)
    ax.fill_between(nb_attackers_to_test, qrwmed_reg_5_mean_list - qrwmed_reg_5_std_list, qrwmed_reg_5_mean_list + qrwmed_reg_5_std_list, color=(0, 0, 1), alpha=0.1)
    ax.fill_between(nb_attackers_to_test, med_thresh_0_mean_list - med_thresh_0_std_list, med_thresh_0_mean_list + med_thresh_0_std_list, color=(1, 0.5, 0.5), alpha=0.1)
    ax.fill_between(nb_attackers_to_test, med_thresh_03_mean_list - med_thresh_03_std_list, med_thresh_03_mean_list + med_thresh_03_std_list, color=(1, 0, 0), alpha=0.1)
    ax.fill_between(nb_attackers_to_test, mean_mean_list - mean_std_list, mean_mean_list + mean_std_list, color=(0, 1, 0), alpha=0.1)
'''
    # Add labels, legend, and grid
    #ax.set_yscale('log')
    ax.set_title(f'{error_func} by the number of attackers of type {attackers_algo} per method')
    ax.set_xlabel('Number of attackers')
    ax.set_ylabel(f'{error_func}')
    #plt.yscale('log')
    ax.legend()
    ax.grid(True)

    # Show plot
    return ax

def plot_for_nbattackers_thresh_reg_compute_means_stds_one_epoch(values_to_test, nb_attackers_to_test, algo, epoch, delts, errors):
    ratings_df, real_grade = generate_random_basic_data(100, 80, 0.1)
    print(f'real_grade: {real_grade}')

    new_delts = np.empty((len(nb_attackers_to_test),len(values_to_test)))
    new_errors = np.empty((len(values_to_test)))

    print(f'Process {epoch} starts computing error')
    for idx, i in enumerate(values_to_test):
        user_sim = create_sim(ratings_df.iloc[0], ratings_df.drop(0), lower_threshold=i) if algo == 'median' else create_sim(ratings_df.iloc[0], ratings_df.drop(0))

        sane_pred = pipeline_median(ratings_df, user_sim, 0, -1, lower_threshold=i) if algo == 'median' else pipeline_qrw_med(ratings_df, user_sim, 0, -1, reg=i) if algo == 'qrmed' else pipeline_mean(ratings_df, user_sim, 0, -1) if algo == 'mean' else 0
        new_errors[idx] = np.abs(real_grade - sane_pred)

    errors[epoch] = new_errors

    print(f'Process {epoch} starts computing delts')

    for idx, j in enumerate(nb_attackers_to_test):
        print(f'Process {epoch} starts computing delta with nb_attackers: {idx}')
        for idy, i in enumerate(values_to_test):
            delta = compute_delta(ratings_df, nb_attackers=j, algo=algo, lower_threshold=i) if algo == 'median' else compute_delta(ratings_df, nb_attackers=j, algo=algo, reg=i) if algo == 'qrmed' else compute_delta(ratings_df, nb_attackers=j, algo=algo)
            new_delts[idx, idy] = delta

    delts[epoch] = new_delts

def plot_for_nbattackers_thresh_reg_compute_means_stds(nb_epochs, values_to_test, nb_attackers_to_test, algo):
    if algo not in ['median', 'qrmed', 'mean']:
        raise ValueError(f'Wooooow! What is {algo}!?')


    jobs = []

    manager = mp.Manager()
    delts = manager.dict()
    errors = manager.dict()


    for epoch in range(nb_epochs):
        p = mp.Process(target=plot_for_nbattackers_thresh_reg_compute_means_stds_one_epoch, args=(values_to_test, nb_attackers_to_test, algo, epoch, delts, errors))
        jobs.append(p)
        p.start()

    for job in jobs:
        job.join()

    delts_np = np.array(list(delts.values()))
    errors_np = np.array(list(errors.values()))

    delt_means = np.mean(delts_np, axis=0)
    delt_stds = np.std(delts_np, axis=0)

    err_means = np.mean(errors_np, axis=0)
    err_stds = np.std(errors_np, axis=0)

    return delt_means, delt_stds, err_means, err_stds



def plot_for_nbattackers_thresh_reg(axs, nb_attackers_to_test, values_to_test, nb_epochs, CI, algo):

    delt_means, delt_stds, err_means, err_stds = plot_for_nbattackers_thresh_reg_compute_means_stds(nb_epochs, values_to_test, nb_attackers_to_test, algo)

    for idx, j in enumerate(nb_attackers_to_test):
        axs[idx].set_title(j)
        axs[idx].set_xlabel('Threshold') if algo=='median' else axs[idx].set_xlabel('Reg')

        axs[idx].plot(values_to_test, err_means, label='error', color='blue')
        axs[idx].fill_between(values_to_test, err_means - CI*(err_stds/np.sqrt(nb_epochs)), err_means + CI*(err_stds/np.sqrt(nb_epochs)), color='blue', alpha=0.1)

        axs[idx].plot(values_to_test, delt_means[idx], label='delta', color='red')
        axs[idx].fill_between(values_to_test, delt_means[idx] - CI*(delt_stds[idx]/np.sqrt(nb_epochs)), delt_means[idx] + CI*(delt_stds[idx]/np.sqrt(nb_epochs)), color='red', alpha=0.1)

        axs[idx].legend()