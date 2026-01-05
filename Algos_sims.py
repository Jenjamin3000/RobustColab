import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.decomposition import NMF

def pipeline_mean(base_rates: pd.DataFrame, user_sim: np.array, user: int, item: int, reg=0.0, lower_threshold=0.0):
    #user_unknown = base_rates.iloc[user]
    base_rates = base_rates.drop(user)
    #user_sim = create_sim(user_unknown, base_rates)
    sum_sim = 0
    rate = 0
    for user_row in range(len(base_rates)):
        if not np.isnan(base_rates.iloc[user_row, item]):
            sum_sim += np.abs(user_sim[user_row])
            rate += user_sim[user_row] * base_rates.iloc[user_row, item]
    if sum_sim == 0:
        return 0
    else:
        return rate/sum_sim

def pipeline_median(base_rates: pd.DataFrame, user_sim: np.array, user: int, item: int, reg=0.0, lower_threshold=0.0):
    #user_unknown = base_rates.iloc[user]
    base_rates = base_rates.drop(user)
    #user_sim = create_sim(user_unknown, base_rates, lower_threshold)

    # Zip weights and rates
    weights_rates = np.stack([user_sim, base_rates.iloc[:,item]], axis=1)

    # Inverse rates and weights for negative weights
    weights_rates = np.apply_along_axis(lambda arr: arr if arr[0] >= 0 else -arr, 1, weights_rates)

    # Sort by rates
    sorted_list = weights_rates[np.argsort(weights_rates[:,1])]

    # Remove nans
    clean_array = sorted_list[~np.isnan(sorted_list).any(axis=1)]

    # Sum the weights of the remaining entries
    sum_weights = np.sum(clean_array[:,0])

    # Accumulate the weights and stopping when reaching the half of the sum
    weights_acc = 0
    for w_user in clean_array:

        weights_acc += w_user[0]
        if weights_acc > sum_weights/2:
            return w_user[1]


    return 0

def pipeline_qrw_med(rates: pd.DataFrame, user_sim: np.array, user: int, item: int, reg=0.0, lower_threshold=0.0):
    #user_unknown = rates.iloc[user]
    base_rates = rates.drop(user)
    #user_sim = create_sim(user_unknown, base_rates)

    # Zip weights and rates
    weights_rates = np.stack([user_sim, base_rates.iloc[:,item]], axis=1)

    # Inverse rates and weights for negative weights
    weights_rates = np.apply_along_axis(lambda arr: arr if arr[0] >= 0 else -arr, 1, weights_rates)

    def qrw_med(x, weights_rates, reg):
        acc = 0.0
        for weight_rate in weights_rates:
            if not np.isnan(weight_rate[1]):
                acc += weight_rate[0]*np.abs(weight_rate[1]-x)

        return acc + (reg * x**2)

    return minimize(qrw_med, np.array([0]), (weights_rates, reg)).x[0]

def pipeline_mf(rates: pd.DataFrame, user_sim: np.array, user: int, item: int, reg=0.0, lower_threshold=0.0):
    inferred_df = pd.DataFrame(infer_mf(rates, 5))

    return inferred_df.iloc[user, item]

def infer_mf(df: pd.DataFrame, n_components: int, learning_rate :float =0.01, reg_param :float =0.1, num_epochs :int = 200, print_mse=False) -> pd.DataFrame:
    pmf = PMF(np.array(df), num_factors=n_components, learning_rate=learning_rate,
              reg_param=reg_param, num_epochs=num_epochs, print_mse=print_mse)
    pmf.train()

    return pmf.full_matrix()

class PMF:
    def __init__(self, R, num_factors, learning_rate=0.01, reg_param=0.01, num_epochs=100, print_mse=False):
        self.R = R
        self.num_users, self.num_items = R.shape
        self.num_factors = num_factors
        self.learning_rate = learning_rate
        self.reg_param = reg_param
        self.num_epochs = num_epochs
        self.print_mse = print_mse

    def train(self):
        self.U = np.random.normal(
            scale=1. / self.num_factors, size=(self.num_users, self.num_factors))
        self.V = np.random.normal(
            scale=1. / self.num_factors, size=(self.num_items, self.num_factors))

        for epoch in range(self.num_epochs):
            '''for i in range(self.num_users):
                for j in range(self.num_items):
                    if self.R[i, j] > 0:
                        prediction = self.predict(i, j)
                        error = self.R[i, j] - prediction

                        self.U[i, :] = self.U[i, :] + self.learning_rate * (error * self.V[j, :] - self.reg_param * self.U[i, :])
                        self.V[j, :] = self.V[j, :] + self.learning_rate * (error * self.U[i, :] - self.reg_param * self.V[j, :])'''

            error = np.abs(np.nan_to_num(self.R - self.full_matrix()))
            self.U += self.learning_rate*(error.dot(self.V) - self.reg_param*self.U)
            self.V += self.learning_rate*(error.T.dot(self.U) - self.reg_param*self.V)

            if self.print_mse:
                print('------------------------')
                print(f'error: {error}')
                print(f'self.U: {self.U}')
                print(f'self.V: {self.V}')
                #mse = self.compute_mse()
                #print(f'Epoch: {epoch+1}, MSE: {mse}')

    def predict(self, i, j):
        return np.dot(self.U[i, :], self.V[j, :])

    def compute_mse(self):
        predicted = self.full_matrix()

        error = np.sum((self.R[~np.isnan(self.R)] - predicted[~np.isnan(self.R)]) ** 2)

        return np.sqrt(error)

    def full_matrix(self):
        return np.dot(self.U, self.V.T)


def create_sim(user_unknown, base_rates, lower_threshold=0.0):
    '''

    Create the similarity vector between user_unknown and each user in base_rates using cosine similarity. The items not rated by any of the two users are ignored.
    (Consider improve this function using numpy built-in functions)

    :param user_unknown:
    :param base_rates:
    :return:
    '''
    user_sim = np.array([])

    for user in range(len(base_rates)):
        sumx = 0
        sumy = 0
        sumxy = 0

        for idx in range(len(base_rates.columns)):
            if not np.isnan(user_unknown.iloc[idx]) and not np.isnan(base_rates.iloc[user, idx]):
                sumx += user_unknown.iloc[idx] * user_unknown.iloc[idx]
                sumy += base_rates.iloc[user, idx] * base_rates.iloc[user, idx]
                sumxy += user_unknown.iloc[idx] * base_rates.iloc[user, idx]

        if np.sqrt(sumx)*np.sqrt(sumy) == 0:
            new_sim = 0
        else:
            new_sim = sumxy/(np.sqrt(sumx)*np.sqrt(sumy))

        if 0 <= new_sim < lower_threshold:
            user_sim = np.append(user_sim, lower_threshold)
        elif 0 > new_sim > -lower_threshold:
            user_sim = np.append(user_sim, -lower_threshold)
        else:
            user_sim = np.append(user_sim, new_sim)

    return user_sim