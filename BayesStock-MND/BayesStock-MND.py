import torch
import numpy as np
import pandas as pd
import os
from collections import Counter
from scipy.stats import multivariate_normal


def load_stock_data(directory):
    """
    Load stock data from CSV files in the given directory.
    """
    stock_files = os.listdir(directory)
    stock_data = {}

    for file_name in stock_files:
        df = pd.read_csv(os.path.join(directory, file_name))
        if len(df['Close']) > 490:
            stock_data[file_name] = df

    return stock_data


def generate_feature_vectors(stock_df, time_windows):
    """
    Generate feature vectors based on given time windows.
    """
    features, labels = [], []
    max_window = max(time_windows)

    for t in range(max_window, len(stock_df) - 1):
        feature_vector = [
            stock_df['Adj Close'][t] / stock_df['Adj Close'][t - window]
            for window in time_windows
        ]

        features.append(feature_vector)
        labels.append(1 if stock_df['Adj Close'][t + 1] >= stock_df['Adj Close'][t] else 0)

    return np.array(features), np.array(labels)


def evaluate_stock_movement(stock_df, time_windows, train_period):
    """
    Bayesian stock movement prediction using a multivariate normal distribution.
    """
    features, labels = generate_feature_vectors(stock_df, time_windows)
    train_size = len(features) - 65

    X_train, X_test = features[train_size - train_period:train_size], features[train_size:]
    y_train, y_test = labels[train_size - train_period:train_size], labels[train_size:]

    prior_up = np.mean(y_train == 1)
    prior_down = 1 - prior_up

    up_features = X_train[y_train == 1]
    down_features = X_train[y_train == 0]

    def compute_likelihood(features, mean, covariance):
        return np.array([multivariate_normal.pdf(x, mean, covariance) for x in features])

    if len(up_features) > 1:
        mu_up, sigma_up = np.mean(up_features, axis=0), np.cov(up_features, rowvar=False)
        sigma_up += np.eye(sigma_up.shape[0]) * 1e-6  # Regularization
        likelihood_up = compute_likelihood(X_test, mu_up, sigma_up)
    else:
        likelihood_up = np.zeros(len(X_test))

    if len(down_features) > 1:
        mu_down, sigma_down = np.mean(down_features, axis=0), np.cov(down_features, rowvar=False)
        sigma_down += np.eye(sigma_down.shape[0]) * 1e-6
        likelihood_down = compute_likelihood(X_test, mu_down, sigma_down)
    else:
        likelihood_down = np.zeros(len(X_test))

    total_probability = likelihood_up * prior_up + likelihood_down * prior_down
    posterior_up = (likelihood_up * prior_up) / (total_probability + 1e-8)
    posterior_down = (likelihood_down * prior_down) / (total_probability + 1e-8)

    return (posterior_up > posterior_down).astype(int), y_test


def majority_voting(predictions_list):
    """
    Perform majority voting among multiple prediction arrays.
    """
    num_samples = len(predictions_list[0])
    return np.array([Counter([pred[i] for pred in predictions_list]).most_common(1)[0][0] for i in range(num_samples)])


if __name__ == "__main__":
    data_directory = 'your_data_path'
    stock_data = load_stock_data(data_directory)

    feature_time_windows = [[1, 3], [4, 6], [6, 18], [15, 31], [61, 66], [11, 18]]
    training_period = 477

    accuracies = []

    for stock_symbol, stock_df in stock_data.items():
        model_predictions = []
        y_test_values = None

        for window in feature_time_windows:
            test_predictions, y_test = evaluate_stock_movement(stock_df, window, training_period)
            model_predictions.append(test_predictions)

            if y_test_values is None:
                y_test_values = y_test

        final_predictions = majority_voting(model_predictions)
        accuracy = np.mean(final_predictions == y_test_values)
        accuracies.append(accuracy)
        print(f"Stock: {stock_symbol}, Accuracy: {round(accuracy, 2)}")