import copy
import multiprocessing
import os
import pandas as pd
from torch import nn
from torch_geometric.nn import GATConv
from torch_geometric import loader
from sklearn.neighbors import BallTree
import torch
import numpy as np
from torch_geometric.data import Data


def hard_voting(predictions):

    """
    Performs hard voting on the predictions from multiple models.
    For each sample, the majority vote from the models is taken (i.e., if
    the mean of the predictions for a sample is greater than 0.5, the result is 1, else 0).

    Args:
        predictions (list or np.array): A list or numpy array containing the predictions from multiple models.
                                        Shape: [num_models, num_samples]

    Returns:
        np.array: An array of binary predictions based on majority voting. Shape: [num_samples]
    """
    # Convert predictions to a numpy array for better performance with vectorized operations
    predictions = np.array(predictions)

    # Compute the mean of predictions along axis 0 (across all models)
    mean_predictions = np.mean(predictions, axis=0)

    # Apply the threshold to determine the final class (1 if mean > 0.5, else 0)
    final_predictions = (mean_predictions > 0.5).astype(int)

    return final_predictions


def get_dict_of_data(path):
    """
    Loads CSV files from the specified directory and returns a dictionary
    where keys are file names and values are the corresponding DataFrames.

    Args:
        path (str): The directory path containing the CSV files.

    Returns:
        dict: A dictionary where keys are file names and values are pandas DataFrames.
    """
    name_of_files = os.listdir(path)
    dict_of_data = {}
    for name in name_of_files:
        df = pd.read_csv(os.path.join(path, name))
        dict_of_data[name] = df
    return dict_of_data


class GATSequence(nn.Module):
    """
    Graph Attention Network (GAT) model for binary classification, processing a sequence of graphs.
    This model uses two GATConv layers followed by a fully connected layer for classification.
    """

    def __init__(self, in_channels, out_channels, num_graphs=5):
        """
        Initializes the GATSequence model.

        Args:
            in_channels (int): The number of input features per node.
            out_channels (int): The number of output channels (classes).
            num_graphs (int): The number of graphs to process (typically 5 for sequence modeling).
        """
        super(GATSequence, self).__init__()
        self.num_graphs = num_graphs
        self.conv1 = GATConv(in_channels, 8, heads=8)
        self.conv2 = GATConv(8 * 8, out_channels, heads=1)
        self.fc = nn.Linear(out_channels * num_graphs, 2)  # Classify based on the output of all 5 graphs

    def forward(self, data_list):
        """
        Forward pass through the sequence of graphs.

        Args:
            data_list (list of Data): A list of `Data` objects representing the graphs to process.

        Returns:
            torch.Tensor: The output from the final fully connected layer, representing class predictions.
        """
        x_list = []
        for data in data_list:
            x, edge_index = data.x, data.edge_index
            x = self.conv1(x, edge_index)
            x = torch.relu(x)
            x = self.conv2(x, edge_index)
            x_list.append(x)

        # Aggregate outputs from 5 graphs and pass through the final FC layer
        x_seq = torch.stack(x_list, dim=1)  # Shape: [batch_size, 5, feature_dim]
        x_seq = x_seq.view(x_seq.size(0), -1)  # Flatten the 5 graph outputs
        return self.fc(x_seq)



def set_seed(seed):
    """
    Sets the seed for random number generators to ensure reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)


def create_graph(features, day_index, device, k=3):
    """
    Creates a graph for a specific day based on its feature data using k-nearest neighbors.

    Args:
        features (torch.Tensor): The feature matrix of shape [num_samples, num_stocks].
        day_index (int): The index for the day to generate the graph.
        device (torch.device): The device on which to place the graph data (e.g., 'cuda' or 'cpu').
        k (int): The number of neighbors to use for the graph construction.

    Returns:
        torch_geometric.data.Data: A Data object representing the graph for the specific day.
    """
    feature_vectors = features[day_index].cpu().numpy()
    tree = BallTree(feature_vectors)
    distances, indices = tree.query(feature_vectors, k=k + 1)

    edge_index_list = []
    for i in range(indices.shape[0]):
        for j in range(1, k + 1):
            edge_index_list.append([i, indices[i][j]])

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous().to(device)
    return Data(x=features[day_index], edge_index=edge_index)


def prepare_graph_data(dict_of_data, key='Adj Close', num_days=500, per=5):
    """
    Prepares the graph data with the past 5 days' features to be used for model training.

    Args:
        dict_of_data (dict): A dictionary where each key is a stock name and the value is a DataFrame.
        key (str): The column key to use for features (e.g., 'Adj Close').
        num_days (int): The number of days to prepare the data for.
        per (int): The number of previous days to consider for each stock (typically 5).

    Returns:
        torch.Tensor, torch.Tensor: Features and labels as tensors.
    """
    features, labels = [], []
    stock_keys = list(dict_of_data.keys())

    for t in range(per, num_days - 1):  # Start at day 5 to have past 5 days
        feature_matrix = []

        # Collect the past 5 closing prices for each stock
        for stock in stock_keys:
            close_prices = list(dict_of_data[stock][key])[t - per:t]
            dif = [(close_prices[l] > close_prices[l - 1]) for l in range(1, len(close_prices))]
            # t-5 to t-1 (5 previous days)
            feature_matrix.append(dif)

        features.append(feature_matrix)

        # Labels for the next day (t+1)
        for stock in stock_keys:
            next_close = list(dict_of_data[stock]['Adj Close'])[t + 1]
            current_close = list(dict_of_data[stock]['Adj Close'])[t]
            labels.append(1 if next_close > current_close else 0)

    features = torch.tensor(features, dtype=torch.float32)  # Shape: [num_samples, 5, num_stocks]
    labels = torch.tensor(labels, dtype=torch.long)  # Shape: [num_samples]

    return features, labels


def get_acc_for_a_day(p):
    """
    Computes accuracy for a given day using the provided data and model.

    Args:
        p (tuple): A tuple containing:
            - dict_of_data (dict): The stock data dictionary.
            - seed (int): The random seed to use for reproducibility.
            - period (int): The training period (number of days).

    Returns:
        tuple: A tuple containing:
            - predictions (np.array): The predicted labels from the model.
            - test_labels (np.array): The actual test labels for accuracy calculation.
    """
    #12
    #15
    num_graph = 19
    dict_of_data, seed, period = p

    set_seed(seed)

    key = 'Adj Close'
    all_key = [ 'Adj Close']

    max_length = max(len(data[key]) for data in dict_of_data.values())
    for stock in list(dict_of_data.keys()):
        df = pd.DataFrame()
        for keys in all_key:
            time_series = list(dict_of_data[stock][keys])
            if len(time_series) <= max_length:
                df[keys] = ((max_length - len(time_series)) * [time_series[0]] + time_series)

            else:
                df[keys] = time_series
        dict_of_data[stock] = df
    per = 4
    features, labels = prepare_graph_data(dict_of_data, 'Adj Close', num_days=500, per=per)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features = features.to(device)
    labels = labels.to(device)

    # Create a list of 5 graphs for each time step
    data_list = []
    for t in range(num_graph, len(features)):  # Start at t = 5 to have 5 previous time steps
        graphs = []
        for i in range(num_graph):  # G_t-1 to G_t-5
            graphs.append(create_graph(features, t - i - 1, device))  # G_t-5 to G_t-1
        data_list.append(graphs)

    # Split into training and testing sets
    train_data = data_list[-64-period:-64]
    test_data = data_list[-64:]
    # Initialize model
    model = GATSequence(in_channels=per-1, out_channels=4, num_graphs=num_graph).to(
        device)  # 5 features (one for each closing price)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    model.train()
    for epoch in range(5):
        for data in loader.DataLoader(train_data, batch_size=32, shuffle=False):
            data = [d.to(device) for d in data]  # Move the graphs to the device
            labels_batch = labels[data[0].batch]  # Assuming all graphs in the batch share the same label
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, labels_batch)
            loss.backward()
            optimizer.step()

    # Evaluate model on the test data
    model.eval()
    predictions = []

    with torch.no_grad():
        for data in loader.DataLoader(test_data, batch_size=1):
            data = [d.to(device) for d in data]
            out = model(data)
            pred = out.argmax(dim=1)
            predictions.append(pred.cpu().numpy())
    predictions = np.concatenate(predictions)
    stock_keys = list(dict_of_data.keys())
    # Collect test labels for accuracy calculation
    test_labels = []
    start_test_index = len(data_list) - len(test_data)
    for t in range(start_test_index, len(data_list)):
        for stock in stock_keys:
            next_close = list(dict_of_data[stock][key])[t + 1]
            current_close = list(dict_of_data[stock][key])[t]
            test_labels.append(1 if next_close > current_close else 0)

    test_labels = np.array(test_labels)

    return predictions, test_labels


if __name__ == '__main__':
    address = 'C:/stock_csv_datas/'
    dict_of_data = get_dict_of_data(address)

    # Remove stocks with insufficient data or NaN values
    for stock in list(dict_of_data):
        if len(dict_of_data[stock]['Adj Close'].values) < 500 or np.isnan(list(dict_of_data[stock]['Close'].values)).any():
            dict_of_data.pop(stock, None)

    num_of_run = 1
    results = []
    all_tuples = []
    train_periods = [100, 200, 400, 150, 250]
    seeds = list(range(num_of_run))
    accuray_of_different_seeds = []

    # Running multiple experiments
    for runs in range(num_of_run):
        all_5_model_ = []
        for train_period in train_periods:
            all_tuples.append((copy.deepcopy(dict_of_data), runs, train_period))

        with multiprocessing.Pool(processes=20) as pool:
            results = pool.map(get_acc_for_a_day, all_tuples)

        predicted_labels_of_different_model = [item[0] for item in results]
        accuray_of_different_seeds.append(
            (np.sum(hard_voting(predicted_labels_of_different_model) == results[0][1]) / len(results[0][1])) * 100
        )

    # Print the average accuracy across all runs
    print(f'Average Accuracy of different seeds: {np.mean(accuray_of_different_seeds):.3f}%')
