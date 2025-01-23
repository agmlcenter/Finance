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


def get_dict_of_data(path):
    """
    Loads CSV files from the specified directory and returns a dictionary
    where keys are file names and values are the corresponding DataFrames.

    Args:
        path (str): Path to the directory containing CSV files.

    Returns:
        dict: A dictionary with file names as keys and DataFrames as values.
    """
    name_of_files = os.listdir(path)
    dict_of_data = {}
    for name in name_of_files:
        df = pd.read_csv(os.path.join(path, name))
        if len(list(df['Close']))>2400:
            dict_of_data[name] = df
    return dict_of_data

class GAT(nn.Module):
    """
    Graph Attention Network (GAT) model for binary classification.
    This model uses two graph attention layers and a fully connected output layer.
    """
    def __init__(self, in_channels, out_channels):
        """
        Initializes the GAT model with two graph attention layers.

        Args:
            in_channels (int): The number of input channels (features).
            out_channels (int): The number of output channels (classes).
        """
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, 8, heads=8)
        self.conv2 = GATConv(8 * 8, out_channels, heads=1)
        self.fc = nn.Linear(out_channels, 2)  # Assuming binary classification

    def forward(self, data):
        """
        Defines the forward pass of the model.

        Args:
            data (Data): The input graph data.

        Returns:
            torch.Tensor: The output of the fully connected layer.
        """
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return self.fc(x)

def set_seed(seed):
    """
    Sets the seed for reproducibility of results.

    Args:
        seed (int): The seed value for random number generators.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

def create_graph(features, day_index, device, k=3):
    """
    Creates an approximate k-NN graph using Ball Tree from the features for a specific day.

    Args:
        features (torch.Tensor): The feature matrix for all days.
        day_index (int): The index of the day for which to create the graph.
        device (torch.device): The device to store the data (CPU or GPU).
        k (int, optional): The number of neighbors to consider for each node. Default is 3.

    Returns:
        Data: The graph data object containing features and edge index.
    """
    feature_vectors = features[day_index].cpu().numpy()

    # Build a Ball Tree on the feature vectors
    tree = BallTree(feature_vectors)

    # Query the Ball Tree for k nearest neighbors for each node
    distances, indices = tree.query(feature_vectors, k=k + 1)  # k + 1 to avoid self-loop

    edge_index_list = []
    for i in range(indices.shape[0]):
        for j in range(1, k + 1):  # Skip the first neighbor (self-loop)
            edge_index_list.append([i, indices[i][j]])

    # Convert to edge index format for PyTorch Geometric
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous().to(device)

    return Data(x=features[day_index], edge_index=edge_index)


def get_acc_for_a_day(p):
    """
    Computes accuracy for a given day using the provided data.

    Args:
        p (tuple): A tuple containing the day index, data dictionary, and random seed.

    Returns:
        float: The accuracy of the model for the given day, as a percentage.
    """
    index, dict_of_data, seed = p
    set_seed(seed)  # Set the seed before any random operation

    key = 'Adj Close'

    # Determine the maximum length of time series for all stocks
    max_length = max(len(data[key]) for data in dict_of_data.values())

    # Fill shorter time series with repeated values of the first element
    for stock in list(dict_of_data.keys()):
        df = pd.DataFrame()
        time_series = list(dict_of_data[stock][key])
        if len(time_series) < max_length:
            df[key] = (max_length - len(time_series)) * [time_series[0]] + time_series
        else:
            df[key] = time_series
        dict_of_data[stock] = df.tail(11 + index).head(13)

    # Prepare features and labels
    features, labels = [], []
    num_days = max(len(data[key]) for data in dict_of_data.values())
    stock_keys = list(dict_of_data.keys())

    # Create data for each day
    for t in range(5, num_days - 1):  # Start from day 5
        feature_matrix = []
        for stock in stock_keys:
            # Get last 5 days of closing prices
            close_prices = list(dict_of_data[stock][key])[t - 5:t]
            feature_matrix.append(close_prices)
        features.append(feature_matrix)
        for stock in stock_keys:
            next_close = list(dict_of_data[stock][key])[t + 1]
            current_close = list(dict_of_data[stock][key])[t]
            labels.append(1 if next_close > current_close else 0)

    # Convert features and labels to tensors and move to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features = torch.tensor(features, dtype=torch.float).to(device)
    labels = torch.tensor(labels, dtype=torch.long).to(device)

    # Create temporal graphs using k-NN
    data_list = [create_graph(features, t, device) for t in range(len(features))]
    train_data = data_list[:-2]
    test_data = data_list[-2:]

    # Initialize and train the GAT model
    model = GAT(in_channels=5, out_channels=4).to(device)  # 5 features from last 5 closing prices
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(6):  # Number of epochs
        for data in loader.DataLoader(train_data, batch_size=32, shuffle=False):
            data = data.to(device)
            labels_batch = labels[data.batch]  # Use labels from the same device
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, labels_batch)  # Compute the loss
            loss.backward()
            optimizer.step()

    # Evaluate model on the test data
    model.eval()
    predictions = []

    with torch.no_grad():
        for data in loader.DataLoader(test_data[:-1], batch_size=32):
            data = data.to(device)
            out = model(data)
            pred = out.argmax(dim=1)
            predictions.append(pred.cpu().numpy())

    # Concatenate predictions
    predictions = np.concatenate(predictions)

    # Collect test labels for accuracy calculation
    test_labels = []
    start_test_index = len(data_list) - (len(test_data))
    for t in range(start_test_index, len(data_list) - 1):
        for stock in stock_keys:
            next_close = list(dict_of_data[stock][key])[t + 1]
            current_close = list(dict_of_data[stock][key])[t]
            if (t - start_test_index) * len(stock_keys) + stock_keys.index(stock) < len(predictions):
                test_labels.append(1 if next_close > current_close else 0)

    # Convert test labels to numpy array
    test_labels = np.array(test_labels)

    # Calculate accuracy
    accuracy = np.sum(predictions == test_labels) / len(test_labels)
    return accuracy * 100


if __name__ == '__main__':
    # Data preparation
    address = 'C:/stock_csv_datas/'
    dict_of_data = get_dict_of_data(address)

    # Remove stocks with insufficient data
    for stock in list(dict_of_data):
        if len(dict_of_data[stock]['Close'].values) < 500 or np.isnan(list(dict_of_data[stock]['Close'].values)).any():
            dict_of_data.pop(stock, None)


    all_run_result = []

    # Number of all runs
    num_of_run = 1
    all_tuple = []
    test_train = 64    # Construct tuples for multiprocessing with 10 different seeds
    seeds = list(range(num_of_run))

    for idx in range(test_train):
        for j in range(num_of_run):
            all_tuple.append((test_train+1 - idx, copy.deepcopy(dict_of_data), seeds[j]))

    # Use multiprocessing to compute accuracy
    results = []
    for tuple_ in all_tuple:
        results.append(get_acc_for_a_day(tuple_))
    # with multiprocessing.Pool(processes=20) as pool:
    #     results = pool.map(get_acc_for_a_day, all_tuple)

    # Calculate average accuracy across runs
    Average_Acc = 0
    for j in range(num_of_run):
        Average_Acc += np.average([results[j + i * num_of_run] for i in range(test_train)]) / num_of_run
    print(f'Accuracy: {Average_Acc:.3f}%')
