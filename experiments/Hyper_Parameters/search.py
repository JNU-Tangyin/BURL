import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

class SimpleDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        return torch.from_numpy(sample).float(), label
class TripletDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anchor = self.data[idx]
        anchor_label = self.labels[idx]
        positive_idx = random.choice([i for i, label in enumerate(self.labels) if label == anchor_label and i != idx])
        positive = self.data[positive_idx]
        negative_idx = random.choice([i for i, label in enumerate(self.labels) if label != anchor_label])
        negative = self.data[negative_idx]
        return torch.from_numpy(anchor).float(), torch.from_numpy(positive).float(), torch.from_numpy(negative).float()

class SiameseNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SiameseNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, output_dim),
        )

    def forward(self, input_data):
        output = self.fc(input_data)
        return output

# XGBoost
def evaluate_embeddings_with_xgboost(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    def xgb_evaluate(max_depth, gamma, colsample_bytree):
        params = {'eval_metric': 'logloss',
                  'max_depth': int(max_depth),
                  'subsample': 0.8,
                  'eta': 0.1,
                  'gamma': gamma,
                  'colsample_bytree': colsample_bytree}
        cv_result = cross_val_score(xgb.XGBClassifier(**params), X_train, Y_train, cv=5, scoring='accuracy')
        return np.mean(cv_result)

    xgb_bo = BayesianOptimization(xgb_evaluate, {'max_depth': (3, 10), 'gamma': (0, 1), 'colsample_bytree': (0.3, 0.9)})
    xgb_bo.maximize(init_points=3, n_iter=5)

    params = xgb_bo.max['params']
    optimal_params = {'max_depth': int(params['max_depth']), 'gamma': params['gamma'], 'colsample_bytree': params['colsample_bytree'], 'eval_metric': 'logloss'}
    model = xgb.XGBClassifier(**optimal_params)
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred, average='macro')
    recall = recall_score(Y_test, Y_pred, average='macro')
    f1 = f1_score(Y_test, Y_pred, average='macro')

    return accuracy, precision, recall, f1

# Random Forest
def evaluate_embeddings_with_random_forest(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    def RF_evaluate(n_estimators, max_depth, min_samples_split):
        params = {'n_estimators': int(n_estimators),
                  'max_depth': int(max_depth),
                  'min_samples_split': int(min_samples_split)}
        cv_result = cross_val_score(RandomForestClassifier(**params), X_train, Y_train, cv=5, scoring='accuracy')
        return np.mean(cv_result)

    RF_bo = BayesianOptimization(RF_evaluate, {'n_estimators': (10, 100), 'max_depth': (3, 10), 'min_samples_split': (2, 10)})
    RF_bo.maximize(init_points=3, n_iter=5)

    params = RF_bo.max['params']
    optimal_params = {'n_estimators': int(params['n_estimators']), 'max_depth': int(params['max_depth']), 'min_samples_split': int(params['min_samples_split'])}
    RF_model = RandomForestClassifier(**optimal_params)
    RF_model.fit(X_train, Y_train)

    Y_pred = RF_model.predict(X_test)

    accuracy = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred, average='macro')
    recall = recall_score(Y_test, Y_pred, average='macro')
    f1 = f1_score(Y_test, Y_pred, average='macro')

    return accuracy, precision, recall, f1

def grid_search_and_evaluate(data):
    evaluation_results_xgb = []
    evaluation_results_rf = []
    num_epochs_options = [1000, 2000, 3000]
    output_dim_options = [10, 50, 100]
    margin_options = [1.0, 3.0, 5.0]
    batch_size_options = [16, 32, 64]

    best_performance_xgb = None
    best_performance_rf = None
    best_params_xgb = {}
    best_params_rf = {}

    X, Y = data.iloc[:, :-1].values, data.iloc[:, -1].values
    Y_encoded, _ = pd.factorize(Y)

    for num_epochs in num_epochs_options:
        for output_dim in output_dim_options:
            for margin in margin_options:
                for batch_size in batch_size_options:
                    print('Num Epochs:', num_epochs, 'Output Dim:', output_dim, 'Margin:', margin, 'Batch Size:', batch_size)
                    model = SiameseNetwork(input_dim=X.shape[1], hidden_dim=64, output_dim=output_dim)
                    optimizer = optim.SGD(model.parameters(), lr=0.01)
                    loss_fn = nn.TripletMarginLoss(margin=margin)
                    dataset = TripletDataset(X, Y_encoded)
                    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
                    # Train CL
                    for epoch in range(num_epochs):
                        model.train()
                        total_loss = 0
                        for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
                            optimizer.zero_grad()
                            anchor_output = model(anchor)
                            positive_output = model(positive)
                            negative_output = model(negative)
                            loss = loss_fn(anchor_output, positive_output, negative_output)
                            total_loss += loss.item()
                            loss.backward()
                            optimizer.step()
                            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader)}")
                    # Embedding
                    embedding_dataset = SimpleDataset(X, Y)
                    embedding_loader = DataLoader(embedding_dataset, batch_size=batch_size, shuffle=False)
                    model.eval()
                    embeddings = []
                    with torch.no_grad():
                        for samples, _ in embedding_loader:
                            embeddings_batch = model(samples)
                            embeddings.append(embeddings_batch.cpu().numpy())

                    embeddings = np.vstack(embeddings)

                    accuracy_xgb, _, precision_xgb, recall_xgb, f1_xgb = evaluate_embeddings_with_xgboost(embeddings, Y_encoded)
                    accuracy_rf, _, precision_rf, recall_rf, f1_rf = evaluate_embeddings_with_random_forest(embeddings, Y_encoded)
                    evaluation_results_xgb.append({
                        'params': {
                            'Num Epochs': num_epochs,
                            'Output Dim': output_dim,
                            'Margin': margin,
                            'Batch Size': batch_size
                        },
                        'performance': {
                            'Accuracy': accuracy_xgb,
                            'Precision': precision_xgb,
                            'Recall': recall_xgb,
                            'F1': f1_xgb
                        }
                    })
                    evaluation_results_rf.append({
                        'params': {
                            'Num Epochs': num_epochs,
                            'Output Dim': output_dim,
                            'Margin': margin,
                            'Batch Size': batch_size
                        },
                        'performance': {
                            'Accuracy': accuracy_rf,
                            'Precision': precision_rf,
                            'Recall': recall_rf,
                            'F1': f1_rf
                       }
                    })

                    if best_performance_xgb is None or accuracy_xgb > best_performance_xgb['Accuracy']:
                        best_performance_xgb = {'Accuracy': accuracy_xgb, 'Precision': precision_xgb, 'Recall': recall_xgb, 'F1': f1_xgb}
                        best_params_xgb = {'Num Epochs': num_epochs, 'Output Dim': output_dim, 'Margin': margin, 'Batch Size': batch_size}

                    if best_performance_rf is None or accuracy_rf > best_performance_rf['Accuracy']:
                        best_performance_rf = {'Accuracy': accuracy_rf, 'Precision': precision_rf, 'Recall': recall_rf, 'F1': f1_rf}
                        best_params_rf = {'Num Epochs': num_epochs, 'Output Dim': output_dim, 'Margin': margin, 'Batch Size': batch_size}

    print("Best XGBoost Performance:", best_performance_xgb)
    print("Best XGBoost Parameters:", best_params_xgb)
    print("Best RandomForest Performance:", best_performance_rf)
    print("Best RandomForest Parameters:", best_params_rf)


    return evaluation_results_xgb, evaluation_results_rf

# Load Dataset
datasets = {
    'Spam': pd.read_csv('../Ablation/without CL/Spam_Embedding_by_Model_without_CL.csv'),
    'News': pd.read_csv('../Ablation/without CL/News_Embedding_by_Model_without_CL.csv'),
    'Malicious': pd.read_csv('../Ablation/without CL/Malicious_Phish_Embedding_by_Model_without_CL.csv'),
    'Classification': pd.read_csv('../Ablation/without CL/Classification_Embedding_by_Model_without_CL.csv'),
    'App': pd.read_csv('../Ablation/without CL/App_Embedding_by_Model_without_CL.csv'),

}

for name, dataset in datasets.items():
    print(f"Evaluating {name} Dataset")
    evaluation_results_xgb, evaluation_results_rf = grid_search_and_evaluate(dataset)
    # Save File
    flattened_results_xgb = []
    for res in evaluation_results_xgb:
        flattened_result = {**res['params'], **res['performance']}
        flattened_results_xgb.append(flattened_result)

    flattened_results_rf = []
    for res in evaluation_results_rf:
        flattened_result = {**res['params'], **res['performance']}
        flattened_results_rf.append(flattened_result)
    df_results_xgb = pd.DataFrame(flattened_results_xgb)
    df_results_rf = pd.DataFrame(flattened_results_rf)

    df_results_xgb.to_csv(f'{name}_evaluation_results_xgb.csv', index=False)
    df_results_rf.to_csv(f'{name}evaluation_results_rf.csv', index=False)