import pickle
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
from catboost import CatBoostClassifier


# Define the ANN model
class ANN(nn.Module):
    def __init__(self, input_size):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Function to get predictions for ANN
def get_predictions(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_preds, all_labels


# Function to evaluate chosen model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, precision, recall, f1

def trainModel(embeddingsFile, modelName):
    # Load embeddings and labels
    with open(embeddingsFile, 'rb') as f:
        data = pickle.load(f)


    embeddings = np.array(data['embeddings'])

    if len(embeddings.shape) == 3:
        embeddings = embeddings.reshape(embeddings.shape[0], -1)

    labels = np.array(data['labels'])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

    if modelName == "xgb":
        # Train XGBoost Classifier
        xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', tree_method='gpu_hist')
        xgb_model.fit(X_train, y_train)
        xgb_accuracy, xgb_precision, xgb_recall, xgb_f1 = evaluate_model(xgb_model, X_test, y_test)
        print(
            f'{modelName} XGBoost - Accuracy: {xgb_accuracy}, Precision: {xgb_precision}, Recall: {xgb_recall}, F1 Score: {xgb_f1}')
        xgb_model.save_model(f'{modelName}_xgb_model.json')

    elif modelName == "cat":
        # Train CatBoost Classifier
        cat_model = CatBoostClassifier(task_type="GPU", verbose=0)
        cat_model.fit(X_train, y_train)
        cat_accuracy, cat_precision, cat_recall, cat_f1 = evaluate_model(cat_model, X_test, y_test)
        print(
            f'{modelName} CatBoost - Accuracy: {cat_accuracy}, Precision: {cat_precision}, Recall: {cat_recall}, F1 Score: {cat_f1}')
        cat_model.save_model(f'{modelName}_cat_model.cbm')

    elif modelName == "ann":
        # Train ANN Model
        train_data = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).long())
        test_data = TensorDataset(torch.tensor(X_test).float(), torch.tensor(y_test).long())
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

        model = ANN(embeddings.shape[1]).to('cuda' if torch.cuda.is_available() else 'cpu')
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(100):  # number of epochs
            for inputs, labels in train_loader:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        y_pred, y_true = get_predictions(model, test_loader)
        ann_accuracy = accuracy_score(y_true, y_pred)
        ann_precision = precision_score(y_true, y_pred, average='weighted')
        ann_recall = recall_score(y_true, y_pred, average='weighted')
        ann_f1 = f1_score(y_true, y_pred, average='weighted')
        print(
            f'{modelName} ANN - Accuracy: {ann_accuracy}, Precision: {ann_precision}, Recall: {ann_recall}, F1 Score: {ann_f1}')
        torch.save(model.state_dict(), f'{modelName}_ann_model.pth')


# "xgb", "cat" or "ann" will be taken as first argument
modelName = sys.argv[1]

# path to embeddings file will be taken as second argument
embeddingsFile = sys.argv[2]

# Train the model, save and evaluate it
trainModel(embeddingsFile, modelName)