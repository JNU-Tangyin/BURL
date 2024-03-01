import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random

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

# Spam
input_dim = 100
hidden_dim = 64
output_dim = 100

data = pd.read_csv('../without CL/Spam_Embedding_by_Model_without_CL.csv')
X, Y = data.iloc[:, :-1].values, data.iloc[:, -1].values
means = np.mean(X, axis=0)
stds = np.std(X, axis=0)
url_data = (X - means) / stds
n_classes = len(np.unique(Y))

labels_encoded, unique = pd.factorize(np.unique(Y))
label_mapping = dict(zip(unique, labels_encoded))
Y = np.vectorize(label_mapping.get)(Y)

triplet_dataset = TripletDataset(X, Y)
triplet_loader = DataLoader(triplet_dataset, batch_size=16, shuffle=True)

siamese_net = SiameseNetwork(input_dim,hidden_dim, output_dim)
optimizer = optim.SGD(siamese_net.parameters(), lr=0.01)

margin = 5.0
triplet_loss = nn.TripletMarginLoss(margin=margin)

num_epochs = 1500
for epoch in range(num_epochs):
    for batch_idx, (anchor, positive, negative) in enumerate(triplet_loader):
        optimizer.zero_grad()

        anchor_output = siamese_net(anchor)
        positive_output = siamese_net(positive)
        negative_output = siamese_net(negative)

        loss = triplet_loss(anchor_output, positive_output, negative_output)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(triplet_loader)}], Loss: {loss.item()}')

simple_dataset = SimpleDataset(X, Y)
simple_loader = DataLoader(simple_dataset, batch_size=32, shuffle=False)

with torch.no_grad():
    features = []
    labels = []

    for input_data, batch_labels in simple_loader:
            output = siamese_net(input_data)
            features.extend(output.numpy())
            labels.extend(batch_labels.numpy())

result = np.array(features)
Spam_Embedding =np.hstack((result, Y.reshape([-1,1])))
pd.DataFrame(Spam_Embedding).to_csv('Spam_Embedding_by_FullModel.csv', index=False)


# News
input_dim = 100
hidden_dim = 64
output_dim = 100

data = pd.read_csv('../without CL//News_Embedding_by_Model_without_CL.csv')
X, Y = data.iloc[:, :-1].values, data.iloc[:, -1].values
means = np.mean(X, axis=0)
stds = np.std(X, axis=0)

url_data = (X - means) / stds
n_classes = len(np.unique(Y))
labels_encoded, unique = pd.factorize(np.unique(Y))
label_mapping = dict(zip(unique, labels_encoded))
Y = np.vectorize(label_mapping.get)(Y)

triplet_dataset = TripletDataset(X, Y)
triplet_loader = DataLoader(triplet_dataset, batch_size=16, shuffle=True)

siamese_net = SiameseNetwork(input_dim,hidden_dim, output_dim)
optimizer = optim.SGD(siamese_net.parameters(), lr=0.01)

margin = 5.0
triplet_loss = nn.TripletMarginLoss(margin=margin)

num_epochs = 1500
for epoch in range(num_epochs):
    for batch_idx, (anchor, positive, negative) in enumerate(triplet_loader):
        optimizer.zero_grad()

        anchor_output = siamese_net(anchor)
        positive_output = siamese_net(positive)
        negative_output = siamese_net(negative)

        loss = triplet_loss(anchor_output, positive_output, negative_output)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(triplet_loader)}], Loss: {loss.item()}')

simple_dataset = SimpleDataset(X, Y)
simple_loader = DataLoader(simple_dataset, batch_size=32, shuffle=False)

with torch.no_grad():
    features = []
    labels = []

    for input_data, batch_labels in simple_loader:
            output = siamese_net(input_data)
            features.extend(output.numpy())
            labels.extend(batch_labels.numpy())

result = np.array(features)
News_Embedding =np.hstack((result, Y.reshape([-1,1])))
pd.DataFrame(News_Embedding).to_csv('News_Embedding_by_FullModel.csv', index=False)


# Malicious
input_dim = 100
hidden_dim = 64
output_dim = 100

data = pd.read_csv('../without CL/Malicious_Phish_Embedding_by_Model_without_CL.csv')
X, Y = data.iloc[:, :-1].values, data.iloc[:, -1].values
means = np.mean(X, axis=0)
stds = np.std(X, axis=0)

url_data = (X - means) / stds

n_classes = len(np.unique(Y))

labels_encoded, unique = pd.factorize(np.unique(Y))
label_mapping = dict(zip(unique, labels_encoded))
Y = np.vectorize(label_mapping.get)(Y)

triplet_dataset = TripletDataset(X, Y)
triplet_loader = DataLoader(triplet_dataset, batch_size=16, shuffle=True)

siamese_net = SiameseNetwork(input_dim,hidden_dim, output_dim)
optimizer = optim.SGD(siamese_net.parameters(), lr=0.01)

margin = 5.0
triplet_loss = nn.TripletMarginLoss(margin=margin)

num_epochs = 2000
for epoch in range(num_epochs):
    for batch_idx, (anchor, positive, negative) in enumerate(triplet_loader):
        optimizer.zero_grad()

        anchor_output = siamese_net(anchor)
        positive_output = siamese_net(positive)
        negative_output = siamese_net(negative)

        loss = triplet_loss(anchor_output, positive_output, negative_output)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(triplet_loader)}], Loss: {loss.item()}')

simple_dataset = SimpleDataset(X, Y)
simple_loader = DataLoader(simple_dataset, batch_size=32, shuffle=False)

with torch.no_grad():
    features = []
    labels = []

    for input_data, batch_labels in simple_loader:
            output = siamese_net(input_data)
            features.extend(output.numpy())
            labels.extend(batch_labels.numpy())

result = np.array(features)
Malicious_Embedding =np.hstack((result, Y.reshape([-1,1])))
pd.DataFrame(Malicious_Embedding).to_csv('Malicious_Phish_Embedding_by_FullModel.csv', index=False)



# Classification
input_dim = 100
hidden_dim = 64
output_dim = 100

data = pd.read_csv('../without CL/Classification_Embedding_by_Model_without_CL.csv')
X, Y = data.iloc[:, :-1].values, data.iloc[:, -1].values
means = np.mean(X, axis=0)
stds = np.std(X, axis=0)
url_data = (X - means) / stds
n_classes = len(np.unique(Y))

labels_encoded, unique = pd.factorize(np.unique(Y))
label_mapping = dict(zip(unique, labels_encoded))
Y = np.vectorize(label_mapping.get)(Y)

triplet_dataset = TripletDataset(X, Y)
triplet_loader = DataLoader(triplet_dataset, batch_size=16, shuffle=True)

siamese_net = SiameseNetwork(input_dim,hidden_dim, output_dim)
optimizer = optim.SGD(siamese_net.parameters(), lr=0.01)

margin = 5.0
triplet_loss = nn.TripletMarginLoss(margin=margin)

num_epochs = 3000
for epoch in range(num_epochs):
    for batch_idx, (anchor, positive, negative) in enumerate(triplet_loader):
        optimizer.zero_grad()

        anchor_output = siamese_net(anchor)
        positive_output = siamese_net(positive)
        negative_output = siamese_net(negative)

        loss = triplet_loss(anchor_output, positive_output, negative_output)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(triplet_loader)}], Loss: {loss.item()}')

simple_dataset = SimpleDataset(X, Y)
simple_loader = DataLoader(simple_dataset, batch_size=32, shuffle=False)

with torch.no_grad():
    features = []
    labels = []

    for input_data, batch_labels in simple_loader:
            output = siamese_net(input_data)
            features.extend(output.numpy())
            labels.extend(batch_labels.numpy())

result = np.array(features)
Classification_Embedding = np.hstack((result, Y.reshape([-1,1])))
pd.DataFrame(Classification_Embedding).to_csv('Classification_Embedding_by_FullModel.csv', index=False)



# App
input_dim = 100
hidden_dim = 64
output_dim = 100

data = pd.read_csv('../without CL/App_Embedding_by_Model_without_CL.csv')
X, Y = data.iloc[:, :-1].values, data.iloc[:, -1].values
means = np.mean(X, axis=0)
stds = np.std(X, axis=0)
url_data = (X - means) / stds
n_classes = len(np.unique(Y))

labels_encoded, unique = pd.factorize(np.unique(Y))
label_mapping = dict(zip(unique, labels_encoded))
Y = np.vectorize(label_mapping.get)(Y)

triplet_dataset = TripletDataset(X, Y)
triplet_loader = DataLoader(triplet_dataset, batch_size=16, shuffle=True)


siamese_net = SiameseNetwork(input_dim,hidden_dim, output_dim)
optimizer = optim.SGD(siamese_net.parameters(), lr=0.01)

margin = 5.0
triplet_loss = nn.TripletMarginLoss(margin=margin)

num_epochs = 1500
for epoch in range(num_epochs):
    for batch_idx, (anchor, positive, negative) in enumerate(triplet_loader):
        optimizer.zero_grad()
        anchor_output = siamese_net(anchor)
        positive_output = siamese_net(positive)
        negative_output = siamese_net(negative)
        loss = triplet_loss(anchor_output, positive_output, negative_output)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(triplet_loader)}], Loss: {loss.item()}')

simple_dataset = SimpleDataset(X, Y)
simple_loader = DataLoader(simple_dataset, batch_size=32, shuffle=False)

with torch.no_grad():
    features = []
    labels = []

    for input_data, batch_labels in simple_loader:
            output = siamese_net(input_data)
            features.extend(output.numpy())
            labels.extend(batch_labels.numpy())

result = np.array(features)
App_Embedding =np.hstack((result, Y.reshape([-1,1])))
pd.DataFrame(App_Embedding).to_csv('App_Embedding_by_FullModel.csv', index=False)