import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import tqdm

from sklearn.metrics import f1_score, balanced_accuracy_score, precision_score, recall_score

# def map_to_scores2(x):
#     if x == 1:
#         return torch.tensor([1., 0.])
#     if x == 2:
#         return torch.tensor([0., 1.])
#
#
# class TSCDataset(Dataset):
#     def __init__(self, X, y):
#         self.x = X
#         self.y = y
#
#     def __len__(self):
#         return len(self.x)
#
#     def __getitem__(self, i):
#         return torch.tensor([self.x.iloc[i][0]]).float(), map_to_scores2(float(self.y[i]))

def print_metrics(predictions, y_test):
	print("PRECISION: ", metrics.precision_score(list(map(lambda x: int(x), y_test)), predictions, average='macro'))
	print("RECALL: ",metrics.recall_score(list(map(lambda x: int(x), y_test)), predictions, average='macro'))
	print("F1: ", metrics.f1_score(list(map(lambda x: int(x), y_test)), predictions, average='macro'))

def map_to_scores(x):
    if x == 0.0:
        return torch.tensor([1., 0., 0., 0., 0.])
    if x == 1.0:
        return torch.tensor([0., 1., 0., 0., 0.])
    if x == 2.0:
        return torch.tensor([0., 0., 1., 0., 0.])
    if x == 3.0:
        return torch.tensor([0., 0., 0., 1., 0.])
    if x == 4.0:
        return torch.tensor([0., 0., 0., 0., 1.])

class TSCDatasetNN(Dataset):
    def __init__(self, X, y):
        self.x = X
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return torch.tensor([self.x[i]]).float(), map_to_scores(float(self.y[i]))

class TSCDataset(Dataset):
    def __init__(self, X, y):
        self.x = X
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return torch.tensor([self.x.iloc[i]]).float(), map_to_scores(float(self.y[i]))


def get_f1_acc_recall(pred, target):
    #print(pred)
    #print(target)
    rec = recall_score(target, pred, average='macro')
    f1 = f1_score(target, pred, average='macro')
    #print(f"RECALL SCORE: {rec}")
    #print(f"ACCURACY SCORE: {balanced_accuracy_score(target, pred)}")
    #print(f"F1 SCORE: {f1}")
    return rec, f1


def evaluate(model, dataloader_test):
    lst = []
    targets = []
    model.eval()
    with torch.no_grad():
        for X, y in dataloader_test:
            pred = model(X)
            #print(pred)
            lst.append((torch.argmax(pred, dim=1)).numpy())
            #print(lst)
            targets.append(torch.argmax(y, dim=1).numpy())
    lst = np.concatenate(lst)
    #print(lst)
    #lst = [map_to_scores(x) for x in lst[0]]
    targets = np.concatenate(targets)
    #print(lst)
    #print(targets)
    return get_f1_acc_recall(lst, targets)


def train_loop(model, epochs, scheduler, optim, loss_fn, dataloader_train, dataloader_test, n_batches):
    best_rec = 0
    best_f1 = 0
    for epoch in range(epochs):
        print("EPOCH: ", epoch)
        for i, (X, y) in tqdm.notebook.tqdm(enumerate(dataloader_train), total=n_batches):
            # print(X, y)
            pred = model(X)
            # print(pred)
            loss = loss_fn(pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            scheduler.step(loss)

            if i % 10 == 0:
                print(f"train loss: {loss.item():>5f}")

        rec, f1 = evaluate(model, dataloader_test)
        print("RECALL: ", rec)
        print("F1 SCORE: ", f1)
        if rec > best_rec: best_rec = rec
        if f1 > best_f1: best_f1 = f1
        if f1 < best_f1 and rec < best_rec:
            print("EARLY STOPPING")
            return
        model.train()

