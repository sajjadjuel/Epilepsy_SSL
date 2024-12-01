import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from argparse import ArgumentParser
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Dataset
import os
import scipy.io
import warnings
from torch.utils.tensorboard import SummaryWriter
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_recall_curve, roc_curve

from model import TaskOriNet  # Assuming your model is in this file
from density import GaussianDensityTorch
from utils import plot_roc, plot_pr  # Assuming these are in your utils

warnings.filterwarnings("ignore")

# Collate function to handle potential None values during data loading
def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.default_collate(batch)

# DataLoader class
class MyData(Dataset):
    def __init__(self, txt_file, datatype="train", transform=None):
        self.txt_file = txt_file
        self.transform = transform
        self.files = []
        self.datatype = datatype

        with open(self.txt_file, 'r') as f:
            files = f.readlines()
            for file in files:
                file = file.split('\n')[0]
                self.files.append(file)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_path = self.files[index]
        try:
            file = scipy.io.loadmat(file_path)
            label = np.array(1) if "abnormal" in file_path else np.array(0)
            transformed_data = np.expand_dims(file['data'], 0)
            return transformed_data, label
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None  # Skip invalid files

# Data loading function
def load_data(batch_size=16):
    splits = ['train', 'validate', 'test']
    drop_last_batch = {'train': False, 'validate': False, 'test': False}
    shuffle = {'train': True, 'validate': True, 'test': False}

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((8.68e-10,), (2.09e-07,))
    ])

    dataset = {}

    dataset['train'] = MyData(txt_file="./data/all_TXT_3s/train.txt", datatype="train", transform=transform_train)
    dataset['train'], dataset['validate'] = torch.utils.data.random_split(dataset['train'],
        [int(0.75 * len(dataset['train']) + 0.5), int(0.25 * len(dataset['train']) + 0.5)])
    
    dataset['test'] = MyData(txt_file="./data/all_TXT_3s/test.txt", datatype="test", transform=transform_train)

    print('Loaded train set: {} eegs'.format(len(dataset['train'])))
    print('Loaded val set: {} eegs'.format(len(dataset['validate'])))
    print('Loaded test set: {} eegs'.format(len(dataset['test'])))

    dataloader = {x: DataLoader(dataset=dataset[x],
                                 batch_size=batch_size,
                                 shuffle=shuffle[x],
                                 num_workers=0,
                                 drop_last=drop_last_batch[x],
                                 pin_memory=True,
                                 collate_fn=collate_fn)
                  for x in splits}

    return dataloader

# Train function
def train(data_loader, epochs, NUM_CLASSES, length, inplane, learning_rate,
          optim_name, model_dir, device, density=GaussianDensityTorch()):

    model = TaskOriNet(num_classes=NUM_CLASSES).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = CosineAnnealingWarmRestarts(optimizer, epochs)

    writer_path = f'./AD_runs/{model_dir}'
    print('TensorBoard runs:', writer_path)
    writer = SummaryWriter(log_dir=writer_path, comment=model_dir)

    best_auc = 0
    for epoch in range(epochs):
        losses = []
        correct_num = 0
        model.train()

        for (x, _) in data_loader['train']:
            x, label = x.to(device), label.to(device)
            x_, label = random_scale(x, num_classes=NUM_CLASSES, length=length)
            embed, out = model(x_)
            output = F.softmax(out, dim=-1)
            _, index = torch.max(output.cpu(), 1)
            correct_num += torch.sum(index == label.cpu()).item()
            loss = F.cross_entropy(out + 1e-8, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if scheduler is not None:
            scheduler.step()

        acc_train = correct_num / len(data_loader['train'].dataset)

        print(f'====> Epoch: {epoch} Average loss: {np.mean(losses):.4f}')
        print(f'====> Epoch: {epoch} Train Accuracy: {acc_train:.4f}')
        writer.add_scalar('Loss', np.mean(losses), global_step=epoch)
        writer.add_scalar('Train Accuracy', acc_train, global_step=epoch)

        if epoch != 0 and epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                train_embed = []
                for (x, _) in data_loader['train']:
                    x = x.to(device, dtype=torch.float32)
                    embed, _ = model(x)
                    train_embed.append(embed.cpu())

                train_embed = torch.cat(train_embed)
                train_embed = torch.nn.functional.normalize(train_embed, p=2, dim=1)
                density.fit(train_embed)

                y_true = []
                embeds = []
                for (x, label) in data_loader['validate']:
                    x = x.to(device, dtype=torch.float32)
                    embed, _ = model(x)
                    embeds.append(embed.cpu())
                    y_true.append(label)

                for (x, label) in data_loader['test']:
                    x = x.to(device, dtype=torch.float32)
                    embed, _ = model(x)
                    embeds.append(embed.cpu())
                    y_true.append(label)

                y_true = np.concatenate(y_true)
                embeds = torch.cat(embeds)
                embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)
                distances = density.predict(embeds)

                roc_auc = plot_roc(y_true, distances)
                print(f'====> Epoch: {epoch} AUC: {roc_auc:.4f}')
                if roc_auc > best_auc:
                    best_auc = roc_auc
                    torch.save(model.state_dict(), f'./AD_models/{model_dir}/best.pth')

        if epoch == epochs - 1:
            model.eval()
            torch.save(model.state_dict(), f'./AD_models/{model_dir}/epochs{epochs}.pth')

    writer.close()

# Evaluation function
def test_ocsvm(data_loader, NUM_CLASSES, inplane, epochs, model_dir, device, save_roc_path, save_pr_path, save_plots):
    model = TaskOriNet(num_classes=NUM_CLASSES).to(device)
    model_name = f'./AD_models/{model_dir}/epochs{epochs}.pth'
    model.load_state_dict(torch.load(model_name))
    model.eval()

    with torch.no_grad():
        train_embed = []
        for (x, _) in data_loader['train']:
            x = x.to(device, dtype=torch.float32)
            embed, _ = model(x)
            train_embed.append(embed.cpu())

        train_embed = torch.cat(train_embed)
        train_embed = torch.nn.functional.normalize(train_embed, p=2, dim=1)

        gamma = 10. / (torch.var(train_embed) * train_embed.shape[1])
        clf = OneClassSVM(kernel='rbf', gamma=gamma).fit(train_embed)

        y_true = []
        embeds = []
        for (x, label) in data_loader['validate']:
            x = x.to(device, dtype=torch.float32)
            embed, _ = model(x)
            embeds.append(embed.cpu())
            y_true.append(label)

        for (x, label) in data_loader['test']:
            x = x.to(device, dtype=torch.float32)
            embed, _ = model(x)
            embeds.append(embed.cpu())
            y_true.append(label)

        y_true = np.concatenate(y_true)
        embeds = torch.cat(embeds)
        embeds = torch.nn.functional.normalize(embeds, p=2, dim=1)

        scores = -clf.score_samples(embeds)

        roc_auc = plot_roc(y_true, scores)
        pr_auc = plot_pr(y_true, scores)

        fpr, tpr, threshold = roc_curve(y_true, scores)
        fnr = 1 - tpr
        eer_threshold = threshold[np.nanargmin(np.abs((fnr - fpr)))]
        EER = fpr[np.nanargmin(np.abs((fnr - fpr)))]

        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        f1_scores = 2 * (precision * recall) / (precision + recall)
        f1_score = f1_scores[np.where(thresholds == eer_threshold)[0][0]]

        print(f"roc auc = {roc_auc:.3f}")
        print(f'f1 score = {f1_score:.3f}')
        print(f'EER = {EER:.3f}')

        return roc_auc, f1_score, EER

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs for training')
    parser.add_argument('--eval', dest='eval', action='store_true', help='Evaluation mode')
    parser.add_argument('--cuda', dest='cuda', type=int, default=1, help='CUDA device')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--num_classes', default=3, type=int, help='Number of classes')
    parser.add_argument('--length', default=769, type=int, help='Length of the input')
    parser.add_argument('--inplane', default=18, type=int, help='Inplane')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='Learning rate')
    parser.add_argument('--optim', default="adam", help='Optimization algorithm (sgd, adam)')
    parser.add_argument('--save_plots', default=True, type=bool, help='Save plots')
    options = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    data_loader = load_data(options.batch_size)

    if options.eval:
        print("Evaluation mode selected.")
        test_ocsvm(data_loader, options.num_classes, options.inplane, options.epochs, '2020', device, None, None, False)
    else:
        print("Training mode selected.")
        train(data_loader, options.epochs, options.num_classes, options.length, options.inplane,
              options.learning_rate, options.optim, '2020', device)
