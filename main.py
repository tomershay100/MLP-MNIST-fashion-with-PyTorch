# Tomer Shay, Roei Gida
import argparse
from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import TensorDataset
import torchvision


class BaseModel(nn.Module):
    def __init__(self, image_size, lr):
        super(BaseModel, self).__init__()

        self.name = 'Base Model'
        self.lr = lr
        self.image_size = image_size

        self.train_accuracies = []
        self.train_loss = []
        self.validate_accuracies = []
        self.validate_loss = []
        self.test_accuracies = []
        self.test_loss = []


class ModelA(BaseModel):
    def __init__(self, image_size, lr):
        super(ModelA, self).__init__(image_size, lr)

        self.name = 'Model A'

        self.fc0 = nn.Linear(image_size, 100)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)

        self.optimizer = optim.SGD(self.parameters(), lr=self.lr)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class ModelB(ModelA):
    def __init__(self, image_size, lr):
        super(ModelB, self).__init__(image_size, lr)

        self.name = 'Model B'
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)


class ModelC(ModelB):
    def __init__(self, image_size, lr, dropout=0.2):
        super(ModelC, self).__init__(image_size, lr)

        self.name = 'Model C'

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class ModelD(ModelB):
    def __init__(self, image_size, lr):
        super(ModelD, self).__init__(image_size, lr)

        self.name = 'Model D'

        self.batch_norm_1 = nn.BatchNorm1d(100)
        self.batch_norm_2 = nn.BatchNorm1d(50)

    def forward(self, x):
        x = x.view(-1, self.image_size)

        x = self.fc0(x)
        x = self.batch_norm_1(x)
        x = F.relu(x)

        x = self.fc1(x)
        x = self.batch_norm_2(x)
        x = F.relu(x)

        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class ModelE(BaseModel):
    def __init__(self, image_size, lr):
        super(ModelE, self).__init__(image_size, lr)

        self.name = 'Model E'

        self.fc0 = nn.Linear(image_size, 128)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 10)
        self.fc3 = nn.Linear(10, 10)
        self.fc4 = nn.Linear(10, 10)

        self.optimizer = optim.SGD(self.parameters(), lr=self.lr)
        self.activation_func = F.relu

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = self.activation_func(self.fc0(x))
        x = self.activation_func(self.fc1(x))
        x = self.activation_func(self.fc2(x))
        x = self.activation_func(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


class ModelF(ModelE):
    def __init__(self, image_size, lr):
        super(ModelF, self).__init__(image_size, lr)

        self.name = 'Model F'

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.activation_func = torch.sigmoid


class BestModel(BaseModel):
    def __init__(self, image_size, lr):
        super(BestModel, self).__init__(image_size, lr=lr)

        self.name = 'Best Model'

        self.batch_norm_0 = nn.BatchNorm1d(image_size)
        self.batch_norm_1 = nn.BatchNorm1d(512)
        self.batch_norm_2 = nn.BatchNorm1d(256)
        self.batch_norm_3 = nn.BatchNorm1d(128)
        self.batch_norm_4 = nn.BatchNorm1d(64)

        self.fc0 = nn.Linear(image_size, 512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

        self.dropout = nn.Dropout(p=0.1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = x.view(-1, self.image_size)
        x = self.dropout(x)

        x = F.leaky_relu(self.batch_norm_1(self.fc0(x)))

        x = F.leaky_relu(self.batch_norm_2(self.fc1(x)))
        x = self.dropout(x)

        x = F.leaky_relu(self.batch_norm_3(self.fc2(x)))

        x = F.leaky_relu(self.batch_norm_4(self.fc3(x)))
        x = self.dropout(x)

        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


def export_plot(model):
    plt.subplot(2, 1, 1)
    plt.title(f'{model.name}')
    plt.plot(model.train_accuracies, label="train")
    plt.plot(model.validate_accuracies, label="validate")
    plt.plot(model.test_accuracies, label="test")
    plt.ylabel("Accuracy")

    plt.subplot(2, 1, 2)
    plt.plot(model.train_loss, label="train")
    plt.plot(model.validate_loss, label="validate")
    plt.plot(model.test_loss, label="test")
    plt.ylabel('Loss')

    plt.xlabel('Epochs')

    plt.legend()
    plt.savefig(f'{model.name}.png')
    print(f'model saved to \'{model.name}.png\'')


def train(model, train_set):
    model.train()
    train_loss = 0
    correct = 0
    for _, (x, y) in enumerate(train_set):
        model.optimizer.zero_grad()
        output = model(x)
        loss = F.nll_loss(output, y.T[0])
        loss.backward()
        model.optimizer.step()
        train_loss += float(loss.data)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(y.view_as(pred)).sum().item()

    model.train_accuracies.append(100 * correct / len(train_set.dataset))
    model.train_loss.append(train_loss / (len(train_set.dataset) / train_set.batch_size))


def validate(model, validate_set, is_test=False):
    model.eval()
    validate_loss = 0
    correct = 0
    with torch.no_grad():
        for _, (x, y) in enumerate(validate_set):
            output = model(x)
            loss = F.nll_loss(output, y.T[0])
            validate_loss += float(loss.data)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()

    if not is_test:
        model.validate_accuracies.append(100 * correct / len(validate_set.dataset))
        model.validate_loss.append(validate_loss / (len(validate_set.dataset) / validate_set.batch_size))
    else:
        model.test_accuracies.append(100 * correct / len(validate_set.dataset))
        model.test_loss.append(validate_loss / (len(validate_set.dataset) / validate_set.batch_size))


def running_epochs(model, epochs, is_best):
    global train_loader, validate_loader, test_loader

    best_acc_model = deepcopy(model)
    best_val_acc = 0

    for i in range(epochs):
        if is_best:
            if i == 14:
                model.optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.975)
        if i % 7 == 0:
            model.optimizer.param_groups[0]['lr'] *= 0.2

        print(f'====== {model.name} EPOCH #{i + 1} ============')

        train(model, train_loader)
        print(f'[train accuracy:]\t\t\t{"{:.2f}".format(model.train_accuracies[-1])}%')
        print(f'[train loss:]\t\t\t\t{"{:.2f}".format(model.train_loss[-1])}')

        validate(model, validate_loader)
        print(f'[validate accuracy:]\t\t{"{:.2f}".format(model.validate_accuracies[-1])}%')
        print(f'[validate loss:]\t\t\t{"{:.2f}".format(model.validate_loss[-1])}')

        if best_val_acc < model.validate_accuracies[-1]:
            best_val_acc = model.validate_accuracies[-1]
            best_acc_model = deepcopy(model)
            if best_val_acc > 90:
                model.optimizer.param_groups[0]['lr'] = 0.0001
            print('---- model saved! ----')

        validate(model, test_loader, is_test=True)
        print(f'[test accuracy:]\t\t\t{"{:.2f}".format(model.test_accuracies[-1])}%')
        print(f'[test loss:]\t\t\t\t{"{:.2f}".format(model.test_loss[-1])}')

    return best_acc_model


def load_original_mnist_fashion(batch_size, validate_percentage):
    print("loading files..")
    transforms = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])

    dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transforms, download=True)
    train_set, val_set = torch.utils.data.random_split(dataset, [len(dataset.data) * (1 - validate_percentage / 100),
                                                                 len(dataset.data) * (validate_percentage / 100)])

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    validate_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.FashionMNIST('./data', download=True, train=False, transform=transforms),
        batch_size=batch_size)

    return train_loader, validate_loader, test_loader


def load_local_mnist_fashion(train_x_file, train_y_file, test_x_file, test_y_file, batch_size, validate_percentage):
    # get data from the files
    print("loading files..")
    train_x = np.loadtxt(train_x_file)
    train_y = np.array([np.loadtxt(train_y_file)]).T
    test_x = np.loadtxt(test_x_file)
    test_y = np.array([np.loadtxt(test_y_file)]).T

    train_x /= 255  # normalize train pixels to 0 - 1
    test_x /= 255  # normalize test pixels to 0 - 1

    # shuffle train data set
    # print("shuffle training set..")
    rand = np.arange(len(train_x))
    np.random.shuffle(rand)
    train_x = train_x[rand]
    train_y = train_y[rand]

    # print("separate into validate..")
    validate_x = train_x[:(len(train_x) * validate_percentage) // 100]
    validate_y = train_y[:(len(train_y) * validate_percentage) // 100]
    train_x = train_x[(len(train_x) * validate_percentage) // 100:]
    train_y = train_y[(len(train_y) * validate_percentage) // 100:]

    # from numpy array to torch tensor
    train_x = torch.from_numpy(train_x).float()
    train_y = torch.from_numpy(train_y).long()

    validate_x = torch.from_numpy(validate_x).float()
    validate_y = torch.from_numpy(validate_y).long()

    test_x = torch.from_numpy(test_x).float()
    test_y = torch.from_numpy(test_y).long()

    train_xy = TensorDataset(train_x, train_y)
    validate_xy = TensorDataset(validate_x, validate_y)
    test_xy = TensorDataset(test_x, test_y)

    train_loader = torch.utils.data.DataLoader(train_xy, batch_size=batch_size, shuffle=True)
    validate_loader = torch.utils.data.DataLoader(validate_xy, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_xy, batch_size=batch_size)

    return train_loader, validate_loader, test_loader


# get arguments
parser = argparse.ArgumentParser()

# -train_x train_x -train_y train_y -test_x test_x -test_y test_y -e epochs...
parser.add_argument("-train_x", dest="train_x_path", default="train_x", help="train_x file path")
parser.add_argument("-train_y", dest="train_y_path", default="train_y", help="train_y file path")
parser.add_argument("-test_x", dest="test_x_path", default="test_x", help="test_x file path")
parser.add_argument("-test_y", dest="test_y_path", default="test_y", help="test_y file path")
parser.add_argument("-e", dest="epochs", default="10", help="Epochs")
parser.add_argument("-batch_size", dest="batch_size", default="64", help="Batch Size")
parser.add_argument("-validate", dest="validate_percentage", default="10", help="Validate Percentage")
parser.add_argument("-model", dest="model", default="BestModel",
                    help="The Model to run (between A to F or \"BestModel\"")
parser.add_argument("-local", dest="is_local", default="False",
                    help="True for using local train and test file, False for using the original MNIST-fashion dataset")
parser.add_argument("-plot", dest="to_export", default="True",
                    help="False to don't export a graph of accuracy and loss values.")

args = parser.parse_args()

if bool(args.is_local):
    train_loader, validate_loader, test_loader = load_local_mnist_fashion(args.train_x_path, args.train_y_path,
                                                                          args.test_x_path, args.test_y_path,
                                                                          int(args.batch_size),
                                                                          int(args.validate_percentage))
else:
    train_loader, validate_loader, test_loader = load_original_mnist_fashion(int(args.batch_size),
                                                                             int(args.validate_percentage))
is_best = False
if args.model == 'A':
    model = ModelA(image_size=28 * 28, lr=0.12)
elif args.model == 'B':
    model = ModelB(image_size=28 * 28, lr=0.0001)
elif args.model == 'C':
    model = ModelC(image_size=28 * 28, lr=0.0001)
elif args.model == 'D':
    model = ModelD(image_size=28 * 28, lr=0.01)
elif args.model == 'E':
    model = ModelE(image_size=28 * 28, lr=0.1)
elif args.model == 'F':
    model = ModelF(image_size=28 * 28, lr=0.001)
else:
    model = BestModel(image_size=28 * 28, lr=0.001)
    is_best = True

best_model = running_epochs(model, int(args.epochs), is_best=is_best)
print("========================================")
print("learn finished.", end=' ')

if bool(args.to_export):
    print("exporting plot..")
    export_plot(best_model)
else:
    print()

print('\nfinal accuracy:')
validate(best_model, test_loader, is_test=True)
print(f'[test accuracy:]\t\t\t{"{:.2f}".format(best_model.test_accuracies[-1])}%')
print(f'[test loss:]\t\t\t\t{"{:.2f}".format(best_model.test_loss[-1])}')
