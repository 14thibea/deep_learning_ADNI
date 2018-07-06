# Reproduction from Payan and Montana

import torch
import torch.nn as nn
import torch.nn.functional as F
from os import path
import os
import numpy as np
import pandas as pd
from copy import copy, deepcopy


class LargeAutoEncoder(nn.Module):
    """
    Sparse Autoencoder for transfer learning

    """
    def __init__(self):
        super(LargeAutoEncoder, self).__init__()
        self.downsample = nn.MaxPool3d(2, 2)
        self.encode = nn.Conv3d(1, 150, 5)
        self.decode = nn.ConvTranspose3d(150, 1, 5)

    def forward(self, x):
        d = self.downsample(x)
        h = F.relu(self.encode(d))
        out = F.relu(self.decode(h))
        return out, h, d


class LargeConvolutionalNetwork(nn.Module):
    """
    Classifier for binary classification task
    """
    def __init__(self, n_classes=2):
        super(LargeConvolutionalNetwork, self).__init__()
        self.downsample = nn.MaxPool3d(2, 2)
        self.encode = nn.Conv3d(1, 150, 5)
        self.pool = nn.MaxPool3d(5, 5)
        self.fc1 = nn.Linear(150 * 11 * 13 * 11, 800)
        self.fc2 = nn.Linear(800, n_classes)

    def forward(self, x):
        d = self.downsample(x)
        h = F.relu(self.encode(d))
        h = self.pool(h)
        h = h.view(-1, 150 * 11 * 13 * 11)
        h = F.relu(self.fc1(h))
        out = self.fc2(h)
        return out


class AdaptativeAutoEncoder(nn.Module):
    """
    Sparse Autoencoder for transfer learning

    """
    def __init__(self, n_filters):
        super(AdaptativeAutoEncoder, self).__init__()
        self.downsample = nn.MaxPool3d(2, 2)
        self.encode = nn.Conv3d(1, n_filters, 5)
        self.decode = nn.ConvTranspose3d(n_filters, 1, 5)

    def forward(self, x):
        d = self.downsample(x)
        h = F.relu(self.encode(d))
        out = F.relu(self.decode(h))
        return out, h, d


class AdaptativeConvolutionalNetwork(nn.Module):
    """
    Classifier for binary classification task
    """
    def __init__(self, n_filters, dropout=0, n_classes=2):
        super(AdaptativeConvolutionalNetwork, self).__init__()
        self.downsample = nn.MaxPool3d(2, 2)
        self.encode = nn.Conv3d(1, n_filters, 5)
        self.pool = nn.MaxPool3d(5, 5)
        self.fc1 = nn.Linear(n_filters * 11 * 13 * 11, 800)
        self.fc2 = nn.Linear(800, n_classes)
        self.n_filters = n_filters
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, train=False):
        d = self.downsample(x)
        h = F.relu(self.encode(d))
        h = self.pool(h)
        h = h.view(-1, self.n_filters * 11 * 13 * 11)
        if train:
            h = self.dropout(h)
        h = F.relu(self.fc1(h))
        out = self.fc2(h)
        return out


def l1_penalty(var):
    return torch.abs(var).sum()


def test_autoencoder(model, dataloader, criterion=nn.MSELoss(), gpu=False):

    total_loss = 0

    with torch.no_grad():
        for sample in dataloader:
            if gpu:
                images, diagnoses = sample['image'].cuda(), sample['diagnosis'].cuda()
            else:
                images, diagnoses = sample['image'], sample['diagnosis']
            outputs, hidden_layer, downsample = model(images)
            loss = criterion(outputs, downsample)
            total_loss += loss

    print('Loss of the model: ' + str(total_loss))

    return total_loss


def save_results(best_params, validloader, test_method, results_path, name, denomination='Accuracy', testloader=None,
                 gpu=False):

    if testloader is not None:
        len_test = len(testloader)
        acc_test = test_method(best_params['best_model'], testloader, gpu=gpu)
    else:
        len_test = 0
        acc_test = 0

    acc_train = test_method(best_params['best_model'], trainloader, gpu=gpu)

    output_name = 'best_' + name + '.txt'
    text_file = open(path.join(results_path, output_name), 'w')
    text_file.write('Best fold: %i \n' % best_params['fold'])
    text_file.write('Best epoch: %i \n' % (best_params['best_epoch'] + 1))
    text_file.write('Time of training: %d s \n' % best_params['training_time'])
    if denomination == 'Accuracy':
        text_file.write(denomination + ' on validation set: %.2f %% \n' % acc_train)
        if testloader is not None:
            text_file.write(denomination + ' on test set: %.2f %% \n' % acc_test)
        text_file.close()
    else:
        text_file.write(denomination + ' on validation set: %.3E \n' % (acc_train / len(trainset)))
        if testloader is not None:
            text_file.write(denomination + ' on test set: %.3E \n' % (acc_test / len(testset)))
        text_file.close()

    if denomination == 'Accuracy':
        print(denomination + ' of the network on the %i validation images: %.2f %%' % (len(trainset), acc_train))
        print(denomination + ' of the network on the %i test images: %.2f %%' % (len_test, acc_test))
    else:
        print(denomination + ' of the network on the %i validation images: %.3E' % (len(trainset), acc_train))
        print(denomination + ' of the network on the %i test images: %.3E' % (len_test, acc_test))

    parameters_name = 'best_parameters_' + name + '.tar'
    torch.save(best_params['best_model'].state_dict(), path.join(results_path, parameters_name))


def load_state_dict(self, state_dict):
    """
    Loads a pretrained layer in a Module instance

    :param self: the Module instance
    :param state_dict: The dictionary of pretrained parameters
    """
    own_state = self.state_dict()
    for name, param in state_dict.items():
        if name in own_state:
            # backwards compatibility for serialized parameters
            param = param.data
            own_state[name].copy_(param)


if __name__ == '__main__':
    from data_loader import MriBrainDataset, ToTensor, GaussianSmoothing
    from training_functions import CrossValidationSplit, cross_validation, test
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from time import time
    import argparse
    import torchvision

    parser = argparse.ArgumentParser()

    # Mandatory arguments
    parser.add_argument("train_path", type=str,
                        help='path to your list of subjects for training')
    parser.add_argument("results_path", type=str,
                        help="where the outputs are stored")
    parser.add_argument("caps_path", type=str,
                        help="path to your caps folder")

    # Network structure
    parser.add_argument('-filters', '--n_filters', type=int, default=150,
                        help='number of filters used in the encoding convolutional layer')
    parser.add_argument('--n_classes', type=int, default=2,
                        help='Number of classes in the dataset')

    # Dataset management
    parser.add_argument('--bids', action='store_true', default=False)
    parser.add_argument('--sigma', type=float, default=0,
                        help='Size of the Gaussian smoothing kernel (preprocessing)')

    # Training arguments
    parser.add_argument("-e", "--epochs", type=int, default=2,
                        help="number of loops on the whole dataset")
    parser.add_argument('-lra', '--learning_rate_auto', type=float, default=1,
                        help='the learning rate of the optimizer of the sparse autoencoder ( * 0.0005)')
    parser.add_argument('-lrc', '--learning_rate_class', type=float, default=1,
                        help='the learning rate of the optimizer of the classifier ( * 0.0005)')
    parser.add_argument("-l1", "--lambda1", type=float, default=1,
                        help="coefficient of the L1 regularization for the sparsity of the autoencoder")
    parser.add_argument('-cv', '--cross_validation', type=int, default=10,
                        help='cross validation parameter')
    parser.add_argument('--dropout', '-d', type=float, default=0.5,
                        help='Dropout rate before FC layers')
    parser.add_argument('--batch_size', '-batch', type=int, default=4,
                        help="The size of the batches to train the network")

    # Managing output
    parser.add_argument("-n", "--name", type=str, default='network',
                        help="name given to the outputs and checkpoints of the parameters")
    parser.add_argument("-save", "--save_interval", type=int, default=1,
                        help="the number of epochs done between the tests and saving")

    # Managing device
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='Uses gpu instead of cpu if cuda is available')
    parser.add_argument('--on_cluster', action='store_true', default=False,
                        help='to work on the cluster of the ICM')

    args = parser.parse_args()
    results_path = path.join(args.results_path, args.name)
    if not path.exists(results_path):
        os.makedirs(results_path)

    if args.gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device("cpu")

    # Autoencoder training
    autoencoder = AdaptativeAutoEncoder(args.n_filters).to(device=device)
    lr_autoencoder = 0.00005 * args.learning_rate_auto
    lr_classifier = 0.00005 * args.learning_rate_class
    batch_size = args.batch_size
    train_prop = 0.85
    val_prop = 0.15
    tol = 1e-2

    composed = torchvision.transforms.Compose([GaussianSmoothing(sigma=args.sigma), ToTensor(gpu=args.gpu)])
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr_autoencoder)

    dataset = MriBrainDataset(args.train_path, args.caps_path, transform=composed, on_cluster=args.on_cluster)

    cross_val = CrossValidationSplit(dataset, cv=train_prop, stratified=True, shuffle_diagnosis=True, val_prop=val_prop)

    trainset, validset, testset = cross_val(dataset)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    validloader = DataLoader(validset, batch_size=batch_size, shuffle=False, num_workers=4)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    epoch = 0
    loss_train = np.inf
    loss_valid_min = np.inf
    best_model = None
    best_epoch = 0
    t0 = time()
    name = 'autoencoder_' + args.name
    filename = path.join(results_path, name + '.tsv')
    criterion = nn.MSELoss()

    results_df = pd.DataFrame(columns=['epoch', 'training_time', 'acc_train', 'acc_validation'])
    with open(filename, 'w') as f:
        results_df.to_csv(f, index=False, sep='\t')

    flag = True

    while flag:

        prev_loss_train = loss_train
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            if args.gpu:
                inputs = data['image'].cuda()
            else:
                inputs = data['image']

            outputs, hidden_layer, downsample = autoencoder(inputs)
            MSEloss = criterion(outputs, downsample)
            l1_regularization = args.lambda1 * l1_penalty(hidden_layer)

            loss = MSEloss + l1_regularization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:  # print every 10 mini-batches
                print('[%d, %d] loss: %f' %
                      (epoch + 1, i + 1, running_loss))
                running_loss = 0.0

        print('Finished Epoch: %d' % (epoch + 1))

        if epoch % args.save_interval == args.save_interval - 1:
            training_time = time() - t0

            loss_train = test_autoencoder(autoencoder, trainloader, gpu=args.gpu)
            loss_valid = test_autoencoder(autoencoder, validloader, gpu=args.gpu)
            row = np.array([epoch + 1, training_time, loss_train, loss_valid]).reshape(1, -1)

            row_df = pd.DataFrame(row, columns=['epoch', 'training_time', 'loss_train', 'loss_validation'])
            with open(filename, 'a') as f:
                row_df.to_csv(f, header=False, index=False, sep='\t')

            if loss_valid < loss_valid_min:
                loss_valid_min = copy(loss_valid)
                best_epoch = copy(epoch)
                best_model = deepcopy(autoencoder)

            epoch += 1
            print('Convergence criterion: ', torch.abs((prev_loss_train - loss_train)/loss_train))
            flag = epoch < args.epochs and torch.abs(prev_loss_train - loss_train)/loss_train > tol

    training_time = time() - t0

    best_params = {'training_time': time() - t0,
                   'best_epoch': best_epoch,
                   'best_model': best_model,
                   'loss_valid_min': loss_valid_min,
                   'fold': -1}

    save_results(best_params, validloader, test_autoencoder, results_path, name, testloader=testloader,
                 denomination='Loss', gpu=args.gpu)

    classifier = AdaptativeConvolutionalNetwork(args.n_filters, args.dropout,
                                                n_classes=args.n_classes).to(device=device)

    # Load pretrained layer in classifier
    load_state_dict(classifier, best_model.state_dict())
    classifier.encode.bias.requires_grad = False
    classifier.encode.weight.requires_grad = False

    name = 'classifier_' + args.name
    best_params = cross_validation(classifier, trainset, batch_size=batch_size, folds=args.cross_validation,
                                   epochs=args.epochs, results_path=results_path, model_name=name,
                                   save_interval=args.save_interval, gpu=args.gpu, lr=lr_classifier,
                                   tol=1.0)
