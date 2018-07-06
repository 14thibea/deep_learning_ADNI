import torch
import torch.nn as nn
import torch.nn.functional as F
from os import path
import os


class ConvLBP(nn.Conv3d):
    """
    A 3D convolution layer with fixed binary weights

    source: https://github.com/dizcza/lbcnn.pytorch/
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, sparsity=0.5):
        super().__init__(in_channels, out_channels, kernel_size, padding=1, bias=False)
        weights = next(self.parameters())
        matrix_proba = torch.FloatTensor(weights.data.shape).fill_(0.5)
        binary_weights = torch.bernoulli(matrix_proba) * 2 - 1
        mask_inactive = torch.rand(matrix_proba.shape) > sparsity
        binary_weights.masked_fill_(mask_inactive, 0)
        weights.data = binary_weights
        weights.requires_grad = False


class DiagnosisClassifier(nn.Module):
    """
    Classifier for a binary classification task

    """
    def __init__(self, n_classes=2):
        super(DiagnosisClassifier, self).__init__()
        self.pool = nn.MaxPool3d(2, 2)  # Subsampling
        self.conv1 = nn.Conv3d(1, 6, 4)
        self.conv2 = nn.Conv3d(6, 16, 4)
        self.conv3 = nn.Conv3d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 12 * 15 * 12, 10000)
        self.fc2 = nn.Linear(10000, 5000)
        self.fc3 = nn.Linear(5000, 1000)
        self.fc4 = nn.Linear(1000, 200)
        self.fc5 = nn.Linear(200, 60)
        self.fc6 = nn.Linear(60, n_classes)

    def forward(self, x, train=False):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, 32 * 12 * 15 * 12)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x


class BasicGPUClassifier(nn.Module):
    """
    Classifier for a binary classification task

    """
    def __init__(self, dropout=0.1, n_classes=2, bids=False):
        super(BasicGPUClassifier, self).__init__()
        self.pool = nn.MaxPool3d(2, 2)  # Subsampling
        self.pool2 = nn.MaxPool3d(2, 2, padding=1)
        self.conv1 = nn.Conv3d(1, 8, 4)
        self.conv2 = nn.Conv3d(8, 16, 4)
        self.conv3 = nn.Conv3d(16, 32, 5)
        self.conv4 = nn.Conv3d(32, 64, 4)
        self.fc1 = nn.Linear(64 * 5 * 7 * 5, 5000)
        self.fc2 = nn.Linear(5000, 1000)
        self.fc3 = nn.Linear(1000, 200)
        self.fc4 = nn.Linear(200, 60)
        self.fc5 = nn.Linear(60, n_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, train=False):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool2(F.relu(self.conv4(x)))

        x = x.view(-1, 64 * 5 * 7 * 5)
        if train:
            x = self.dropout(x)
        x = F.relu(self.fc1(x))
        if train:
            x = self.dropout(x)
        x = F.relu(self.fc2(x))
        if train:
            x = self.dropout(x)
        x = F.relu(self.fc3(x))
        if train:
            x = self.dropout(x)
        x = F.relu(self.fc4(x))
        if train:
            x = self.dropout(x)
        x = self.fc5(x)
        return x


class SimonyanClassifier(nn.Module):
    """
    Classifier for a binary classification task

    """
    def __init__(self, n_classes=2):
        super(SimonyanClassifier, self).__init__()
        self.pool = nn.MaxPool3d(2, 2, padding=1)  # Subsampling
        self.conv1 = nn.Conv3d(1, 8, 3)
        self.conv2 = nn.Conv3d(8, 16, 3)
        self.conv3 = nn.Conv3d(16, 32, 3)
        self.conv4 = nn.Conv3d(32, 32, 3)
        self.conv5 = nn.Conv3d(32, 64, 3)
        self.conv6 = nn.Conv3d(64, 64, 3)
        self.fc1 = nn.Linear(64 * 6 * 7 * 6, 5000)
        self.fc2 = nn.Linear(5000, 1000)
        self.fc3 = nn.Linear(1000, 200)
        self.fc4 = nn.Linear(200, 60)
        self.fc5 = nn.Linear(60, n_classes)

    def forward(self, x, train=False):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = self.conv2(x)
        x = self.pool(F.relu(x))

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)

        x = x.view(-1, 64 * 6 * 7 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class SimpleClassifier(nn.Module):
    """
    Classifier for a binary classification task

    """
    def __init__(self, n_classes=2):
        super(SimpleClassifier, self).__init__()
        self.pool = nn.MaxPool3d(2, 2)  # Subsampling
        self.conv1 = nn.Conv3d(1, 8, 4)
        self.conv2 = nn.Conv3d(8, 8, 5)
        self.conv3 = nn.Conv3d(8, 16, 4)
        self.conv4 = nn.Conv3d(16, 16, 5)
        self.conv5 = nn.Conv3d(16, 32, 4)
        self.conv6 = nn.Conv3d(32, 32, 5)
        self.fc1 = nn.Linear(32 * 9 * 12 * 9, 5000)
        self.fc2 = nn.Linear(5000, 800)
        self.fc3 = nn.Linear(800, 100)
        self.fc4 = nn.Linear(100, 10)
        self.fc5 = nn.Linear(10, n_classes)

    def forward(self, x, train=False):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)

        x = x.view(-1, 32 * 9 * 12 * 9)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class SimpleLBP(nn.Module):
    """
    Classifier for a binary classification task

    """
    def __init__(self, n_classes=2):
        super(SimpleLBP, self).__init__()
        # Warning: LBP layers have automatically padding = 1

        self.pool = nn.MaxPool3d(2, 2)  # Subsampling
        self.conv1 = ConvLBP(1, 8, 4)
        self.conv2 = nn.Conv3d(8, 8, 5)
        self.conv3 = ConvLBP(8, 16, 4)
        self.conv4 = nn.Conv3d(16, 16, 5)
        self.conv5 = ConvLBP(16, 32, 4)
        self.conv6 = nn.Conv3d(32, 32, 5)
        self.fc1 = nn.Linear(32 * 10 * 13 * 10, 5000)
        self.fc2 = nn.Linear(5000, 800)
        self.fc3 = nn.Linear(800, 100)
        self.fc4 = nn.Linear(100, 10)
        self.fc5 = nn.Linear(10, n_classes)

    def forward(self, x, train=False):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)

        x = x.view(-1, 32 * 10 * 13 * 10)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class SimpleLBCNN(nn.Module):
    """
    Classifier for a binary classification task

    """
    def __init__(self, n_classes=2):
        super(SimpleLBCNN, self).__init__()
        self.pool = nn.MaxPool3d(2, 2)  # Subsampling
        self.conv1 = ConvLBP(1, 8, 6)
        self.conv2 = nn.Conv3d(8, 8, 5)

        self.conv3 = ConvLBP(8, 32, 6)
        self.conv3_1 = nn.Conv3d(32, 16, 1)

        self.conv4 = ConvLBP(16, 32, 5)
        self.conv4_1 = nn.Conv3d(32, 16, 1)

        self.conv5 = ConvLBP(16, 32, 6)
        self.conv5_1 = nn.Conv3d(32, 32, 1)

        self.conv6 = ConvLBP(32, 32, 5)
        self.conv6_1 = nn.Conv3d(32, 32, 1)

        self.fc1 = nn.Linear(32 * 10 * 13 * 10, 5000)
        self.fc2 = nn.Linear(5000, 800)
        self.fc3 = nn.Linear(800, 100)
        self.fc4 = nn.Linear(100, 10)
        self.fc5 = nn.Linear(10, n_classes)

    def forward(self, x, train=False):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = self.conv3_1(F.relu(self.conv3(x)))
        x = self.conv4_1(F.relu(self.conv4(x)))
        x = self.pool(x)

        x = self.conv5_1(F.relu(self.conv5(x)))
        x = self.conv6_1(F.relu(self.conv6(x)))
        x = self.pool(x)

        x = x.view(-1, 32 * 10 * 13 * 10)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class LocalBriefNet(nn.Module):

    def __init__(self, n_classes=2):
        super(LocalBriefNet, self).__init__()
        self.pool = nn.MaxPool3d(2, 2)
        self.conv5x5 = nn.Conv3d(1, 32, 5)
        self.conv3x3 = nn.Conv3d(32, 32, 3)
        self.last_conv = nn.Conv3d(32, 2, 3)
        self.fc = nn.Linear(2 * 24 * 30 * 24, n_classes)

    def forward(self, x, train=False):
        x = F.relu(self.conv5x5(x))
        x = self.pool(x)
        x = F.relu(self.conv3x3(x))
        x = F.relu(self.conv3x3(x))
        x = F.relu(self.conv3x3(x))
        x = self.pool(x)

        x = F.relu(self.last_conv(x))
        x = x.view(-1, 2 * 24 * 30 * 24)
        x = self.fc(x)
        return x


class LocalBriefNet2(nn.Module):

    def __init__(self, n_classes=2):
        super(LocalBriefNet2, self).__init__()
        self.pool = nn.MaxPool3d(2, 2)
        self.conv5x5 = nn.Conv3d(1, 32, 5)
        self.conv3x3 = nn.Conv3d(32, 32, 3)
        self.last_conv = nn.Conv3d(32, 2, 3)
        self.fc = nn.Linear(2 * 23 * 29 * 23, n_classes)

    def forward(self, x, train=False):
        x = F.relu(self.conv5x5(x))
        x = F.relu(self.conv3x3(x))
        x = self.pool(x)
        x = F.relu(self.conv3x3(x))
        x = F.relu(self.conv3x3(x))
        x = F.relu(self.conv3x3(x))
        x = self.pool(x)

        x = F.relu(self.last_conv(x))
        x = x.view(-1, 2 * 23 * 29 * 23)
        x = self.fc(x)
        return x


class LocalBriefNet3(nn.Module):

    def __init__(self, n_classes=2):
        super(LocalBriefNet3, self).__init__()
        self.pool = nn.MaxPool3d(2, 2)
        self.conv5x5 = nn.Conv3d(1, 32, 5)
        self.conv3x3 = nn.Conv3d(32, 32, 3)
        self.last_conv = nn.Conv3d(32, 2, 3)
        self.fc = nn.Linear(2 * 24 * 30 * 24, n_classes)

    def forward(self, x, train=False):
        x = F.relu(self.conv5x5(x))
        x = F.relu(self.conv3x3(x))
        x = self.pool(x)
        x = F.relu(self.conv3x3(x))
        x = F.relu(self.conv3x3(x))
        x = self.pool(x)

        x = F.relu(self.last_conv(x))
        x = x.view(-1, 2 * 24 * 30 * 24)
        x = self.fc(x)
        return x


class VGG(nn.Module):
    def __init__(self, n_classes=2):
        super(VGG, self).__init__()
        self.pool = nn.MaxPool3d(2, 2)
        self.conv1 = nn.Conv3d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv3d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv3d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv3d(128, 128, 3, padding=1)
        self.conv5 = nn.Conv3d(128, 256, 3, padding=1)
        self.conv6 = nn.Conv3d(256, 256, 3, padding=1)
        self.conv7 = nn.Conv3d(256, 32, 1)
        self.fc1 = nn.Linear(32 * 3 * 4 * 3, 100)
        self.fc2 = nn.Linear(100, n_classes)

    def forward(self, x, train=False):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)

        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)

        x = F.relu(self.conv6(x))
        x = F.relu(self.conv6(x))
        x = self.pool(x)

        x = F.relu(self.conv7(x))
        x = x.view(-1, 32 * 3 * 4 * 3)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    from data_loader import MriBrainDataset, BidsMriBrainDataset, ToTensor, GaussianSmoothing
    from training_functions import cross_validation
    import torchvision
    import argparse

    parser = argparse.ArgumentParser()

    # Mandatory arguments
    parser.add_argument("train_path", type=str,
                        help='path to your list of subjects for training')
    parser.add_argument("results_path", type=str,
                        help="where the outputs are stored")
    parser.add_argument("caps_path", type=str,
                        help="path to your caps folder")

    # Network structure
    parser.add_argument("--classifier", type=str, default='basic',
                        help='classifier selected')
    parser.add_argument('--n_classes', type=int, default=2,
                        help='Number of classes in the dataset')

    # Dataset management
    parser.add_argument('--bids', action='store_true', default=False)
    parser.add_argument('--sigma', type=float, default=0,
                        help='Size of the Gaussian smoothing kernel (preprocessing)')
    parser.add_argument('--rescale', type=str, default='crop',
                        help='Action to rescale the BIDS without deforming the images')

    # Training arguments
    parser.add_argument("-e", "--epochs", type=int, default=2,
                        help="number of loops on the whole dataset")
    parser.add_argument('-lr', '--learning_rate', type=float, default=1,
                        help='the learning rate of the optimizer (*0.00005)')
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

    if args.gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    lr = args.learning_rate * 0.00005

    results_path = path.join(args.results_path, args.name)
    if not path.exists(results_path):
        os.makedirs(results_path)

    composed = torchvision.transforms.Compose([GaussianSmoothing(sigma=args.sigma), ToTensor(gpu=args.gpu)])

    if args.bids:
        trainset = BidsMriBrainDataset(args.train_path, args.caps_path, classes=args.n_classes,
                                       transform=composed, rescale=args.rescale)
    else:
        trainset = MriBrainDataset(args.train_path, args.caps_path, classes=args.n_classes,
                                   transform=composed, on_cluster=args.on_cluster)

    if args.classifier == 'basic':
        classifier = DiagnosisClassifier(n_classes=args.n_classes).to(device=device)
    elif args.classifier == 'simonyan':
        classifier = SimonyanClassifier(n_classes=args.n_classes).to(device=device)
    elif args.classifier == 'simple':
        classifier = SimpleClassifier(n_classes=args.n_classes).to(device=device)
    elif args.classifier == 'basicgpu':
        classifier = BasicGPUClassifier(n_classes=args.n_classes, dropout=args.dropout).to(device=device)
    elif args.classifier == 'simpleLBP':
        classifier = SimpleLBP(n_classes=args.n_classes).to(device=device)
    elif args.classifier == 'simpleLBCNN':
        classifier = SimpleLBCNN(n_classes=args.n_classes).to(device=device)
    elif args.classifier == 'localbriefnet':
        classifier = LocalBriefNet(n_classes=args.n_classes).to(device=device)
    elif args.classifier == 'localbriefnet2':
        classifier = LocalBriefNet2(n_classes=args.n_classes).to(device=device)
    elif args.classifier == 'localbriefnet3':
        classifier = LocalBriefNet3(n_classes=args.n_classes).to(device=device)
    elif args.classifier == 'vgg':
        classifier = VGG(n_classes=args.n_classes).to(device=device)
    else:
        raise ValueError('Unknown classifier')

    # Initialization
    # classifier.apply(weights_init)
    # Training
    best_params = cross_validation(classifier, trainset, batch_size=args.batch_size, folds=args.cross_validation,
                                   epochs=args.epochs, results_path=results_path, model_name=args.name,
                                   save_interval=args.save_interval, gpu=args.gpu, lr=lr)
