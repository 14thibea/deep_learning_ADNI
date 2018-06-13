import torch
import torch.nn as nn
import torch.nn.functional as F
from os import path
import os


class DiagnosisClassifier(nn.Module):
    """
    Classifier for a binary classification task

    """
    def __init__(self):
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
        self.fc6 = nn.Linear(60, 2)

    def forward(self, x):
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
    def __init__(self):
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
        self.fc5 = nn.Linear(60, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool2(F.relu(self.conv4(x)))

        x = x.view(-1, 64 * 5 * 7 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x


class SimonyanClassifier(nn.Module):
    """
    Classifier for a binary classification task

    """
    def __init__(self):
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
        self.fc5 = nn.Linear(60, 2)

    def forward(self, x):
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
    def __init__(self):
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
        self.fc5 = nn.Linear(10, 2)

    def forward(self, x):
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


if __name__ == '__main__':
    from data_loader import MriBrainDataset, ToTensor, MeanNormalization
    from training_functions import test, cross_validation
    from torch.utils.data import DataLoader
    import torchvision
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("train_path", type=str,
                        help='path to your list of subjects for training')
    parser.add_argument("results_path", type=str,
                        help="where the outputs are stored")
    parser.add_argument("caps_path", type=str,
                        help="path to your caps folder")
    parser.add_argument("-test", "--test_path", type=str, default=None,
                        help='path to your list of subjects for testing')
    parser.add_argument("-e", "--epochs", type=int, default=2,
                        help="number of loops on the whole dataset")
    parser.add_argument("-n", "--name", type=str, default='network',
                        help="name given to the outputs and checkpoints of the parameters")
    parser.add_argument("-c", "--cpu", type=int, default=1,
                        help="number of CPUs used")
    parser.add_argument("--classifier", type=str, default='basic',
                        help='classifier selected')
    parser.add_argument("-save", "--save_interval", type=int, default=1,
                        help="the number of epochs done between the tests and saving")
    parser.add_argument('--gpu', action='store_true', default=False)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001,
                        help='the learning rate of the optimizer')
    parser.add_argument('-cv', '--cross_validation', type=int, default=10,
                        help='cross validation parameter')

    args = parser.parse_args()

    if args.gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device("cpu")
    args = parser.parse_args()

    batch_size = 4
    torch.set_num_threads(args.cpu)
    model_name = args.name
    train_tsv_path = args.train_path
    test_tsv_path = args.test_path
    caps_path = args.caps_path
    results_path = path.join(args.results_path, model_name)
    if not path.exists(results_path):
        os.makedirs(results_path)

    mean_path = '/teams/ARAMIS/PROJECTS/elina.thibeausutre/data/mean_brain.nii'
    composed = torchvision.transforms.Compose([MeanNormalization(mean_path),
                                               ToTensor(gpu=args.gpu)])

    trainset = MriBrainDataset(train_tsv_path, caps_path, transform=composed)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    if args.classifier == 'basic':
        classifier = DiagnosisClassifier().to(device=device)
    elif args.classifier == 'simonyan':
        classifier = SimonyanClassifier().to(device=device)
    elif args.classifier == 'simple':
        classifier = SimpleClassifier().to(device=device)
    elif args.classifier == 'basicgpu':
        classifier = BasicGPUClassifier().to(device=device)
    else:
        raise ValueError('Unknown classifier')

    # Initialization
    # classifier.apply(weights_init)
    # Training
    best_params = cross_validation(classifier, trainset, batch_size=batch_size, folds=args.cross_validation,
                                   epochs=args.epochs, results_path=results_path, model_name=model_name,
                                   save_interval=args.save_interval, gpu=args.gpu, lr=args.learning_rate)

    if test_tsv_path is not None:
        testset = MriBrainDataset(test_tsv_path, caps_path, transform=composed)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)
        len_test = len(testset)
        acc_test = test(best_params['best_model'], testloader, gpu=args.gpu)
    else:
        len_test = 0
        acc_test = 0

    acc_train = test(best_params['best_model'], trainloader, gpu=args.gpu)

    output_name = 'best_' + model_name + '_epoch-' + str(best_params['best_epoch'] + 1) + '.txt'
    text_file = open(path.join(results_path, output_name), 'w')
    text_file.write('Best fold: %i \n' % best_params['fold'])
    text_file.write('Time of training: %d s \n' % best_params['training_time'])
    text_file.write('Accuracy on training set: %.2f %% \n' % acc_train)
    if test_tsv_path is not None:
        text_file.write('Accuracy on test set: %.2f %% \n' % acc_test)
    text_file.close()

    print('Accuracy of the network on the %i train images: %.2f %%' % (len(trainset), acc_train))
    print('Accuracy of the network on the %i test images: %.2f %%' % (len_test, acc_test))

    parameters_name = 'best_parameters_' + model_name + '_epochs-' + str(best_params['best_epoch'] + 1) + '.tar'
    torch.save(best_params['best_model'].state_dict(), path.join(results_path, parameters_name))
