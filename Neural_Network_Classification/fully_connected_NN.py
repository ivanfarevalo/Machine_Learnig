import torch
from torch.utils.data import Dataset
import sample_generation
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class GaussianDataSet(Dataset):
    def __init__(self, points, labels=None, transforms=None):
        self.X = points
        self.y = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        data = self.X[i, :]
        data = torch.from_numpy(data)

        if self.transforms:
            data = self.transforms(data)

        if self.y is not None:
            return data, self.y[i]
        else:
            return data


class LinearNetwork(nn.Module):
    def __init__(self, layer1_size, layer2_size=None, nonlinear_tanh=False):
        super(LinearNetwork, self).__init__()
        self.nonlinear_tanh = nonlinear_tanh
        self.fc1 = nn.Linear(2, layer1_size)
        if layer2_size is not None:
            self.fc2 = nn.Linear(layer1_size, layer2_size)
            self.fc3 = nn.Linear(layer2_size, 2)
        else:
            self.fc2 = nn.Linear(layer1_size, 2)

    def nonlinearity(self, x):
        return torch.tanh(x) if self.nonlinear_tanh else F.relu(x)

    def forward(self, x):
        x = self.fc1(x)
        x = self.nonlinearity(x)
        x = self.fc2(x)
        if hasattr(self, 'fc3'):
            x = self.nonlinearity(x)
            x = self.fc3(x)
        return x


def generate_data(num_samples, training=False, batch_size=100):

    if training:
        seed0, seed1A, seed1B = 255, 400, 545
    else:
        seed0, seed1A, seed1B = 456, 25, 62

    class_0, class_1, class_1A, class_1B = \
        sample_generation.generate_samples(num_samples, seed0=seed0, seed1A=seed1A, seed1B=seed1B)
    samples = np.r_[class_0.data, class_1.data]  # Concatenate class 0 and 1 samples
    data_labels = np.r_[np.zeros((num_samples, 1), dtype=int), np.ones((num_samples, 1), dtype=int)]
    data_set = GaussianDataSet(samples, data_labels)

    if training:
        data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, shuffle=True)
    else:
        data_loader = torch.utils.data.DataLoader(data_set, batch_size=2*num_samples, shuffle=False)

    return data_loader


def create_network(layer1_size, layer2_size=2, nonlinear_tanh=False):

    def weights_init(model):
        for m in model.modules():
            if isinstance(m, nn.Linear):
                # initialize the weight tensor, here we use a normal distribution
                m.weight.data.normal_(0, 1)
                # m.weight.data.uniform_(0, 1)

    network = LinearNetwork(layer1_size, layer2_size, nonlinear_tanh)
    network.apply(weights_init)
    print(network)
    return network


def train_network(network, criterion, optimizer, trainloader, num_epocs=50):

    network.train()  # Stop calculating gradients

    for epoch in range(num_epocs):  # loop over the dataset multiple times

        running_loss = 0.0

        for i, data in enumerate(trainloader, 0):

            inputs, labels = data[0], data[1]  # get the inputs
            optimizer.zero_grad()  # zero the parameter gradients

            # forward + backward + optimize
            outputs = network(inputs.float())
            labels = labels.squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100== 99:  # print every 100 mini-batches
                print('[%d, %5d] loss: %.5f' %
                      (epoch + 1, i + 1, running_loss/100))
                running_loss = 0.0

    print('Finished Training')


def test_network(network, testloader, plot_decision_errors=True):

    network.eval()

    correct = 0
    total = 0
    with torch.no_grad():

        data = next(iter(testloader))
        test_points, labels = data
        outputs = network(test_points.float())
        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        num_class_samples = int(total/2)
        correct = (predicted == labels.squeeze()).sum().item()
        class_0_error_idx = np.not_equal(0, predicted[:num_class_samples].numpy())
        class_1_error_idx = np.not_equal(1, predicted[num_class_samples:].numpy())
        class_0_errors = test_points.numpy()[np.flatnonzero(class_0_error_idx)]
        class_1_errors = test_points.numpy()[np.flatnonzero(class_1_error_idx) + num_class_samples] # Foward index to second half of samples

    print('Percent error of Class 0:', 100 * class_0_errors.shape[0] / num_class_samples)
    print('Percent error of Class 1:', 100 * class_1_errors.shape[0] / num_class_samples)
    print('Total percent error: %d %%' % (100 * (1 - correct / total)))

    if plot_decision_errors:
        plot_performance(network, class_0_errors, class_1_errors, test_points)


def plot_performance(network, class_0_errors, class_1_errors, test_points):
    num_pts = 200
    limit = 10
    X = Y = np.linspace(-limit, limit, num_pts)
    XX, YY = np.meshgrid(X, Y)
    decisions = np.zeros([num_pts, num_pts])
    grid_samples = np.c_[np.ravel(XX), np.ravel(YY)]
    testset = GaussianDataSet(grid_samples)
    testloader = torch.utils.data.DataLoader(testset, batch_size=num_pts*num_pts,
                                             shuffle=False)
    decisions = np.ravel(decisions)
    with torch.no_grad():

        for data in testloader:
            # test_points = data
            outputs = network(data.float())
            _, decisions = torch.max(outputs.data, 1)

    decisions = decisions.numpy().reshape(XX.shape)

    # Plot decision boundaries
    plt.figure()
    plt.contour(XX, YY, decisions, alpha=1, cmap=plt.cm.coolwarm, antialiased=True)
    plt.contourf(XX, YY, decisions, alpha=.15, cmap=plt.cm.coolwarm, antialiased=True)
    plt.title('Neural Network decision boundaries')

    # Plot test samples
    test_points = test_points.numpy()
    plt.plot(test_points[:1000, 0], test_points[:1000, 1], '.', alpha=1)
    plt.plot(test_points[1000:, 0], test_points[1000:, 1], 'r.', alpha=1)

    # Plot test errors
    plt.plot(class_0_errors[:, 0], class_0_errors[:, 1], 'y+')
    plt.plot(class_1_errors[:, 0], class_1_errors[:, 1], 'k+')
    plt.axis('equal')
    plt.legend(
        ['Class 0 Samples', 'Class 1 Samples', 'Class 0 Decision Errors', 'Class 1 Decision Errors'])
    plt.show()

    return  plt


if __name__ == '__main__':

    # Hyperparameters
    number_of_training_samples = 5000
    number_of_testing_samples = 1000
    layer1_size, layer2_size = 5, 2
    learning_rate = 0.01
    regularization_weight = 0
    num_epocs = 100
    training_batch_size = 100
    use_nonlinear_tanh = False

    trainloader = generate_data(number_of_training_samples, training=True, batch_size=training_batch_size)
    network = create_network(layer1_size, layer2_size, nonlinear_tanh=use_nonlinear_tanh)  # Can set nonlinear_tanh=True
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), learning_rate, regularization_weight)

    train_network(network, criterion, optimizer, trainloader, num_epocs=num_epocs)

    testloader = generate_data(number_of_testing_samples, training=False)
    test_network(network, testloader, plot_decision_errors=True)
