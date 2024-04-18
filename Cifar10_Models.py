import time

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from MNISTClassifier import one_hot_encode
from Utility import prepare_for_training


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, drop_rate):
        super(_DenseLayer, self).__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = self.conv1(self.relu1(self.norm1(x)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class SmallDenseNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SmallDenseNet, self).__init__()
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 48, kernel_size=5, stride=1, padding=2)
        self.norm1 = nn.BatchNorm2d(48)
        self.relu = nn.ReLU(inplace=True)
        self.lrn = nn.LocalResponseNorm(2)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        growth_rate = 32
        block_config = [2, 2, 2]  # Number of layers in each block
        num_init_features = 48

        self.features = nn.Sequential()
        # Dense Blocks
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                growth_rate=growth_rate, drop_rate=0.5)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:  # do not add transition layer after the last block
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final Batch Norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Pooling and Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(num_features, num_classes)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.test_accuracy_list = []

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        self.testloader = DataLoader(testset, batch_size=64, shuffle=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.lrn(x)
        x = self.pool(x)

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def train_model(self, train_loader, criterion, optimizer, n_epochs=10, print_every=500,
                    model_name="Alexnet_surrogate_model"):
        model, best_val_accuracy, best_model_state_dict, start_epoch = prepare_for_training(self, model_name, optimizer)
        for epoch in range(start_epoch, n_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            start_time = time.time()
            start_time_epoch = time.time()
            for i, (inputs, labels) in enumerate(train_loader, 0):
                inputs = inputs.view(inputs.shape[0], 3, 32, 32).detach().clone()
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if torch.cuda.is_available():
                    outputs = model(inputs)
                else:
                    outputs = self(inputs)
                try:
                    labels = labels.view(-1, outputs.size()[1]).float().detach().clone().requires_grad_(True)
                except:
                    labels = torch.tensor(list(
                        map(lambda x: one_hot_encode(x, outputs.size()[1]), labels))).detach().clone().requires_grad_(
                        True)

                # # Get the maximum values along the third dimension (dimension with size 10)
                # fitnesses, _ = torch.max(labels_copy, dim=2)
                # # Sum all the elements in the tensor to get a scalar
                # sum_fitnesses = torch.sum(torch.max(-torch.log(fitnesses), torch.tensor(100))).requires_grad_()
                outputs, labels = outputs.to(self.device), labels.to(self.device)
                loss = criterion(outputs, labels)  # + sum_fitnesses

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                _, labels = torch.max(labels.data, 1)
                correct += (predicted == labels).sum().item()

                if i % print_every == print_every - 1:
                    finish_time = time.time()
                    total_time = finish_time - start_time
                    print('[%d, %5d] loss: %.3f accuracy: %.3f the time it took: %.3f seconds' % (
                        epoch + 1, i + 1, running_loss / print_every, 100 * correct / total, total_time))
                    running_loss = 0.0
                    correct = 0
                    total = 0
                    start_time = time.time()
            finish_time_epoch = time.time()
            total_time_epoch = finish_time_epoch - start_time_epoch
            print(f'Epoch {epoch + 1} took {total_time_epoch} seconds')
            validation_accuracy = self.validate_model()
            self.test_accuracy_list.append(validation_accuracy)
            # Save model after each epoch
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_accuracy_list': self.test_accuracy_list
            }
            directory = f"checkpoints/{self.__class__.__name__}"
            torch.save(checkpoint, f'./{directory}/{model_name}.pth')
            if validation_accuracy > best_val_accuracy:
                best_val_accuracy = validation_accuracy
                best_model_state_dict = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'test_accuracy_list': self.test_accuracy_list
                }
                torch.save(best_model_state_dict, f'./{directory}/best_accuracy_{model_name}.pth')
            print("Saved model checkpoint!")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(
            f"The maximal accuracy during training was: {max(self.test_accuracy_list)} on epoch: {self.test_accuracy_list.index(max(self.test_accuracy_list))}")
        # self.plot_accuracy_graph()

        accuracy_list = self.test_accuracy_list
        # Plotting using Seaborn
        sns.set(style="darkgrid")
        sns.lineplot(x=range(len(accuracy_list)), y=accuracy_list, marker='X')

        # # Add accuracy values as annotations
        # for i, accuracy in enumerate(accuracy_list):
        #     plt.annotate(f"{accuracy:.2f}", (i, accuracy), textcoords="offset points", xytext=(0, 10), ha='center')

        # Set labels and title
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Over Epochs")

        # Set x-axis ticks as integers
        # plt.xticks(range(len(accuracy_list)))

        # Display the plot
        plt.show()

    def validate_model(self):
        # Test the model
        correct = 0
        total = 0
        model = None
        if torch.cuda.is_available():
            num_of_gpus = torch.cuda.device_count()
            gpu_list = list(range(num_of_gpus))
            model = nn.DataParallel(self, device_ids=gpu_list).to(self.device)
        # Don't need to keep track of gradients
        with torch.no_grad():
            for images, labels in self.testloader:
                images = images.view(images.shape[0], 3, 32, 32).detach().clone()
                if torch.cuda.is_available():
                    outputs = model(images)
                else:
                    outputs = self(images)
                images, labels, outputs = images.to(self.device), labels.to(self.device), outputs.to(self.device)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Accuracy on test set is: {100 * correct / total}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return correct / total

    def test_model(self):
        # Test the model
        # self.eval()
        correct = 0
        total = 0
        model = None
        if torch.cuda.is_available():
            num_of_gpus = torch.cuda.device_count()
            gpu_list = list(range(num_of_gpus))
            model = nn.DataParallel(self, device_ids=gpu_list).to(self.device)
        # Don't need to keep track of gradients
        with torch.no_grad():
            for images, labels in self.testloader:
                images = images.view(images.shape[0], 3, 32, 32).detach().clone()
                if torch.cuda.is_available():
                    outputs = model(images)
                else:
                    outputs = self(images)
                images, labels, outputs = images.to(self.device), labels.to(self.device), outputs.to(self.device)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Accuracy on test set is: {100 * correct / total}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # self.train()
        return correct / total


class BasicCNN_BN(nn.Module):
    def __init__(self):
        super(BasicCNN_BN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 10)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        self.testloader = DataLoader(testset, batch_size=64, shuffle=False)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout(x)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.fc2(x)
        return x


class LeNet5_BN(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5_BN, self).__init__()
        self.conv1 = nn.Conv2d(3, 48, kernel_size=5, stride=1, padding=2)  # Adjusted to match AlexNetSurrogate
        self.bn1 = nn.BatchNorm2d(48)
        self.lrn = nn.LocalResponseNorm(size=2)  # Adding LRN to mimic AlexNetSurrogate
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # Adjusted pooling to match

        self.conv2 = nn.Conv2d(48, 128, 5, padding=2)  # Adjusted to increase filter count
        self.bn2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)  # Additional conv layer to increase depth
        self.bn3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256 * 4 * 4,
                             512)  # Adjusted to account for additional conv layer and increased depth
        self.drop1 = nn.Dropout(p=0.5)
        self.bn4 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 256)
        self.drop2 = nn.Dropout(p=0.5)
        self.bn5 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, num_classes)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.test_accuracy_list = []

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        self.testloader = DataLoader(testset, batch_size=64, shuffle=False)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.lrn(x)
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        x = x.view(-1, 256 * 4 * 4)
        x = self.drop1(F.relu(self.bn4(self.fc1(x))))
        x = self.drop2(F.relu(self.bn5(self.fc2(x))))
        x = self.fc3(x)
        return x

    def train_model(self, train_loader, criterion, optimizer, n_epochs=10, print_every=500,
                    model_name="Alexnet_surrogate_model"):
        model, best_val_accuracy, best_model_state_dict, start_epoch = prepare_for_training(self, model_name, optimizer)
        for epoch in range(start_epoch, n_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            start_time = time.time()
            start_time_epoch = time.time()
            for i, (inputs, labels) in enumerate(train_loader, 0):
                inputs = inputs.view(inputs.shape[0], 3, 32, 32).detach().clone()
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if torch.cuda.is_available():
                    outputs = model(inputs)
                else:
                    outputs = self(inputs)
                try:
                    labels = labels.view(-1, outputs.size()[1]).float().detach().clone().requires_grad_(True)
                except:
                    labels = torch.tensor(list(
                        map(lambda x: one_hot_encode(x, outputs.size()[1]), labels))).detach().clone().requires_grad_(
                        True)

                # # Get the maximum values along the third dimension (dimension with size 10)
                # fitnesses, _ = torch.max(labels_copy, dim=2)
                # # Sum all the elements in the tensor to get a scalar
                # sum_fitnesses = torch.sum(torch.max(-torch.log(fitnesses), torch.tensor(100))).requires_grad_()
                outputs, labels = outputs.to(self.device), labels.to(self.device)
                loss = criterion(outputs, labels)  # + sum_fitnesses

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                _, labels = torch.max(labels.data, 1)
                correct += (predicted == labels).sum().item()

                if i % print_every == print_every - 1:
                    finish_time = time.time()
                    total_time = finish_time - start_time
                    print('[%d, %5d] loss: %.3f accuracy: %.3f the time it took: %.3f seconds' % (
                        epoch + 1, i + 1, running_loss / print_every, 100 * correct / total, total_time))
                    running_loss = 0.0
                    correct = 0
                    total = 0
                    start_time = time.time()
            finish_time_epoch = time.time()
            total_time_epoch = finish_time_epoch - start_time_epoch
            print(f'Epoch {epoch + 1} took {total_time_epoch} seconds')
            validation_accuracy = self.validate_model()
            self.test_accuracy_list.append(validation_accuracy)
            # Save model after each epoch
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_accuracy_list': self.test_accuracy_list
            }
            directory = f"checkpoints/{self.__class__.__name__}"
            torch.save(checkpoint, f'./{directory}/{model_name}.pth')
            if validation_accuracy > best_val_accuracy:
                best_val_accuracy = validation_accuracy
                best_model_state_dict = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'test_accuracy_list': self.test_accuracy_list
                }
                torch.save(best_model_state_dict, f'./{directory}/best_accuracy_{model_name}.pth')
            print("Saved model checkpoint!")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(
            f"The maximal accuracy during training was: {max(self.test_accuracy_list)} on epoch: {self.test_accuracy_list.index(max(self.test_accuracy_list))}")
        # self.plot_accuracy_graph()

        accuracy_list = self.test_accuracy_list
        # Plotting using Seaborn
        sns.set(style="darkgrid")
        sns.lineplot(x=range(len(accuracy_list)), y=accuracy_list, marker='X')

        # # Add accuracy values as annotations
        # for i, accuracy in enumerate(accuracy_list):
        #     plt.annotate(f"{accuracy:.2f}", (i, accuracy), textcoords="offset points", xytext=(0, 10), ha='center')

        # Set labels and title
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Over Epochs")

        # Set x-axis ticks as integers
        # plt.xticks(range(len(accuracy_list)))

        # Display the plot
        plt.show()

    def validate_model(self):
        # Test the model
        correct = 0
        total = 0
        model = None
        if torch.cuda.is_available():
            num_of_gpus = torch.cuda.device_count()
            gpu_list = list(range(num_of_gpus))
            model = nn.DataParallel(self, device_ids=gpu_list).to(self.device)
        # Don't need to keep track of gradients
        with torch.no_grad():
            for images, labels in self.testloader:
                images = images.view(images.shape[0], 3, 32, 32).detach().clone()
                if torch.cuda.is_available():
                    outputs = model(images)
                else:
                    outputs = self(images)
                images, labels, outputs = images.to(self.device), labels.to(self.device), outputs.to(self.device)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Accuracy on test set is: {100 * correct / total}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return correct / total

    def test_model(self):
        # Test the model
        # self.eval()
        correct = 0
        total = 0
        model = None
        if torch.cuda.is_available():
            num_of_gpus = torch.cuda.device_count()
            gpu_list = list(range(num_of_gpus))
            model = nn.DataParallel(self, device_ids=gpu_list).to(self.device)
        # Don't need to keep track of gradients
        with torch.no_grad():
            for images, labels in self.testloader:
                images = images.view(images.shape[0], 3, 32, 32).detach().clone()
                if torch.cuda.is_available():
                    outputs = model(images)
                else:
                    outputs = self(images)
                images, labels, outputs = images.to(self.device), labels.to(self.device), outputs.to(self.device)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Accuracy on test set is: {100 * correct / total}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # self.train()
        return correct / total


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class MiniResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(MiniResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64 * block.expansion, num_classes)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.test_accuracy_list = []

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        self.testloader = DataLoader(testset, batch_size=64, shuffle=False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def train_model(self, train_loader, criterion, optimizer, n_epochs=10, print_every=500,
                    model_name="Alexnet_surrogate_model"):
        model, best_val_accuracy, best_model_state_dict, start_epoch = prepare_for_training(self, model_name, optimizer)
        for epoch in range(start_epoch, n_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            start_time = time.time()
            start_time_epoch = time.time()
            for i, (inputs, labels) in enumerate(train_loader, 0):
                inputs = inputs.view(inputs.shape[0], 3, 32, 32).detach().clone()
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if torch.cuda.is_available():
                    outputs = model(inputs)
                else:
                    outputs = self(inputs)
                try:
                    labels = labels.view(-1, outputs.size()[1]).float().detach().clone().requires_grad_(True)
                except:
                    labels = torch.tensor(list(
                        map(lambda x: one_hot_encode(x, outputs.size()[1]), labels))).detach().clone().requires_grad_(
                        True)

                # # Get the maximum values along the third dimension (dimension with size 10)
                # fitnesses, _ = torch.max(labels_copy, dim=2)
                # # Sum all the elements in the tensor to get a scalar
                # sum_fitnesses = torch.sum(torch.max(-torch.log(fitnesses), torch.tensor(100))).requires_grad_()
                outputs, labels = outputs.to(self.device), labels.to(self.device)
                loss = criterion(outputs, labels)  # + sum_fitnesses

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                _, labels = torch.max(labels.data, 1)
                correct += (predicted == labels).sum().item()

                if i % print_every == print_every - 1:
                    finish_time = time.time()
                    total_time = finish_time - start_time
                    print('[%d, %5d] loss: %.3f accuracy: %.3f the time it took: %.3f seconds' % (
                        epoch + 1, i + 1, running_loss / print_every, 100 * correct / total, total_time))
                    running_loss = 0.0
                    correct = 0
                    total = 0
                    start_time = time.time()
            finish_time_epoch = time.time()
            total_time_epoch = finish_time_epoch - start_time_epoch
            print(f'Epoch {epoch + 1} took {total_time_epoch} seconds')
            validation_accuracy = self.validate_model()
            self.test_accuracy_list.append(validation_accuracy)
            # Save model after each epoch
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_accuracy_list': self.test_accuracy_list
            }
            directory = f"checkpoints/{self.__class__.__name__}"
            torch.save(checkpoint, f'./{directory}/{model_name}.pth')
            if validation_accuracy > best_val_accuracy:
                best_val_accuracy = validation_accuracy
                best_model_state_dict = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'test_accuracy_list': self.test_accuracy_list
                }
                torch.save(best_model_state_dict, f'./{directory}/best_accuracy_{model_name}.pth')
            print("Saved model checkpoint!")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(
            f"The maximal accuracy during training was: {max(self.test_accuracy_list)} on epoch: {self.test_accuracy_list.index(max(self.test_accuracy_list))}")
        # self.plot_accuracy_graph()

        accuracy_list = self.test_accuracy_list
        # Plotting using Seaborn
        sns.set(style="darkgrid")
        sns.lineplot(x=range(len(accuracy_list)), y=accuracy_list, marker='X')

        # # Add accuracy values as annotations
        # for i, accuracy in enumerate(accuracy_list):
        #     plt.annotate(f"{accuracy:.2f}", (i, accuracy), textcoords="offset points", xytext=(0, 10), ha='center')

        # Set labels and title
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Over Epochs")

        # Set x-axis ticks as integers
        # plt.xticks(range(len(accuracy_list)))

        # Display the plot
        plt.show()

    def validate_model(self):
        # Test the model
        correct = 0
        total = 0
        model = None
        if torch.cuda.is_available():
            num_of_gpus = torch.cuda.device_count()
            gpu_list = list(range(num_of_gpus))
            model = nn.DataParallel(self, device_ids=gpu_list).to(self.device)
        # Don't need to keep track of gradients
        with torch.no_grad():
            for images, labels in self.testloader:
                images = images.view(images.shape[0], 3, 32, 32).detach().clone()
                if torch.cuda.is_available():
                    outputs = model(images)
                else:
                    outputs = self(images)
                images, labels, outputs = images.to(self.device), labels.to(self.device), outputs.to(self.device)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Accuracy on test set is: {100 * correct / total}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return correct / total

    def test_model(self):
        # Test the model
        # self.eval()
        correct = 0
        total = 0
        model = None
        if torch.cuda.is_available():
            num_of_gpus = torch.cuda.device_count()
            gpu_list = list(range(num_of_gpus))
            model = nn.DataParallel(self, device_ids=gpu_list).to(self.device)
        # Don't need to keep track of gradients
        with torch.no_grad():
            for images, labels in self.testloader:
                images = images.view(images.shape[0], 3, 32, 32).detach().clone()
                if torch.cuda.is_available():
                    outputs = model(images)
                else:
                    outputs = self(images)
                images, labels, outputs = images.to(self.device), labels.to(self.device), outputs.to(self.device)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Accuracy on test set is: {100 * correct / total}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # self.train()
        return correct / total


def MiniResNet9():
    return MiniResNet(BasicBlock, [2, 2, 2])


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class TeacherVGG(nn.Module):
    def __init__(self):
        super(TeacherVGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.test_accuracy_list = []

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        self.testloader = DataLoader(testset, batch_size=64, shuffle=False)

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 128 * 8 * 8)
        x = self.classifier(x)
        return x

    def train_model(self, train_loader, criterion, optimizer, n_epochs=10, print_every=500,
                    model_name="Alexnet_surrogate_model"):
        model, best_val_accuracy, best_model_state_dict, start_epoch = prepare_for_training(self, model_name, optimizer)
        for epoch in range(start_epoch, n_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            start_time = time.time()
            start_time_epoch = time.time()
            for i, (inputs, labels) in enumerate(train_loader, 0):
                inputs = inputs.view(inputs.shape[0], 3, 32, 32).detach().clone()
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if torch.cuda.is_available():
                    outputs = model(inputs)
                else:
                    outputs = self(inputs)
                try:
                    labels = labels.view(-1, outputs.size()[1]).float().detach().clone().requires_grad_(True)
                except:
                    labels = torch.tensor(list(
                        map(lambda x: one_hot_encode(x, outputs.size()[1]), labels))).detach().clone().requires_grad_(
                        True)

                # # Get the maximum values along the third dimension (dimension with size 10)
                # fitnesses, _ = torch.max(labels_copy, dim=2)
                # # Sum all the elements in the tensor to get a scalar
                # sum_fitnesses = torch.sum(torch.max(-torch.log(fitnesses), torch.tensor(100))).requires_grad_()
                outputs, labels = outputs.to(self.device), labels.to(self.device)
                loss = criterion(outputs, labels)  # + sum_fitnesses

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                _, labels = torch.max(labels.data, 1)
                correct += (predicted == labels).sum().item()

                if i % print_every == print_every - 1:
                    finish_time = time.time()
                    total_time = finish_time - start_time
                    print('[%d, %5d] loss: %.3f accuracy: %.3f the time it took: %.3f seconds' % (
                        epoch + 1, i + 1, running_loss / print_every, 100 * correct / total, total_time))
                    running_loss = 0.0
                    correct = 0
                    total = 0
                    start_time = time.time()
            finish_time_epoch = time.time()
            total_time_epoch = finish_time_epoch - start_time_epoch
            print(f'Epoch {epoch + 1} took {total_time_epoch} seconds')
            validation_accuracy = self.validate_model()
            self.test_accuracy_list.append(validation_accuracy)
            # Save model after each epoch
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_accuracy_list': self.test_accuracy_list
            }
            directory = f"checkpoints/{self.__class__.__name__}"
            torch.save(checkpoint, f'./{directory}/{model_name}.pth')
            if validation_accuracy > best_val_accuracy:
                best_val_accuracy = validation_accuracy
                best_model_state_dict = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'test_accuracy_list': self.test_accuracy_list
                }
                torch.save(best_model_state_dict, f'./{directory}/best_accuracy_{model_name}.pth')
            print("Saved model checkpoint!")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(
            f"The maximal accuracy during training was: {max(self.test_accuracy_list)} on epoch: {self.test_accuracy_list.index(max(self.test_accuracy_list))}")
        # self.plot_accuracy_graph()

        accuracy_list = self.test_accuracy_list
        # Plotting using Seaborn
        sns.set(style="darkgrid")
        sns.lineplot(x=range(len(accuracy_list)), y=accuracy_list, marker='X')

        # # Add accuracy values as annotations
        # for i, accuracy in enumerate(accuracy_list):
        #     plt.annotate(f"{accuracy:.2f}", (i, accuracy), textcoords="offset points", xytext=(0, 10), ha='center')

        # Set labels and title
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Over Epochs")

        # Set x-axis ticks as integers
        # plt.xticks(range(len(accuracy_list)))

        # Display the plot
        plt.show()

    def validate_model(self):
        # Test the model
        correct = 0
        total = 0
        model = None
        if torch.cuda.is_available():
            num_of_gpus = torch.cuda.device_count()
            gpu_list = list(range(num_of_gpus))
            model = nn.DataParallel(self, device_ids=gpu_list).to(self.device)
        # Don't need to keep track of gradients
        with torch.no_grad():
            for images, labels in self.testloader:
                images = images.view(images.shape[0], 3, 32, 32).detach().clone()
                if torch.cuda.is_available():
                    outputs = model(images)
                else:
                    outputs = self(images)
                images, labels, outputs = images.to(self.device), labels.to(self.device), outputs.to(self.device)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Accuracy on test set is: {100 * correct / total}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return correct / total

    def test_model(self):
        # Test the model
        # self.eval()
        correct = 0
        total = 0
        model = None
        if torch.cuda.is_available():
            num_of_gpus = torch.cuda.device_count()
            gpu_list = list(range(num_of_gpus))
            model = nn.DataParallel(self, device_ids=gpu_list).to(self.device)
        # Don't need to keep track of gradients
        with torch.no_grad():
            for images, labels in self.testloader:
                images = images.view(images.shape[0], 3, 32, 32).detach().clone()
                if torch.cuda.is_available():
                    outputs = model(images)
                else:
                    outputs = self(images)
                images, labels, outputs = images.to(self.device), labels.to(self.device), outputs.to(self.device)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Accuracy on test set is: {100 * correct / total}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # self.train()
        return correct / total


class ImprovedAlexnetSurrogate(nn.Module):
    def __init__(self, num_classes=10):
        super(ImprovedAlexnetSurrogate, self).__init__()

        self.conv1 = nn.Conv2d(3, 48, kernel_size=5, stride=1, padding=2)
        self.conv1.bias.data.normal_(0, 0.01)
        self.conv1.bias.data.fill_(0)
        self.relu = nn.ReLU()
        self.lrn = nn.LocalResponseNorm(2)
        self.pad = nn.MaxPool2d(3, stride=2)
        self.batch_norm1 = nn.BatchNorm2d(48, eps=0.001)

        self.layer1 = self.make_residue_block(48, 128, stride=1, with_se=True)
        self.layer2 = self.make_residue_block(128, 192, stride=2, with_se=True)
        self.layer3 = self.make_residue_block(192, 192, stride=1, with_se=True)
        self.layer4 = self.make_residue_block(192, 128, stride=2, with_se=True)
        # Additional layers
        self.layer5 = self.make_residue_block(128, 128, stride=1, with_se=True)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 512)
        self.fc1.bias.data.normal_(0, 0.01)
        self.fc1.bias.data.fill_(0)
        self.drop = nn.Dropout(p=0.5)
        self.batch_norm2 = nn.BatchNorm1d(512, eps=0.001)

        self.fc2 = nn.Linear(512, 256)
        self.fc2.bias.data.normal_(0, 0.01)
        self.fc2.bias.data.fill_(0)
        self.batch_norm3 = nn.BatchNorm1d(256, eps=0.001)

        self.fc3 = nn.Linear(256, num_classes)
        self.fc3.bias.data.normal_(0, 0.01)
        self.fc3.bias.data.fill_(0)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.test_accuracy_list = []

        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        self.testloader = DataLoader(testset, batch_size=512, shuffle=False)

    def make_residue_block(self, in_channels, out_channels, stride, with_se=False):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if with_se:
            layers.append(SEBlock(out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.lrn(x)
        x = self.pad(x)
        x = self.batch_norm1(x)

        # Residue blocks with SE
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)  # Additional layer

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = self.drop(x)
        x = self.batch_norm2(x)

        x = self.fc2(x)
        x = self.batch_norm3(x)

        x = self.fc3(x)

        return x

    def train_model(self, train_loader, criterion, optimizer, n_epochs=10, print_every=500,
                    model_name="Alexnet_surrogate_model"):
        model, best_val_accuracy, best_model_state_dict, start_epoch = prepare_for_training(self, model_name, optimizer)
        for epoch in range(start_epoch, n_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            start_time = time.time()
            start_time_epoch = time.time()
            for i, (inputs, labels) in enumerate(train_loader, 0):
                inputs = inputs.view(inputs.shape[0], 3, 32, 32).detach().clone()
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if torch.cuda.is_available():
                    outputs = model(inputs)
                else:
                    outputs = self(inputs)
                try:
                    labels = labels.view(-1, outputs.size()[1]).float().detach().clone().requires_grad_(True)
                except:
                    labels = torch.tensor(list(
                        map(lambda x: one_hot_encode(x, outputs.size()[1]), labels))).detach().clone().requires_grad_(
                        True)

                # # Get the maximum values along the third dimension (dimension with size 10)
                # fitnesses, _ = torch.max(labels_copy, dim=2)
                # # Sum all the elements in the tensor to get a scalar
                # sum_fitnesses = torch.sum(torch.max(-torch.log(fitnesses), torch.tensor(100))).requires_grad_()
                outputs, labels = outputs.to(self.device), labels.to(self.device)
                loss = criterion(outputs, labels)  # + sum_fitnesses

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                _, labels = torch.max(labels.data, 1)
                correct += (predicted == labels).sum().item()

                if i % print_every == print_every - 1:
                    finish_time = time.time()
                    total_time = finish_time - start_time
                    print('[%d, %5d] loss: %.3f accuracy: %.3f the time it took: %.3f seconds' % (
                        epoch + 1, i + 1, running_loss / print_every, 100 * correct / total, total_time))
                    running_loss = 0.0
                    correct = 0
                    total = 0
                    start_time = time.time()
            finish_time_epoch = time.time()
            total_time_epoch = finish_time_epoch - start_time_epoch
            print(f'Epoch {epoch + 1} took {total_time_epoch} seconds')
            validation_accuracy = self.validate_model()
            self.test_accuracy_list.append(validation_accuracy)
            # Save model after each epoch
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_accuracy_list': self.test_accuracy_list
            }
            directory = f"checkpoints/{self.__class__.__name__}"
            torch.save(checkpoint, f'./{directory}/{model_name}.pth')
            if validation_accuracy > best_val_accuracy:
                best_val_accuracy = validation_accuracy
                best_model_state_dict = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'test_accuracy_list': self.test_accuracy_list
                }
                torch.save(best_model_state_dict, f'./{directory}/best_accuracy_{model_name}.pth')
            print("Saved model checkpoint!")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(
            f"The maximal accuracy during training was: {max(self.test_accuracy_list)} on epoch: {self.test_accuracy_list.index(max(self.test_accuracy_list))}")
        # self.plot_accuracy_graph()

        accuracy_list = self.test_accuracy_list
        # Plotting using Seaborn
        sns.set(style="darkgrid")
        sns.lineplot(x=range(len(accuracy_list)), y=accuracy_list, marker='X')

        # # Add accuracy values as annotations
        # for i, accuracy in enumerate(accuracy_list):
        #     plt.annotate(f"{accuracy:.2f}", (i, accuracy), textcoords="offset points", xytext=(0, 10), ha='center')

        # Set labels and title
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Over Epochs")

        # Set x-axis ticks as integers
        # plt.xticks(range(len(accuracy_list)))

        # Display the plot
        plt.show()

    def validate_model(self):
        # Test the model
        correct = 0
        total = 0
        model = None
        if torch.cuda.is_available():
            num_of_gpus = torch.cuda.device_count()
            gpu_list = list(range(num_of_gpus))
            model = nn.DataParallel(self, device_ids=gpu_list).to(self.device)
        # Don't need to keep track of gradients
        with torch.no_grad():
            for images, labels in self.testloader:
                images = images.view(images.shape[0], 3, 32, 32).detach().clone()
                if torch.cuda.is_available():
                    outputs = model(images)
                else:
                    outputs = self(images)
                images, labels, outputs = images.to(self.device), labels.to(self.device), outputs.to(self.device)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Accuracy on test set is: {100 * correct / total}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return correct / total

    def test_model(self):
        # Test the model
        # self.eval()
        correct = 0
        total = 0
        model = None
        if torch.cuda.is_available():
            num_of_gpus = torch.cuda.device_count()
            gpu_list = list(range(num_of_gpus))
            model = nn.DataParallel(self, device_ids=gpu_list).to(self.device)
        # Don't need to keep track of gradients
        with torch.no_grad():
            for images, labels in self.testloader:
                images = images.view(images.shape[0], 3, 32, 32).detach().clone()
                if torch.cuda.is_available():
                    outputs = model(images)
                else:
                    outputs = self(images)
                images, labels, outputs = images.to(self.device), labels.to(self.device), outputs.to(self.device)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Accuracy on test set is: {100 * correct / total}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # self.train()
        return correct / total


def train_and_validate(model, optimizer, trainloader, testloader, criterion, num_epochs=10,
                       save_path='best_model.pth'):
    best_val_accuracy = 0.0
    directory = "checkpoints/TargetModels"
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        model.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            if criterion.__class__.__name__ == 'MSELoss':
                # Apply the one_hot_encode function to each element of the tensor
                one_hot_encoded_tensor = [one_hot_encode(x.item(), 10) for x in labels]
                one_hot_encoded_tensor = torch.tensor(one_hot_encoded_tensor)
                loss = criterion(outputs, one_hot_encoded_tensor)
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(trainloader)
        train_accuracy = correct / total

        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for data in testloader:
                inputs, labels = data
                outputs = model(inputs)
                if criterion.__class__.__name__ == 'MSELoss':
                    # Apply the one_hot_encode function to each element of the tensor
                    one_hot_encoded_tensor = [one_hot_encode(x.item(), 10) for x in labels]
                    one_hot_encoded_tensor = torch.tensor(one_hot_encoded_tensor)
                    loss = criterion(outputs, one_hot_encoded_tensor)
                else:
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(testloader)
        val_accuracy = val_correct / val_total

        print(
            f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        # Save the best model
        path = f"{directory}/{save_path}"
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), path)
            print(f'Model improved and saved to {path}')

    print('Finished Training')


def evaluate_model(model, testloader):
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy of {model.__class__.__name__}: {accuracy}%')
    model.train()
