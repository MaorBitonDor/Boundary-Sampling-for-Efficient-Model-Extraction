import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.parallel
from Config import Config
from Utility import prepare_for_training
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from MNISTClassifier import one_hot_encode


class Alexnet(nn.Module):
    def __init__(self, name="surrogate_model", n_outputs=10):
        super(Alexnet, self).__init__()

        self.name = name
        self.num_classes = n_outputs

        self.conv1 = nn.Conv2d(3, 48, 5, stride=1, padding=2)
        self.conv1.bias.data.normal_(0, 0.01)
        self.conv1.bias.data.fill_(0)

        self.relu = nn.ReLU()
        self.lrn = nn.LocalResponseNorm(2)
        self.pad = nn.MaxPool2d(3, stride=2)

        self.batch_norm1 = nn.BatchNorm2d(48, eps=0.001)

        self.conv2 = nn.Conv2d(48, 128, 5, stride=1, padding=2)
        self.conv2.bias.data.normal_(0, 0.01)
        self.conv2.bias.data.fill_(1.0)

        self.batch_norm2 = nn.BatchNorm2d(128, eps=0.001)

        self.conv3 = nn.Conv2d(128, 192, 3, stride=1, padding=1)
        self.conv3.bias.data.normal_(0, 0.01)
        self.conv3.bias.data.fill_(0)

        self.batch_norm3 = nn.BatchNorm2d(192, eps=0.001)

        self.conv4 = nn.Conv2d(192, 192, 3, stride=1, padding=1)
        self.conv4.bias.data.normal_(0, 0.01)
        self.conv4.bias.data.fill_(1.0)

        self.batch_norm4 = nn.BatchNorm2d(192, eps=0.001)

        self.conv5 = nn.Conv2d(192, 128, 3, stride=1, padding=1)
        self.conv5.bias.data.normal_(0, 0.01)
        self.conv5.bias.data.fill_(1.0)

        self.batch_norm5 = nn.BatchNorm2d(128, eps=0.001)

        self.fc1 = nn.Linear(1152, 512)
        self.fc1.bias.data.normal_(0, 0.01)
        self.fc1.bias.data.fill_(0)

        self.drop = nn.Dropout(p=0.5)

        self.batch_norm6 = nn.BatchNorm1d(512, eps=0.001)

        self.fc2 = nn.Linear(512, 256)
        self.fc2.bias.data.normal_(0, 0.01)
        self.fc2.bias.data.fill_(0)

        self.batch_norm7 = nn.BatchNorm1d(256, eps=0.001)

        self.fc3 = nn.Linear(256, 10)
        self.fc3.bias.data.normal_(0, 0.01)
        self.fc3.bias.data.fill_(0)

        self.soft = nn.Softmax()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.test_accuracy_list = []
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        self.testloader = DataLoader(testset, batch_size=512, shuffle=False)
        # file_path_test = "CIFAR10_testset"
        # if not os.path.exists(file_path_test):
        #     # Define a transform to normalize the data
        #     transform = transforms.Compose([transforms.ToTensor(),
        #                                     transforms.Normalize((0.5,), (0.5,))])
        #     # Download and load the test data
        #     testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        #     self.testloader = DataLoader(testset, batch_size=512, shuffle=False)
        #     torch.save(testset, file_path_test)
        # else:
        #     testset = torch.load(file_path_test)
        #     self.testloader = DataLoader(testset, batch_size=512, shuffle=False)

    def forward(self, x):
        if isinstance(x, np.ndarray):
            # Convert NumPy array to PyTorch tensor
            x = torch.tensor(x)
        x = x.to(self.device)
        layer1 = self.batch_norm1(self.pad(self.lrn(self.relu(self.conv1(x)))))
        layer2 = self.batch_norm2(self.pad(self.lrn(self.relu(self.conv2(layer1)))))
        layer3 = self.batch_norm3(self.relu(self.conv3(layer2)))
        layer4 = self.batch_norm4(self.relu(self.conv4(layer3)))
        layer5 = self.batch_norm5(self.pad(self.relu(self.conv5(layer4))))
        flatten = layer5.view(-1, 128 * 3 * 3)
        fully1 = self.relu(self.fc1(flatten))
        fully1 = self.batch_norm6(self.drop(fully1))
        fully2 = self.relu(self.fc2(fully1))
        fully2 = self.batch_norm7(self.drop(fully2))
        logits = self.fc3(fully2)
        # softmax_val = self.soft(logits)
        # del x
        return logits

    def train_model(self, train_loader, criterion, optimizer, n_epochs=10, print_every=500, model_name="Alexnet_model"):
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
            Config.log.info(
                f"Epoch {epoch + 1} took {total_time_epoch} seconds, Accuracy on test set is: {validation_accuracy}")
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
        self.plot_accuracy_graph()

    def plot_accuracy_graph(self):
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
        correct = 0
        total = 0
        self.eval()
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
        self.train()
        return correct / total
