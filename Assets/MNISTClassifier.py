import time

import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST

from BAM_Code.Utility import prepare_for_training


class MNISTClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # BatchNorm layer after the first convolution
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)  # BatchNorm layer after the second convolution
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)  # BatchNorm layer after the third convolution
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.bn4 = nn.BatchNorm1d(
            256
        )  # BatchNorm layer after the first fully connected layer
        self.fc2 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(
            128
        )  # BatchNorm layer after the second fully connected layer
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        test_dataset = FashionMNIST(
            root="./data", train=False, transform=transform, download=True
        )
        self.testloader = DataLoader(test_dataset, batch_size=512, shuffle=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_accuracy_list = []

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        x = self.softmax(x)
        return x

    def real_forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        x = self.softmax(x)
        return x

    def train_model(
        self,
        train_loader,
        criterion,
        optimizer,
        n_epochs=10,
        print_every=500,
        model_name="mnist_model",
    ):
        model, best_val_accuracy, best_model_state_dict, start_epoch = (
            prepare_for_training(self, model_name, optimizer)
        )
        for epoch in range(start_epoch, n_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            start_time = time.time()
            start_time_epoch = time.time()
            for i, (inputs, labels) in enumerate(train_loader, 0):
                inputs = inputs.view(inputs.shape[0], 1, 28, 28).detach().clone()
                # inputs = inputs.view(inputs.shape[0], -1).detach().clone()
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    outputs = model(inputs)
                else:
                    outputs = self(inputs)
                try:
                    labels = (
                        labels.view(-1, outputs.size()[1])
                        .float()
                        .detach()
                        .clone()
                        .requires_grad_(True)
                    )
                except:
                    labels = (
                        torch.tensor(
                            list(
                                map(
                                    lambda x: one_hot_encode(x, outputs.size()[1]),
                                    labels,
                                )
                            )
                        )
                        .detach()
                        .clone()
                        .requires_grad_(True)
                    )

                outputs, labels = outputs.to(self.device), labels.to(self.device)
                loss = criterion(outputs, labels)  # + sum_fitnesses
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
                    print(
                        "[%d, %5d] loss: %.3f accuracy: %.3f the time it took: %.3f seconds"
                        % (
                            epoch + 1,
                            i + 1,
                            running_loss / print_every,
                            100 * correct / total,
                            total_time,
                        )
                    )
                    running_loss = 0.0
                    correct = 0
                    total = 0
                    start_time = time.time()
            finish_time_epoch = time.time()
            total_time_epoch = finish_time_epoch - start_time_epoch
            print(f"Epoch {epoch + 1} took {total_time_epoch} seconds")
            validation_accuracy = self.validate_model()
            self.test_accuracy_list.append(validation_accuracy)
            # Save model after each epoch
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "test_accuracy_list": self.test_accuracy_list,
            }
            directory = f"checkpoints/{self.__class__.__name__}"
            torch.save(checkpoint, f"./{directory}/{model_name}.pth")
            if validation_accuracy > best_val_accuracy:
                best_val_accuracy = validation_accuracy
                best_model_state_dict = {
                    "epoch": epoch + 1,
                    "model_state_dict": self.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "test_accuracy_list": self.test_accuracy_list,
                }
                torch.save(
                    best_model_state_dict,
                    f"./{directory}/best_accuracy_{model_name}.pth",
                )
            print("Saved model checkpoint!")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(
            f"The maximal accuracy during training was: {max(self.test_accuracy_list)} on epoch: {self.test_accuracy_list.index(max(self.test_accuracy_list))}"
        )

    def plot_accuracy_graph(self):
        accuracy_list = self.test_accuracy_list
        # Plotting using Seaborn
        sns.set(style="darkgrid")
        sns.lineplot(x=range(len(accuracy_list)), y=accuracy_list, marker="X")

        # Set labels and title
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Over Epochs")

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
                images = images.view(images.shape[0], 1, 28, 28).detach().clone()
                # images = images.view(images.shape[0], -1).detach().clone()
                if torch.cuda.is_available():
                    outputs = model(images)
                else:
                    outputs = self(images)
                images, labels, outputs = (
                    images.to(self.device),
                    labels.to(self.device),
                    outputs.to(self.device),
                )
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
        # Define a transform to normalize the data
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        self.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            for images, labels in self.testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = self(images)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Accuracy: {100 * correct / total}")
        self.train(True)
        return correct / total

    @staticmethod
    def generate_random_data(num, tile_size=5, num_tiles=50):
        image_size = (1, 1, 28, 28)
        data = [torch.rand(image_size) for _ in range(num)]
        return data


def one_hot_encode(x, num_classes):
    vec = [0.0] * num_classes
    vec[x] = 1.0
    return vec
