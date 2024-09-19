import time

import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets

from BAM_Code.Config import Config
from Assets.MNISTClassifier import one_hot_encode
from BAM_Code.Utility import prepare_for_training, prepare_config_and_log


# Define the CNN model
class TinyImagenteClassifier(nn.Module):
    def __init__(self, num_classes=200, model_type="resnet18"):
        super(TinyImagenteClassifier, self).__init__()
        # resnet = models.resnet50(weights=None)
        self.image_size = 64
        if model_type == "resnet50":
            backbone = models.resnet50(
                pretrained=False
            )  # Set pretrained to True if needed
        elif model_type == "resnet50_pretrained":
            backbone = models.resnet50(pretrained=True)
        elif model_type == "resnet18_pretrained":
            backbone = models.resnet18(pretrained=True)
        elif model_type == "resnet18":
            backbone = models.resnet18(pretrained=False)
        else:
            backbone = models.resnet101(
                pretrained=False
            )  # Set pretrained to True if needed
        # Remove the fully connected layers at the end
        self.features = nn.Sequential(*list(backbone.children())[:-2])

        # Add adaptive pooling instead of fixed size
        self.global_avg_pooling = nn.AdaptiveAvgPool2d((1, 1))

        # Add a fully connected layer with the desired number of output classes
        num_ftrs = backbone.fc.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Define transforms for data augmentation
        train_transform = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Load the training dataset
        prepare_config_and_log()
        batch_size = Config.instance["batch_size"]
        train_dir = "../data/TinyImagenet/tiny-imagenet-200/train"
        valid_dir = "../data/TinyImagenet/tiny-imagenet-200/test"
        train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
        self.train_loader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=16
        )
        self.test_accuracy_list = []
        # Load the validation dataset
        valid_transform = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.valid_dataset = datasets.ImageFolder(
            root=valid_dir, transform=valid_transform
        )
        self.testloader = DataLoader(
            dataset=self.valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=16,
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def train_model(
        self,
        train_loader,
        criterion,
        optimizer,
        n_epochs=10,
        print_every=500,
        model_name="TinyImagnet_model",
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
                inputs = (
                    inputs.view(inputs.shape[0], 3, self.image_size, self.image_size)
                    .detach()
                    .clone()
                )
                inputs, labels = inputs.to(self.device), labels.to(self.device)
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
                loss = criterion(outputs, labels)
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
            learning_rate = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch + 1} took {total_time_epoch:.3f} seconds, Learning rate: {learning_rate}"
            )
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
        self.plot_accuracy_graph()

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
        correct = 0
        total = 0
        model = None
        if torch.cuda.is_available():
            num_of_gpus = torch.cuda.device_count()
            gpu_list = list(range(num_of_gpus))
            model = nn.DataParallel(self, device_ids=gpu_list).to(self.device)
        # Don't need to keep track of gradients
        batch_size = Config.instance["batch_size"]
        testloader = DataLoader(
            dataset=self.valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=16,
        )
        with torch.no_grad():
            for images, labels in testloader:
                images = (
                    images.view(images.shape[0], 3, self.image_size, self.image_size)
                    .detach()
                    .clone()
                )
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

        print(f"Accuracy on test set is: {100 * correct / total:.3f}%")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return correct / total

    def test_model(self):
        self.eval()
        correct = 0
        total = 0
        test_loss = 0
        model = None
        if torch.cuda.is_available():
            num_of_gpus = torch.cuda.device_count()
            gpu_list = list(range(num_of_gpus))
            model = nn.DataParallel(self, device_ids=gpu_list).to(self.device)
        prepare_config_and_log()
        batch_size = Config.instance["batch_size"]
        testloader = DataLoader(
            dataset=self.valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=16,
        )
        with torch.no_grad():
            for images, labels in testloader:
                images = (
                    images.view(images.shape[0], 3, self.image_size, self.image_size)
                    .detach()
                    .clone()
                )
                if torch.cuda.is_available():
                    outputs = model(images)
                else:
                    outputs = self(images)
                images, labels, outputs = (
                    images.to(self.device),
                    labels.to(self.device),
                    outputs.to(self.device),
                )
                loss = nn.CrossEntropyLoss()(outputs, labels)
                test_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Accuracy on test set is: {100 * correct / total}%")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.train(True)
        return test_loss / total, correct / total

    def step_lr(self, lr_max, epoch, num_epochs):
        """Step Scheduler"""
        ratio = epoch / float(num_epochs)
        if ratio < 0.3:
            return lr_max
        elif ratio < 0.6:
            return lr_max * 0.2
        elif ratio < 0.8:
            return lr_max * 0.2 * 0.2
        else:
            return lr_max * 0.2 * 0.2 * 0.2

    def lr_scheduler(self, epochs, lr_mode, lr_min, lr_max):
        """Learning Rate Scheduler Options"""
        if lr_mode == 1:
            lr_schedule = lambda t: np.interp(
                [t], [0, epochs // 2, epochs], [lr_min, lr_max, lr_min]
            )[0]
        elif lr_mode == 0:
            lr_schedule = lambda t: self.step_lr(lr_max, t, epochs)
        return lr_schedule
