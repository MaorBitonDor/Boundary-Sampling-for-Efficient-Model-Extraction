import os
import time

import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import models, transforms, datasets

from Config import Config
from Utility import prepare_for_training, prepare_config_and_log
from MNISTClassifier import one_hot_encode


# Define the CNN model
class MonkeyClassifier(nn.Module):
    def __init__(self, num_classes=10, model_type="resnet18"):
        super(MonkeyClassifier, self).__init__()
        # resnet = models.resnet50(weights=None)
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
                transforms.RandomResizedCrop(size=(224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Load the training dataset
        prepare_config_and_log()
        batch_size = Config.instance["batch_size"]
        train_dir = "data/MonkeySpicies/training/training"
        valid_dir = "data/MonkeySpicies/validation/validation"
        train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
        self.train_loader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=16
        )
        self.test_accuracy_list = []
        # Load the validation dataset
        valid_transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        self.valid_dataset = datasets.ImageFolder(
            root=valid_dir, transform=valid_transform
        )
        # Assuming self.valid_dataset is your original dataset
        datasets_list = [
            self.valid_dataset for _ in range(10)
        ]  # List containing 10 references to your dataset
        self.valid_dataset = ConcatDataset(datasets_list)
        prepare_config_and_log()
        # batch_size = int(len(self.valid_dataset) / 50)
        # batch_size = Config.instance["batch_size"]
        batch_size = 32
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
        model_name="MonkeySpecies_model",
    ):
        model, best_val_accuracy, best_model_state_dict, start_epoch = (
            prepare_for_training(self, model_name, optimizer)
        )
        # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i for i in range(100, 300, 50)],
        #                                                     last_epoch=-1)
        # lr_scheduler = self.lr_scheduler(n_epochs, 1, 0., 0.1)
        for epoch in range(start_epoch, n_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            if torch.cuda.is_available():
                num_of_gpus = torch.cuda.device_count()
                gpu_list = list(range(num_of_gpus))
                model = nn.DataParallel(self, device_ids=gpu_list).to(self.device)
                # model.train(True)
            start_time = time.time()
            start_time_epoch = time.time()
            # Manually clip the learning rate to a lower limit
            # lower_limit = 1e-5
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = max(param_group['lr'], lower_limit)
            # if epoch == 5:
            #     batch_size = 512
            #     train_loader = DataLoader(train_loader.dataset, batch_size=batch_size, shuffle=False, pin_memory=True,
            #                               num_workers=32)
            for i, (inputs, labels) in enumerate(train_loader, 0):
                inputs = inputs.view(inputs.shape[0], 3, 224, 224).detach().clone()
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
                # outputs *= 1e2
                loss = criterion(outputs, labels)  # + sum_fitnesses
                # lr = lr_scheduler(epoch + (i + 1) / len(train_loader))
                # optimizer.param_groups[0].update(lr=lr)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                _, labels = torch.max(labels.data, 1)
                correct += (predicted == labels).sum().item()
                del inputs
                del labels
                del outputs
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
            # lr_scheduler(epoch)
            learning_rate = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch + 1} took {total_time_epoch:.3f} seconds, Learning rate: {learning_rate}"
            )
            validation_accuracy = self.validate_model()
            loss_test, test_acc = self.test_model()
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
            if test_acc > best_val_accuracy:
                best_val_accuracy = test_acc
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
            checkpoint = torch.load(f"./{directory}/{model_name}.pth")[
                "model_state_dict"
            ]
            self.load_state_dict(checkpoint)
            print("Saved model checkpoint!")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        directory = f"checkpoints/{self.__class__.__name__}"
        checkpoint = torch.load(f"./{directory}/best_accuracy_{model_name}.pth")[
            "model_state_dict"
        ]
        self.load_state_dict(checkpoint)
        print(
            f"The maximal accuracy during training was: {max(self.test_accuracy_list)} on epoch: {self.test_accuracy_list.index(max(self.test_accuracy_list))}"
        )
        self.plot_accuracy_graph()

    def plot_accuracy_graph(self):
        accuracy_list = self.test_accuracy_list
        # Plotting using Seaborn
        sns.set(style="darkgrid")
        sns.lineplot(x=range(len(accuracy_list)), y=accuracy_list, marker="X")

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
        # self.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        correct = 0
        total = 0
        test_loss = 0
        model = None
        if torch.cuda.is_available():
            num_of_gpus = torch.cuda.device_count()
            gpu_list = list(range(num_of_gpus))
            model = nn.DataParallel(self, device_ids=gpu_list).to(self.device)
            # model.eval()

        batch_size = len(self.valid_dataset)
        self.testloader = DataLoader(
            dataset=self.valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=16,
        )
        with torch.no_grad():
            for images, labels in self.testloader:
                images = (
                    images.view(images.shape[0], 3, 224, 224)
                    .detach()
                    .clone()
                    .to(self.device)
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

        print(f"Accuracy on test set is: {100 * correct / total:.3f}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # model.train(True)
        # self.train(True)
        return correct / total

    def test_model(self):
        self.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        correct = 0
        total = 0
        test_loss = 0
        model = None
        if torch.cuda.is_available():
            num_of_gpus = torch.cuda.device_count()
            gpu_list = list(range(num_of_gpus))
            model = nn.DataParallel(self, device_ids=gpu_list).to(self.device)
            # model.eval()
        with torch.no_grad():
            for images, labels in self.testloader:
                images = (
                    images.view(images.shape[0], 3, 224, 224)
                    .detach()
                    .clone()
                    .to(self.device)
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

        print(f"Accuracy on test set is: {100 * correct / total:.3f}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # model.train(True)
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


# # Define data transformations
# image_size = 224
# image_crop = 224
# train_transform = transforms.Compose([
#     transforms.Resize((image_size, image_size)),  # Resize the image to 224x224
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
# ])
#
# val_transform = transforms.Compose([
#     transforms.Resize((image_size, image_size)),  # Resize the image to 224x224
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
# ])
# # train_transform = transforms.Compose([
# #     transforms.Resize(image_size),
# #     # transforms.CenterCrop(image_crop),
# #     # transforms.RandomHorizontalFlip(),
# #     # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
# #     # transforms.RandomRotation(degrees=45),
# #     transforms.ToTensor(),
# #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
# # ])
# #
# # val_transform = transforms.Compose([
# #     transforms.Resize(image_size),
# #     transforms.CenterCrop(image_crop),
# #     transforms.ToTensor(),
# #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
# # ])
#
# # Load the dataset
# torch.manual_seed(42)
# train_dir = 'data/MonkeySpicies/training/training'
# valid_dir = 'data/MonkeySpicies/validation/validation'
# train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
# test_dataset = datasets.ImageFolder(root=valid_dir, transform=val_transform)
#
# # Create data loaders
# prepare_config_and_log()
# batch_size = Config.instance["batch_size"]
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
#
# # Initialize the model, loss function, and optimizer
# model = MonkeyClassifier(num_classes=10, model_type="resnet18")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# criterion = nn.MSELoss()
# optimizer = optim.AdamW(model.parameters(), lr=0.005)
# model.train_model(train_loader, criterion, optimizer, 1000)
# model.test_model()
# #
# # # Training loop
# # epochs = 500
# # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # model.to(device)
# # if torch.cuda.is_available():
# #     num_of_gpus = torch.cuda.device_count()
# #     gpu_list = list(range(num_of_gpus))
# #     model = nn.DataParallel(model, device_ids=gpu_list).to(device)
# # for epoch in range(epochs):
# #     model.train()
# #     start_time = time.time()
# #     for inputs, labels in train_loader:
# #         inputs, labels = inputs.to(device), labels.to(device)
# #
# #         optimizer.zero_grad()
# #         outputs = model(inputs)
# #         try:
# #             labels = labels.view(-1, outputs.size()[1]).float().detach().clone().requires_grad_(True)
# #         except:
# #             labels = torch.tensor(list(
# #                 map(lambda x: one_hot_encode(x, outputs.size()[1]), labels))).detach().clone().requires_grad_(
# #                 True).to(device)
# #         loss = criterion(outputs, labels)
# #         loss.backward()
# #         optimizer.step()
# #
# #     # Validation loop
# #     finish_time = time.time()
# #     total_time = finish_time - start_time
# #     model.eval()
# #     correct = 0
# #     total = 0
# #
# #     with torch.no_grad():
# #         for inputs, labels in test_loader:
# #             inputs, labels = inputs.to(device), labels.to(device)
# #             outputs = model(inputs)
# #             # try:
# #             #     labels_loss = labels.view(-1, outputs.size()[1]).float().detach().clone().requires_grad_(True)
# #             # except:
# #             #     labels_loss = torch.tensor(list(
# #             #         map(lambda x: one_hot_encode(x, outputs.size()[1]), labels))).detach().clone().requires_grad_(
# #             #         True).to(device)
# #             _, predicted = torch.max(outputs.data, 1)
# #             total += labels.size(0)
# #             correct += (predicted == labels).sum().item()
# #
# #     accuracy = correct / total
# #     print(f'Epoch {epoch + 1}/{epochs}, Validation Accuracy: {accuracy * 100:.2f}%, It took {total_time}seconds')
# #
# directory = f"checkpoints/TargetModels"
# if model.__class__.__name__ == 'DataParallel':
#     checkpoint = {
#         'model_state_dict': model.module.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict()
#     }
# else:
#     checkpoint = {
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict()
#     }
# model_name = "MonkeyClassifier_target224_4"
# torch.save(checkpoint, f'./{directory}/{model_name}.pth')
