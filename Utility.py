from __future__ import print_function

import warnings
from typing import Optional

import pandas as pd
from art.attacks.evasion import ProjectedGradientDescentPyTorch
from art.estimators.classification import PyTorchClassifier

from BAM_Algorithm import BasicModelGeneticAlgorithm
from Config import Config
from DatasetLoader import DatasetLoader
from SmallDatasetsDatasetLoader import SmallDatasetDatasetLoader

warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 200)

import os
import random
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


def generate_random_data_cifar_old(num_images: int) -> list[torch.Tensor]:
    """
    Generate a list in length of `num_images` of random input data in the shape of (1, 3, 32, 32)

    :param num_images: The number of random images to generate.
    :return: A list of randomly generated tensors representing images.
    """
    image_size = (1, 3, 32, 32)
    data = [torch.rand(image_size).cpu() for _ in range(num_images)]
    return data


def generate_random_data_tiny_imagenet(num_images: int) -> list[torch.Tensor]:
    """
    Generate a list in length of `num_images` of random input data in the shape of (1, 3, 64, 64)

    :param num_images: The number of random images to generate.
    :return: A list of randomly generated tensors representing images.
    """
    return [torch.rand(1, 3, 64, 64).cpu() for _ in range(num_images)]


def generate_random_food11(num_images: int) -> list[torch.Tensor]:
    """
    Generate a list in length of `num_images` of random input data in the shape of (1, 3, 224, 224).

    :param num_images: The number of random images to generate.
    :return: A list of randomly generated tensors representing images.
    """
    # Generate a batch of random images
    batch_of_images = [torch.rand(1, 3, 224, 224) for _ in range(num_images)]
    return batch_of_images


def train_surrogate_model_generic(dataloader: any, num_epochs: int, model_class: classmethod, criterion: any,
                                  optimizer_name: str = "AdamW") -> "SurrogateModel":
    """
    Train a surrogate model using the provided dataloader.

    :param dataloader: The dataloader containing the training data.
    :param num_epochs: The number of epochs to train the model.
    :param model_class: The class of the model to train.
    :param criterion: The loss criterion used for training.
    :param optimizer_name: The name of the optimizer to use (default is "AdamW").
    :return: The trained surrogate model.
    """
    surrogate_model = model_class()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    surrogate_model.to(device)
    if optimizer_name == "Adam":
        optimizer = optim.Adam(surrogate_model.parameters(), lr=1e-2)
    elif optimizer_name == "SGD":
        # optimizer = optim.SGD(surrogate_model.parameters(), lr=0.01, momentum=0.9)
        optimizer = optim.SGD(surrogate_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(surrogate_model.parameters(), lr=1e-2, alpha=0.9, momentum=0.5)
    else:
        optimizer = optim.AdamW(surrogate_model.parameters(), lr=5e-3)

    # Train the model
    start_time = time.time()
    surrogate_model.train_model(dataloader, criterion, optimizer, n_epochs=num_epochs)
    finish_time = time.time()
    total_time = finish_time - start_time
    print(f"The time it took to train the model was: {total_time}seconds")

    return surrogate_model


def create_config() -> None:
    """
    Create a configuration object.
    """
    Config()


def BAM_main_algorithm(model: "VictimModel", model_class: classmethod, criterion: any,
                       random_data_generator_function: callable, num_of_classes: int, k: int = 300,
                       epsilon: float = 0.05, population_size: int = 1000, generations: int = 20,
                       search_spread: int = 10, epochs: int = 50, optimizer_name: str = "AdamW",
                       small_dataset: bool = False, save_path: Optional[str] = None) -> tuple[float, float]:
    """
    This code execute all the whole model extraction process from end to end by first creating inputs that will be near
    the victim model decision boundaries, then training the surrogate model on those inputs and test the surrogate model
    performance.

    :param model: The victim model to copy.
    :param model_class: The class of the surrogate model.
    :param criterion: The loss criterion used for training the surrogate model.
    :param random_data_generator_function: The function used to generate random data for the evolutionary algorithm.
    :param num_of_classes: The number of classes in victim model.
    :param k: The top k individuals that will be the start of the following generation (default is 300).
    :param epsilon: The epsilon value the evolutionary algorithm to use in the fitness function.
    :param population_size: The size of the population for the evolutionary algorithm (default is 1000).
    :param generations: The number of generations for the evolutionary algorithm (default is 20).
    :param search_spread: The search spread parameter for the evolutionary algorithm (default is 10).
    :param epochs: The number of epochs for training the surrogate model (default is 50).
    :param optimizer_name: The name of the optimizer to use (default is "AdamW").
    :param small_dataset: Whether to use a small dataset (default is False).
    :param save_path: The path to save the model (default is None).
    :return: A tuple containing the accuracy of the model and the attack success rate.
    """
    data_directory = Config.instance["data_directory"]
    destination_folder = Config.instance["destination_folder"].format(data_directory=data_directory,
                                                                      model_class=f"{model_class.__name__}",
                                                                      generations=f"{generations}")
    file_path_confidence_batch = Config.instance["file_path_confidence_batch"].format(
        destination_folder=destination_folder, generations=f"{generations}")
    if Config.instance["dont_get_from_disk"]:
        import shutil
        try:
            shutil.rmtree(destination_folder)
            print(f"Folder '{destination_folder}' deleted successfully.")
        except OSError as e:
            print(f"Error deleting folder '{destination_folder}': {e}")
    # Here we are using Evolutionary algorithm in order to extract data to train copy model
    ga = BasicModelGeneticAlgorithm(model, random_data_generator_function, num_of_classes, model_class.__name__)
    if not os.path.exists(file_path_confidence_batch) or Config.instance["dont_get_from_disk"]:
        best_individuals1 = ga.run_genetic_algorithm(generations, k, epsilon, population_size, search_spread)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"The number of queries was: {ga.query_counter}")

    batch_size = Config.instance["batch_size"]
    if small_dataset:
        dataset = SmallDatasetDatasetLoader(destination_folder, file_size=population_size)
    else:
        dataset = DatasetLoader(destination_folder, file_size=population_size)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=32)

    print(f"Now we are training the model with the data extracted by using evolutionary algorithms:")
    surrogate_model = train_surrogate_model_generic(dataloader, epochs, model_class, criterion,
                                                    optimizer_name=optimizer_name)
    surrogate_model.validate_model()
    model_acc = surrogate_model.test_model()

    attack_success_rate = test_trans(surrogate_model, model, criterion, num_of_classes, model.testloader)
    print(attack_success_rate)
    # Save model after each epoch

    if surrogate_model.__class__.__name__ == 'DataParallel':
        checkpoint = {
            'model_state_dict': surrogate_model.module.state_dict()
        }
    else:
        checkpoint = {
            'model_state_dict': surrogate_model.state_dict(),
        }
    if save_path is None:
        model_name = "phase1"
        directory = f"checkpoints/phases_tests/{surrogate_model.__class__.__name__}"
    else:
        model_name = save_path
        directory = f"checkpoints/model_test/{surrogate_model.__class__.__name__}"

    # Check if the directory exists
    if not os.path.exists(directory):
        # Create the directory
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    torch.save(checkpoint, f'./{directory}/{model_name}.pth')
    # del proxy_dataset
    del surrogate_model
    del dataset
    # release all GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model_acc, attack_success_rate


def prepare_config_and_log() -> None:
    """
    Prepare the configuration and logging.
    """
    create_config()


# TODO check if need to stay
def get_new_data_loader(model: "VictimModel", data_loader: DataLoader, device: str) -> tuple[DataLoader, tuple[int]]:
    """
    Get a new data loader with correct predictions.

    :param model: The model to use for predictions.
    :param data_loader: The data loader to use.
    :param device: The device to use (cpu or cuda).
    :return: A tuple containing the new data loader and the input shape.
    """
    correct_images = []
    correct_labels = []
    input_shape = None
    for inputs, labels in data_loader:
        # Assuming inputs is a batch of images
        inputs, labels = inputs.to(device), labels.to(device)
        if input_shape is None:
            input_shape = tuple(inputs.shape[1:])
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        correct_mask = predicted == labels
        correct_images.append(inputs[correct_mask])
        correct_labels.append(labels[correct_mask])

    correct_images = torch.cat(correct_images, dim=0)
    correct_labels = torch.cat(correct_labels, dim=0)

    correct_dataset = TensorDataset(correct_images, correct_labels)
    correct_data_loader = DataLoader(correct_dataset, batch_size=data_loader.batch_size, shuffle=False)
    return correct_data_loader, input_shape


# TODO check if need to stay
def test_trans(surrogate_model: "SurrogateModel", victim: "VictimModel", num_classes: int,
               data_loader: DataLoader) -> float:
    """
    Test the transferability of an attack.

    :param surrogate_model: The surrogate model.
    :param victim: The victim model.
    :param loss: The loss function.
    :param num_classes: The number of classes.
    :param data_loader: The data loader.
    :return: The attack success rate.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    new_data_loader, input_shape = get_new_data_loader(victim, data_loader, device)
    equality_tensor_p = 0
    loss = nn.CrossEntropyLoss()
    for batch, (images, labels) in enumerate(new_data_loader):
        clf = PyTorchClassifier(model=surrogate_model, loss=loss,
                                input_shape=input_shape, nb_classes=num_classes)

        PGD_attack = ProjectedGradientDescentPyTorch(estimator=clf, max_iter=random.randint(10, 20), eps=30 / 255,
                                                     num_random_init=1)  # uses L infinity

        victim = victim.to(device)
        x_n = images.cpu().numpy()

        # random pertubations
        x_p = PGD_attack.generate(x=x_n)
        x_p = torch.from_numpy(x_p)
        x_p = x_p.to(device)
        adv_label_p = victim(x_p).argmax(dim=1)
        # adv_label_p = adv_label_p.to('cpu')

        equality_tensor_p += torch.sum((labels != adv_label_p).int()).item()
    attack_success_rate = equality_tensor_p / len(new_data_loader.sampler)
    return attack_success_rate


def prepare_for_training(self_model: any, model_name: str, optimizer: any) -> tuple[any, float, dict[str, any], int]:
    """
    Prepare for training.

    :param self_model: The model to train.
    :param model_name: The name of the model.
    :param optimizer: The optimizer to use.
    :return: A tuple containing the model, best validation accuracy, best model state dict, and start epoch.
    """
    start_epoch = 0
    directory = f"checkpoints/{self_model.__class__.__name__}"
    if Config.instance["delete_checkpoints"]:
        import shutil
        try:
            shutil.rmtree(directory)
            print(f"Folder '{directory}' deleted successfully.")
        except OSError as e:
            print(f"Error deleting folder '{directory}': {e}")

    # Check if the directory exists
    if not os.path.exists(directory):
        # Create the directory
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")
    # Reload saved model and epoch number
    if os.path.exists(f'./{directory}/{model_name}.pth'):
        checkpoint = torch.load(f'./{directory}/{model_name}.pth')
        self_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        self_model.test_accuracy_list = checkpoint['test_accuracy_list']
        print("Successfully reloaded model checkpoint!")
    else:
        print("Model checkpoint not found. Starting from the beginning...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model = None
    if torch.cuda.is_available():
        num_of_gpus = torch.cuda.device_count()
        gpu_list = list(range(num_of_gpus))
        model = nn.DataParallel(self_model, device_ids=gpu_list).to(self_model.device)
    # Reload saved model and epoch number
    if os.path.exists(f'./{directory}/best_accuracy_{model_name}.pth'):
        best_model_state_dict = torch.load(f'./{directory}/best_accuracy_{model_name}.pth')
        best_val_accuracy = best_model_state_dict['test_accuracy_list'][-1]  # Track the best validation accuracy
        print("Successfully reloaded best model checkpoint!")
    else:
        print("Best model checkpoint not found.")
        best_val_accuracy = 0.0  # Track the best validation accuracy
        best_model_state_dict = None  # Track the state_dict of the best model
    model.to(self_model.device)
    # model.train(True)
    # self_model.to(self_model.device)
    return model, best_val_accuracy, best_model_state_dict, start_epoch
