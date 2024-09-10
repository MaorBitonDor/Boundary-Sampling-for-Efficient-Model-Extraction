import torch
from torch import nn

from Alexnet import Alexnet
from AlexnetSurrogate import AlexnetSurrogate
from Config import Config
from Utility import (
    prepare_config_and_log,
    generate_random_data_cifar_old,
    BAM_main_algorithm,
)

if __name__ == "__main__":
    # release all GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    prepare_config_and_log()
    config = Config.instance
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_of_classes = config["num_of_classes"]
    name = "teacher_alexnet_for_cifar10"
    ckpt_path = "teacher_alexnet_for_cifar10_state_dict"
    alex = Alexnet(name, num_of_classes)
    alex.load_state_dict(torch.load(ckpt_path, map_location=device))
    alex.to(device)
    print(f"Accuracy of target model is:")
    alex.test_model()
    k = config["k"]
    epsilon = config["epsilon"]
    population_size = config["population_size"]
    generations = config["generations"]
    search_spread = config["search_spread"]
    epochs = config["epochs"]
    dont_get_from_dist = config["dont_get_from_disk"]
    criterion = nn.MSELoss().to(device)
    optimizer_name = config["optimizer_name"]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    surrogate_model = BAM_main_algorithm(
        alex,
        AlexnetSurrogate,
        criterion,
        generate_random_data_cifar_old,
        num_of_classes,
        k,
        epsilon,
        population_size,
        generations,
        search_spread,
        epochs,
        optimizer_name,
    )

    surrogate_acc = surrogate_model.test_model()
    print(f"The surrogate model accuracy is {surrogate_acc}")
