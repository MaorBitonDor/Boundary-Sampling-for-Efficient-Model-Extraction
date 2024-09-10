# Boundary Sampling for Efficient Model Extraction
## Overview
This repository contains an implementation of a model extraction algorithm that harnesses the power of evolutionary algorithms to efficiently extract the victim model with zero assumptions on the victim model training data. The algorithm is designed to extract black box models by only querying the model and sending input and receiving the output.

![Overview Image](main_fig.png)

## Getting Started

To use the algorithm, follow these steps:

1. **Clone the Repository:** Clone this repository to your local machine using `git clone`.
2. **Install Dependencies:** Install the required dependencies listed in `requirements.txt`.
3. **Configure Parameters:** Adjust the algorithm parameters you send to the algorithm to suit your needs.
4. **Run the Algorithm:** Execute the main script to start the model extraction process.
5. **Evaluate Results:** Evaluate the extracted models based on their performance metrics and refine as necessary.

## Example Usage

```python
from Config import Config
from Utility import (
    BAM_main_algorithm_tabular,
    prepare_config_and_log,
    generate_random_tabular_data_function, # This is the function that will generate a random data in the shape of the input for victim model.
)

prepare_config_and_log()
config = Config.instance # Setup the configuration file

# Define BAM algorithm parameters
num_of_classes = 12
k = 3000
epsilon = 0.05
population_size = 10000
generations = 30
search_spread = 10

# Load the victim model
victim_model = load_model() # This will need to be replaced by the user to load the relevant model to extract

# Run BAM algorithm
surrogate_model = BAM_main_algorithm_tabular(
    victim_model,
    SurrogateModelClass,
    generate_random_tabular_data_function,
    num_of_classes=num_of_classes,
    k=k,
    epsilon=epsilon,
    population_size=population_size,
    generations=generations,
    search_spread=search_spread,
)

surrogate_acc = surrogate_model.test_model() # This will need to be replaced by the user test function
print(f"The surrogate model accuracy is {surrogate_acc}")

