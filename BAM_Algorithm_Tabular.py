import gc
import math
import os
import random
import time
from statistics import mean
from typing import List, Tuple, Callable

import numpy as np
import torch

from Config import Config


class BasicModelGeneticAlgorithmTabular:
    """
    A model extraction method that uses genetic algorithms to reverse engineer black-box models without assumptions
    about the victim's training data. This class is tailored for extracting neural network models used in
    classification tasks.

    The genetic algorithm simulates natural selection through random data mutations and selections,
    enabling the exploration and exploitation of the input space to effectively approximate the decision boundaries
    of black-box models. Optimized to leverage the parallel computation capabilities of GPUs, this class enhances
    performance and efficiency during the model extraction process.

    :param model (torch.nn.Module): The neural network model that will be victim. This model is expected to be a
             pre-trained classifier whose decision-making process is to be studied or replicated.
    :param random_data_generator_function (callable): Function to generate random data for model inputs. It initiates
            the genetic algorithm with a diverse set of inputs to optimize the model behavior.
    :param num_of_classes (int): Specifies the number of distinct classes the model can classify. It is crucial for
            configuring the output dimensions of any surrogate models used during extraction.
    :param model_name (str): A descriptive name for the model used for logging and managing output files during the
            extraction process.
    """

    def __init__(
        self,
        model: "VictimModel",
        random_data_generator_function: Callable,
        num_of_classes: int,
        model_name: str,
    ) -> None:
        self.model = model

        # Initialize query counter and loss function
        self.query_counter = 0

        # Set the data generator, number of classes, and model name
        self.random_data_generator_function = random_data_generator_function
        self.num_of_classes = num_of_classes
        self.model_name = model_name

    def run_genetic_algorithm(
        self,
        generations: int,
        k: int,
        epsilon: float,
        population_size: int,
        search_spread: float,
    ) -> list:
        """
        Executes the genetic algorithm over a specified number of generations, evolving the population
        towards optimal solutions as defined by the fitness function.

        :param generations: The total number of generations to evolve through.
        :param k: The number of top individuals to select for breeding in each generation.
        :param epsilon: A parameter used in the fitness function to adjust model output probabilities.
        :param population_size: The number of individuals in each generation.
        :param search_spread: The variance or spread of the noise added during generation mutation.
        :return: The final evolved population after all generations have been processed.
        """
        population = self.generate_population(population_size, generations)

        for generation in range(generations):
            gc.collect()  # Optimize memory by clearing garbage
            start_time = time.time()  # Timing each generation for performance metrics

            # Evaluate the fitness of each individual in the population
            fitnesses = self.fitness(population, epsilon)
            fitness_avg = mean(fitnesses)  # Calculate average fitness for analytics
            # self.avg_fitness.append(fitness_avg)  # Store the average fitness

            # Select the top k individuals based on their fitness
            top_k_population, top_k_population_with_fitness = self.select(
                population, fitnesses, k
            )
            gc.collect()  # Clear any garbage left in memory

            # Create a new generation by mutating the selected top k individuals
            new_population = self.create_new_generation_with_noise(
                top_k_population, population, population_size, search_spread
            )
            gc.collect()  # Clean up memory after creating the new population

            # Update the population for the next generation
            population = self.predict_and_create_proxy_dataset(
                new_population, generation, generations
            )
            gc.collect()  # Final garbage collection for the loop

            finish_time = time.time()
            total_time = finish_time - start_time
            print(
                f"Generation number: {generation}, Average fitness: {fitness_avg:.5f}, Time taken: {total_time:.3f} seconds"
            )

        return population

    def fitness(
        self, population: List[Tuple[torch.Tensor, torch.Tensor]], epsilon: float
    ) -> List[float]:
        """
        Evaluate the fitness of each individual in a population based on the model's output confidence levels.

        The goal of this fitness function is to adjust the confidence levels of the model's predictions so they are as close
        to a uniform distribution as possible. This is achieved by minimizing the maximum probability among the predicted
        class probabilities for each sample, post an epsilon adjustment. A lower maximum probability post-adjustment
        indicates a distribution closer to uniform, reflecting higher fitness.

        :param population: A list where each tuple contains an individual's data and the model's prediction
                           for that data.
        :param epsilon: A small value subtracted from predictions to adjust confidence levels.
        :return: A list of fitness values for each individual in the population, where each fitness value
                 is represented as a float indicating the adjusted maximum probability.
        """
        # Concatenate tensors of adjusted predictions
        predictions_tensor = torch.cat([ind[1] - epsilon for ind in population])

        # Compute the maximum probability for each prediction across classes
        max_probs, _ = torch.max(predictions_tensor, dim=1)

        # Ensure fitness values are non-negative by taking the maximum with zero
        new_max_probs = torch.max(max_probs, torch.tensor(0.0))

        # Convert tensor of fitness values to a list of floats for easier handling
        list_of_fitnesses = [
            x.item()
            for x in torch.split(new_max_probs, split_size_or_sections=1, dim=0)
        ]

        return list_of_fitnesses

    def generate_population(self, size: int, generations: int) -> List:
        """
        Generate a population of model data based on random data inputs. This method creates a proxy dataset by
        generating random data, predicting outputs using the model, and processing the results to create a population
        for use in the genetic algorithm.

        The population is intended for further genetic algorithm operations, where the goal is to evolve the population
        towards a more optimal set of inputs with respect to the model's fitness criteria.

        :param size: The number of individuals in the population. This determines the number of random data points generated.
        :param generations: The number of generations for which the population is to be evolved. This parameter is used
                            within the `predict_and_create_proxy_dataset` to track generational data.
        :return: A list representing the population, where each element is generated by the model predictions based on
                 random data inputs.
        """
        # Generate random data for the initial population
        random_data = self.random_data_generator_function(size)

        # Predict outputs and create the initial proxy dataset for the genetic algorithm
        population = self.predict_and_create_proxy_dataset(random_data, 0, generations)

        return population

    def select(
        self, population: List, fitnesses: List[float], k: int
    ) -> Tuple[List, List[Tuple]]:
        """
        Select the top k individuals from the population based on their fitness. This method sorts the population
        by fitness and returns the best k individuals for further genetic operations.

        :param population: The current population of individuals. Each individual is assumed to be a data point
                           that can be evaluated by the model.
        :param fitnesses: A list of fitness scores corresponding to each individual in the population.
                          Higher scores indicate better fitness.
        :param k: The number of top individuals to select from the population.
        :return: A tuple containing two lists:
                 - The first list comprises the top k individuals from the population.
                 - The second list consists of tuples, each containing an individual and its corresponding fitness score.
        """
        # Pair each individual with its corresponding fitness and create a list of tuples
        fitnesses_weights = [(i, f) for i, f in enumerate(fitnesses)]

        # Sort the list of tuples by fitness in ascending order
        sorted_fitnesses = sorted(fitnesses_weights, key=lambda x: x[1])

        # Extract the top k individuals with the highest fitness
        top_k_fitness = sorted_fitnesses[:k]

        # Retrieve the top k individuals from the population
        top_k_population = [population[x[0]] for x in top_k_fitness]

        # Create a list of tuples for the top k individuals with their fitness values
        top_k_population_with_fitness = [
            (population[x[0]], x[1]) for x in top_k_fitness
        ]

        return top_k_population, top_k_population_with_fitness

    def predict_and_create_proxy_dataset(
        self, population: List[torch.Tensor], cur_generation: int, generations: int
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Processes a population through the model to predict outputs and creates a proxy dataset from these predictions.
        The method manages batch processing of data, collates model predictions, and stores the result to disk.

        :param population: A list of tensors representing the current population's input data.
        :param cur_generation: The current generation number, used for tracking and logging.
        :param generations: The total number of generations expected, used for directory structuring.
        :return: A list where each tuple contains a tensor from the population and its corresponding model prediction.
        """
        # Determine the appropriate batch size based on configuration and population size
        max_batch_size = Config.instance["genetic_alg_prediction_max_batch_size"]
        batch_size = (
            max_batch_size if len(population) > max_batch_size else len(population)
        )
        num_samples = len(population)

        model_prediction_list = []

        # Process population in batches
        for i in range(0, num_samples, batch_size):
            batch = torch.cat(population[i : i + batch_size], dim=0).numpy()
            batch_prediction = self.model.predict(
                batch
            )  # Detach predictions to prevent gradient tracking
            model_prediction_list.append(torch.Tensor(batch_prediction))

        # Concatenate all batch predictions into a single tensor
        model_prediction = torch.cat(model_prediction_list, dim=0)
        self.query_counter += model_prediction.shape[
            0
        ]  # Update query counter with the number of samples processed

        # Convert model predictions into a list of tensors
        list_of_confidence = list(
            torch.split(model_prediction, split_size_or_sections=1, dim=0)
        )
        # Pair each input tensor with its corresponding prediction
        cur_dataset = [
            (torch.tensor(population[i]), list_of_confidence[i])
            for i in range(len(list_of_confidence))
        ]

        # Prepare directory for saving the dataset
        data_directory = Config.instance["data_directory"]
        directory = f"{data_directory}/{self.model_name}/{generations}_generations"
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Directory '{directory}' created.")

        # Save the dataset to file
        file_path_confidence = f"{directory}/Toward_proxy_dataset_confidence_{generations}_batch_{cur_generation}"
        save_tensor_list_to_file(cur_dataset, file_path_confidence)

        # Clear GPU cache if available to optimize memory usage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return cur_dataset

    def create_new_generation_with_noise(
        self,
        top_k_population: List[torch.Tensor],
        population: List[torch.Tensor],
        population_size: int,
        search_spread: float,
    ) -> List[torch.Tensor]:
        """
        Generate a new population by introducing noise to the top individuals from the previous generation.
        This simulates mutation in genetic algorithms, helping to explore new areas of the model boundaries prevent
        premature convergence.

        :param top_k_population: The top individuals from the previous generation that have been selected for breeding.
        :param population: The entire population from the previous generation, used for calculating noise bounds.
        :param population_size: The desired size of the new population to be generated.
        :param search_spread: Controls the magnitude of the noise relative to the data spread; larger values result in smaller noise.
        :return: The newly generated population of the same size as defined by `population_size`.
        """
        # Retrieve only the tensor data from the top_k_population
        new_population = [x[0] for x in top_k_population]

        # Calculate the maximum and minimum values across all features in the population
        max_values, _ = torch.max(torch.stack([x[0][0] for x in population]), dim=0)
        min_values, _ = torch.min(torch.stack([x[0][0] for x in population]), dim=0)

        # Define the scale of the noise to be added based on the feature range and the search spread
        ss_vector = ((max_values - min_values) / search_spread).detach()

        # Stack the tensors from new_population to form a single tensor
        population_tensor = torch.cat(new_population)
        population_original_shape = tuple(population_tensor.shape)

        # Calculate the number of copies needed to reach the desired population size
        factor = math.ceil((float(population_size) / float(len(top_k_population))))

        new_population_list = []

        # Generate new individuals by applying scaled noise
        noise_size = 2  # Determines the magnitude of the noise
        for i in range(factor - 1):
            random_number = random.choice([1, -1])
            noise = random_number * (
                noise_size * ss_vector * torch.rand(population_original_shape)
                - 0.5 * noise_size * ss_vector
            )
            new_population_list.append(noise + population_tensor)

        # Split the list of noisy population tensors back into individual tensors
        splitted_population_list = [
            list(torch.split(x, 1, dim=0)) for x in new_population_list
        ]
        flattened_list = [
            item for sublist in splitted_population_list for item in sublist
        ]

        # Combine the new noisy individuals with the original top individuals
        splitted_population = flattened_list + new_population

        # Randomly sample from the combined list to form the final new population
        new_population = random.sample(splitted_population, population_size)

        return new_population

    def update_query_counter(self, amount: int) -> None:
        """
        Increment the query counter by a specified amount. This method is used to track the number of queries or
        model evaluations made, which can be crucial for understanding the computational cost or for adhering to
        query budgets in optimization processes.

        :param amount: The amount by which to increment the query counter.
        :return: None
        """
        # Increase the query counter by the specified amount
        self.query_counter += amount


def save_tensor_list_to_file(tensor_list: List, file_path: str) -> None:
    """
    Saves a list of tensor pairs to separate NumPy files, one for inputs and one for labels. This function is
    useful for persisting model data to disk for later use (training the surrogate model).

    :param tensor_list: A list where each tuple contains two elements:
        1. A tensor representing the input data to a model.
        2. A tensor representing the labels.
    :param file_path: The base file path to which the input and label files will be saved. This path will be
        appended with suffixes '_input' and '_labels' for saving the respective files.
    :return: None
    """
    # Prepare the list of input tensors: Detach from graph, move to CPU, and convert to NumPy
    tensor_array_input = [x[0].detach().cpu().numpy() for x in tensor_list]

    # Prepare the list of label tensors: Check type to handle potential constants or int labels
    tensor_array_labels = [
        x[1].detach().cpu().numpy() if hasattr(x[1], "detach") else np.array([x[1]])
        for x in tensor_list
    ]

    # Save the input tensors to a file with '_input' suffix
    np.save(file_path + "_input", tensor_array_input)

    # Save the label tensors to a file with '_labels' suffix
    np.save(file_path + "_labels", tensor_array_labels)
