import argparse
import logging
from typing import Dict, Any


class Config:
    instance = None
    log = None

    def __new__(cls, log_file: str = "model_log.log") -> None:
        """
        Create a new instance of the Config class.

        :param log_file: The path to the log file (default is "model_log.log").
        :return: A dictionary containing the configuration options.
        """
        if cls.instance is None:
            ins = cls.create_instance()
            cls.instance = ins
        if cls.log is None:
            cls.log = cls.configure_logging(log_file=log_file)

    @staticmethod
    def create_instance() -> Dict[str, Any]:
        """
        Create an instance of the Config class.

        :return: A dictionary containing the configuration options.
        """
        parser = argparse.ArgumentParser(description="Your program description here")

        parser.add_argument("--k", type=int, default=6000, help="Value for k")
        parser.add_argument(
            "--epsilon", type=float, default=0.0005, help="Value for epsilon"
        )
        parser.add_argument(
            "--population_size", type=int, default=20000, help="Population size"
        )
        parser.add_argument(
            "--generations", type=int, default=40, help="Number of generations"
        )
        parser.add_argument(
            "--search_spread", type=int, default=10, help="Search spread"
        )
        parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
        parser.add_argument(
            "--dont_get_from_disk",
            action="store_false",
            help="Flag to not read data from the disk",
        )
        parser.add_argument(
            "--num_of_classes", type=int, default=10, help="Number of classes"
        )
        parser.add_argument(
            "--learning_rate", type=float, default=0.3, help="Learning rate"
        )
        parser.add_argument(
            "--optimizer_name", type=str, default="AdamW", help="Optimizer name"
        )
        parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
        parser.add_argument(
            "--delete_checkpoints",
            action="store_false",
            help="Flag to delete checkpoints",
        )
        parser.add_argument(
            "--genetic_alg_prediction_max_batch_size",
            type=int,
            default=100,
            help="Max batch size for genetic algorithm prediction",
        )
        parser.add_argument(
            "--data_directory",
            type=str,
            default="/ssd/SaveDataset/Batches",
            help="Data directory",
        )
        parser.add_argument(
            "--destination_folder",
            type=str,
            default="{data_directory}/{model_class}/{generations}_generations",
            help="Destination folder",
        )
        parser.add_argument(
            "--file_path_confidence_batch",
            type=str,
            default="{destination_folder}/Toward_proxy_dataset_confidence_{generations}_batch_{gen_minus_one}_input.npy",
            help="File path confidence batch",
        )

        args = parser.parse_args()
        return vars(args)

    @staticmethod
    def configure_logging(
        log_file: str = "model_log.log", log_level: str = "INFO"
    ) -> logging.Logger:
        """
        Configure logging for the application.

        :param log_file: The path to the log file (default is "model_log.log").
        :param log_level: The logging level (default is "INFO").
        :return: A logger object.
        """
        logger = logging.getLogger("my_logger")
        logger.setLevel(log_level)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger
