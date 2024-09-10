from Config import Config
from TabularModels import RTIoTModel
from Utility import (
    BAM_main_algorithm_tabular,
    prepare_config_and_log,
    generate_random_tabular_data_rt_iot,
)

if __name__ == "__main__":
    prepare_config_and_log()
    config = Config.instance

    model_path = "rt_iot_xgb_model.json"
    preprocessor_path = "rt_iot_preprocessor.pkl"

    loaded_model = RTIoTModel()
    loaded_model.load(model_path, preprocessor_path)

    loaded_model.test_model()

    surrogate_model = BAM_main_algorithm_tabular(
        loaded_model,
        RTIoTModel,
        generate_random_tabular_data_rt_iot,
        num_of_classes=12,
        k=3000,
        epsilon=0.2,
        population_size=10000,
        generations=30,
        search_spread=10,
    )

    surrogate_acc = surrogate_model.test_model()
    print(f"The surrogate model accuracy is {surrogate_acc}")
