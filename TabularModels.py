import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

from Utility import create_train_test_sets, preprocess_income


class BaseModel:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.params = {}

    def fit_preprocessor(self, X):
        self.preprocessor.fit(X)
        print("Preprocessor fitted.")

    def preprocess(self, X):
        return self.preprocessor.transform(X) if self.preprocessor else X

    def train_model(self, X, y):
        print("Start Training...")
        dtrain = xgb.DMatrix(X, label=y)
        self.model = xgb.train(self.params, dtrain, num_boost_round=100)
        print("Finished Training...")

    def predict(self, X):
        dtest = xgb.DMatrix(X)
        # Predict using softmax probability
        predictions = self.model.predict(dtest)
        return predictions  # This will be a 2D array where each row gives probabilities of each class

    def save(
            self,
            model_path,
            preprocessor_path,
    ):
        self.model.save_model(model_path)
        with open(preprocessor_path, "wb") as f:
            pickle.dump(self.preprocessor, f)

    def load(
            self,
            model_path,
            preprocessor_path,
    ):
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        with open(preprocessor_path, "rb") as f:
            self.preprocessor = pickle.load(f)


class AdultModel(BaseModel):
    def __init__(self):
        super().__init__()
        numerical_features = [
            "age",
            "fnlwgt",
            "education-num",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
        ]
        categorical_features = [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ]
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_features),
                ("cat", OneHotEncoder(), categorical_features),
            ],
            sparse_threshold=0,
        )  # Ensure output is a dense array
        self.params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 5,  # Slightly deeper trees
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "lambda": 1,  # L2 regularization
            "alpha": 0.1,  # L1 regularization
            "scale_pos_weight": 1,
            "learning_rate": 0.25,  # Lower learning rate
            "gamma": 0,  # Minimum loss reduction
        }
        self.train_file = "Adults/adults_train_set.csv"
        self.test_file = "Adults/adults_test_set.csv"
        self.adult_url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        )
        self.adult_columns = [
            "age",
            "workclass",
            "fnlwgt",
            "education",
            "education-num",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
            "native-country",
            "income",
        ]
        self.adult_target = "income"

    def predict(self, X):
        dtest = xgb.DMatrix(X)
        predictions = self.model.predict(dtest)

        # Check if it's binary classification and adjust accordingly
        # This step is necessary only if the output does not already include two probabilities,
        # which can depend on the training setup.
        if predictions.ndim == 1 or (
                predictions.ndim == 2 and predictions.shape[1] == 1
        ):
            # Generate probabilities for the negative class (1 - probability of the positive class)
            predictions = np.column_stack((1 - predictions, predictions))

        return predictions
        # return self.model.predict(dtest)

    def validate_model(self):
        X_train_adult, X_test_adult, y_train_adult, y_test_adult = (
            create_train_test_sets(
                self.adult_url,
                self.train_file,
                self.test_file,
                self.adult_target,
                self.adult_columns,
                na_values="?",
                preprocess_target=preprocess_income,
            )
        )
        self.preprocessor.fit(X_test_adult)
        X_test_adult_preprocessed = self.preprocess(X_test_adult)

        adult_preds = self.predict(X_test_adult_preprocessed)
        adult_preds = np.argmax(
            adult_preds, axis=1
        )  # Convert probabilities to binary output
        acc = accuracy_score(y_test_adult, adult_preds)
        print("Adult Prediction Accuracy:", acc)
        return acc

    def test_model(self):
        return self.validate_model()


class RTIoTModel(BaseModel):
    def __init__(self):
        super().__init__()
        numerical_features = ["flow_duration", "fwd_pkts_per_sec"]
        categorical_features = ["proto", "service"]

        # Configure preprocessing
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_features),
                ("cat", OneHotEncoder(), categorical_features),
            ],
            remainder="passthrough",  # Keep other variables without transformation
            sparse_threshold=0,  # Output dense matrix
        )

        # Setup parameters for the XGBoost model
        self.params = {
            "objective": "multi:softprob",
            "num_class": 12,  # Adjust based on the actual number of classes
            "eval_metric": "mlogloss",
            "max_depth": 6,
            "min_child_weight": 1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "lambda": 1,
            "alpha": 0.1,
            "learning_rate": 0.1,
            "gamma": 0,
        }
        self.train_file = "RT_IoT/rt_iot2022_train_set.csv"
        self.test_file = "RT_IoT/rt_iot2022_test_set.csv"
        self.target_name = "Attack_type"

    def validate_model(self):
        X_train, X_test, y_train, y_test = create_train_test_sets(
            data_link=942,
            train_file=self.train_file,
            test_file=self.test_file,
            target_name=self.target_name,
            retrieval_type="fetch",
        )

        adult_preds = self.predict(X_test.values)
        adult_preds = np.argmax(
            adult_preds, axis=1
        )  # Convert probabilities to binary output
        acc = accuracy_score(y_test, adult_preds)
        print("Prediction Accuracy:", acc)
        return acc

    def test_model(self):
        return self.validate_model()


def main():
    train_file = "RT_IoT/rt_iot2022_train_set.csv"
    test_file = "RT_IoT/rt_iot2022_test_set.csv"
    target_name = "Attack_type"
    X_train, X_test, y_train, y_test = create_train_test_sets(
        data_link=942,
        train_file=train_file,
        test_file=test_file,
        target_name=target_name,
        retrieval_type="fetch",
    )
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")

    model = RTIoTModel()
    # model.fit_preprocessor(X_train)

    model.train_model(X_train.values, y_train.values)

    probs = model.predict(X_test.values)  # This will output the probabilities for each class
    predicted_classes = np.argmax(probs, axis=1)
    # Calculate the accuracy
    accuracy = np.mean(predicted_classes == y_test)
    print("Prediction Accuracy:", accuracy)

    model_path = "rt_iot_xgb_model.json"
    preprocessor_path = "rt_iot_preprocessor.pkl"

    model.save(model_path, preprocessor_path=preprocessor_path)

    # train_file = "Adults/adults_train_set.csv"
    # test_file = "Adults/adults_test_set.csv"
    # adult_url = (
    #     "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    # )
    # adult_columns = [
    #     "age",
    #     "workclass",
    #     "fnlwgt",
    #     "education",
    #     "education-num",
    #     "marital-status",
    #     "occupation",
    #     "relationship",
    #     "race",
    #     "sex",
    #     "capital-gain",
    #     "capital-loss",
    #     "hours-per-week",
    #     "native-country",
    #     "income",
    # ]
    # adult_target = "income"
    # X_train_adult, X_test_adult, y_train_adult, y_test_adult = (
    #     create_train_test_sets(
    #         adult_url,
    #         train_file,
    #         test_file,
    #         adult_target,
    #         adult_columns,
    #         na_values="?",
    #         preprocess_target=preprocess_income,
    #     )
    # )
    # print(f"Training set size: {X_train_adult.shape[0]} samples")
    # print(f"Test set size: {X_test_adult.shape[0]} samples")
    #
    # adult_model = AdultModel()
    # adult_model.preprocessor.fit(X_train_adult)
    # X_train_adult_preprocessed = adult_model.preprocess(X_train_adult)
    # X_test_adult_preprocessed = adult_model.preprocess(X_test_adult)
    #
    # adult_model.train_model(X_train_adult_preprocessed, y_train_adult)
    # adult_preds = adult_model.predict(X_test_adult_preprocessed)
    # adult_preds = (adult_preds > 0.5).astype(int)  # Convert probabilities to binary output
    # print("Adult Prediction Accuracy:", accuracy_score(y_test_adult, adult_preds))

    # model_path = 'adult_xgb_model.json'
    # preprocessor_path = 'adult_preprocessor.pkl'
    #
    # adult_model.save(model_path, preprocessor_path)
    #
    # loaded_model = AdultModel()
    # loaded_model.load(model_path, preprocessor_path)
    #
    # adult_preds = loaded_model.predict(X_test_adult_preprocessed)
    # adult_preds = (adult_preds > 0.5).astype(int)  # Convert probabilities to binary output
    # print("Adult Prediction Accuracy:", accuracy_score(y_test_adult, adult_preds))


# # Set a seed value
# seed_value = 42
#
# # Set `torch` random seed for all devices (CPU and CUDA)
# torch.manual_seed(seed_value)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(seed_value)  # For CUDA devices
#
# # Set `numpy` random seed
# np.random.seed(seed_value)
#
# # Set `random` seed for Python standard library functions
# random.seed(seed_value)
if __name__ == "__main__":
    main()
#     model_path = 'adult_xgb_model.json'
#     preprocessor_path = 'adult_preprocessor.pkl'
#
#     loaded_model = AdultModel()
#     loaded_model.load(model_path, preprocessor_path)
#
#     X_train_adult, X_test_adult, y_train_adult, y_test_adult = create_train_test_sets()
#     print(f"Training set size: {X_train_adult.shape[0]} samples")
#     print(f"Test set size: {X_test_adult.shape[0]} samples")
#
#     loaded_model.preprocessor.fit(X_train_adult)
#     X_train_adult_preprocessed = loaded_model.preprocess(X_train_adult)
#     X_test_adult_preprocessed = loaded_model.preprocess(X_test_adult)
#
#     adult_preds = loaded_model.predict(X_test_adult_preprocessed)
#     adult_preds = (adult_preds > 0.5).astype(int)  # Convert probabilities to binary output
#     print("Adult Prediction Accuracy:", accuracy_score(y_test_adult, adult_preds))
