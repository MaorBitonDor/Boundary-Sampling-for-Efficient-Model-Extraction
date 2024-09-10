import os
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from ucimlrepo import fetch_ucirepo

from Utility import create_train_test_sets, preprocess_income, preprocess_nsl_kdd


class BaseModel:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.params = {}
        # self.feature_names = None

    def fit_preprocessor(self, X):
        self.preprocessor.fit(X)
        print("Preprocessor fitted.")

    def preprocess(self, X):
        return self.preprocessor.transform(X) if self.preprocessor else X

    def train_model(self, X, y):
        print("Start Training...")

        # Ensure categorical columns are of type 'category'
        # X = self._convert_categorical_columns(X)

        # Create DMatrix
        dtrain = xgb.DMatrix(X, label=y)
        self.model = xgb.train(self.params, dtrain, num_boost_round=100)
        # self.feature_names = X.columns.tolist()  # Save the feature names
        print("Finished Training...")

    def predict(self, X):
        # Ensure categorical columns are of type 'category'
        # X = self._convert_categorical_columns(X)
        # X = X[self.feature_names]

        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)

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
            "max_depth": 10,  # Slightly deeper trees
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "lambda": 1,  # L2 regularization
            "alpha": 0.1,  # L1 regularization
            "scale_pos_weight": 1,
            "learning_rate": 0.25,  # Lower learning rate
            "gamma": 0,  # Minimum loss reduction
            "device": "cuda",
            "tree_method": "hist",
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
            "max_depth": 11,
            "min_child_weight": 15,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "lambda": 1,
            "alpha": 0.1,
            "learning_rate": 0.1,
            "gamma": 0,
            "device": "cuda",
            "tree_method": "hist",
        }
        self.train_file = "RT_IoT/rt_iot2022_train_set.csv"
        self.test_file = "RT_IoT/rt_iot2022_test_set.csv"
        self.target_name = "Attack_type"

    def validate_model(self, X_test, y_test):
        preds = self.predict(X_test)
        preds = np.argmax(preds, axis=1)  # Convert probabilities to binary output
        acc = accuracy_score(y_test, preds)
        print("Prediction Accuracy:", acc)
        return acc

    def test_model(self):
        X_train, X_test, y_train, y_test = create_train_test_sets(
            data_link=942,
            train_file=self.train_file,
            test_file=self.test_file,
            target_name=self.target_name,
            retrieval_type="fetch",
        )
        self.fit_preprocessor(X_test)
        # self.fit_preprocessor(X_train)
        # X_test_preprocessed = self.preprocess(X_test)
        return self.validate_model(X_test, y_test)


class CovTypeModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.preprocessor = None

        # Setup parameters for the XGBoost model
        self.params = {
            "objective": "multi:softprob",
            "num_class": 7,  # CovType has 7 classes
            "eval_metric": "mlogloss",
            "max_depth": 11,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "lambda": 1,
            "alpha": 0.1,
            "learning_rate": 0.1,
            "gamma": 0,
            "device": "cuda",
            "tree_method": "hist",
        }

    def load_dataset(self):
        covertype = fetch_ucirepo(id=31)
        X = covertype.data.features
        y = covertype.data.targets.values.flatten() - 1  # Adjust target to be 0-indexed
        return X, y

    def split_and_save_data(
        self,
        test_size=0.2,
        random_state=42,
        train_file="CovType/covtype_train.csv",
        test_file="CovType/covtype_test.csv",
    ):
        if not os.path.exists(train_file) or not os.path.exists(test_file):
            X, y = self.load_dataset()
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            X_train = X_train.reset_index(drop=True)
            y_train = pd.Series(y_train).reset_index(drop=True)
            X_test = X_test.reset_index(drop=True)
            y_test = pd.Series(y_test).reset_index(drop=True)
            train_data = pd.concat([X_train, y_train.rename("Cover_Type")], axis=1)
            test_data = pd.concat([X_test, y_test.rename("Cover_Type")], axis=1)
            train_data.to_csv(train_file, index=False)
            test_data.to_csv(test_file, index=False)
        else:
            train_data = pd.read_csv(train_file)
            test_data = pd.read_csv(test_file)
            X_train = train_data.drop("Cover_Type", axis=1)
            y_train = train_data["Cover_Type"]
            X_test = test_data.drop("Cover_Type", axis=1)
            y_test = test_data["Cover_Type"]
        return X_train, X_test, y_train, y_test

    def fit_preprocessor(self, X):
        numerical_features = [
            "Elevation",
            "Aspect",
            "Slope",
            "Horizontal_Distance_To_Hydrology",
            "Vertical_Distance_To_Hydrology",
            "Horizontal_Distance_To_Roadways",
            "Hillshade_9am",
            "Hillshade_Noon",
            "Hillshade_3pm",
            "Horizontal_Distance_To_Fire_Points",
        ]
        categorical_features = [
            col
            for col in X.columns
            if col.startswith("Soil_Type_") or col.startswith("Wilderness_Area_")
        ]

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_features),
                ("cat", OneHotEncoder(), categorical_features),
            ],
            sparse_threshold=0,  # Output dense matrix
        )
        self.preprocessor.fit(X)
        print("Preprocessor fitted.")

    def validate_model(self, X_test, y_test):
        X_test_preprocessed = self.preprocess(X_test)
        preds = self.predict(X_test_preprocessed)
        preds = np.argmax(preds, axis=1)
        acc = accuracy_score(y_test, preds)
        print("CovType Prediction Accuracy:", acc)
        return acc

    def test_model(self, test_file="CovType/covtype_test.csv"):
        test_data = pd.read_csv(test_file)
        X_test = test_data.drop("Cover_Type", axis=1)
        y_test = test_data["Cover_Type"]
        self.fit_preprocessor(X_test)
        return self.validate_model(X_test, y_test)


class NSLKDDModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.preprocessor = None

        # Setup parameters for the XGBoost model
        self.params = {
            "objective": "multi:softprob",
            "num_class": 23,  # NSL-KDD has 23 classes (Normal, DoS, Probe, R2L, U2R, etc.)
            "eval_metric": "mlogloss",
            "max_depth": 15,
            "min_child_weight": 25,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "lambda": 1,
            "alpha": 0.1,
            "learning_rate": 0.25,
            "gamma": 0,
            "device": "cuda",
            "tree_method": "hist"
        }

        # # Initial parameters for the XGBoost model
        # self.params = {
        #     "objective": "multi:softprob",
        #     "num_class": 23,  # NSL-KDD has 23 classes (Normal, DoS, Probe, R2L, U2R, etc.)
        #     "eval_metric": "mlogloss",
        #     "colsample_bytree": 1.0,
        #     "learning_rate": 0.25,
        #     "max_depth": 20,
        #     "min_child_weight": 1,
        #     "subsample": 0.8,
        #     "device": "cuda",
        #     "tree_method": "hist",
        # }

    def load_dataset(self):
        # Load the dataset from the nsl-kdd folder
        X_train, X_test, y_train, y_test = self.read_data()

        # Encode the labels
        label_mapping = {
            "normal": 0,
            "neptune": 1,
            "satan": 2,
            "ipsweep": 3,
            "portsweep": 4,
            "nmap": 5,
            "smurf": 6,
            "phf": 7,
            "teardrop": 8,
            "warezclient": 9,
            "pod": 10,
            "back": 11,
            "guess_passwd": 12,
            "warezmaster": 13,
            "imap": 14,
            "ftp_write": 15,
            "multihop": 16,
            "rootkit": 17,
            "buffer_overflow": 18,
            "land": 19,
            "loadmodule": 20,
            "perl": 21,
            "spy": 22,
        }
        y_train = y_train.replace(label_mapping).astype(int)
        y_test = y_test.replace(label_mapping).astype(int)

        return X_train, X_test, y_train, y_test

    def read_data(self):
        train_data = pd.read_csv("nsl-kdd/KDDTrain+.txt", header=None)
        test_data = pd.read_csv("nsl-kdd/KDDTest+.txt", header=None)

        columns = [
            "duration",
            "protocol_type",
            "service",
            "flag",
            "src_bytes",
            "dst_bytes",
            "land",
            "wrong_fragment",
            "urgent",
            "hot",
            "num_failed_logins",
            "logged_in",
            "num_compromised",
            "root_shell",
            "su_attempted",
            "num_root",
            "num_file_creations",
            "num_shells",
            "num_access_files",
            "num_outbound_cmds",
            "is_host_login",
            "is_guest_login",
            "count",
            "srv_count",
            "serror_rate",
            "srv_serror_rate",
            "rerror_rate",
            "srv_rerror_rate",
            "same_srv_rate",
            "diff_srv_rate",
            "srv_diff_host_rate",
            "dst_host_count",
            "dst_host_srv_count",
            "dst_host_same_srv_rate",
            "dst_host_diff_srv_rate",
            "dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate",
            "dst_host_serror_rate",
            "dst_host_srv_serror_rate",
            "dst_host_rerror_rate",
            "dst_host_srv_rerror_rate",
            "outcome",
            "level",
        ]

        train_data.columns = columns
        test_data.columns = columns

        train_data.loc[train_data['outcome'] == "normal", "outcome"] = 'normal'
        test_data.loc[test_data['outcome'] != 'normal', "outcome"] = 'attack'

        # Separate features and labels
        X_train = train_data.iloc[:, :-1]
        y_train = train_data.iloc[:, -1]
        X_test = test_data.iloc[:, :-1]
        y_test = test_data.iloc[:, -1]

        X_train = preprocess_nsl_kdd(X_train)
        X_test = preprocess_nsl_kdd(X_test)
        return X_train, X_test, y_train, y_test

    def fit_preprocessor(self, X):
        numerical_features = [
            col for col in X.columns if X[col].dtype in ["int64", "float64"]
        ]
        categorical_features = [col for col in X.columns if X[col].dtype == "object"]

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ],
            sparse_threshold=0,  # Output dense matrix
        )
        self.preprocessor.fit(X)
        print("Preprocessor fitted.")

    def validate_model(self, X_test, y_test):
        # X_test_preprocessed = self.preprocess(X_test)
        preds = self.predict(X_test)
        preds = np.argmax(preds, axis=1)
        acc = accuracy_score(y_test, preds)
        print("NSL-KDD Prediction Accuracy:", acc)
        return acc

    def test_model(self):
        X_train, X_test, y_train, y_test = self.load_dataset()
        # X_test = self._convert_categorical_columns(X_test)
        self.fit_preprocessor(X_train)
        X_test_preprocessed = self.preprocess(X_test)
        return self.validate_model(X_test_preprocessed, y_test)

    def tune_hyperparameters(self, X_train, y_train):
        # Ensure categorical columns are of type 'category'
        # X_train = self._convert_categorical_columns(X_train)

        param_grid = {
            "max_depth": [20],
            "min_child_weight": [1],
            "subsample": [0.8],
            "colsample_bytree": [1.0],
            "learning_rate": [0.25],
        }

        xgb_model = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=23,
            eval_metric="mlogloss",
            use_label_encoder=False,
            tree_method="hist",
            gpu_id=0,
            # enable_categorical=True,  # Enable categorical data handling
        )

        grid_search = GridSearchCV(
            estimator=xgb_model, param_grid=param_grid, cv=3, verbose=1
        )
        grid_search.fit(X_train, y_train)

        self.params.update(grid_search.best_params_)
        print("Best parameters found: ", grid_search.best_params_)

    # def _convert_categorical_columns(self, df):
    #     categorical_columns = df.select_dtypes(include=["object"]).columns
    #     for col in categorical_columns:
    #         df[col] = df[col].astype("category")
    #     return df


# def main():
#     train_file = "RT_IoT/rt_iot2022_train_set.csv"
#     test_file = "RT_IoT/rt_iot2022_test_set.csv"
#     target_name = "Attack_type"
#     X_train, X_test, y_train, y_test = create_train_test_sets(
#         data_link=942,
#         train_file=train_file,
#         test_file=test_file,
#         target_name=target_name,
#         retrieval_type="fetch",
#     )
#     print(f"Training set size: {X_train.shape[0]} samples")
#     print(f"Test set size: {X_test.shape[0]} samples")
#
#     model = RTIoTModel()
#     # model.fit_preprocessor(X_train)
#
#     model.train_model(X_train.values, y_train.values)
#
#     probs = model.predict(
#         X_test.values
#     )  # This will output the probabilities for each class
#     predicted_classes = np.argmax(probs, axis=1)
#     # Calculate the accuracy
#     accuracy = np.mean(predicted_classes == y_test)
#     print("Prediction Accuracy:", accuracy)
#
#     model_path = "rt_iot_xgb_model.json"
#     preprocessor_path = "rt_iot_preprocessor.pkl"
#
#     model.save(model_path, preprocessor_path=preprocessor_path)
#
#     # train_file = "Adults/adults_train_set.csv"
#     # test_file = "Adults/adults_test_set.csv"
#     # adult_url = (
#     #     "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
#     # )
#     # adult_columns = [
#     #     "age",
#     #     "workclass",
#     #     "fnlwgt",
#     #     "education",
#     #     "education-num",
#     #     "marital-status",
#     #     "occupation",
#     #     "relationship",
#     #     "race",
#     #     "sex",
#     #     "capital-gain",
#     #     "capital-loss",
#     #     "hours-per-week",
#     #     "native-country",
#     #     "income",
#     # ]
#     # adult_target = "income"
#     # X_train_adult, X_test_adult, y_train_adult, y_test_adult = (
#     #     create_train_test_sets(
#     #         adult_url,
#     #         train_file,
#     #         test_file,
#     #         adult_target,
#     #         adult_columns,
#     #         na_values="?",
#     #         preprocess_target=preprocess_income,
#     #     )
#     # )
#     # print(f"Training set size: {X_train_adult.shape[0]} samples")
#     # print(f"Test set size: {X_test_adult.shape[0]} samples")
#     #
#     # adult_model = AdultModel()
#     # adult_model.preprocessor.fit(X_train_adult)
#     # X_train_adult_preprocessed = adult_model.preprocess(X_train_adult)
#     # X_test_adult_preprocessed = adult_model.preprocess(X_test_adult)
#     #
#     # adult_model.train_model(X_train_adult_preprocessed, y_train_adult)
#     # adult_preds = adult_model.predict(X_test_adult_preprocessed)
#     # adult_preds = (adult_preds > 0.5).astype(int)  # Convert probabilities to binary output
#     # print("Adult Prediction Accuracy:", accuracy_score(y_test_adult, adult_preds))
#
#     # model_path = 'adult_xgb_model.json'
#     # preprocessor_path = 'adult_preprocessor.pkl'
#     #
#     # adult_model.save(model_path, preprocessor_path)
#     #
#     # loaded_model = AdultModel()
#     # loaded_model.load(model_path, preprocessor_path)
#     #
#     # adult_preds = loaded_model.predict(X_test_adult_preprocessed)
#     # adult_preds = (adult_preds > 0.5).astype(int)  # Convert probabilities to binary output
#     # print("Adult Prediction Accuracy:", accuracy_score(y_test_adult, adult_preds))
#
#
# # # Set a seed value
# # seed_value = 42
# #
# # # Set `torch` random seed for all devices (CPU and CUDA)
# # torch.manual_seed(seed_value)
# # if torch.cuda.is_available():
# #     torch.cuda.manual_seed_all(seed_value)  # For CUDA devices
# #
# # # Set `numpy` random seed
# # np.random.seed(seed_value)
# #
# # # Set `random` seed for Python standard library functions
# # random.seed(seed_value)
# if __name__ == "__main__":
#     main()
# #     model_path = 'adult_xgb_model.json'
# #     preprocessor_path = 'adult_preprocessor.pkl'
# #
# #     loaded_model = AdultModel()
# #     loaded_model.load(model_path, preprocessor_path)
# #
# #     X_train_adult, X_test_adult, y_train_adult, y_test_adult = create_train_test_sets()
# #     print(f"Training set size: {X_train_adult.shape[0]} samples")
# #     print(f"Test set size: {X_test_adult.shape[0]} samples")
# #
# #     loaded_model.preprocessor.fit(X_train_adult)
# #     X_train_adult_preprocessed = loaded_model.preprocess(X_train_adult)
# #     X_test_adult_preprocessed = loaded_model.preprocess(X_test_adult)
# #
# #     adult_preds = loaded_model.predict(X_test_adult_preprocessed)
# #     adult_preds = (adult_preds > 0.5).astype(int)  # Convert probabilities to binary output
# #     print("Adult Prediction Accuracy:", accuracy_score(y_test_adult, adult_preds))
