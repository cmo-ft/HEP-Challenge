# ------------------------------
# Dummy Sample Submission
# ------------------------------

from reverse_systs import reverse_parameterize_systs
from my_stat import StatisticalAnalysis
from sklearn.model_selection import train_test_split as sk_train_test_split
import pandas as pd
import numpy as np
import os

current_file = os.path.dirname(os.path.abspath(__file__))

class Model:
    """
    This is a model class to be submitted by the participants in their submission.

    Atributes:
        * get_train_set (callable): A function that returns a dictionary with data, labels, weights, detailed_labels and settings.
        * systematics (object): A function that can be used to get a dataset with systematics added.
        * model (object): The model object.
        * name (str): The name of the model.
        * stat_analysis (object): The statistical analysis object.
        
    Methods:
        * fit(): Trains the model.
        * predict(test_set): Predicts the values for the test set.

    """

    def __init__(self, get_train_set=None, systematics=None):
        """
        Initializes the Model class.

        Args:
            * get_train_set (callable, optional): A function that returns a dictionary with data, labels, weights,detailed_labels and settings.
            * systematics (object, optional): A function that can be used to get a dataset with systematics added.

        Returns:
            None
        """

        self.systematics = systematics

        self.get_train_set = get_train_set

        self.re_train = True
        self.re_compute = True

        from boosted_decision_tree import BoostedDecisionTree
        self.name = "model_systematic_aware_XGB"
        self.model = BoostedDecisionTree()
        module_file = current_file + f"/{self.name}.json"
        if os.path.exists(module_file):
            self.model.load(module_file)
            self.re_train = False  # if model is already trained, no need to retrain

        print("Model is ", self.name)

        # self.stat_analysis = StatisticalAnalysis(
        #     self.model, bins=100, systematics=self.systematics
        # )
        self.stat_analysis = StatisticalAnalysis(
            self.model, bins=100, systematics=self.systematics
        )

        saved_info_file_dir = current_file + "/saved_info_" + self.name
        if os.path.exists(saved_info_file_dir):
            self.re_compute = not self.stat_analysis.load(saved_info_file_dir)
        else:
            os.makedirs(saved_info_file_dir, exist_ok=True)

        self.random_state = np.random.RandomState(42)
        self.train_systs = {
            "tes": True,
            "jes": True,
            "soft_met": True,
        }


    def random_syst_generator(self):
        tes, jes, soft_met = 1.0, 1.0, 0.0
        if self.train_systs.get('tes', False):
            tes = np.clip(self.random_state.normal(loc=1.0, scale=0.01), a_min=0.9, a_max=1.1)
        if self.train_systs.get('jes', False):
            jes = np.clip(self.random_state.normal(loc=1.0, scale=0.01), a_min=0.9, a_max=1.1)
        if self.train_systs.get('soft_met', False):
            soft_met = np.clip(self.random_state.normal(loc=0.0, scale=1), a_min=0.0, a_max=5.0)
        return {"tes": tes, "jes": jes, "soft_met": soft_met}


    def apply_random_systs(self, data_set, nominal_only=False):
        """
        Applies systematics to the data.

        Args:
            * data_set (dict): A dictionary containing the data.

        Returns:
            dict: A dictionary containing the data with systematics applied.
        """

        if nominal_only:
            syst = {"tes": 1.0, "jes": 1.0, "soft_met": 0.0}
            data_set = self.systematics(data_set, **syst)
        else:
            syst = self.random_syst_generator()
            data_set = self.systematics(data_set, **syst)

        return data_set, syst


    def syst_augment_data_set(self, data_set, factor=1, nominal_only=False):
        """
        Augments the data set. In each augmentation:
            1. apply syst 
            2. manage data
        
        Args:
            * data_set (dict): A dictionary containing the data.
            * factor (int, optional): The factor by which to augment the data set. Defaults to 1.
            
        Returns:
            dict: A dictionary with keys: "data", "labels" and "weights".
        """
        data_set_original = data_set.copy()
        data, label, weights = [], [], []
        for i in range(factor):
            data_set_i, syst = self.apply_random_systs(data_set_original, nominal_only=nominal_only)
            data.append(reverse_parameterize_systs(data_set_i["data"], **syst))
            label.append(data_set_i["labels"])
            weights.append(data_set_i["weights"])
        
        output = {
            "data": pd.concat(data),
            "labels": np.concatenate(label),
            "weights": np.concatenate(weights)
        }
        return output


    def fit(self):
        """
        Trains the model.

        Functionality:
            This function can be used to train a model. If `re_train` is True, it balances the dataset,
            fits the model using the balanced dataset, and saves the model. If `re_train` is False, it
            loads the saved model and calculates the saved information. The saved information is used
            to compute the train results.

        Returns:
            None
        """

        saved_info_file_dir = current_file + "/saved_info_" + self.name

        def print_set_info(name, dataset):
            print(f"{name} Set:")
            print(f"{'-' * len(name)} ")
            print(f"  Data Shape:          {dataset['data'].shape}")
            print(f"  Labels Shape:        {dataset['labels'].shape}")
            print(f"  Weights Shape:       {dataset['weights'].shape}")
            print(
                f"  Sum Signal Weights:  {dataset['weights'][dataset['labels'] == 1].sum():.2f}"
            )
            print(
                f"  Sum Background Weights: {dataset['weights'][dataset['labels'] == 0].sum():.2f}"
            )
            print(f"  Sample keys:         {dataset['data'].keys()}")
            print("\n")

        if self.re_train or self.re_compute:
            train_set = self.get_train_set()

            # # reduce the train set size to avoid out of memory
            # original_train_size = len(train_set["data"])
            # for key in ["data", "labels", "weights", "detailed_labels"]:
            #     train_set[key] = train_set[key][: int(original_train_size * 0.01)]

            print("Full data: ", train_set["data"].shape)
            print("Full Labels: ", train_set["labels"].shape)
            print("Full Weights: ", train_set["weights"].shape)
            print(
                "sum_signal_weights: ",
                train_set["weights"][train_set["labels"] == 1].sum(),
            )
            print(
                "sum_bkg_weights: ",
                train_set["weights"][train_set["labels"] == 0].sum(),
            )
            print(" \n ")

            # train : validation : template = 3 : 1 : 6
            temp_set, holdout_set = train_test_split(
                train_set, test_size=0.6, random_state=42, reweight=True
            )
            temp_set["data"] = temp_set["data"].reset_index(drop=True)
            training_set, valid_set = train_test_split(
                temp_set, test_size=0.2, random_state=42, reweight=True
            )

            del train_set

            """
            1. Apply random systematics
            2. Reverse TES and JES
            3. Decorelate the data with soft_met syst
            4. Repeat the above steps to augment the data
            """
            # training_set_aug = self.syst_augment_data_set(training_set, factor=4)
            # valid_set_aug = self.syst_augment_data_set(valid_set, factor=4)
            # since TES and JES have no effect, and soft_met is not considered for now, we only use nominal
            training_set_aug = self.syst_augment_data_set(training_set, factor=1, nominal_only=True)
            valid_set_aug = self.syst_augment_data_set(valid_set, factor=1, nominal_only=True)

            print_set_info("Training", training_set_aug)
            print_set_info("Validation", valid_set_aug)
            print_set_info("Holdout (For Statistical Template)", holdout_set)

            if self.re_train:
                balanced_set = balance_set(training_set_aug)

                self.model.fit(
                    balanced_set["data"],
                    balanced_set["labels"],
                    balanced_set["weights"],
                    valid_set=[
                        valid_set_aug["data"],
                        valid_set_aug["labels"],
                        valid_set_aug["weights"],
                    ],
                )

                self.model.save(current_file + "/" + self.name)

            self.stat_analysis.calculate_template(holdout_set, saved_info_file_dir)

        def predict_and_analyze(
                dataset_name, data_set, fig_name
        ):
            print_set_info(name=dataset_name, dataset=data_set)

            data_set = self.syst_augment_data_set(data_set, factor=1, nominal_only=True)
            results = self.stat_analysis.compute_mu(
                data_set["data"],
                data_set["weights"],
            )

            print(f"{dataset_name} Results:")
            print(f"{'-' * len(dataset_name)} Results:")
            for key, value in results.items():
                print(f"\t{key} : {value}")
            print("\n")

        if self.re_train or self.re_compute:
            datasets = [
                # ("Training", training_set, "train_mu"),
                ("Validation", valid_set, "valid_mu"),
                # ("Holdout", holdout_set, "holdout_mu"),
            ]

            for name, dataset, plot_name in datasets:
                predict_and_analyze(
                    name,
                    dataset,
                    plot_name,
                )
        

    def predict(self, test_set):
        """
        Predicts the values for the test set.

        Args:
            * test_set (dict): A dictionary containing the data and weights. test_set should contain the DER features.

        Returns:
            dict: A dictionary with the following keys:
            - 'mu_hat': The predicted value of mu.
            - 'delta_mu_hat': The uncertainty in the predicted value of mu.
            - 'p16': The lower bound of the 16th percentile of mu.
            - 'p84': The upper bound of the 84th percentile of mu.
        """

        test_data = test_set["data"]
        test_weights = test_set["weights"]

        print("[*] -> test weights sum = ", test_weights.sum())
        # test_data = self.systematics(test_set, tes=1.0, jes=1.0, soft_met=0.0)

        result = self.stat_analysis.compute_mu(
            test_data,
            test_weights
        )

        print("Test Results: ", result)
        print(flush=True)

        return result


def train_test_split(data_set, test_size=0.2, random_state=42, reweight=False):
    """
    Splits the data into training and testing sets.

    Args:
        * data_set (dict): A dictionary containing the data, labels, weights, detailed_labels, and settings
        * test_size (float, optional): The size of the testing set. Defaults to 0.2.
        * random_state (int, optional): The random state. Defaults to 42.
        * reweight (bool, optional): Whether to reweight the data. Defaults to False.

    Returns:
        tuple: A tuple containing the training and testing
    """
    data = data_set["data"].copy()
    train_set = {}
    test_set = {}
    full_size = len(data)

    print(f"Full size of the data is {full_size}")

    for key in data_set.keys():
        if (key != "data") and (key != "settings"):
            data[key] = np.array(data_set[key])

    train_data, test_data = sk_train_test_split(
        data, test_size=test_size, random_state=random_state
    )

    for key in data_set.keys():
        if (key != "data") and (key != "settings"):
            train_set[key] = np.array(train_data.pop(key))
            test_set[key] = np.array(test_data.pop(key))

    train_set["data"] = train_data
    test_set["data"] = test_data

    if reweight is True:
        signal_weight = np.sum(data_set["weights"][data_set["labels"] == 1])
        background_weight = np.sum(data_set["weights"][data_set["labels"] == 0])
        signal_weight_train = np.sum(train_set["weights"][train_set["labels"] == 1])
        background_weight_train = np.sum(train_set["weights"][train_set["labels"] == 0])
        signal_weight_test = np.sum(test_set["weights"][test_set["labels"] == 1])
        background_weight_test = np.sum(test_set["weights"][test_set["labels"] == 0])

        train_set["weights"][train_set["labels"] == 1] = train_set["weights"][
                                                             train_set["labels"] == 1
                                                             ] * (signal_weight / signal_weight_train)
        test_set["weights"][test_set["labels"] == 1] = test_set["weights"][
                                                           test_set["labels"] == 1
                                                           ] * (signal_weight / signal_weight_test)

        train_set["weights"][train_set["labels"] == 0] = train_set["weights"][
                                                             train_set["labels"] == 0
                                                             ] * (background_weight / background_weight_train)
        test_set["weights"][test_set["labels"] == 0] = test_set["weights"][
                                                           test_set["labels"] == 0
                                                           ] * (background_weight / background_weight_test)

    return train_set, test_set


def balance_set(training_set):
    """
    Balances the training set by equalizing the number of background and signal events.

    Args:
        training_set (dict): A dictionary containing the data, labels, weights, detailed_labels, and settings.

    Returns:
        dict: A dictionary containing the balanced training set.
    """

    balanced_set = training_set.copy()

    weights_train = training_set["weights"].copy()
    train_labels = training_set["labels"].copy()
    class_weights_train = (
        weights_train[train_labels == 0].sum(),
        weights_train[train_labels == 1].sum(),
    )

    for i in range(len(class_weights_train)):  # loop on B then S target
        # training dataset: equalize number of background and signal
        weights_train[train_labels == i] *= (
                max(class_weights_train) / class_weights_train[i]
        )
        # test dataset : increase test weight to compensate for sampling

    balanced_set["weights"] = weights_train

    return balanced_set
