import numpy as np

from training import Model, RandomForestClassifierModel
from data_processing import DataFile
import ray
from ray import tune
import os


class HyperparametersOptimization:
    def __init__(self, model_class_object, data:DataFile, encode_list, num_workers_of_single_task
                 , hyperparameters_to_optimize_dict):
        self.model_class_object = model_class_object
        self.data = data
        self.encode_list = encode_list
        self.num_workers_of_single_task = num_workers_of_single_task
        self.hyperparameters_to_optimize_dict = hyperparameters_to_optimize_dict
        self.x_data_series, self.y_data_series, self.x_valid_data_series = self.__get_data()

    def __get_data(self):
        x_data_series = self.data.get_encoded_series(is_train=True, encode_list=self.encode_list)
        y_data_series = self.data.get_label_series()
        x_valid_data_series = self.data.get_encoded_series(is_train=False, encode_list=self.encode_list)
        return x_data_series, y_data_series, x_valid_data_series

    def optimize(self, config: dict):
        model = self.model_class_object(*self.__get_data(), **config)
        average_auc, average_f1, average_accuracy = model.get_average_evaluate_criterion()
        ray.train.report({'auc': average_auc, 'f1': average_f1, 'accuracy':average_accuracy})
        return {'auc': average_auc, 'f1': average_f1, 'accuracy':average_accuracy}

    def tune(self):
        hyperparameters_to_optimize_dict = self.hyperparameters_to_optimize_dict
        analysis = tune.run(
            run_or_experiment=self.optimize,
            config=hyperparameters_to_optimize_dict,
            resources_per_trial={"cpu": self.num_workers_of_single_task, 'gpu': 0},
            scheduler=tune.schedulers.ASHAScheduler(),
            metric='f1',
            mode='max'
        )
        print('The best hyperparameter:', analysis.best_config)


if __name__ == '__main__':
    # ray.init()
    num_workers_of_single_task = 4
    hyperparameters_to_optimize_dict = {
        "n_estimators": tune.grid_search([5000, 8000, 10000, 20000, 30000]),
        "max_depth": tune.grid_search([None]),
        "threshold": tune.grid_search([0.4, 0.5, 0.55, 0.6, 0.65]),
        "num_workers": num_workers_of_single_task
    }
    model_class_object = RandomForestClassifierModel
    data = DataFile(training_data_file_path='../dataset/traindata-new - new 24.07.05.xlsx'
                , testing_data_file_path='../dataset/testdata-new.xlsx'
                , num_workers=num_workers_of_single_task)
    # data = DataFile(training_data_file_path='./dataset/protac.xlsx'
    #                 , testing_data_file_path='./dataset/testdata-new.xlsx'
    #                 , num_workers=num_workers_of_single_task)
    encode_list = [
        'AAC',
        "topological_fingerprints",
        "maccs",
        "atom_pairs",
        "morgan_fingerprints",
        "uniprot",
        "target",
        'e3_ligase'
    ]
    hyperparametersOptimization = (
        HyperparametersOptimization(model_class_object=model_class_object,
                                    data=data,
                                    encode_list=encode_list,
                                    num_workers_of_single_task=num_workers_of_single_task,
                                    hyperparameters_to_optimize_dict=hyperparameters_to_optimize_dict)
    )
    hyperparametersOptimization.tune()
