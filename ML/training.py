from data_processing import DataFile
import numpy as np
import pandas as pd
import os
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold


# warnings.filterwarnings("ignore")


class Model:
    def __init__(self, x_data_series, y_data_series, x_valid_data_series, model, fold=5, num_workers=8, threshold=0.6):
        self.x_data_series = x_data_series
        self.y_data_series = y_data_series
        self.x_valid_data_series = x_valid_data_series
        self.model = model
        self.threshold = threshold
        self.fold = fold
        self.num_workers = num_workers

    def __get_split_k_fold_data(self):
        x_train_data_fold_list, y_train_data_fold_list, x_test_data_fold_list, y_test_data_fold_list = [], [], [], []
        x_data_series, y_data_series = self.x_data_series, self.y_data_series
        skf = StratifiedKFold(n_splits=self.fold, shuffle=True)
        for train_index, test_index in skf.split(x_data_series, y_data_series):
            x_train_data_list, y_train_data_list = (x_data_series[train_index].tolist()
                                                    , y_data_series[train_index].tolist())
            x_test_data_list, y_test_data_list = x_data_series[test_index].tolist(), y_data_series[test_index].tolist()
            x_train_data_fold_list.append(x_train_data_list)
            y_train_data_fold_list.append(y_train_data_list)
            x_test_data_fold_list.append(x_test_data_list)
            y_test_data_fold_list.append(y_test_data_list)
        return x_train_data_fold_list, y_train_data_fold_list, x_test_data_fold_list, y_test_data_fold_list

    def __train(self, x_train_data_list, y_train_data_list):
        self.model.fit(x_train_data_list, y_train_data_list)

    def get_predict_probability_array(self, x_test_data_list):
        try:
            return self.model.predict_proba(x_test_data_list)[:, 1]
        except AttributeError:
            return self.model.decision_function(x_test_data_list)

    def get_auc_and_f1_score_evaluate_tuple(self, x_test_data_list, y_test_data_list):
        y_probability_array = self.get_predict_probability_array(x_test_data_list)
        y_predict_label_list = (y_probability_array >= self.threshold).astype(int)
        auc = roc_auc_score(y_test_data_list, y_probability_array)
        f1 = f1_score(y_test_data_list, y_predict_label_list)
        accuracy = np.sum(np.array(y_test_data_list) == np.array(y_predict_label_list)) / len(x_test_data_list)
        # [print(y_test_data_list[index], y_predict_label_list[index]) for index in range(len(y_test_data_list))]
        return auc, f1, accuracy

    def __save_predict_result(self, index):
        os.makedirs(f'result/{self.model.__class__.__name__}', exist_ok=True)
        y_probability_array = self.get_predict_probability_array(self.x_valid_data_series.tolist())
        y_predict_label_list = (y_probability_array >= self.threshold).astype(int)
        result_df = pd.DataFrame({"uuid": np.arange(1, len(y_predict_label_list) + 1), "Label": y_predict_label_list})
        result_df.to_csv(f'./result/{self.model.__class__.__name__}/{index}.csv', index=False)
        return result_df

    def get_average_evaluate_criterion(self):
        # get the average evaluate criterion of auc, f1, accuracy.
        auc_list, f1_list, accuracy_list = [], [], []
        x_train_data_fold_list, y_train_data_fold_list, x_test_data_fold_list, y_test_data_fold_list = (
            self.__get_split_k_fold_data())
        for index in range(len(x_train_data_fold_list)):
            x_train_data_list = x_train_data_fold_list[index]
            x_test_data_list = x_test_data_fold_list[index]
            y_train_data_list = y_train_data_fold_list[index]
            y_test_data_list = y_test_data_fold_list[index]
            self.__train(x_train_data_list, y_train_data_list)
            auc, f1, accuracy = self.get_auc_and_f1_score_evaluate_tuple(x_test_data_list, y_test_data_list)
            auc_list.append(auc)
            f1_list.append(f1)
            accuracy_list.append(accuracy)
        average_auc, average_f1, average_accuracy = np.mean(auc_list), np.mean(f1_list), np.mean(accuracy_list)
        return average_auc, average_f1, average_accuracy

    def main(self):
        x_train_data_fold_list, y_train_data_fold_list, x_test_data_fold_list, y_test_data_fold_list = (
            self.__get_split_k_fold_data())
        for index in range(len(x_train_data_fold_list)):
            x_train_data_list = x_train_data_fold_list[index]
            x_test_data_list = x_test_data_fold_list[index]
            y_train_data_list = y_train_data_fold_list[index]
            y_test_data_list = y_test_data_fold_list[index]
            self.__train(x_train_data_list, y_train_data_list)
            auc, f1, accuracy = self.get_auc_and_f1_score_evaluate_tuple(x_test_data_list, y_test_data_list)
            print(f'Fold: {index} - AUC: {auc} - F1: {f1} - Accuracy: {accuracy}')
            self.__save_predict_result(index)


class RandomForestClassifierModel(Model):
    def __init__(self, x_data_series, y_data_series, x_valid_data_series, n_estimators=10000, max_depth=None
                 , threshold=0.4, n_fold=5, num_workers=4):
        self.model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, n_jobs=num_workers)
        super(RandomForestClassifierModel, self).__init__(x_data_series, y_data_series, x_valid_data_series
                                                          , self.model, n_fold, threshold=threshold)


class GradientBoostingClassifierModel(Model):
    def __init__(self, x_data_series, y_data_series, x_valid_data_series, n_fold=5):
        self.model = GradientBoostingClassifier()
        super(GradientBoostingClassifierModel, self).__init__(x_data_series, y_data_series, x_valid_data_series
                                                              , self.model, n_fold)


class DecisionTreeClassifierModel(Model):
    def __init__(self, x_data_series, y_data_series, x_valid_data_series, n_fold=5):
        self.model = DecisionTreeClassifier()
        super(DecisionTreeClassifierModel, self).__init__(x_data_series, y_data_series, x_valid_data_series
                                                          , self.model, n_fold)


class KNeighborsClassifierModel(Model):
    def __init__(self, x_data_series, y_data_series, x_valid_data_series, n_fold=5, num_workers=8):
        self.model = KNeighborsClassifier(n_jobs=num_workers)
        super().__init__(x_data_series, y_data_series, x_valid_data_series, self.model, n_fold)


class GaussianNBClassifierModel(Model):
    def __init__(self, x_data_series, y_data_series, x_valid_data_series, n_fold=5):
        self.model = GaussianNB()
        super().__init__(x_data_series, y_data_series, x_valid_data_series, self.model, n_fold)


class PerceptronModel(Model):
    def __init__(self, x_data_series, y_data_series, x_valid_data_series, n_fold=5, num_workers=8):
        self.model = Perceptron(n_jobs=num_workers)
        super().__init__(x_data_series, y_data_series, x_valid_data_series, self.model, n_fold)


class LogisticRegressionModel(Model):
    def __init__(self, x_data_series, y_data_series, x_valid_data_series, n_fold=5, num_workers=8):
        self.model = LogisticRegression(max_iter=10000, n_jobs=num_workers)
        super().__init__(x_data_series, y_data_series, x_valid_data_series, self.model, n_fold)


class SGDClassifierModel(Model):
    def __init__(self, x_data_series, y_data_series, x_valid_data_series, n_fold=5, num_workers=8):
        self.model = SGDClassifier(n_jobs=num_workers)
        super().__init__(x_data_series, y_data_series, x_valid_data_series, self.model, n_fold)


class LinearSVCModel(Model):
    def __init__(self, x_data_series, y_data_series, x_valid_data_series, n_fold=5):
        self.model = LinearSVC(max_iter=5000)
        super().__init__(x_data_series, y_data_series, x_valid_data_series, self.model, n_fold)


class SVCModel(Model):
    def __init__(self, x_data_series, y_data_series, x_valid_data_series, n_fold=5):
        self.model = SVC(max_iter=5000)
        super().__init__(x_data_series, y_data_series, x_valid_data_series, self.model, n_fold)


def main(data: DataFile, num_workers, encode_list):
    x_data_series = data.get_encoded_series(is_train=True, encode_list=encode_list)
    y_data_series = data.get_label_series()
    x_valid_data_series = data.get_encoded_series(is_train=False, encode_list=encode_list)
    model_list = [
        RandomForestClassifierModel(x_data_series, y_data_series, x_valid_data_series, num_workers=num_workers)
        , GradientBoostingClassifierModel(x_data_series, y_data_series, x_valid_data_series)
        , DecisionTreeClassifierModel(x_data_series, y_data_series, x_valid_data_series)
        , KNeighborsClassifierModel(x_data_series, y_data_series, x_valid_data_series, num_workers=num_workers)
        , GaussianNBClassifierModel(x_data_series, y_data_series, x_valid_data_series)
        , PerceptronModel(x_data_series, y_data_series, x_valid_data_series, num_workers=num_workers)
        , LogisticRegressionModel(x_data_series, y_data_series, x_valid_data_series, num_workers=num_workers)
        , SGDClassifierModel(x_data_series, y_data_series, x_valid_data_series, num_workers=num_workers)
        , LinearSVCModel(x_data_series, y_data_series, x_valid_data_series)
        , SVCModel(x_data_series, y_data_series, x_valid_data_series)
    ]
    for model in model_list:
        print(f'Processing: {model.__class__.__name__}')
        model.main()


if __name__ == '__main__':
    num_workers = os.cpu_count() // 2  # 线程数
    # data = Data(training_data_file_path='./dataset/traindata-new.xlsx'
    #             , testing_data_file_path='./dataset/testdata-new.xlsx'
    #             , num_workers=num_workers)
    data = DataFile(training_data_file_path='../dataset/traindata-new - new 24.07.05.xlsx'
                    , testing_data_file_path='../dataset/testdata-new.xlsx'
                    , num_workers=num_workers)
    # data = DataFile(training_data_file_path='./dataset/protac.xlsx'
    #                 , testing_data_file_path='./dataset/testdata-new.xlsx'
    #                 , num_workers=num_workers)
    encode_list = [
                    'AAC',
                    "topological_fingerprints",
                    "maccs",
                    "atom_pairs",
                    "morgan_fingerprints",
                    "uniprot",
                    "target",
                    'e3_ligase',
                    'assay',
                    'qualitative_trait',
                    'targeting_ligand_and_e3_ligase_recruiter'
                   ]
    main(data, num_workers, encode_list)

