import os
import re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold, chi2
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RepeatedStratifiedKFold, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier, AdaBoostClassifier
from sklearn.metrics import roc_curve, roc_auc_score, auc, classification_report, f1_score, \
    confusion_matrix, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, PowerTransformer, RobustScaler, \
    QuantileTransformer
from sklearn.decomposition import PCA

# Change to own directory; this code assumes that all feature types are stored in a single folder
# titled "keras_data_OUTPUT, and that all labels are stored in a single .csv file called keras_labels.csv. "
feature_dir = r"/Users/matejloncaric/Documents/UvA/Year_4/MSc Thesis/Matej_all_stuff/keras_data_OUTPUT/"
labels_dir = r"/Users/matejloncaric/Documents/UvA/Year_4/MSc Thesis/Matej_all_stuff/keras_data_OUTPUT/keras_labels.csv"
save_dir = r"/Users/matejloncaric/Documents/UvA/Year_4/MSc Thesis/Matej_all_stuff/model_outputs/"
output_path = r"/Users/matejloncaric/Documents/UvA/Year_4/MSc Thesis/Matej_all_stuff/results"

pipelines = {
    'knn': Pipeline([('clf', KNeighborsClassifier())]),
    'svc': Pipeline([('clf', SVC(probability=True, random_state=42))]),
    'logreg': Pipeline([('clf', LogisticRegression(random_state=42))]),
    'xgb': Pipeline([('clf', XGBClassifier(random_state=42))]),
    'lgbm': Pipeline([('clf', LGBMClassifier(random_state=42))]),
    'rf': Pipeline([('clf', RandomForestClassifier(random_state=42))]),
    'adaboost': Pipeline([('clf', AdaBoostClassifier(random_state=42))])
}

param_grids = {
    'knn': {'clf__n_neighbors': [1, 3, 5, 7, 9], 'clf__weights': ['uniform', 'distance'],
            'clf__metric': ['euclidean', 'manhattan', 'minkowski']},
    'svc': {'clf__C': [0.1, 1, 10],
            'clf__gamma': [0.1, 1, 10]},
    'logreg': {'clf__solver': ['newton-cg', 'lbfgs', 'liblinear'],
               'clf__penalty': ['l2'], 'clf__C': [100, 10, 1.0, 0.1, 0.01]},
    'xgb': {
        'clf__n_estimators': [50, 100],
        'clf__max_depth': [3, 5],
        'clf__learning_rate': [0.01, 0.1],
        'clf__subsample': [0.5, 1],
    },
    'lgbm': {
        'clf__max_depth': [3, 5],
        'clf__n_estimators': [50, 100],
        'clf__learning_rate': [0.01, 0.1],
        'clf__is_unbalance': [True],
        'clf__boosting_type': ['gbdt', 'dart'],
    },
    'rf': {'clf__n_estimators': [20, 50, 100],
           'clf__max_depth': [None, 5, 10],
           'clf__class_weight': ['balanced', 'balanced_subsample']},
    'adaboost': {
        'clf__n_estimators': [50, 100],
        'clf__learning_rate': [0.01, 0.1],
    }
}


def load_features(feature_type, feature_dirs, labels_dirs):
    """
    Load features and labels from specified directories based on the CNN architecture. The architecture files have to
    be named as "ARCHITECTURE_features.csv".

    Parameters:
        :param feature_type: Specify the CNN architecture from which the features were extracted.
        :param feature_dirs: Directory containing feature files.
        :param labels_dirs: File path for the labels CSV file.

    :return: Returns X (features) and y (labels) as NumPy arrays.
    """
    architecture_files = dict(VGG16="VGG16_features.csv",
                              VGG19="VGG19_features.csv",
                              DLRF="ResNet50_2.csv",
                              InceptionResNet="IncRes_features.csv",
                              HCRF="radiomics_clean_1.csv",
                              HRF="merged.csv")

    features_file = os.path.join(feature_dirs, architecture_files[feature_type])
    labels = pd.read_csv(labels_dirs, index_col=0)
    features = pd.read_csv(features_file, index_col=0)

    common_indices = features.index.intersection(labels.index)
    features = features.loc[common_indices]
    labels = labels.loc[common_indices]

    X = features.values
    y = labels.values.ravel()

    return X, y, feature_type


def get_scaler(scaler):
    """
    Returns the scaler object based on the specified scaler name.

    :param: scaler (str): Name of the scaler to be used.
    :return: object: Scaler object based on the specified name.
    :raise: AssertionError if scaler is not one of the available scalers.
    """
    scalers = {
        'Standard': StandardScaler(),
        'MinMax': MinMaxScaler(),
        'MaxAbs': MaxAbsScaler(),
        'Power': PowerTransformer(),
        'Robust': RobustScaler(),
        'Quantile': QuantileTransformer()
    }

    assert scaler in scalers, f"scaler must be one of {list(scalers.keys())}"
    return scalers[scaler]


def variance_threshold_selector(data, threshold=0.5):
    """
    Selects features with variance greater than the specified threshold.

    Parameters:
        :param: data (array-like): Input data, where rows represent samples and columns represent features.
        :param: threshold (float, optional): Variance threshold. Features with variance below this threshold
                will be removed. Defaults to 0.5.

    :return: ndarray: Transformed data with low variance features removed.
    """
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[:, selector.get_support(indices=True)]


def remove_variance(arr):
    """
    Removes features with low variance from the input array.

    :param: arr (array-like): Input data, where rows represent samples and columns represent features.
    :return: arr (array-like): Transformed data with low variance features removed.
    """
    min_variance = 1e-3
    low_variance = variance_threshold_selector(arr, min_variance)
    return low_variance


def get_imbalance_solver(solver):
    """
    Returns the specified imbalance solver object.

    :param: solver (str): Name of the imbalance solver.
    :return: object: Imbalance solver object.
    :raise: AssertionError: If the specified solver is not one of the available solvers.
    """
    solvers = {
        "SMOTE": SMOTE(random_state=42),
        "SMOTEENN": SMOTEENN(random_state=42, sampling_strategy="not majority"),
        "RandomOverSampler": RandomOverSampler(sampling_strategy=0.90),
        "RandomUnderSampler": RandomUnderSampler(sampling_strategy='majority'),
        "ADASYN": ADASYN(random_state=42),
        "BorderlineSMOTE": BorderlineSMOTE(random_state=42)
    }

    assert solver in solvers, f"solver has to be one of {list(solvers.keys())}"
    return solvers[solver]


def preprocess_data(X_train, X_test, y_train, feature_type, solver=None, use_skb=False, skb_number=None, use_pca=False,
                    pca_components=None, use_scaler=False, scaler=None):
    """
    Preprocesses the training and test data based on the specified preprocessing options.

    Parameters:
        :param: X_train (np.ndarray): Training features.
        :param: X_test (np.ndarray): Test features.
        :param: y_train (np.ndarray): Training labels.
        :param: feature_type (str): Type of architecture used for feature extraction.
        :param: solver (str, optional): Which solving technique to apply for class imbalances. Defaults to None.
        :param: use_skb (bool, optional): Whether to use SelectKBest for feature selection. Defaults to False.
        :param: skb_number (int, optional): Number of top features to select using SelectKBest. Defaults to None.
        :param: use_pca (bool, optional): Whether to use Principal Component Analysis (PCA) for dimensionality
                reduction. Defaults to False.
        :param: use_scaler (bool, optional): Whether to apply feature scaling using a scaler. Defaults to False.
        :param: scaler (object, optional): Feature scaler object to use for scaling. Defaults to None.

    :return: tuple of three np.ndarray: Preprocessed X_train, X_test, and y_train.
    """
    # if feature_type in ["HRF"]:  # Significant negative impact on AUC - not used
    #
    #     X_train = remove_variance(X_train)
    #     selected_features = np.arange(X_train.shape[1])
    #     X_test = X_test[:, selected_features]
    #     # print("Variance removed!")

    # if use_scaler and scaler is not None:
    #
    #     scaler_obj = get_scaler(scaler)
    #     X_train = scaler_obj.fit_transform(X_train)
    #     X_test = scaler_obj.transform(X_test)
    #     # print(X_train)

    if use_skb and skb_number is not None:

        # selector = SelectKBest(f_classif, k=skb_number) # Use for negative features
        selector = SelectKBest(chi2, k=skb_number) # Use for positive features with categorical targets
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)

    if solver is not None:

        solver = get_imbalance_solver(solver)
        X_train, y_train = solver.fit_resample(X_train, y_train)

    if use_scaler and scaler is not None:

        scaler_obj = get_scaler(scaler)
        X_train = scaler_obj.fit_transform(X_train)
        X_test = scaler_obj.transform(X_test)

    if use_pca and pca_components is not None:

        # pca = PCA()
        # explained_variance_ratio_cumsum = np.cumsum(pca.fit(X_train).explained_variance_ratio_)
        # pca_components = np.argmax(explained_variance_ratio_cumsum >= 0.95) + 1
        pca = PCA(n_components=pca_components)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    return X_train, X_test, y_train


def select_best_threshold(classifier, X_test, y_test):
    """
    Selects the best threshold for a given classifier based on Youden's J statistic on the test set.

    Parameters:
        :param: classifier (sklearn estimator): Classifier object with predict_proba() method.
        :param: X_test (numpy.ndarray): Test features.
        :param: y_test (numpy.ndarray): Test labels.

    :return: float: Best threshold value.
    """
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    j_scores = tpr - fpr

    # Find the threshold index that maximizes Youden's J statistic
    best_threshold_index = np.argmax(j_scores)
    best_threshold = thresholds[best_threshold_index]

    return best_threshold


def plot_curve(feature_type, skb_number, name, y_true_list, y_scores_list, save_dir, overwrite=True):
    """
    Plots the average ROC (including the Average AUC with st. dev) for a classifier and saves it as an image.

    Parameters:
        :param: feature_type (str): Type of CNN architecture used.
        :param: skb_number (int): Number of features selected.
        :param: name (str): Name or label of the classifier.
        :param: y_true_list (list): List of true labels for each fold.
        :param: y_scores_list (list): List of predicted probabilities for each fold.
        :param: save_dir (str): Directory to save the image.
        :param: overwrite (bool): Whether to overwrite the ROC plot if it already exists. Defaults to True.

    :return: None
    """
    plt.figure()

    # Calculate mean and standard deviation for each false positive rate
    mean_fpr = np.linspace(0, 1, 100)
    tprs = []
    aucs = []

    for i in range(len(y_true_list)):
        fpr, tpr, _ = roc_curve(y_true_list[i], y_scores_list[i])
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = roc_auc_score(y_true_list[i], y_scores_list[i])
        aucs.append(roc_auc)

    # Plot random baseline
    plt.plot([0, 1], [0, 1], 'r--', label='Random')

    # Calculate mean and standard deviation of AUC scores
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    # Plot mean ROC curve
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    plt.plot(mean_fpr, mean_tpr, color='b', label=f"Mean ROC (AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})")

    # Plot standard deviation shaded area
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"ROC Curve for Features Extracted from {feature_type}")
    plt.legend(loc='lower right')

    save_path = os.path.join(save_dir, "graphs", f"{feature_type}_{skb_number}_{name}_roc_curve.png")

    if os.path.exists(save_path) and not overwrite:
        print(f"Warning: File already exists at {save_path}. Skipping saving the image.")
    else:
        plt.savefig(save_path)
        print(f"ROC curve plot saved at {save_path}.")

    plt.close()


def save_results(name, feature_type, auc, sensitivity, specificity, f1, save_dirs, solver_used, skb_used, skb_number,
                 pca_used, pca_components, scaler_name):
    """
    Saves the evaluation results of a classifier to a .txt file, and prints a confirmation.

    Parameters:
        :param: name (str): Name or label of the classifier.
        :param: feature_type (str): Type of CNN architecture used.
        :param: auc (float): AUC score.
        :param: sensitivity (float): Sensitivity.
        :param: specificity (float): Specificity.
        :param: f1_score (float): F1 score.
        :param: save_dirs (str): Directory to save the .txt file.
        :param: solver_used (str): Name of the solver used.
        :param: skb_used (bool): Whether SelectKBest was used.
        :param: skb_number (int): Number of top features selected with SelectKBest.
        :param: pca_used (bool): Whether PCA was used.
        :param: pca_components (int): Number of principal components used with PCA.
        :param: scaler_name (str): Name of the scaler used.

    :return: None

    """
    save_path = os.path.join(save_dirs, f"{feature_type}_{name}_results.txt")

    header = f"{name} classifier with features extracted from {feature_type}"
    preprocessing_info = (
        f"Preprocessing: Class imbalance solver={solver_used}, SKB={skb_used}, SKB features={skb_number}, "
        f"PCA={pca_used}, PCA components={pca_components}, Scaler={scaler_name}"
    )

    results_table = pd.DataFrame({
        'Metric': ['AUC', 'Sensitivity', 'Specificity', 'F1 Score'],
        'Value': [auc, sensitivity, specificity, f1]
    })

    mode = 'a' if os.path.exists(save_path) else 'w'
    with open(save_path, mode) as file:
        file.write(f"\n\n{header}\n")
        file.write(f"{preprocessing_info}\n\n")
        file.write('Results:\n')
        file.write(results_table.to_string(index=False))

    print(f"Results saved to: {save_path}")


def print_results(name, skb_number, solver, scaler, avg_auc, avg_sens, avg_spec, avg_f1):
    """
    Prints the average performance metrics for each classifier to the console, along with details of the preprocessing
    steps that were applied. Called only if the results are not exported to a .txt file.

    Parameters:
        :param: name (str): Type of classifier used.
        :param: skb_number (int): Number of features used.
        :param: solver (str): Type of class imbalance solver used.
        :param: scaler (str): Type of scaler used.
        :param: avg_auc (float): Average AUC across all folds.
        :param: avg_sens (float): Average sensitivity across all folds.
        :param: avg_spec (float): Average specificity across all folds.
        :param: avg_f1 (float): Average F1 score across all folds.

    :return: None
    """
    print(f"\n{name} test scores for {skb_number} features with {solver} and {scaler} scaler:\n"
          f"AUC: {avg_auc:.4f}\n"
          f"sensitivity: {avg_sens:.4f}\n"
          f"specificity: {avg_spec:.4f}\n"
          f"f1 score: {avg_f1:.4f}\n")


def get_metrics(y_true, y_pred_proba, threshold):
    """
    Calculate performance metrics based on predicted probabilities and true labels.

    Parameters:
        :param: y_true (array-like): True labels.
        :param: y_pred_proba (array-like): Predicted probabilities of the positive class.
        :param: threshold (float): Threshold for binary classification.

    :return: tuple of 4 floats and 1 Scikit-Learn object : (AUC, sensitivity, specificity, F1, cm)
    """
    y_pred = (y_pred_proba > threshold).astype(int)
    auc = roc_auc_score(y_true, y_pred_proba)
    cm = confusion_matrix(y_true, y_pred)
    sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    f1 = f1_score(y_true, y_pred)
    return auc, sensitivity, specificity, f1, cm


def get_avg_metrics(classifier_auc_scores, classifier_sens_scores, classifier_spec_scores, classifier_f1_scores):
    """
    Calculates the average performance metrics (AUC, Sensitivity, Specificity, F1 Score) for a classifier
    given lists of scores.

    Parameters:
        :param: classifier_auc_scores (list): List of AUC scores for the classifier across multiple runs/folds.
        :param: classifier_sens_scores (list): List of Sensitivity scores for the classifier across multiple runs/folds.
        :param: classifier_spec_scores (list): List of Specificity scores for the classifier across multiple runs/folds.
        :param: classifier_f1_scores (list): List of F1 scores for the classifier across multiple runs/folds.

    :return: tuple: A tuple containing the average AUC, average Sensitivity, average Specificity, and average F1 Score.
    """
    avg_auc = np.mean(classifier_auc_scores)
    avg_spec = np.mean(classifier_sens_scores)
    avg_sens = np.mean(classifier_spec_scores)
    avg_f1 = np.mean(classifier_f1_scores)

    return avg_auc, avg_spec, avg_sens, avg_f1


def get_classifier_dicts():
    """
    Creates four dictionaries to store average performance metrics per classifier.

    :return: tuple of dictionaries: classifiers_avg_auc, classifiers_avg_sens, classifiers_avg_spec, classifiers_avg_f1
    """
    classifiers_avg_auc = {}
    classifiers_avg_sens = {}
    classifiers_avg_spec = {}
    classifiers_avg_f1 = {}
    return classifiers_avg_auc, classifiers_avg_sens, classifiers_avg_spec, classifiers_avg_f1


def get_scores_lists():
    """
    Creates six lists to store the true labels and scores in each fold, as well as to store the AUC, sensitivity,
    specificity, and F1 score in each fold.

    :return: tuple of 6 lists: y_true_list, y_scores_list, classifier_auc_scores, classifier_sens_scores,
             classifier_spec_scores, classifier_f1_scores
    """
    y_true_list = []
    y_scores_list = []
    classifier_auc_scores = []
    classifier_sens_scores = []
    classifier_spec_scores = []
    classifier_f1_scores = []
    return y_true_list, y_scores_list, classifier_auc_scores, classifier_sens_scores, classifier_spec_scores, \
        classifier_f1_scores


def perform_grid_search(X_train, y_train, pipeline, name):
    """
    Performs a grid search to determine the best parameters for the given pipeline using 10-fold cross-validation,
    and fits the model with the optimal parameters.

    Parameters:
        :param: X_train (array-like): Feature matrix of the training data.
        :param: y_train (array-like): Target variable vector of the training data.
        :param: pipeline (Pipeline object): An instance of a scikit-learn Pipeline, specifying the steps for
                preprocessing and model training.
        :param: name (str): A key from the param_grids dictionary to access the grid of hyperparameters
                to be used in the search.

    :return: GridSearchCV object: A trained GridSearchCV object with the optimal parameters for the given pipeline.
    """
    grid_search = GridSearchCV(pipeline, param_grids[name], cv=10, scoring="roc_auc")
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_


def append_fold_scores(y_true_list, y_scores_list, classifier_auc_scores, classifier_sens_scores,
                       classifier_spec_scores, classifier_f1_scores, y_test, y_pred_proba, auc, sensitivity,
                       specificity, f1):
    """
    Appends the performance metrics, true labels, and prediction probabilities of each fold to the corresponding lists.
    This function is called inside a cross-validation loop to keep track of scores for each fold.

    Parameters:
        :param: y_true_list (list): List to which the true labels of each fold will be appended.
        :param: y_scores_list (list): List to which the predicted probabilities of each fold will be appended.
        :param: classifier_auc_scores (list): List to which the AUC scores of each fold will be appended.
        :param: classifier_sens_scores (list): List to which the sensitivity scores of each fold will be appended.
        :param: classifier_spec_scores (list): List to which the specificity scores of each fold will be appended.
        :param: classifier_f1_scores (list): List to which the F1 scores of each fold will be appended.
        :param: y_test (array-like): True labels of the current fold.
        :param: y_pred_proba (array-like): Predicted probabilities of the current fold.
        :param: auc (float): AUC score of the current fold.
        :param: sensitivity (float): Sensitivity score of the current fold.
        :param: specificity (float): Specificity score of the current fold.
        :param: f1 (float): F1 score of the current fold.

    :return: None
    """
    # Append the scores to the respective lists
    classifier_auc_scores.append(auc)
    classifier_sens_scores.append(sensitivity)
    classifier_spec_scores.append(specificity)
    classifier_f1_scores.append(f1)

    # Append the predictions and labels to the fold-specific lists
    y_true_list.append(y_test)
    y_scores_list.append(y_pred_proba)


def store_average_scores(classifiers_avg_auc, classifiers_avg_sens, classifiers_avg_spec, classifiers_avg_f1,
                         name, avg_auc, avg_sens, avg_spec, avg_f1):
    """
    Stores the average performance metrics of a classifier in corresponding dictionaries. This function is  called after
    cross-validation is completed for a classifier, to store its average AUC, sensitivity, specificity, and F1 scores.

    Parameters:
        :param: classifiers_avg_auc (dict): Dictionary to store the average AUC scores of classifiers.
        :param: classifiers_avg_sens (dict): Dictionary to store the average sensitivity scores of classifiers.
        :param: classifiers_avg_spec (dict): Dictionary to store the average specificity scores of classifiers.
        :param: classifiers_avg_f1 (dict): Dictionary to store the average F1 scores of classifiers.
        :param: name (str): Type of classifier used.
        :param: avg_auc (float): The average AUC score of the classifier.
        :param: avg_sens (float): The average sensitivity score of the classifier.
        :param: avg_spec (float): The average specificity score of the classifier.
        :param: avg_f1 (float): The average F1 score of the classifier.

    :return: None
    """
    classifiers_avg_auc[name] = avg_auc
    classifiers_avg_spec[name] = avg_spec
    classifiers_avg_sens[name] = avg_sens
    classifiers_avg_f1[name] = avg_f1


def train_classifiers(X, y, feature_type, solver=None, use_skb=False, skb_number=None, use_pca=False,
                      pca_components=None, use_scaler=False, scaler=None, plot_roc=True, export_results=True):
    """
    Trains and evaluates multiple classifiers on the given dataset with optional preprocessing techniques.

    Parameters:
        :param: X (numpy.ndarray): Input features.
        :param: y (numpy.ndarray): Target labels.
        :param: feature_type (string): Type of CNN architecture used.
        :param: solver (str, optional): Which solving technique to apply for class imbalances. Defaults to None.
        :param: use_skb (bool, optional): Whether to use SelectKBest for feature selection. Defaults to False.
        :param: skb_number (int, optional): Number of top features to select using SelectKBest. Defaults to None.
        :param: use_pca (bool, optional): Whether to use Principal Component Analysis (PCA) for dimensionality
                reduction. Defaults to False.
        :param: pca_components (int, optional): Number of components to keep with PCA. Defaults to None.
        :param: use_scaler (bool, optional): Whether to apply feature scaling using a scaler. Defaults to False.
        :param: scaler (object, optional): Feature scaler object to use for scaling. Defaults to None.
        :param: plot_roc (bool, optional): Whether to plot the Receiver Operating Curve. Defaults to False.
        :param: export_results (bool, optional): Whether to export results to .txt files. Defaults to True.

    :return: None
    """

    rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=2, random_state=42)

    # Dictionaries for storing the average results
    classifiers_avg_auc, classifiers_avg_sens, classifiers_avg_spec, classifiers_avg_f1 = get_classifier_dicts()

    for name, pipeline in pipelines.items():

        # Lists for storing the results during the training
        y_true_list, y_scores_list, classifier_auc_scores, classifier_sens_scores, classifier_spec_scores, \
            classifier_f1_scores = get_scores_lists()

        for train_index, test_index in rskf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            X_train, X_test, y_train = preprocess_data(X_train, X_test, y_train, feature_type, solver,
                                                       use_skb, skb_number, use_pca, pca_components, use_scaler, scaler)

            # Perform hyperparameter tuning using GridSearchCV
            clf = perform_grid_search(X_train, y_train, pipeline, name)
            clf.fit(X_train, y_train)

            # Select the best threshold
            best_threshold = select_best_threshold(clf, X_test, y_test)

            # Apply the best threshold and get the final predictions
            y_pred_proba = clf.predict_proba(X_test)[:, 1]

            # Calculate the evaluation metrics for the current fold
            auc, sensitivity, specificity, f1, cm = get_metrics(y_true=y_test, y_pred_proba=y_pred_proba,
                                                                threshold=best_threshold)

            append_fold_scores(y_true_list, y_scores_list, classifier_auc_scores, classifier_sens_scores,
                               classifier_spec_scores, classifier_f1_scores, y_test, y_pred_proba, auc, sensitivity,
                               specificity, f1)

        # Calculate the average scores across all folds for the current classifier
        avg_auc, avg_spec, avg_sens, avg_f1 = get_avg_metrics(classifier_auc_scores, classifier_sens_scores,
                                                              classifier_spec_scores, classifier_f1_scores)

        store_average_scores(classifiers_avg_auc, classifiers_avg_sens, classifiers_avg_spec, classifiers_avg_f1,
                             name, avg_auc, avg_sens, avg_spec, avg_f1)

        if export_results:
            save_results(name=name, feature_type=feature_type, auc=avg_auc, sensitivity=avg_sens, specificity=avg_spec,
                         f1=avg_f1, save_dirs=save_dir, solver_used=solver, skb_used=use_skb, skb_number=skb_number,
                         pca_used=use_pca, pca_components=pca_components, scaler_name=scaler)

        else:
            print_results(name, skb_number, solver, scaler, avg_auc, avg_sens, avg_spec, avg_f1)

        if plot_roc:
            plot_curve(feature_type, skb_number, name, y_true_list, y_scores_list, save_dir)


def get_unique_filename(filename, output_path):
    """
    Generates a unique filename by appending a suffix if the file already exists. This function is used to avoid
    overwriting existing files. If the specified filename already exists in the directory, it appends a suffix in the
    format _1, _2, ... to the filename and returns the modified filename.

    Parameters:
        :param: filename (str): The original filename (including extension) to be checked.
        :param: output_path (str): The directory in which the file will be saved.

    :return: str: A unique filename with a suffix appended if necessary.
    """
    base_name, extension = os.path.splitext(filename)
    counter = 1
    new_filename = os.path.join(output_path, filename)

    while os.path.exists(new_filename):
        new_filename = os.path.join(output_path, f"{base_name}_{counter}{extension}")
        counter += 1

    return new_filename


def get_best_results(save_dir, output_path):
    """
    Extracts the best results from text files and saves them in CSV files. For each classifier and feature type, this
    function scans through the results stored in text files, extracts the best performance scores based on AUC,
    and stores them in a CSV file. It generates a separate CSV file for each feature type.

    Note: The text files should be named in the format "{feature_type}_{classifier}_results.txt" and
    contain performance metrics including AUC, sensitivity, specificity, and F1 Score.

    Parameters:
        :param: save_dir (str): Directory where the results text files are stored.
        :param: output_path (str): Directory where the CSV files will be saved.

    :return: None
    """
    feature_types = ["ResNet50", "Radiomics", "Merged"]
    classifiers = ["adaboost", "knn", "rf", "svc", "xgb", "lgbm", "logreg"]

    for feature_type in feature_types:

        results = []

        for classifier in classifiers:

            file_name = f"{feature_type}_{classifier}_results.txt"
            file_path = os.path.join(save_dir, file_name)

            max_auc = 0
            corresponding_scores = {}
            with open(file_path, 'r') as file:
                lines = file.readlines()

                for i, line in enumerate(lines):
                    if "AUC" in line:
                        auc = float(re.search(r"\d+\.\d+", line).group())
                        if auc > max_auc:
                            max_auc = round(auc, 4)
                            sens = round(float(re.search(r"\d+\.\d+", lines[i + 1]).group()), 4)
                            spec = round(float(re.search(r"\d+\.\d+", lines[i + 2]).group()), 4)
                            f1 = round(float(re.search(r"\d+\.\d+", lines[i + 3]).group()), 4)
                            corresponding_scores = {'AUC': max_auc, 'Sensitivity': sens, 'Specificity': spec, 'F1 Score': f1}

            results.append(corresponding_scores)

        filename = f'{feature_type}_results.csv'
        unique_filename = get_unique_filename(filename, output_path)
        df = pd.DataFrame(results, index=classifiers)
        df.to_csv(unique_filename)

    print('Results saved to CSV files in', output_path)


def main():
    X, y, feature_type = load_features("DLRF", feature_dir, labels_dir)

    # For radiomics, better results with PCA on LGBM, MinMax (with scaling before everythin) and BorderlineSMOTE
    # For deep features, 30-110 + SMOTEENN + chi2 for feature selection
    for number in [20, 30, 40, 50, 60, 70, 80, 90, 100, 110]:
        for scaler in ["Standard", "MinMax"]:
            for solver in ["BorderlineSMOTE", "ADASYN", "SMOTEENN"]:
                train_classifiers(X, y, feature_type, solver=solver, use_scaler=True, scaler=scaler, use_skb = True,
                                  skb_number=number, use_pca=True, export_results=False, plot_roc=False)

    # Add feature types to the function
    get_best_results(save_dir, output_path)


if __name__ == "__main__":
    main()
