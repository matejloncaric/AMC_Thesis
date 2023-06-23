import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

feature_dir = r"/Users/matejloncaric/Documents/UvA/Year_4/MSc Thesis/Matej_all_stuff/keras_data_OUTPUT/"
labels_dir = r"/Users/matejloncaric/Documents/UvA/Year_4/MSc Thesis/Matej_all_stuff/keras_data_OUTPUT/keras_labels.csv"
save_dir = r"/Users/matejloncaric/Documents/UvA/Year_4/MSc Thesis/Matej_all_stuff/scree_plots"
feature_types = ['VGG16', 'VGG19', 'ResNet50', 'InceptionResNet', 'Radiomics', "Merged"]


def load_features(feature_type, feature_dirs, labels_dirs):
    """
    Load features and labels from specified directories based on the CNN architecture. The architecture files have to
    be named as "ARCHITECTURE_features.csv".

    Arguments:
        :param feature_type: Specify the CNN architecture from which the features were extracted.
        :param feature_dirs: Directory containing feature files.
        :param labels_dirs: File path for the labels CSV file.

    :return: Returns X (features) and y (labels) as NumPy arrays.

    """

    architecture_files = dict(VGG16="VGG16_features.csv",
                              VGG19="VGG19_features.csv",
                              ResNet50="ResNet50_2.csv",
                              InceptionResNet="IncRes_features.csv",
                              Radiomics="radiomics_clean.csv",
                              Merged="merged.csv")

    features_file = os.path.join(feature_dirs, architecture_files[feature_type])
    labels = pd.read_csv(labels_dirs, index_col=0)
    features = pd.read_csv(features_file, index_col=0)

    common_indices = features.index.intersection(labels.index)
    features = features.loc[common_indices]
    labels = labels.loc[common_indices]

    X = features.values
    y = labels.values.ravel()

    return X, y, feature_type


def create_scree_plot(feature_type, save_dirs, pca, figsize=(8, 6), overwrite=True):
    """
    Create a scree plot for PCA eigenvalues.

    Parameters:
        :param: pca (sklearn.decomposition.PCA): PCA object fitted on the data.
        :param: figsize (tuple, optional): Figure size. Defaults to (8, 6).
        :param: overwrite (Boolean): whether to overwrite an existing image if one exists.

    :return: None
    """
    # Calculate the eigenvalues
    eigenvalues = pca.explained_variance_
    cum_eigenvalues = np.cumsum(eigenvalues)

    # Plot the scree plot
    plt.figure(figsize=figsize)
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, marker='o')
    plt.xlabel('Principal Components')
    plt.ylabel('Eigenvalues')
    plt.title(f'Scree Plot for {feature_type}')
    plt.grid(True)

    # Save the plot as an image; overwrite the image if the overwrite argument is not set to False
    save_path = os.path.join(save_dirs, f"{feature_type}_scree_plot.png")
    if os.path.exists(save_path) and not overwrite:
        print(f"Warning: File already exists at {save_path}. Skipping saving the image.")
    else:
        plt.savefig(save_path)
        print(f"Scree plot saved for {feature_type} at {save_path}")

    plt.close()


def main():
    for feature_type in feature_types:
        X, y, feature_type = load_features(feature_type, feature_dir, labels_dir)
        pca = PCA()
        pca.fit(X)
        create_scree_plot(feature_type, save_dir, pca)


if __name__ == "__main__":
    main()
