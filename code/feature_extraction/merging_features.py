import os
import pandas as pd

radiomics_path = r"/Users/matejloncaric/Documents/UvA/Year_4/MSc Thesis/Matej_all_stuff/keras_data_OUTPUT/radiomics_clean_1.csv"
resnet_path = r"/Users/matejloncaric/Documents/UvA/Year_4/MSc Thesis/Matej_all_stuff/keras_data_OUTPUT/ResNet50_2.csv"

radiomics = pd.read_csv(radiomics_path, index_col=0)
resnet = pd.read_csv(resnet_path, index_col=0)

# Keep only the rows with common indices and merge the dataframes
merged_df = radiomics.merge(resnet, left_index=True, right_index=True)

merged_df.to_csv(r"/Users/matejloncaric/Documents/UvA/Year_4/MSc Thesis/Matej_all_stuff/keras_data_OUTPUT/merged.csv")