import pandas as pd

df = pd.read_excel(r"L:\basic\divi\jstoker\slicer_pdac\Master Students SS 23\Matej\keras_data_OUTPUT\radiomics_2.xlsx")

# Get the index of the column "original_shape_Elongation"
index_of_column = df.columns.get_loc('original_shape_Elongation')

# Get the column names starting from the first column after "original_shape_Elongation"
selected_columns = df.columns[index_of_column:]

# Create the new DataFrame with the selected columns
radiomics_clean = df.iloc[:, [0] + list(range(index_of_column+1, len(df.columns)))]

# Save the new DataFrame as a new Excel file
radiomics_clean.to_excel('radiomics_clean_2.xlsx', index=True)