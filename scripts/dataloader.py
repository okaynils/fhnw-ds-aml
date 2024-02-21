import pandas as pd
import os

class DataLoader:
    def __init__(self, base_path='data'):
        """
        Initializes the DataLoader with a base path for the datasets.
        :param base_path: The base path where all CSV files are stored, default is 'data'.
        """
        self.base_path = base_path

    def load_csv(self, file_name, sep=';', parse_dates=None):
        """
        Loads a CSV file into a pandas DataFrame.
        :param file_name: Name of the CSV file to load (without .csv extension).
        :param sep: Separator used in the CSV file.
        :param parse_dates: Columns that should be parsed as dates.
        :return: pandas DataFrame containing the data from the CSV file.
        """
        path = os.path.join(self.base_path, f"{file_name}.csv")
        return pd.read_csv(path, sep=sep, parse_dates=parse_dates)

    def list_datasets(self):
        """
        Lists all available CSV datasets in the base path.
        :return: List of dataset names (file names without the .csv extension).
        """
        files = os.listdir(self.base_path)
        return [file.split('.')[0] for file in files if file.endswith('.csv')]

    def preprocess_data(self, df, preprocessing_steps=None):
        """
        Applies preprocessing steps to a DataFrame.
        :param df: DataFrame to preprocess.
        :param preprocessing_steps: List of preprocessing functions to apply to the DataFrame.
        :return: Preprocessed DataFrame.
        """
        if preprocessing_steps is not None:
            for step in preprocessing_steps:
                df = step(df)
        return df

    def merge_dataframes(self, df1, df2, on, how='inner'):
        """
        Merges two DataFrames.
        :param df1: First DataFrame.
        :param df2: Second DataFrame.
        :param on: Column or index level names to join on.
        :param how: Type of merge to be performed.
        :return: Merged DataFrame.
        """
        return pd.merge(df1, df2, on=on, how=how)