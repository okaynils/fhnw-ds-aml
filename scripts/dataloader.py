import pandas as pd
import os

class DataLoader:
    def __init__(self, base_path='data'):
        self.base_path = base_path

    def load_csv(self, file_name, sep=';', parse_dates=None):
        path = os.path.join(self.base_path, f"{file_name}.csv")

        df = pd.read_csv(path, sep=sep)

        if isinstance(parse_dates, dict):
            for column, fmt in parse_dates.items():
                df[column] = pd.to_datetime(df[column], format=fmt)
        elif parse_dates is not None:
            for column in parse_dates:
                df[column] = pd.to_datetime(df[column], errors='coerce')

        return df

    def list_datasets(self):
        files = os.listdir(self.base_path)
        return [file.split('.')[0] for file in files if file.endswith('.csv')]

    def preprocess_data(self, df, preprocessing_steps=None):
        if preprocessing_steps is not None:
            for step in preprocessing_steps:
                df = step(df)
        return df

    def merge_dataframes(self, df1, df2, on, how='inner'):
        return pd.merge(df1, df2, on=on, how=how)