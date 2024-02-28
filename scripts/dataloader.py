import pandas as pd
import os
import json

class DataLoader:
    def __init__(self, base_path='data', translations_name=None):
        self.base_path = base_path
        self.translations = self.load_translations(path=f'{base_path}/{translations_name}') if translations_name else None

    def load_csv(self, file_name, sep=';', parse_dates=None):
        path = os.path.join(self.base_path, f"{file_name}.csv")
        df = pd.read_csv(path, sep=sep, low_memory=False)

        if isinstance(parse_dates, dict):
            for column, fmt in parse_dates.items():
                df[column] = df[column].astype(str).apply(lambda x: x.split(' ')[0])
                df[column] = pd.to_datetime(df[column], format=fmt, errors='coerce')
        elif parse_dates is not None:
            for column in parse_dates:
                df[column] = df[column].astype(str).apply(lambda x: x.split(' ')[0])
                df[column] = pd.to_datetime(df[column], errors='coerce')

        if self.translations:
            self.apply_translations(df, file_name)

        return df
    
    def load_translations(self, path):
        if path and os.path.exists(path):
            with open(path, 'r') as file:
                return json.load(file)
        else: 
            print(f'Translation File {path} not found!')
        return None

    def apply_translations(self, df, dataset_name):
        if dataset_name in self.translations:
            dataset_translations = self.translations[dataset_name]
            for variable, mapping in dataset_translations.items():
                if variable in df.columns:
                    df[variable] = df[variable].map(mapping)
                    print(f'Mapped {variable}:')
                    print(json.dumps(mapping, indent=4))


    def list_datasets(self, index=False):
        files = os.listdir(self.base_path)

        df = pd.DataFrame({
            'Dataset': [file.split('.')[0] for file in files if file.endswith('.csv')],
            'Number of Rows': [pd.read_csv(os.path.join(self.base_path, file)).shape[0] for file in files if file.endswith('.csv')]
            })

        return df

    def preprocess_data(self, df, preprocessing_steps=None):
        if preprocessing_steps is not None:
            for step in preprocessing_steps:
                df = step(df)
        return df

    def merge_dataframes(self, df1, df2, on, how='inner'):
        return pd.merge(df1, df2, on=on, how=how)