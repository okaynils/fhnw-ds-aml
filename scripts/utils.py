import pandas as pd

def add_prefix_except_id(df, prefix):
    """
    Adds a prefix to all column names in the DataFrame except those containing '_id' or 'id_'.

    Parameters:
    - df: pandas.DataFrame to be modified.
    - prefix: String prefix to be added.

    Returns:
    - A new DataFrame with updated column names.
    """
    new_df = df.copy()
    
    new_df.columns = [f'{prefix}{col}' if '_id' not in col and 'id_' not in col else col for col in df.columns]
    
    return new_df