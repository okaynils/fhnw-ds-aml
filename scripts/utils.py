import pandas as pd

def add_prefix_except_id(df, prefix, id_exceptions=[]):
    """
    Adds a prefix to all column names in the DataFrame except those containing '_id' or 'id_', 
    with an exception for any column names explicitly listed in id_exceptions.

    Parameters:
    - df: pandas.DataFrame to be modified.
    - prefix: String prefix to be added.
    - id_exceptions: List of column names that contain '_id' or 'id_' but should still have the prefix added.

    Returns:
    - A new DataFrame with updated column names.
    """
    # Define a new DataFrame to avoid modifying the original one
    new_df = df.copy()
    
    # Rename columns, adding prefix based on the conditions
    new_df.columns = [
        f'{prefix}{col}' if (('_id' not in col and 'id_' not in col) or col in id_exceptions) else col 
        for col in df.columns
    ]
    
    return new_df