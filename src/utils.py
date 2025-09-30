# utils.py for useful functions

import pandas as pd

# Function to set preferred display formatting options for pandas in Jupyter Notebooks
def format_notebook():
    """
    Configures pandas display settings for cleaner output in Jupyter notebooks.
    Adjusts row/column limits and character width for better readability.
    """
    pd.set_option('display.max_rows', 25)         # Show up to 25 rows
    pd.set_option('display.max_columns', 25)      # Show up to 25 columns
    pd.set_option('display.max_colwidth', 50)     # Limit each cell content to 50 characters
    pd.set_option('display.width', 150)           # Set total output width to 150 characters
