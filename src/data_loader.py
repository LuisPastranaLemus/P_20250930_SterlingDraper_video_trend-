# data_loader.py for opening dataset files

import pandas as pd
import os
import zipfile

# Function to load a dataset from a ZIP file, with optional read_zip arguments
def load_dataset_from_zip(zip_path: str, filename: str, **kwargs) -> pd.DataFrame:
    """
    Loads a CSV or Excel file from within a ZIP archive into a DataFrame.
    
    Args:
        zip_path (str): Path to the ZIP file.
        filename (str): Name of the CSV or Excel file inside the ZIP.
        kwargs: Additional parameters passed to pd.read_csv or pd.read_excel.
    
    Returns:
        pd.DataFrame: Loaded DataFrame.
    
    Raises:
        FileNotFoundError: If the ZIP file does not exist.
        KeyError: If the specified file is not found in the ZIP.
        ValueError: If the file extension is not supported.
    """
    if not os.path.exists(zip_path):
        print(f"\n*** Error *** \nFile not found: {zip_path}")
        print("Current working directory:", os.getcwd())
        raise FileNotFoundError(f"File not found: {zip_path}")

    with zipfile.ZipFile(zip_path) as z:
        if filename not in z.namelist():
            raise KeyError(f"The file '{filename}' was not found in the ZIP archive.")
        
        with z.open(filename) as file:
            ext = os.path.splitext(filename)[1].lower()
            if ext == '.csv':
                df = pd.read_csv(file, **kwargs)
            elif ext in ['.xls', '.xlsx']:
                df = pd.read_excel(file, **kwargs)
            else:
                raise ValueError(f"Unsupported file extension '{ext}'. Only .csv, .xls and .xlsx are supported.")
    return df

# Function to load a dataset from a CSV file, with optional read_csv arguments
def load_dataset_from_csv(path, filename: str, **kwargs):
    """
    Loads a CSV file into a pandas DataFrame from a given path and filename.

    Parameters:
    path (Path or str): Directory path where the file is located.
    filename (str): Name of the CSV file.
    **kwargs: Additional keyword arguments to pass to pd.read_csv() (e.g., delimiter, encoding, dtype).

    Returns:
    DataFrame: Loaded dataset.

    Raises:
    FileNotFoundError: If the specified file does not exist.
    """

    full_path = path / filename

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"*** Error ***\nFile not found: {full_path}\nCurrent working directory: {os.getcwd()}")

    df = pd.read_csv(full_path, **kwargs)

    return df

# Function to load a dataset from an Excel file with optional read_excel arguments
def load_dataset_from_excel(path, filename: str, **kwargs):
    """
    Loads an Excel file into a pandas DataFrame from a specified directory and filename.

    Parameters:
    path (Path or str): Directory path where the Excel file is stored.
    filename (str): Name of the Excel file (e.g., 'data.xlsx').
    **kwargs: Additional arguments passed to pd.read_excel() (e.g., sheet_name, dtype, engine).

    Returns:
    DataFrame: Loaded dataset.

    Raises:
    FileNotFoundError: If the file does not exist at the specified location.
    """

    full_path = path / filename

    if not os.path.exists(full_path):
        raise FileNotFoundError(f"*** Error ***\nFile not found: {full_path}\nCurrent working directory: {os.getcwd()}")

    df = pd.read_excel(full_path, **kwargs)

    return df

# Function to convert a list of records into a pandas DataFrame
def load_dataset_from_list(data_list):
    """
    Converts a list of dictionaries or tuples into a pandas DataFrame.

    Parameters:
    data_list (list): A list of dictionaries (preferred) or records to be converted.

    Returns:
    DataFrame: A pandas DataFrame containing the provided data.
    """
    
    df = pd.DataFrame(data_list)
    return df

# Function to convert a dictionary into a pandas DataFrame
def load_dataset_from_dict(data_dict):
    """
    Converts a dictionary into a pandas DataFrame.

    Parameters:
    data_dict (dict): A dictionary where keys represent column names and values are lists of column data.

    Returns:
    DataFrame: A pandas DataFrame constructed from the dictionary.
    """
    
    df = pd.DataFrame.from_dict(data_dict, orient='columns')
    return df
