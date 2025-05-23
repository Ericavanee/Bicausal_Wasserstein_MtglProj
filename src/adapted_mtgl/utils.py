"""
Utils functions.
"""
import numpy as np

# utils for general
def get_params(rho,x,y,sigma = 1):
    params = {
    'rho': rho,
    'x': x,
    'y': y,
    'sigma': sigma}
    return params 

def save_list_to_txt(result, save_dir):
    with open(save_dir, 'w') as f:
        for item in result:
            f.write(f"{item}\n")
    print('Your list has been saved.')

def load_txt_as_matrix(file_path, delimiter=None, dtype=float):
    """
    Loads a 2D data text file as a NumPy ndarray.

    Parameters:
    -----------
    file_path : str
        Path to the text file.
    delimiter : str, optional
        The string used to separate values (default: None, which means any whitespace).
    dtype : type, optional
        Data type of the resulting NumPy array (default: float).

    Returns:
    --------
    np.ndarray
        A NumPy 2D array (matrix) with the loaded data.
    
    Raises:
    -------
    ValueError if the file contains inconsistent row lengths.
    IOError if the file cannot be read.
    """
    try:
        matrix = np.loadtxt(file_path, delimiter=delimiter, dtype=dtype)
        if len(matrix.shape) != 2:
            raise ValueError("Loaded data is not 2D. Check the file format.")
        return matrix
    except Exception as e:
        raise IOError(f"Error loading file {file_path}: {e}")

