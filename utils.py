import datetime as dt
import random
from functools import wraps

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import yaml


def plot_images(tensor: torch.Tensor, name: str = None):
    """
    This function takes a tensor of images and plots them in a grid.

    Parameters:
        tensor (torch.Tensor): The tensor containing the images to be plotted.
        name (str, optional): The name of the file to save the plot. If not provided, the plot will be displayed.

    Returns:
        None
    """
    plt.style.use("dark_background")
    im = torchvision.utils.make_grid(tensor, normalize=True, scale_each=True)
    nrows = max(1, len(tensor) // 8)
    plt.figure(figsize=((len(tensor) * 2) / nrows, 2 * nrows))
    plt.imshow(im.permute(1, 2, 0), aspect="auto")
    plt.axis("off")
    plt.tight_layout()
    if name is not None:
        plt.savefig(name)
        plt.close()
    else:
        plt.show()


def get_gpu() -> torch.device:
    """
    Checks if a GPU is available.
    If GPU is available, it returns the device as 'cuda', otherwise it returns 'cpu' or 'mps' if it is macos.

    Returns:
        torch.device: The device available for computation.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(torch.cuda.get_device_name(0))
        print("Memory Usage:")
        print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB")
        print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 1), "GB")

    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Metal Performance Shaders framework (MPS)")

    else:
        device = torch.device("cpu")
        print("GPU is not available")

    devices = torch.cuda.device_count()
    print(f"{devices} Number of Devices Exists")

    return device


def load_yaml(path: str) -> dict:
    """
    Loads a YAML file from the specified `path` and returns the data.

    Parameters:
        path (str): The path to the YAML file.

    Returns:
        dict: The loaded YAML data.
    """
    with open(path, "r") as file:
        data = yaml.load(file, yaml.SafeLoader)
        return data


def log_step(func):
    """
    Decorator that logs the execution time and dataset size of a function.
    ---
    The `log_step` decorator wraps a function and prints information about the function's execution.
    It logs the function name, dataset size (length of the result), and the time taken to complete the function.
    This function can be improved by adding more information to the log depending on the use case.

    Parameters:
        func (function): The function to be wrapped.

    Returns:
        function: The wrapped function.

    Example:
        @log_step
        def my_function(*args, **kwargs):
            # Function code here

        result = my_function(*args, **kwargs)
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        tic = dt.datetime.now()
        result = func(*args, **kwargs)
        time_taken = str(dt.datetime.now() - tic)
        print("--------------------------------------------------")
        print(f"Function Name: {func.__name__}")
        print(f"Dataset size: {len(result)}")
        print(f"Took: {time_taken} seconds to complete")
        print("--------------------------------------------------")
        return result

    return wrapper


def setup_seed(seed: int = 42):
    """
    Set up the random seed for reproducibility.

    Parameters:
        seed (int): The seed value to set. Default is 42.

    Returns:
        None
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
