import datetime as dt
import random
from functools import wraps

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchinfo
import torchvision
import yaml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm


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
    im = torchvision.utils.make_grid(tensor, normalize=True, scale_each=True).cpu()
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

def load_and_preprocess_images(root_path: str, dataset_name: str, transform_train: torchvision.transforms.Compose, transform_val: torchvision.transforms.Compose):
    """
    Loads and preprocesses images from the specified dataset.

    Parameters:
        root_path (str): The root path of the dataset.
        dataset_name (str): The name of the dataset.
        transform (torchvision.transforms.Compose): The transformation to apply to the images.
    Returns:
        tuple: A tuple containing the train dataset and the validation dataset.
    """
    # Common parameters
    common_params_train = {"root": root_path, "transform": transform_train}
    common_params_val = {"root": root_path, "transform": transform_val}
    
    # Dataset-specific parameters
    if dataset_name == "cifar10":
        dataset_class = torchvision.datasets.CIFAR10
        common_params_train.update({"download": True, "train": True})
        common_params_val.update({"download": False, "train": False})
    elif dataset_name == "cifar100":
        dataset_class = torchvision.datasets.CIFAR100
        common_params_train.update({"download": True, "train": True})
        common_params_val.update({"download": False, "train": False})
    elif dataset_name == "stl10":
        dataset_class = torchvision.datasets.STL10
        common_params_train.update({"download": True, "split": "train"})
        common_params_val.update({"download": False, "split": "test"})
    elif dataset_name == "imagenet":
        dataset_class = torchvision.datasets.Imagenette
        common_params_train.update({"download": True, "split": "train"})
        common_params_val.update({"download": False, "split": "val"})
    elif dataset_name == "custom":
        dataset_class = torchvision.datasets.ImageFolder
        common_params_train["root"] = f"data/{dataset_name}/train"
        common_params_val["root"] = f"data/{dataset_name}/val"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create datasets
    train_dataset = dataset_class(**common_params_train)
    val_dataset = dataset_class(**common_params_val)

    return train_dataset, val_dataset

def visualize_features(features: torch.Tensor, labels: torch.Tensor, 
                       class_names: list, tsne_save_config: tuple, 
                       n_components_tsne: int = 2, pca_components: int = None):
    """
    Visualizes features using t-SNE dimensionality reduction technique.

    Parameters:
        features (torch.Tensor): Input features.
        labels (torch.Tensor): Labels corresponding to the features.
        class_names (list): List of class names.
        tsne_save_config (tuple): Configuration for saving t-SNE visualization.
            Should contain folder name, dataset name, and PCA mode.
        n_components_tsne (int, optional): Number of components for t-SNE. Defaults to 2.
        pca_components (int, optional): Number of components for PCA preprocessing. Defaults to 50.
    Returns:
        None
    """

    # Get the folder name, dataset name, and PCA mode from the config
    folder_name, dataset_name, pca_mode = tsne_save_config

    # Convert features and labels to numpy arrays
    features = features.cpu().numpy()
    labels = labels.cpu().numpy()

    # Optional PCA preprocessing
    if pca_components is not None:
        '''
        It is highly recommended to use another dimensionality reduction method 
        (e.g. PCA for dense data or TruncatedSVD for sparse data) to reduce the number 
        of dimensions to a reasonable amount (e.g. 50) if the number of features is very high.
        (Taken from t-SNE sklean documentation)
        '''
        pca = PCA(n_components=pca_components)
        features = pca.fit_transform(features)

    # Perform t-SNE
    tsne = TSNE(n_components=n_components_tsne, random_state=0)
    tsne_result = tsne.fit_transform(features)
    
    # Plot the t-SNE visualization
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='tab10', alpha=0.6)

    # Create a legend with class names
    legend_elements = scatter.legend_elements()[0]
    class_labels = [class_names[int(label)] for label in set(labels)]
    plt.legend(handles=legend_elements, labels=class_labels)

    plt.title("t-SNE visualization of features")
    plt.savefig(f"{folder_name}/tsne_visualize_dataset_{dataset_name}_with_pca_{pca_mode}.png")
    plt.show()

def extract_variance_components(cfg, train_dataset, val_dataset, device):
    """
    Extracts the variance components from the given datasets.
    Args:
        cfg (dict): Configuration dictionary.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        device (str): Device to be used for computation.
    Returns:
        tuple: A tuple containing the modified training dataset and validation dataset.
    """

    def fast_gram_eigh(X, major="C"):
        """
        compute the eigendecomposition of the Gram matrix:
        - XX.T using column (C) major notation
        - X.T@X using row (R) major notation
        """
        if major == "C":
            X_view = X.T
        else:
            X_view = X

        if X_view.shape[1] < X_view.shape[0]:
            print("Warning: using the slow path to compute the eigendecomposition")
            # this case is the usual formula
            U, S = torch.linalg.eigh(X_view.T @ X_view)
        else:
            print("Using the fast path to compute the eigendecomposition")
            # in this case we work in the tranpose domain
            U, S = torch.linalg.eigh(X_view @ X_view.T)
            S = X_view.T @ S
            S[:, U > 0] /= torch.sqrt(U[U > 0])

        return U, S

    # merge the train and val dataset
    dataset = train_dataset + val_dataset

    # Initialize a tensor to store flattened images
    images = torch.zeros(len(dataset), 3 * cfg["MAE"]["MODEL"]["image_size"] * cfg["MAE"]["MODEL"]["image_size"])

    # Load and flatten each image
    for i, (im, y) in tqdm(enumerate(dataset)):
        images[i] = im.flatten()

    # move images to device
    images = images.to(device)

    # Standardize by subtracting the mean
    mean_image = images.mean(0)
    # zero mean-centered (mean is subtracted from each vector).
    images -= mean_image

    # get spectral decomposition and normalize eigenvalues
    eigen_values, eigen_vectors = fast_gram_eigh(images, "R")
    # normalize the eigenvalues (amount of variance)
    eigen_values /= eigen_values.sum()
    # get the cumulative sum of the eigenvalues
    cumulative_variance = eigen_values.cumsum(dim=0)
    # get the number of components that explain the variance
    num_bottom_components = torch.count_nonzero(cumulative_variance < cfg["PCA"]["variance_cutoff"])
    num_top_components = torch.count_nonzero(cumulative_variance > cfg["PCA"]["variance_cutoff"])

    # get the bottom part (x% variance cutoff)
    bottom_images_train = ((images[list(range(len(train_dataset)))]) @ eigen_vectors[:, :num_bottom_components]) @ eigen_vectors[:, :num_bottom_components].T + mean_image
    bottom_images_val = ((images[list(range(len(train_dataset), len(dataset), 1))]) @ eigen_vectors[:, :num_bottom_components]) @ eigen_vectors[:, :num_bottom_components].T + mean_image

    top_images_train = ((images[list(range(len(train_dataset)))]) @ eigen_vectors[:, -num_top_components:]) @ eigen_vectors[:, -num_top_components:].T + mean_image
    top_images_val = ((images[list(range(len(train_dataset), len(dataset), 1))]) @ eigen_vectors[:, -num_top_components:]) @ eigen_vectors[:, -num_top_components:].T + mean_image

    # reshape the images
    bottom_images_train = bottom_images_train.reshape(-1, 3, cfg["MAE"]["MODEL"]["image_size"], cfg["MAE"]["MODEL"]["image_size"])
    bottom_images_val = bottom_images_val.reshape(-1, 3, cfg["MAE"]["MODEL"]["image_size"], cfg["MAE"]["MODEL"]["image_size"])

    top_images_train = top_images_train.reshape(-1, 3, cfg["MAE"]["MODEL"]["image_size"], cfg["MAE"]["MODEL"]["image_size"])
    top_images_val = top_images_val.reshape(-1, 3, cfg["MAE"]["MODEL"]["image_size"], cfg["MAE"]["MODEL"]["image_size"])

    assert len(train_dataset) == len(bottom_images_train)
    assert len(val_dataset) == len(bottom_images_val)
    assert len(train_dataset) == len(top_images_train)
    assert len(val_dataset) == len(top_images_val)

    if "bottom" in cfg["MAE"]["pca_mode"]:
        images_train = bottom_images_train
        images_val = bottom_images_val
    elif "top" in cfg["MAE"]["pca_mode"]:
        images_train = top_images_train
        images_val = top_images_val
    else:
        raise ValueError(f"Unknown PCA mode: {cfg['MAE']['pca_mode']}")

    train_dataset_new = tuple()
    val_dataset_new = tuple()
    for number_of_images in range(len(train_dataset)):
        train_dataset_new += ((train_dataset[number_of_images][0], images_train[number_of_images]),)
    for number_of_images in range(len(val_dataset)):
        val_dataset_new += ((val_dataset[number_of_images][0], images_val[number_of_images]),)

    return train_dataset_new, val_dataset_new

def summary(cfg, model, device, batch_size) -> None:
    """
    Generate a summary of the model using torchinfo.

    Args:
        cfg (dict): Configuration dictionary.
        model (nn.Module or tuple): The model to generate the summary for. If the model is wrapped in a tuple, only the first element will be used.
        device (str): The device to run the model on.
        batch_size (int): The batch size for the input.

    Returns:
        None
    """
    # some models are wrapped in a tuple
    if type(model) == tuple:
        model = model[0]
    device = device
    batch_size = batch_size
    channels = 3
    img_height = cfg["MAE"]["MODEL"]["image_size"]
    img_width = cfg["MAE"]["MODEL"]["image_size"]
    torchinfo.summary(
        model, 
        device=device, 
        input_size=[batch_size, channels, img_height, img_width],
        row_settings=["var_names"]
    )
