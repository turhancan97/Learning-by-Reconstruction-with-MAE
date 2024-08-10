import os
import random
from argparse import ArgumentParser

from icecream import ic
import numpy as np
import torch
import tqdm
from torchvision import datasets
from torchvision.transforms import v2

import utils

# device configuration (use GPU if available)
device = utils.get_gpu()
print(f"Using device: {device}")


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

def sklearn_pca(X, n_components=None):
    """
    Compute the PCA of a dataset using sklearn.
    """
    from sklearn.decomposition import PCA
    X_np = X.numpy()
    pca = PCA(n_components=n_components, svd_solver="auto")
    pca.fit(X_np)
    components = pca.components_
    explained_variance = pca.singular_values_

    explained_variance = torch.tensor(explained_variance)
    components = torch.tensor(components).permute(1, 0)
    return explained_variance, components


@utils.log_step
def main(cfg: dict) -> torch.Tensor:
    """
    - Computes the PCA of the dataset
    - Plots the original images, the bottom PCA images and the top PCA images.
    """
    with torch.no_grad():
        # Transformations to apply to each image
        transform = v2.Compose(
            [
                v2.PILToTensor(),
                v2.Resize(cfg["PCA"]["resize"]),
                v2.CenterCrop(cfg["PCA"]["crop"]),
                v2.ToDtype(torch.float32, scale=True),
            ]
        )

        dataset = datasets.ImageFolder(
            f"datasets/{cfg['PCA']['dataset']}/{cfg['PCA']['split']}",
            transform=transform,
        )
        print("Dataset size:")
        ic(len(dataset))

        # Initialize a tensor to store flattened images
        images = torch.zeros(len(dataset), 3 * cfg["PCA"]["crop"] * cfg["PCA"]["crop"])

        # Load and flatten each image
        for i, (im, y) in tqdm.tqdm(enumerate(dataset)):
            images[i] = im.flatten()

        # # interative debugging
        # import code; code.interact(local=locals())

        # Standardize by subtracting the mean
        mean_image = images.mean(0)
        # zero mean-centered (mean is subtracted from each vector).
        images -= mean_image
        print("Flattened and zero mean centered images:")
        ic(images.shape)

        if cfg["PCA"]["use_sklearn"]:
            print("Using sklearn PCA - with numpy")
            eigen_values, eigen_vectors = sklearn_pca(images)
            ic(eigen_values.shape, eigen_vectors.shape)

            # # flip the eigen_values
            # eigen_values = eigen_values.flip(0)
            eigen_vectors = eigen_vectors.flip(1)

            # normalize the eigenvalues (amount of variance)
            eigen_values /= eigen_values.sum()
            
            # get the cumulative sum of the eigenvalues
            cumulative_variance = eigen_values.cumsum(dim=0)

        else:
            print("Using fast PCA - with torch")
            # get spectral decomposition and normalize eigenvalues
            eigen_values, eigen_vectors = fast_gram_eigh(images, "R")
            ic(eigen_values.shape, eigen_vectors.shape)
            # normalize the eigenvalues (amount of variance)
            eigen_values /= eigen_values.sum() 
            # get the cumulative sum of the eigenvalues
            cumulative_variance = eigen_values.cumsum(dim=0)

        # check the shapes
        assert cumulative_variance.shape[0] == len(dataset)
        assert eigen_vectors.shape[1] == len(dataset)
        assert eigen_vectors.shape[0] == 3 * cfg["PCA"]["crop"] * cfg["PCA"]["crop"]

        # we get the bottom and top part (25% variance cutoff)
        if cfg["PCA"]["use_sklearn"]:
            topk = torch.count_nonzero(cumulative_variance < cfg["PCA"]["variance_cutoff"])
            bottomk = torch.count_nonzero(cumulative_variance > cfg["PCA"]["variance_cutoff"])
        else:
            bottomk = torch.count_nonzero(cumulative_variance < cfg["PCA"]["variance_cutoff"])
            topk = torch.count_nonzero(cumulative_variance > cfg["PCA"]["variance_cutoff"])

        ic(bottomk, topk)

        # create a folder to save the images if it does not exist
        os.makedirs("images/pca_result", exist_ok=True)

        # we select some samples to plot by hand
        pick = [
            random.randint(0, len(dataset)),
            random.randint(0, len(dataset)),
            random.randint(0, len(dataset)),
            random.randint(0, len(dataset)),
        ]
        utils.plot_images(
            (images[pick] + mean_image).reshape(
                -1, 3, cfg["PCA"]["crop"], cfg["PCA"]["crop"]
            ),
            "images/pca_result/original_images.png",
        )
        bottom = ((images[pick]) @ eigen_vectors[:, :bottomk]) @ eigen_vectors[
            :, :bottomk
        ].T + mean_image
        utils.plot_images(
            bottom.reshape(-1, 3, cfg["PCA"]["crop"], cfg["PCA"]["crop"]),
            "images/pca_result/bottom_pca_images.png",
        )
        top = ((images[pick]) @ eigen_vectors[:, -topk:]) @ eigen_vectors[
            :, -topk:
        ].T + mean_image
        utils.plot_images(
            top.reshape(-1, 3, cfg["PCA"]["crop"], cfg["PCA"]["crop"]),
            "images/pca_result/top_pca_images.png",
        )

        pick = np.random.permutation(len(images))[:64]
        utils.plot_images(
            (images[pick] + mean_image).reshape(
                -1, 3, cfg["PCA"]["crop"], cfg["PCA"]["crop"]
            ),
            "images/pca_result/some_random_original_images.png",
        )
        bottom = ((images[pick]) @ eigen_vectors[:, :bottomk]) @ eigen_vectors[
            :, :bottomk
        ].T + mean_image
        utils.plot_images(
            bottom.reshape(-1, 3, cfg["PCA"]["crop"], cfg["PCA"]["crop"]),
            "images/pca_result/some_random_bottom_pca_images.png",
        )
        top = ((images[pick]) @ eigen_vectors[:, -topk:]) @ eigen_vectors[
            :, -topk:
        ].T + mean_image
        utils.plot_images(
            top.reshape(-1, 3, cfg["PCA"]["crop"], cfg["PCA"]["crop"]),
            "images/pca_result/some_random_top_pca_images.png",
        )

    return images


if __name__ == "__main__":
    # python pca.py -c config/config_file.yaml
    parser = ArgumentParser()
    parser.add_argument("-c", "--config")
    args = parser.parse_args()
    print("Read Config File....")
    cfg = utils.load_yaml(args.config)
    ic(cfg)
    utils.setup_seed(cfg["seed"])  # set seed
    print("Start PCA Figure Generation....")
    main(cfg)
