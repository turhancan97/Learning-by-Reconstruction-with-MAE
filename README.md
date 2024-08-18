# Masked Autoencoders for Image Reconstruction

This repository carry out the implementation of a Masked Autoencoder for Image Reconstruction (MAE) using the several different datasets. The difference of this implementation from the original one is that this repository verifies the idea which is obtained from the paper "Learning by Reconstruction Produces Uninformative Features For Perception" by Randall Balestriero and Yann LeCun. The paper is available at [arXiv](https://arxiv.org/abs/2402.11337).

## About the Paper

### Feature Alignment and Misalignment:

* The paper presents that features learned by reconstructing the input data are not aligned with those required for downstream tasks (e.g. classification, detection, segmentation, etc.). Specifically, they show that the principal components of the data (those that explain the most variance and are learned first in reconstruction tasks) are the least useful for downstream tasks.
* For instance, in their experiments with TinyImagenet, projecting images onto the subspace that explains 90% of the pixel variance results in a test accuracy of only 45% in a supervised classification task. In contrast, using the bottom subspace that accounts for only 20% of the pixel variance yields a higher accuracy of 55%.

<figure align="center"> 
  <img src="images/paper/bottom_top_test_accuracy.png" alt="drawing" width="400"/>
  <figcaption>Paper show that if we keep only the low-variance components of images, we can still train good classifiers on this data</figcaption>
</figure>

#### How to split the data into high-frequency and low-frequency components?

* Use PCA on the dataset of images to split the data into high-frequency and low-frequency components. PCA gives us the principal components of
the image dataset and sorts them in terms of explained variance.

* Then, we can filter out from images the components of our choice (e.g. top 90% of the variance, bottom 10% of the variance). 

<figure align="center"> 
  <img src="images/paper/example-top-bottom-variance-image.png" alt="drawing" width="400"/>
  <figcaption>Example of image with components with high explained variance and components with low explained variance</figcaption>
</figure>

As you can see from the image above:
* Components with high explained variance tend to be very blurry, colorful images.
* Components with low explained variance tend to be more detailed, less colorful images.

### Order of Feature Learning:

* The paper demonstrates that reconstruction-based models, such as autoencoders, learn features in an order that prioritizes reconstructing the input data accurately rather than focusing on features that are informative for downstream tasks. This results in the most perceptually relevant features being learned last, which explains why these models often require long training times to perform well in tasks like classification.
* The authors visualize this process by showing how the training loss evolves over time, with features in the top eigenspace (those explaining the most variance) being learned first. Only after these are learned do features in the bottom eigenspace (which are more useful for downstream tasks) begin to be learned.

<figure align="center"> 
  <img src="images/paper/feature_learning_order.png" alt="drawing" width="400"/>
  <figcaption>Order of feature learning in reconstruction-based models</figcaption>
</figure>

### Implications for Model Design:
* MAE, which selectively removes parts of the input data, forces the model to focus on the remaining parts, potentially improving the learning of features that are useful for perception.
* The paper suggests that models like MAE, which encourage the learning of informative features, may be more effective for downstream tasks than other denoising autoencoders.

## About the Repository

As paper above suggests, the top principal components (those that explain the most variance) are the least useful for downstream tasks, while the bottom principal components (those that explain the least variance) are the most useful. We used the bottom principal components which are the most useful for downstream tasks to pretrain the MAE model.

* Input images are the original images.
* Target images are the images that high-variance components are filtered out from the original images.
* The model is trained to reconstruct the target images from the input images.

## Training the Model
### Installing Packages
```bash
conda create -n mae python=3.8
conda activate mae
pip install -r requirements.txt
```

### Training the Model
```bash
python mae_pretrain.py -c config/config_file.yaml
```

### Pretrained models
You can download the weights of the pretrained masked autoencoder models from the following links:

<table>
  <tr>
    <th>Epochs</th>
    <th>Model</th>
    <th>Pretrained Dataset</th>
    <th colspan="5">Download</th>
  </tr>
  <tr>
    <td>400</td>
    <th>ViT-Tiny</th>
    <th>CIFAR10</th>
    <td><a href="x">MAE Model</a></td>
  </tr>
  <tr>
    <td>400</td>
    <th>ViT-Tiny</th>
    <th>CIFAR100</th>
    <td><a href="x">MAE Model</a></td>
  </tr>
  <tr>
    <td>200</td>
    <th>ViT-Tiny</th>
    <th>Imagenette</th>
    <td><a href="x">MAE Model</a></td>
   </tr>
  <tr>
    <td>200</td>
    <th>ViT-Tiny</th>
    <th>STL10</th>
    <td><a href="x">MAE Model</a></td>
   </tr>
</table>

## Evaluation of the Model
### Fine-Tuning and Linear Probing
```bash
cd eval/
python eval_finetune.py -c ../config/config_file.yaml
python eval_linprobe.py -c ../config/config_file.yaml
```
### Classification models
You can download the weights of the classification models from the following links:

<table>
  <tr>
    <th>PCA Mode</th>
    <th>Evaluation Type</th>
    <th>Pretrained Dataset</th>
    <th colspan="5">Download</th>
  </tr>
  <tr>
    <td>No Mode</td>
    <th>Fine-Tuning</th>
    <th>CIFAR10</th>
    <td><a href="x">MAE Encoder</a></td>
  </tr>
  <tr>
    <td>No Mode</td>
    <th>Linear Probing</th>
    <th>CIFAR10</th>
    <td><a href="x">MAE Encoder</a></td>
  </tr>
  <tr>
    <td>bottom 25</td>
    <th>Fine-Tuning</th>
    <th>CIFAR10</th>
    <td><a href="x">MAE Encoder</a></td>
   </tr>
  <tr>
    <td>bottom 25</td>
    <th>Linear Probing</th>
    <th>CIFAR10</th>
    <td><a href="x">MAE Encoder</a></td>
   </tr>
</table>

## About configuration file

## Results

## Huggingface Model and Space

## References

* [Learning by Reconstruction Produces Uninformative Features For Perception](https://arxiv.org/abs/2402.11337)
* [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377)
* The training code is inspired by the [ViT-tiny PyTorch implementation of Masked Autoencoder](https://github.com/IcarusWizard/MAE)