# Train and Test Pre-trained ResNet Models

This repository contains a script to train and test pre-trained ResNet models on different datasets using various attribution methods. The script supports adversarial training and provides options for customizing the training and evaluation process.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- matplotlib
- argparse

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/nielseni6/EvalAttAI.git
    cd train_and_test_resnet
    ```

2. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

## Usage

The script can be run from the command line with various arguments to customize the training and evaluation process.

### Command Line Arguments

- `--dataset`: Dataset to use ([`cifar10`], [`cifar100`], or [`mnist`]). Default is [`cifar10`].
- `--device`: CUDA device to use (e.g., [`0`], `0,1,2,3`, or [`cpu`]). Default is [`1`].
- `--train`: Flag to indicate whether to train the model. Default is [`False`].
- `--save-folder`: Path to save the model weights. Default is [`model_weights`].
- `--load-path`: Path to load the model weights. Default is an empty string.
- `--train-batch-size`: Batch size for training. Default is [`128`].
- `--test-batch-size`: Batch size for testing. Default is [`32`].
- `--num-workers`: Number of worker threads for data loading. Default is [`8`].
- `--alpha`: Alpha value for input modification. Default is [`0.1`].
- `--N`: Number of iterations for input modification. Default is [`10`].
- `--num-epochs`: Number of epochs for training. Default is [`10`].
- `--early-stop-patience`: Number of epochs with no improvement after which training will be stopped. Default is [`5`].
- `--model`: Model to use (e.g., [`resnet50`], [`resnet18`]). Default is [`resnet50`].
- `--adversarial`: Flag to indicate whether to use adversarial training. Default is [`False`].
- `--epsilon`: Epsilon value for adversarial attack. Default is [`0.1`].
- `--attack-type`: Type of adversarial attack ([`fgsm`], [`gaussian`], or [`pgd`]). Default is [`gaussian`].
- `--pgd-alpha`: Alpha value for PGD attack. Default is [`0.03`].
- `--pgd-num-iter`: Number of iterations for PGD attack. Default is [`10`].
- `--attr_methods`: List of attribution methods to use ([`VG`], [`GB`], [`IG`], [`SG`], [`GC`], [`random`], [`gradximage`]). Default is `['random', 'VG', 'gradximage', 'GB', 'IG', 'SG', 'GC']`.
- `--norm`: Flag to indicate whether to normalize accuracies to the random method. Default is [`True`].

### Example Usage

To train a ResNet-50 model on CIFAR-10 with adversarial training:

```sh
python train_and_test_evalattai.py --dataset cifar10 --train True --adversarial True --model resnet50
```

To test a pre-trained ResNet-50 model on CIFAR-10:

```sh
python train_and_test_evalattai.py --dataset cifar10 --train False --model resnet50 --load-path path/to/weights.pt
```

## Output

The script will save the trained model weights to the specified folder and generate a plot of accuracies vs. number of iterations with confidence intervals for different attribution methods. The plot will be saved in the [`figure`] folder.

## Acknowledgments

- This script uses models from the [torchvision](https://pytorch.org/vision/stable/models.html) library.
- The adversarial training and evaluation methods are inspired by various research papers in the field of machine learning and computer vision.
