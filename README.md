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
    git clone https://github.com/yourusername/train_and_test_resnet.git
    cd train_and_test_resnet
    ```

2. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```
    
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

The script will save the trained model weights to the specified folder and generate a plot of accuracies vs. number of iterations with confidence intervals for different attribution methods. The plot will be saved in the [`figure`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fhome%2Fnielseni6%2FPythonScripts%2FEvalAttAI%2Ftrain_and_test_evalattai.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A444%2C%22character%22%3A8%7D%7D%5D%2C%2204bfddc6-e688-4331-8880-8b607a5481ec%22%5D "Go to definition") folder.

## Acknowledgments

- This script uses models from the [torchvision](https://pytorch.org/vision/stable/models.html) library.
- The adversarial training and evaluation methods are inspired by various research papers in the field of machine learning and computer vision.
