import torch
from torchvision import models, datasets, transforms
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import argparse
import os
from tqdm import tqdm  # Import tqdm for the loading bar
from scipy import stats
from captum.attr import GuidedBackprop, IntegratedGradients, NoiseTunnel, LayerGradCam, Saliency
import matplotlib.pyplot as plt

def adversarial_attack(inputs, labels, model, criterion, epsilon, attack_type='fgsm', alpha=0.01, num_iter=40):
    inputs.requires_grad = True
    if attack_type == 'fgsm':
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()
        data_grad = inputs.grad.data
        sign_data_grad = data_grad.sign()
        perturbed_data = inputs + epsilon * sign_data_grad
    elif attack_type == 'gaussian':
        noise = torch.randn_like(inputs) * epsilon
        perturbed_data = inputs + noise
    elif attack_type == 'pgd':
        perturbed_data = inputs.clone().detach()
        perturbed_data.requires_grad = True
        for _ in range(num_iter):
            outputs = model(perturbed_data)
            loss = criterion(outputs, labels)
            model.zero_grad()
            loss.backward()
            data_grad = perturbed_data.grad.data
            perturbed_data = perturbed_data + alpha * data_grad
            perturbed_data = torch.clamp(perturbed_data, inputs - epsilon, inputs + epsilon)
            perturbed_data = torch.clamp(perturbed_data, 0, 1).detach_()
            perturbed_data.requires_grad = True
    else:
        raise ValueError(f"Attack type {attack_type} is not supported.")

    perturbed_data = torch.clamp(perturbed_data, 0, 1)
    return perturbed_data

def train_model(model, train_loader, criterion, optimizer, 
                device, num_epochs=10, save_path='model_weights.pt', 
                early_stop_patience=5, adversarial=False, epsilon=0.1, 
                attack_type='fgsm', alpha=0.01, num_iter=40):
    print('Starting Training')
    model.train()
    best_accuracy = 0.0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        # Use tqdm to create a loading bar for the training loop
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                if adversarial:
                    inputs = adversarial_attack(inputs, labels, model, criterion, epsilon, attack_type, alpha, num_iter)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update the loading bar
                pbar.update(1)
                pbar.set_postfix(loss=running_loss/len(train_loader), accuracy=100 * correct / total)
        
        accuracy = 100 * correct / total
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%')

        # Save the model weights if the accuracy is better than the previous best accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), save_path)
            print(f'New best accuracy: {accuracy:.2f}%. Model weights saved to {save_path}')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        # Check early stopping condition
        if epochs_no_improve >= early_stop_patience:
            print(f'Early stopping triggered after {epoch+1} epochs.')
            break
    
    print('Finished Training')
    return save_path

def vanilla_grad(model, inputs, labels=None, criterion=None):
        # Enable gradients for inputs
        inputs = inputs.requires_grad_(True)

        # Forward pass
        outputs = model(inputs)
        if labels == None:
            labels = int(torch.max(outputs[0], 0)[1])
        out = outputs[:, labels].requires_grad_(requires_grad=True)
        # Get the attribution maps
        grad = torch.autograd.grad(out, inputs, 
                                   grad_outputs=torch.ones_like(out), 
                                   create_graph=True)[0]

        return grad

# Function to test the model
def evalattai(model, test_loader, criterion, args, alpha=0.1, N=10, debug=False):
    print('Starting Testing')
    model.eval()
    test_losses = [0.0] * N
    correct_list = [0] * N
    total_list = [0] * N
    accuracies = []
    confidence_intervals = {n: [0.0, 0.0] for n in range(N)}
    input_count = 0

    # Select attribution method
    if args.attr_method == 'VG':
        # attr_method = vanilla_grad
        attr_method = Saliency(model)
    elif args.attr_method == 'GB':
        attr_method = GuidedBackprop(model)
    elif args.attr_method == 'IG':
        attr_method = IntegratedGradients(model)
    elif args.attr_method == 'SG':
        attr_method = NoiseTunnel(Saliency(model))
    elif args.attr_method == 'GC':
        # List the layers with their names
        l = [module for module in model.modules() if type(module) == nn.Conv2d]
        attr_method = LayerGradCam(model, l[-1])
    elif args.attr_method == 'random':
        attr_method = 'random'
    elif args.attr_method == 'gradximage':
        attr_method = 'gradximage'
    else:
        raise ValueError(f"Attribution method {args.attr_method} is not supported.")
    
    # Use tqdm to create a loading bar for the testing loop
    with tqdm(total=len(test_loader), desc='Testing', unit='batch') as pbar:
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            if args.attr_method == 'VG':
                # attr = attr_method(model, inputs)
                attr = attr_method.attribute(inputs, target=labels, abs=False)
            elif args.attr_method == 'GB':
                attr = attr_method.attribute(inputs, target=labels)
            elif args.attr_method == 'IG':
                attr = attr_method.attribute(inputs, target=labels, n_steps = 10)
            elif args.attr_method == 'SG':
                attr = attr_method.attribute(inputs, target=labels, abs=False, nt_type='smoothgrad', nt_samples=10, stdevs=0.1)
            elif args.attr_method == 'GC':
                attr = attr_method.attribute(inputs, target=labels)
                attr = torch.nn.functional.interpolate(attr, size=inputs.shape[2:], mode='bilinear', align_corners=False)
                # attr = torch.mean(attr, dim=1, keepdim=True)  # GradCam returns a heatmap, average over channels
            elif args.attr_method == 'random':
                attr = torch.randn_like(inputs)
            elif args.attr_method == 'gradximage':
                grad = vanilla_grad(model, inputs, labels)
                attr = grad * inputs
            
            # Ensure attr is a FloatTensor
            attr = attr.float()
            
            # Normalize attr
            attr = (attr - attr.min()) / (attr.max() - attr.min())

            for n in range(N):
                # Modify inputs
                inputs_ = inputs - (alpha * attr * n)

                # Forward pass with modified inputs
                outputs = model(inputs_)
                loss = criterion(outputs, labels)

                test_losses[n] += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_list[n] += labels.size(0)
                correct_list[n] += (predicted == labels).sum().item()
                
                # Plot inputs_ if debug is True
                if debug:
                    plt.figure(figsize=(10, 6))
                    plt.imshow(inputs_[0].clone().detach().cpu().permute(1, 2, 0).numpy())
                    plt.title(f'Modified Input at Iteration {n+1}')
                    plt.savefig('debug_input.png')
            
            # Update the loading bar
            pbar.update(1)
            pbar.set_postfix(accuracy=100 * correct_list[input_count % N] / total_list[input_count % N])

    for n in range(N):
        accuracy = 100 * correct_list[n] / total_list[n]
        correct_tensor = torch.cat((torch.ones(correct_list[n]), torch.zeros(total_list[n] - correct_list[n])))
        accuracies.append(accuracy)

        # Calculate 95% confidence interval
        confidence_interval = stats.t.interval(0.95, total_list[n]-1, loc=accuracy, scale=stats.sem(correct_tensor * 100))
        confidence_intervals[n][0] += confidence_interval[0]
        confidence_intervals[n][1] += confidence_interval[1]
        
    print(f'Test Loss: {sum(test_losses)/len(test_loader):.4f}, Test Accuracies: {accuracies}')
    print(f'95% Confidence Interval of Accuracy: {confidence_intervals}')
    return accuracies, confidence_intervals

if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser(description='Train or test a pre-trained ResNet model on different datasets.')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'mnist'], default='cifar10', help='Dataset to use: cifar10, cifar100, or mnist')
    parser.add_argument('--device', default='1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--train', type=bool, default=False, help='Flag to indicate whether to train the model')
    parser.add_argument('--save-folder', type=str, default='model_weights', help='Path to save the model weights')
    parser.add_argument('--load-path', type=str, default='', help='Path to load the model weights')
    parser.add_argument('--train-batch-size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of worker threads for data loading')
    parser.add_argument('--alpha', type=float, default=0.1, help='Alpha value for input modification')
    parser.add_argument('--N', type=int, default=10, help='Number of iterations for input modification')
    parser.add_argument('--num-epochs', type=int, default=10, help='Number of epochs for training')
    parser.add_argument('--early-stop-patience', type=int, default=5, help='Number of epochs with no improvement after which training will be stopped')
    parser.add_argument('--model', type=str, default='resnet50', help='Model to use: resnet50, resnet18, etc.')
    parser.add_argument('--adversarial', type=bool, default=False, help='Flag to indicate whether to use adversarial training')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon value for adversarial attack')
    parser.add_argument('--attack-type', type=str, choices=['fgsm', 'gaussian', 'pgd'], default='gaussian', help='Type of adversarial attack: fgsm, gaussian, or pgd')
    parser.add_argument('--pgd-alpha', type=float, default=0.03, help='Alpha value for PGD attack')
    parser.add_argument('--pgd-num-iter', type=int, default=10, help='Number of iterations for PGD attack')
    parser.add_argument('--attr_methods', type=str, nargs='+', default=['random', 'VG', 'gradximage', 'GB', 'IG', 'SG', 'GC'], help='List of attribution methods to use: VG, GB, IG, SG, GC, random, gradximage')
    parser.add_argument('--norm', type=bool, default=True, help='Flag to indicate whether to normalize accuracies to the random method')
    args = parser.parse_args()

    # Set CUDA device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Load the pre-trained model
    available_models = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
        'alexnet': models.alexnet,
        'vgg11': models.vgg11,
        'vgg13': models.vgg13,
        'vgg16': models.vgg16,
        'vgg19': models.vgg19,
        'squeezenet1_0': models.squeezenet1_0,
        'squeezenet1_1': models.squeezenet1_1,
        'densenet121': models.densenet121,
        'densenet169': models.densenet169,
        'densenet201': models.densenet201,
        'densenet161': models.densenet161,
        'inception_v3': models.inception_v3,
        'googlenet': models.googlenet,
        'shufflenet_v2_x0_5': models.shufflenet_v2_x0_5,
        'shufflenet_v2_x1_0': models.shufflenet_v2_x1_0,
        'mobilenet_v2': models.mobilenet_v2,
        'mobilenet_v3_large': models.mobilenet_v3_large,
        'mobilenet_v3_small': models.mobilenet_v3_small,
        'resnext50_32x4d': models.resnext50_32x4d,
        'resnext101_32x8d': models.resnext101_32x8d,
        'wide_resnet50_2': models.wide_resnet50_2,
        'wide_resnet101_2': models.wide_resnet101_2,
        'mnasnet0_5': models.mnasnet0_5,
        'mnasnet0_75': models.mnasnet0_75,
        'mnasnet1_0': models.mnasnet1_0,
        'mnasnet1_3': models.mnasnet1_3
    }

    if args.model in available_models:
        model = available_models[args.model](pretrained=True)
    else:
        raise ValueError(f"Model {args.model} is not supported.")
    
    # Modify the final layer to match the number of classes in the dataset
    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100
    elif args.dataset == 'mnist':
        num_classes = 10
    else:
        raise ValueError(f"Dataset {args.dataset} is not supported.")
    
    if 'resnet' in args.model or 'resnext' in args.model or 'wide_resnet' in args.model:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif 'alexnet' in args.model or 'vgg' in args.model:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif 'squeezenet' in args.model:
        model.classifier[1] = nn.Conv2d(model.classifier[1].in_channels, num_classes, kernel_size=(1, 1))
        model.num_classes = num_classes
    elif 'densenet' in args.model:
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif 'inception' in args.model:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif 'googlenet' in args.model:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif 'shufflenet' in args.model:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif 'mobilenet' in args.model:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif 'mnasnet' in args.model:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Model {args.model} is not supported for modification.")
    
    model = model.to(device)

    # Define the data transforms
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the dataset
    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=test_transform)
    elif args.dataset == 'mnist':
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229]),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229]),
        ])
        train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
        test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

    # Shrink the test set size
    test_dataset, _ = torch.utils.data.random_split(test_dataset, [len(test_dataset) // 10, len(test_dataset) - len(test_dataset) // 10])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Generate dynamic save path
    model_name = type(model).__name__
    if args.adversarial:
        save_path = f"{args.save_folder}/{model_name}_{args.attack_type}_weights.pt"
    else:
        save_path = f"{args.save_folder}/{model_name}_weights.pt"

    # Ensure the save folder exists only if it doesn't already exist
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    if args.load_path == '':
        args.load_path = save_path

    if args.train:
        save_path = train_model(model, train_loader, criterion, optimizer, device, num_epochs=args.num_epochs, 
                    save_path=save_path, early_stop_patience=args.early_stop_patience, 
                    adversarial=args.adversarial, epsilon=args.epsilon, attack_type=args.attack_type, 
                    alpha=args.pgd_alpha, num_iter=args.pgd_num_iter)
    
    print(f'Model weights saved/loaded at {save_path}')
    model.load_state_dict(torch.load(args.load_path))
    
    # Initialize a dictionary to store accuracies and confidence intervals for each method
    results = {}

    # Run the test for each attribution method
    for attr_method in args.attr_methods:
        args.attr_method = attr_method
        accuracies, confidence_intervals = evalattai(model, test_loader, criterion, args=args)

        # Store the results
        results[attr_method] = {
            'accuracies': accuracies,
            'confidence_intervals': confidence_intervals
        }

    if args.norm:
        # Normalize accuracies to 'random' method
        random_accuracies = results['random']['accuracies']
        for attr_method in results:
            # if attr_method != 'random':
            results[attr_method]['accuracies'] = [acc - rand_acc for acc, rand_acc in zip(results[attr_method]['accuracies'], random_accuracies)]
            ci_ = results[attr_method]['confidence_intervals']
            results[attr_method]['confidence_intervals'] = [
                (ci_[ix][0] - random_accuracies[ix], ci_[ix][1] - random_accuracies[ix]) for ix in range(len(results[attr_method]['confidence_intervals']))
            ]

    # Plot accuracies vs N with confidence intervals for all methods
    N_values = list(range(1, args.N + 1))
    plt.figure(figsize=(10, 6))

    for attr_method, data in results.items():
        # if attr_method != 'random':
        accuracies = data['accuracies']
        confidence_intervals = data['confidence_intervals']
        lower_bounds = [confidence_intervals[ix][0] for ix in range(len(confidence_intervals))]
        upper_bounds = [confidence_intervals[ix][1] for ix in range(len(confidence_intervals))]

        plt.plot(N_values, accuracies, label=f'{attr_method} Accuracy', marker='o')
        plt.fill_between(N_values, lower_bounds, upper_bounds, alpha=0.2)

    plt.xlabel('N')
    plt.ylabel('Normalized Accuracy (%)' if args.norm else 'Accuracy (%)')
    plt.title('Normalized Accuracy vs N with 95% Confidence Intervals for Different Attribution Methods' if args.norm else 'Accuracy vs N with 95% Confidence Intervals for Different Attribution Methods')
    plt.legend()
    plt.grid(True)

    # Ensure the 'figure' folder exists
    if not os.path.exists('figure'):
        os.makedirs('figure')

    save_path = os.path.join('figure', 'normalized_accuracy_vs_N_all_methods.png' if args.norm else 'accuracy_vs_N_all_methods.png')
    plt.savefig(save_path)
    print(f'Plot saved as {save_path}')
