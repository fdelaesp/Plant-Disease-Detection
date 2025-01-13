import os
import json
import sys
from datetime import datetime
import random

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from logging.handlers import RotatingFileHandler
from deap import base, creator, tools, algorithms
import numpy as np

# -----------------------------
# Configuration Parameters
# -----------------------------
DATASET_NAME = 'plant-pathology-2020-fgvc7'  # Competition dataset identifier
DOWNLOAD_PATH = './PlantVillage_Download'  # Path to the downloaded and extracted dataset
BASE_MODEL_PATH = 'plant_disease_model.pth'  # Existing base model (if any)
CHECKPOINT_DIR = 'CHECKPOINT'  # Directory to save checkpoints
FINAL_MODEL_SAVE_DIR = os.path.join(CHECKPOINT_DIR, 'FINAL_MODELS')  # Directory to save final models
NUM_EPOCHS = 5  # Total number of epochs
EARLY_STOPPING_PATIENCE = 2  # Patience for early stopping
HYPERPARAMETER_TUNING = True  # Toggle hyperparameter tuning
USE_GENETIC_ALGORITHM = True  # Toggle genetic algorithm for hyperparameter tuning
POPULATION_SIZE = 2  # Population size for GA
GENERATIONS = 1  # Number of generations for GA
CX_PROB = 0.5  # Crossover probability
MUT_PROB = 0.2  # Mutation probability


# -----------------------------
# Setup Logging
# -----------------------------
def setup_logging(log_file='training.log'):
    """
    Sets up logging with both console and rotating file handlers.

    Args:
        log_file (str): Path to the log file.

    Returns:
        logger: Configured logger object.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Capture all levels of logs

    # Rotating File Handler
    file_handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=5)  # 5 MB per file
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Display INFO level and above on console
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


# Initialize Logging
logger = setup_logging()


# -----------------------------
# Custom Dataset Class
# -----------------------------
class PlantVillageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        """
        Initializes the dataset.

        Args:
            image_paths (list): List of image file paths.
            labels (list): List of labels corresponding to the images.
            transform: Transformations to apply to the images.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of samples.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves the image and label at the specified index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            image: The transformed image tensor.
            label: The label of the image.
        """
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            # Return a tensor filled with zeros if image processing fails
            image = torch.zeros(3, 224, 224)
        label = self.labels[idx]
        return image, label


# -----------------------------
# Data Preparation Function
# -----------------------------
def prepare_data(csv_path, images_folder, test_size=0.2, random_seed=42):
    """
    Prepares the data for training and validation.

    Args:
        csv_path (str): Path to the CSV file containing image IDs and labels.
        images_folder (str): Path to the folder containing images.
        test_size (float): The proportion of the dataset to include in the validation split.
        random_seed (int): Random seed for reproducibility.

    Returns:
        train_dataset, val_dataset: The training and validation datasets.
        num_classes (int): The number of classes.
        class_names (list): List of class names.
    """
    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Label columns in the CSV file
    label_columns = ['healthy', 'multiple_diseases', 'rust', 'scab']

    # Create a list to store labels as strings
    labels_str = []

    # Process labels
    for idx, row in df.iterrows():
        # Find the label column with value 1
        for label in label_columns:
            if row[label] == 1:
                labels_str.append(label)
                break  # Only one label is 1 in this dataset

    # Extract image IDs
    image_ids = df['image_id'].tolist()

    # Create list of image paths and corresponding labels
    image_paths = []
    labels_final = []
    for img_id, label in zip(image_ids, labels_str):
        image_filename = f"{img_id}.jpg"  # Adjust extension if different
        image_path = os.path.join(images_folder, image_filename)
        if os.path.exists(image_path):
            image_paths.append(image_path)
            labels_final.append(label)
        else:
            logger.warning(f"Image file '{image_path}' does not exist.")

    if not image_paths:
        raise FileNotFoundError(f"No images found in the dataset directory: {images_folder}")

    # Encode labels
    class_names = sorted(list(set(labels_final)))
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
    encoded_labels = [class_to_idx[label] for label in labels_final]

    # Save class names to JSON for later use
    with open('class_names.json', 'w') as f:
        json.dump(class_names, f)
    logger.info("Class names saved to 'class_names.json'.")

    # Split into training and validation sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, encoded_labels, test_size=test_size, random_state=random_seed, stratify=encoded_labels)

    # Define transformations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.1),  # Moved after ToTensor
        transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet mean
                             [0.229, 0.224, 0.225])  # ImageNet std
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Create datasets
    train_dataset = PlantVillageDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = PlantVillageDataset(val_paths, val_labels, transform=val_transform)

    num_classes = len(class_names)

    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Class names: {class_names}")
    logger.info(f"Number of training samples: {len(train_dataset)}")
    logger.info(f"Number of validation samples: {len(val_dataset)}")

    return train_dataset, val_dataset, num_classes, class_names


# -----------------------------
# Model Building Function
# -----------------------------
def build_model(num_classes, dropout_rate=0.5):
    """
    Builds and returns the CNN model.

    Args:
        num_classes (int): The number of output classes.
        dropout_rate (float): Dropout rate for regularization.

    Returns:
        model: The PyTorch model.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False  # Freeze the feature extractor

    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(num_features, num_classes)
    )

    return model


# -----------------------------
# Checkpoint Management Functions
# -----------------------------
def save_checkpoint(model, optimizer, epoch, stage, file_path):
    """
    Saves the model and optimizer states to a checkpoint file.

    Args:
        model: The trained PyTorch model.
        optimizer: The optimizer used during training.
        epoch: The current epoch number.
        stage: The current stage ('tuning' or 'training').
        file_path (str): Path to save the checkpoint.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'stage': stage
    }
    torch.save(checkpoint, file_path)
    logger.info(f"Checkpoint saved to '{file_path}'.")


def load_checkpoint(file_path, model, optimizer=None):
    """
    Loads a model and optimizer state from a checkpoint file.

    Args:
        file_path (str): Path to the checkpoint file.
        model: The model to load the state into.
        optimizer: (Optional) The optimizer to load the state into.

    Returns:
        model, optimizer, epoch, stage: The model, optimizer (if provided), the starting epoch, and stage.
    """
    if not os.path.exists(file_path):
        logger.error(f"Checkpoint file '{file_path}' does not exist.")
        return model, optimizer, 0, 'tuning'

    checkpoint = torch.load(file_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint.get('epoch', 0)
    stage = checkpoint.get('stage', 'tuning')
    logger.info(f"Checkpoint loaded from '{file_path}', resuming from epoch {start_epoch}, stage '{stage}'.")
    return model, optimizer, start_epoch, stage


# -----------------------------
# Training Function
# -----------------------------
def train_model(model, dataloaders, criterion, optimizer, num_epochs, device, start_epoch=0, checkpoint_path=None,
                stage='training', early_stopping_patience=5):
    """
    Trains the model.

    Args:
        model: The PyTorch model.
        dataloaders (dict): Dictionary containing 'train' and 'val' DataLoader objects.
        criterion: Loss function.
        optimizer: Optimizer for training.
        num_epochs (int): Number of training epochs.
        device: Device to run the training on ('cpu' or 'cuda').
        start_epoch (int): The epoch to start training from.
        checkpoint_path (str): Path to save the checkpoint.
        stage (str): Current stage ('tuning' or 'training').
        early_stopping_patience (int): Number of epochs with no improvement after which training will be stopped.

    Returns:
        model: The trained model.
        history: Dictionary containing training and validation loss and accuracy.
    """
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    best_acc = 0.0
    best_model_wts = model.state_dict()

    epochs_no_improve = 0

    model.to(device)

    try:
        for epoch in range(start_epoch, start_epoch + num_epochs):
            logger.info(f"Epoch {epoch + 1}/{start_epoch + num_epochs}")
            logger.info("-" * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluation mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data
                for inputs, labels in tqdm(dataloaders[phase], desc=phase.capitalize(), leave=False):
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # Zero the parameter gradients
                    optimizer.zero_grad()

                    # Forward pass
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # Backward pass and optimize
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # Statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                # Calculate epoch loss and accuracy
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double().item() / len(dataloaders[phase].dataset)

                history[f'{phase}_loss'].append(epoch_loss)
                history[f'{phase}_acc'].append(epoch_acc)

                logger.info(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

                # Deep copy the model
                if phase == 'val':
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = model.state_dict()
                        epochs_no_improve = 0
                        logger.info(f"Validation accuracy improved to {best_acc:.4f}.")
                    else:
                        epochs_no_improve += 1
                        logger.info(f"No improvement in validation accuracy for {epochs_no_improve} epoch(s).")

            # Save checkpoint at the end of each epoch
            if checkpoint_path:
                save_checkpoint(model, optimizer, epoch + 1, stage, checkpoint_path)

            # Early Stopping Check
            if epochs_no_improve >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epochs_no_improve} epochs with no improvement.")
                break

            # Add an empty line for better readability in logs
            logger.info("")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user. Saving current state...")
        if checkpoint_path:
            save_checkpoint(model, optimizer, epoch + 1, stage, checkpoint_path)
        logger.info("Current state saved. Exiting gracefully.")
        sys.exit(0)

    # Load best model weights
    model.load_state_dict(best_model_wts)
    logger.info(f"Best Validation Accuracy: {best_acc:.4f}")

    return model, history


# -----------------------------
# Hyperparameter Tuning with Genetic Algorithm
# -----------------------------
def evaluate_individual(individual, train_dataset, val_dataset, class_names, device):
    """
    Evaluates an individual in the genetic algorithm.

    Args:
        individual (list): List containing hyperparameters [learning_rate, batch_size, dropout_rate].
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset): Validation dataset.
        class_names (list): List of class names.
        device: Device to run the training on.

    Returns:
        tuple: Fitness value (validation accuracy).
    """
    lr, bs, dr = individual
    bs = int(bs)  # Ensure batch_size is an integer
    logger.info(f"Evaluating Individual - Learning Rate: {lr:.6f}, Batch Size: {bs}, Dropout Rate: {dr:.4f}")

    # Define DataLoaders with current batch size
    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=4)
    dataloaders = {'train': train_loader, 'val': val_loader}

    # Build model with current dropout rate
    model = build_model(len(class_names), dropout_rate=dr)

    # Define Loss Function and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)

    try:
        # Train the model for a small number of epochs for evaluation
        model, history = train_model(
            model,
            dataloaders,
            criterion,
            optimizer,
            num_epochs=1,  # Limited epochs for faster evaluation
            device=device,
            start_epoch=0,
            checkpoint_path=None,
            stage='tuning',
            early_stopping_patience=1
        )
    except KeyboardInterrupt:
        logger.info("Hyperparameter evaluation interrupted by user. Exiting gracefully.")
        sys.exit(0)

    # Evaluate on validation set
    model.eval()
    model.to(device)
    val_corrects = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            val_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

    val_acc = val_corrects.double().item() / total
    logger.info(f"Validation Accuracy for Individual: {val_acc:.4f}")

    return (val_acc,)


def mut_hyperparameters(individual, indpb):
    """
    Custom mutation function to ensure batch_size remains an integer.
    """
    # Mutate learning rate and dropout rate using Gaussian mutation
    if random.random() < indpb:
        individual[0] += random.gauss(0, 0.001)  # learning_rate
        individual[0] = max(0.0001, min(individual[0], 0.01))
    if random.random() < indpb:
        individual[2] += random.gauss(0, 0.05)  # dropout_rate
        individual[2] = max(0.3, min(individual[2], 0.7))
    # Mutate batch_size by randomly selecting from available options
    if random.random() < indpb:
        individual[1] = random.choice([16, 32, 64])  # batch_size remains an integer
    return (individual,)


def cx_hyperparameters(ind1, ind2):
    """
    Custom crossover function to ensure batch_size remains an integer.
    """
    # Crossover for learning_rate and dropout_rate using arithmetic crossover
    if random.random() < 0.5:
        ind1[0], ind2[0] = ind2[0], ind1[0]  # Swap learning_rate
    if random.random() < 0.5:
        ind1[2], ind2[2] = ind2[2], ind1[2]  # Swap dropout_rate
    # Crossover for batch_size by swapping
    if random.random() < 0.5:
        ind1[1], ind2[1] = ind2[1], ind1[1]  # Swap batch_size
    return ind1, ind2


def genetic_algorithm_hyperparameter_tuning(train_dataset, val_dataset, class_names, device):
    """
    Performs hyperparameter tuning using a genetic algorithm.

    Args:
        train_dataset (Dataset): Training dataset.
        val_dataset (Dataset): Validation dataset.
        class_names (list): List of class names.
        device: Device to run the training on.

    Returns:
        best_model: The model with the best validation accuracy.
        best_params: The hyperparameters corresponding to the best model.
    """
    # Define the problem as maximizing validation accuracy
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # Define the toolbox
    toolbox = base.Toolbox()

    # Define attribute generators
    toolbox.register("learning_rate", random.uniform, 0.0001, 0.01)
    toolbox.register("batch_size", random.choice, [16, 32, 64])
    toolbox.register("dropout_rate", random.uniform, 0.3, 0.7)

    # Define structure initializers
    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.learning_rate, toolbox.batch_size, toolbox.dropout_rate), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register the evaluation function
    toolbox.register("evaluate", evaluate_individual, train_dataset=train_dataset, val_dataset=val_dataset,
                     class_names=class_names, device=device)

    # Register the genetic operators
    toolbox.register("mate", cx_hyperparameters)
    toolbox.register("mutate", mut_hyperparameters, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Initialize population
    population = toolbox.population(n=POPULATION_SIZE)

    # Define statistics to collect
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    try:
        logger.info("Starting Genetic Algorithm for Hyperparameter Tuning...")

        # Run the genetic algorithm
        population, logbook = algorithms.eaSimple(population, toolbox, cxpb=CX_PROB, mutpb=MUT_PROB,
                                                  ngen=GENERATIONS, stats=stats, verbose=True)

    except KeyboardInterrupt:
        logger.info("Hyperparameter tuning interrupted by user. Exiting gracefully.")
        sys.exit(0)

    # Select the best individual
    best_ind = tools.selBest(population, 1)[0]
    best_params = {
        'learning_rate': best_ind[0],
        'batch_size': int(best_ind[1]),  # Ensure batch_size is an integer
        'dropout_rate': best_ind[2]
    }
    best_fitness = best_ind.fitness.values[0]
    logger.info(f"Best Individual: {best_ind}")
    logger.info(f"Best Fitness (Validation Accuracy): {best_fitness:.4f}")
    logger.info(f"Best Hyperparameters: {best_params}")

    # Save best hyperparameters to a file for resumption
    best_params_path = os.path.join(CHECKPOINT_DIR, 'best_params.json')
    with open(best_params_path, 'w') as f:
        json.dump(best_params, f)
    logger.info(f"Best hyperparameters saved to '{best_params_path}'.")

    # Train the best model with the best hyperparameters for final training
    logger.info("Training the best model with optimal hyperparameters...")

    # Define DataLoaders with best batch size
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False, num_workers=4)
    dataloaders = {'train': train_loader, 'val': val_loader}

    # Build model with best dropout rate
    best_model = build_model(len(class_names), dropout_rate=best_params['dropout_rate'])

    # Define Loss Function and Optimizer with best learning rate
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(best_model.fc.parameters(), lr=best_params['learning_rate'])

    try:
        # Train the model for the remaining epochs
        # Since total epochs are limited to 5 and hyperparameter tuning used POPULATION_SIZE * GENERATIONS * EVALUATION_EPOCHS = 2*1*1=2 epochs,
        # we train for 3 more epochs
        remaining_epochs = NUM_EPOCHS - (POPULATION_SIZE * GENERATIONS * 1)
        remaining_epochs = max(remaining_epochs, 1)  # Ensure at least 1 epoch
        best_model, history = train_model(
            best_model,
            dataloaders,
            criterion,
            optimizer,
            num_epochs=remaining_epochs,
            device=device,
            start_epoch=0,
            checkpoint_path=os.path.join(CHECKPOINT_DIR, 'training_checkpoint.pth'),
            stage='training',
            early_stopping_patience=EARLY_STOPPING_PATIENCE
        )
    except KeyboardInterrupt:
        logger.info("Final training interrupted by user. Exiting gracefully.")
        sys.exit(0)

    return best_model, best_params


# -----------------------------
# Plotting Functions
# -----------------------------
def plot_training_history(history):
    """
    Plots the training and validation accuracy and loss.

    Args:
        history (dict): Dictionary containing training history.
    """
    epochs_range = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(14, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_acc'], label='Training Accuracy')
    plt.plot(epochs_range, history['val_acc'], label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_loss'], label='Training Loss')
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(model, dataloader, class_names, device):
    """
    Plots the confusion matrix for the model on the validation set.

    Args:
        model: The trained PyTorch model.
        dataloader (DataLoader): DataLoader for the validation set.
        class_names (list): List of class names.
        device: Device to run the prediction on.
    """
    model.eval()
    model.to(device)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Confusion Matrix", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()


# -----------------------------
# Prediction Function (Optional)
# -----------------------------
def predict_image(model, image_path, class_names, device):
    """
    Predicts the class of a given image.

    Args:
        model: The trained PyTorch model.
        image_path (str): The path to the image to predict.
        class_names (list): List of class names.
        device: Device to run the prediction on.

    Returns:
        predicted_class (str): The name of the predicted class.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None

    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension

    model.eval()
    model.to(device)
    with torch.no_grad():
        outputs = model(image.to(device))
        _, preds = torch.max(outputs, 1)
        predicted_class = class_names[preds.item()]

    return predicted_class


# -----------------------------
# Utility Functions
# -----------------------------
def get_available_checkpoints(checkpoint_dir, tuning_checkpoint='best_params.json',
                              training_checkpoint='training_checkpoint.pth'):
    """
    Retrieves a list of available checkpoints.

    Args:
        checkpoint_dir (str): Directory where checkpoints are stored.
        tuning_checkpoint (str): Filename for hyperparameter tuning checkpoint.
        training_checkpoint (str): Filename for training checkpoint.

    Returns:
        dict: Dictionary with 'tuning' and 'training' keys pointing to their respective checkpoint paths.
    """
    tuning_path = os.path.join(checkpoint_dir, tuning_checkpoint)
    training_path = os.path.join(checkpoint_dir, training_checkpoint)
    checkpoints = {}
    if os.path.exists(tuning_path):
        checkpoints['tuning'] = tuning_path
    if os.path.exists(training_path):
        checkpoints['training'] = training_path
    return checkpoints


def create_new_run_folder(checkpoint_dir):
    """
    Creates a new run folder inside the checkpoint directory.

    Args:
        checkpoint_dir (str): Directory where checkpoints are stored.

    Returns:
        Path to the new run folder.
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Determine the next run number
    existing_runs = [d for d in os.listdir(checkpoint_dir)
                     if os.path.isdir(os.path.join(checkpoint_dir, d)) and d.startswith('run_')]
    run_numbers = [int(d.split('_')[1]) for d in existing_runs if d.split('_')[1].isdigit()]
    next_run_number = max(run_numbers) + 1 if run_numbers else 1
    new_run_name = f"run_{next_run_number}"
    new_run_path = os.path.join(checkpoint_dir, new_run_name)

    os.makedirs(new_run_path, exist_ok=True)
    logger.info(f"Created new run folder: '{new_run_path}'")
    return new_run_path


# -----------------------------
# Main Execution with Input Menu
# -----------------------------
if __name__ == '__main__':
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Detect if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Step 1: Verify Dataset Directory Exists
    if not os.path.exists(DOWNLOAD_PATH):
        logger.error(f"Download path '{DOWNLOAD_PATH}' does not exist.")
        logger.error("Please ensure you've downloaded and extracted the dataset to this directory.")
        sys.exit(1)
    else:
        logger.info(f"Download path '{DOWNLOAD_PATH}' exists. Proceeding...")

    # Step 2: Identify CSV and Images Folder
    # Assuming 'train.csv' is in the root of DOWNLOAD_PATH and images are in 'images' subfolder
    csv_path = os.path.join(DOWNLOAD_PATH, 'train.csv')
    images_folder = os.path.join(DOWNLOAD_PATH, 'images')  # Adjust if different

    if not os.path.exists(csv_path):
        logger.error(f"CSV file '{csv_path}' does not exist.")
        logger.error("Please ensure 'train.csv' is present in the download directory.")
        sys.exit(1)

    if not os.path.exists(images_folder):
        logger.error(f"Images folder '{images_folder}' does not exist.")
        logger.error("Please ensure images are extracted to the 'images' subfolder within the download directory.")
        sys.exit(1)

    logger.info(f"CSV path: {csv_path}")
    logger.info(f"Images folder: {images_folder}")

    # Step 3: Prepare the Data
    logger.info("Preparing the data...")
    try:
        train_dataset, val_dataset, num_classes, class_names = prepare_data(csv_path, images_folder,
                                                                            test_size=0.2,
                                                                            random_seed=42)
    except FileNotFoundError as e:
        logger.error(e)
        sys.exit(1)

    # Step 4: Input Menu for Starting or Resuming
    available_checkpoints = get_available_checkpoints(CHECKPOINT_DIR)
    start_new = True  # Default to starting new run

    if available_checkpoints:
        print("\nAvailable Checkpoints:")
        option_number = 1
        options = {}
        for stage, path in available_checkpoints.items():
            if stage == 'tuning':
                print(f"{option_number}: Use Best Hyperparameters from '{path}' for Training")
            elif stage == 'training':
                print(f"{option_number}: Resume Training from '{path}'")
            options[option_number] = (stage, path)
            option_number += 1
        print(f"{option_number}: Start a New Run")
        options[option_number] = ('new_run', None)

        # Prompt user for selection
        while True:
            try:
                user_input = input("\nEnter the option number to proceed: ").strip()
                selected_option = int(user_input)
                if selected_option in options:
                    selected_stage, selected_path = options[selected_option]
                    if selected_stage == 'new_run':
                        start_new = True
                        break
                    elif selected_stage == 'tuning':
                        # Use best hyperparameters for training
                        if os.path.exists(selected_path):
                            with open(selected_path, 'r') as f:
                                best_params = json.load(f)
                            # Ensure batch_size is integer
                            best_params['batch_size'] = int(best_params['batch_size'])
                            logger.info("Loaded best hyperparameters from tuning.")
                            # Define DataLoaders with best batch size
                            train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True,
                                                      num_workers=4)
                            val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False,
                                                    num_workers=4)
                            dataloaders = {'train': train_loader, 'val': val_loader}

                            # Build model with best dropout rate
                            model = build_model(num_classes, dropout_rate=best_params['dropout_rate'])

                            # Define Loss Function and Optimizer with best learning rate
                            criterion = nn.CrossEntropyLoss()
                            optimizer = optim.Adam(model.fc.parameters(), lr=best_params['learning_rate'])

                            # Check if training checkpoint exists
                            training_checkpoint_path = os.path.join(CHECKPOINT_DIR, 'training_checkpoint.pth')
                            if os.path.exists(training_checkpoint_path):
                                # Load training checkpoint
                                model, optimizer, start_epoch, stage = load_checkpoint(training_checkpoint_path, model,
                                                                                       optimizer)
                                remaining_epochs = NUM_EPOCHS - start_epoch
                                if remaining_epochs <= 0:
                                    logger.info(
                                        "Training has already been completed for the specified number of epochs.")
                                    sys.exit(0)
                                logger.info("Resuming training from the last checkpoint.")
                            else:
                                start_epoch = 0
                                remaining_epochs = NUM_EPOCHS
                                training_checkpoint_path = os.path.join(CHECKPOINT_DIR, 'training_checkpoint.pth')

                            # Train the Model
                            try:
                                model, history = train_model(
                                    model,
                                    dataloaders,
                                    criterion,
                                    optimizer,
                                    num_epochs=remaining_epochs,
                                    device=device,
                                    start_epoch=start_epoch,
                                    checkpoint_path=training_checkpoint_path,
                                    stage='training',
                                    early_stopping_patience=EARLY_STOPPING_PATIENCE
                                )
                            except KeyboardInterrupt:
                                logger.info("Training interrupted by user. Exiting gracefully.")
                                sys.exit(0)

                            best_model = model
                            start_new = False
                            break
                        else:
                            logger.error(f"Best hyperparameters file '{selected_path}' does not exist.")
                            start_new = True
                            break
                    elif selected_stage == 'training':
                        # Resume training from checkpoint
                        model = build_model(num_classes,
                                            dropout_rate=0.5)  # Default dropout; will be loaded from checkpoint
                        optimizer = optim.Adam(model.fc.parameters(),
                                               lr=0.001)  # Placeholder LR; will be loaded from checkpoint
                        model, optimizer, start_epoch, stage = load_checkpoint(selected_path, model, optimizer)

                        # Define DataLoaders with best batch size (assumed to be saved in best_params.json)
                        best_params_path = os.path.join(CHECKPOINT_DIR, 'best_params.json')
                        if os.path.exists(best_params_path):
                            with open(best_params_path, 'r') as f:
                                best_params = json.load(f)
                            # Ensure batch_size is integer
                            best_params['batch_size'] = int(best_params['batch_size'])
                            batch_size = best_params.get('batch_size', 32)
                            dropout_rate = best_params.get('dropout_rate', 0.5)
                        else:
                            # Default parameters if best_params.json is missing
                            batch_size = 32
                            dropout_rate = 0.5
                            logger.warning(
                                "best_params.json not found. Using default batch_size=32 and dropout_rate=0.5.")

                        # Update model with loaded dropout rate
                        model = build_model(num_classes, dropout_rate=dropout_rate)
                        model, optimizer, start_epoch, stage = load_checkpoint(selected_path, model, optimizer)

                        # Define DataLoaders
                        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
                        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
                        dataloaders = {'train': train_loader, 'val': val_loader}

                        # Define Loss Function and Optimizer with best learning rate
                        criterion = nn.CrossEntropyLoss()
                        optimizer = optim.Adam(model.fc.parameters(), lr=best_params.get('learning_rate', 0.001))

                        # Calculate remaining epochs
                        remaining_epochs = NUM_EPOCHS - start_epoch
                        if remaining_epochs <= 0:
                            logger.info("Training has already been completed for the specified number of epochs.")
                            sys.exit(0)

                        # Train the Model
                        try:
                            model, history = train_model(
                                model,
                                dataloaders,
                                criterion,
                                optimizer,
                                num_epochs=remaining_epochs,
                                device=device,
                                start_epoch=start_epoch,
                                checkpoint_path=selected_path,
                                stage='training',
                                early_stopping_patience=EARLY_STOPPING_PATIENCE
                            )
                        except KeyboardInterrupt:
                            logger.info("Training interrupted by user. Exiting gracefully.")
                            sys.exit(0)

                        best_model = model
                        start_new = False
                        break
                else:
                    print("Invalid option. Please try again.")
            except ValueError:
                print("Invalid input. Please enter a valid option number.")
    else:
        logger.info("No existing checkpoints found. Starting a new run.")
        start_new = True

    if start_new:
        # Create a new run folder
        run_folder = create_new_run_folder(CHECKPOINT_DIR)
        tuning_checkpoint_path = os.path.join(run_folder, 'best_params.json')
        training_checkpoint_path = os.path.join(run_folder, 'training_checkpoint.pth')

        # Remove existing best_params.json and training_checkpoint.pth if any
        if os.path.exists(tuning_checkpoint_path):
            os.remove(tuning_checkpoint_path)
            logger.info(f"Removed existing best hyperparameters file '{tuning_checkpoint_path}'.")
        if os.path.exists(training_checkpoint_path):
            os.remove(training_checkpoint_path)
            logger.info(f"Removed existing training checkpoint '{training_checkpoint_path}'.")

        # Initialize hyperparameter tuning
        if HYPERPARAMETER_TUNING and USE_GENETIC_ALGORITHM:
            best_model, best_params = genetic_algorithm_hyperparameter_tuning(train_dataset, val_dataset, class_names,
                                                                              device)
        elif HYPERPARAMETER_TUNING and not USE_GENETIC_ALGORITHM:
            # Implement Grid Search or another tuning method if needed
            logger.error(
                "Grid Search is not implemented. Please set USE_GENETIC_ALGORITHM=True or disable HYPERPARAMETER_TUNING.")
            sys.exit(1)
        else:
            # If hyperparameter tuning is disabled, proceed with predefined hyperparameters
            best_params = {'learning_rate': 0.001, 'batch_size': 32, 'dropout_rate': 0.5}
            logger.info("Hyperparameter tuning is disabled. Using predefined hyperparameters.")
            # Define DataLoaders with predefined batch size
            train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False, num_workers=4)
            dataloaders = {'train': train_loader, 'val': val_loader}

            # Build the Model
            logger.info("Building the model...")
            model = build_model(num_classes, dropout_rate=best_params['dropout_rate'])

            # Define Loss Function and Optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.fc.parameters(), lr=best_params['learning_rate'])

            # Train the Model
            try:
                model, history = train_model(
                    model,
                    dataloaders,
                    criterion,
                    optimizer,
                    num_epochs=NUM_EPOCHS,
                    device=device,
                    start_epoch=0,
                    checkpoint_path=training_checkpoint_path,
                    stage='training',
                    early_stopping_patience=EARLY_STOPPING_PATIENCE
                )
            except KeyboardInterrupt:
                logger.info("Training interrupted by user. Exiting gracefully.")
                sys.exit(0)

            best_model = model

    # Step 5: Save the Final Model
    if not os.path.exists(FINAL_MODEL_SAVE_DIR):
        os.makedirs(FINAL_MODEL_SAVE_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    final_model_path = os.path.join(FINAL_MODEL_SAVE_DIR, f"plant_disease_model_final_{timestamp}.pth")
    torch.save(best_model.state_dict(), final_model_path)
    logger.info(f"Final model saved to '{final_model_path}'.")

    # Step 6: Plot Training History and Confusion Matrix
    if 'history' in locals():
        logger.info("Plotting training history...")
        plot_training_history(history)

    # Plot Confusion Matrix
    logger.info("Plotting confusion matrix...")
    # Ensure batch_size is integer
    batch_size = best_params.get('batch_size', 32)
    batch_size = int(batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    plot_confusion_matrix(best_model, val_loader, class_names, device)

    # Step 7: (Optional) Predict on a New Image
    # Uncomment and set the path to your test image to use the predict function
    """
    test_image_path = 'path_to_your_test_image.jpg'  # Replace with your test image path
    predicted_class = predict_image(best_model, test_image_path, class_names, device)
    if predicted_class:
        print(f"The image is predicted to be: {predicted_class}")
    """
