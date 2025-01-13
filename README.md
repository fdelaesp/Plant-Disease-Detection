# Plant-Disease-Detection

## Overview
This repository provides a machine learning pipeline for detecting plant diseases using PyTorch. The implementation focuses on building a robust classification model based on the ResNet50 architecture, optimized for plant disease detection tasks. It includes features for data preparation, hyperparameter tuning, and model evaluation.

---

Features
- Data Preparation**: Preprocesses and augments image datasets for robust training.
- Model Architecture: Fine-tunes a ResNet50 model for plant disease classification.
- Hyperparameter Tuning: Uses a Genetic Algorithm to optimize parameters like learning rate, batch size, and dropout rate.
- Logging: Logs training progress, errors, and other critical events for monitoring.
- Checkpointing: Saves model states for resuming interrupted training sessions.
- Visualization: Provides plots of training/validation accuracy, loss, and confusion matrices.

---

Requirements
To run this project, you need:

- Python 3.8 or above
- PyTorch
- torchvision
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tqdm
- deap
- Pillow (PIL)

Install the dependencies using:
```bash
pip install -r requirements.txt
```

---

Getting Started
1. Dataset
   - Ensure the dataset is downloaded and extracted. Place the CSV file (`train.csv`) and images folder (`images/`) in the specified directory.

2. Configuration:
   - Modify parameters in `Model_1.py` as needed (e.g., dataset path, epochs, batch size).

3. Run the Script:
   ```bash
   python Model_1.py
   ```

---

Key Functions in `Model_1.py`

 1. Data Preparation
The `prepare_data()` function:
- Loads image paths and labels from a CSV file.
- Splits data into training and validation sets.
- Applies data augmentation and normalization.

2. Model Building
The `build_model()` function:
- Fine-tunes a pre-trained ResNet50 model.
- Adds a custom fully connected layer with dropout for classification.

 3. Hyperparameter Tuning
- Uses a Genetic Algorithm for optimizing learning rate, batch size, and dropout rate.
- Evaluates each parameter set based on validation accuracy.

 4. Training
The `train_model()` function:
- Trains the model over multiple epochs.
- Supports early stopping to prevent overfitting.

 5. Evaluation
- Plots training and validation accuracy/loss.
- Generates confusion matrices for classification performance.

---

 Outputs
- Model Checkpoints: Saved during and after training for resumption or deployment.
- Visualizations: Accuracy and loss plots, confusion matrices.

---

 Usage Notes
- Modify the hyperparameters and dataset paths to match your setup.
- Use a GPU for faster training, especially on larger datasets.
- Ensure all dependencies are installed for seamless execution.

---

Contributing
Feel free to fork this repository, create a branch, and submit a pull request. Contributions are welcome!

---

 License
This project is licensed under the MIT License.

