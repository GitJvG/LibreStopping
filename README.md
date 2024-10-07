# LibreStopping | LibRecommender Early Stopping

This repository provides an implementation of early stopping functionality for models in the **LibRecommender** framework. Early stopping is a form of regularization used to avoid overfitting when training a model. This implementation allows users to monitor a specified performance metric and halt training when performance ceases to improve.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Features

- **Customizable Monitoring**: Choose the metric to monitor for early stopping (e.g., ROC AUC, loss, precision).
- **Patience Parameter**: Specify the number of epochs with no improvement before training is halted.
- **Model Saving**: Automatically saves the best model based on the monitored metric.

## Installation

To install the repository, install it using pip:

```bash
pip install LibreStopping
```

or clone it to your local machine using:

```bash
git clone https://github.com/GitJvG/LibreStopping.git
```

## Usage

### Required Support Functions and Variables
Before using the `EarlyStopping` class, ensure you have the required support functions and variables set up:

```python
from libreco.algorithms import TwoTower
from libreco.evaluation import evaluate
from libreco.data import DatasetFeat

# Define example data (replace with your actual data loading)
train_data, user_col, item_col, sparse_col, dense_col, eval_data = ['...']
train_data, data_info = DatasetFeat.build_trainset(train_data, user_col, item_col, sparse_col, dense_col)
eval_data = DatasetFeat.build_evalset(eval_data)

# Example functions for creating, fitting, and evaluating a model
def create_model(data_info, n_epochs):
    # Create and return a TwoTower model
    return TwoTower(
        "ranking",
        data_info,
        n_epochs=n_epochs,
        embed_size=16,
        norm_embed=True,
    )

def fit_model(model, train_data, eval_data):
    # Fit the model using training and evaluation data
    model.fit(train_data, 
              neg_sampling=True, 
              eval_data=eval_data, 
              shuffle=True, 
              metrics=["roc_auc"])

def evaluate_model(model, eval_data):
    # Evaluate the model and return evaluation results
    evaluation_results = evaluate(
        model=model,
        data=eval_data,
        neg_sampling=True,
        metrics=["loss", "roc_auc", "precision", "recall", "ndcg"],
    )
    print(f"Evaluation Results: {evaluation_results}")
    return evaluation_results
```

### Using EarlyStopping to Train Your Model

```python
from LibreStopping.LibreStopping import EarlyStopping  # Import the EarlyStopping class

# Initialize EarlyStopping
early_stopping = EarlyStopping(
    model_path="path/to/save/model",  # Path to save the best model
    model_name="best_model",           # Name of the model
    data_info=data_info,               # Dataset information
    patience=5,                        # Number of epochs to wait before stopping
    monitor_metric="loss"           # Metric to monitor
)

# Start training with early stopping
best_model = early_stopping.train_with_early_stopping(
    create_model=create_model,  # Function to create the model
    fit_model=fit_model,        # Function to fit the model
    train_data=train_data,      # Training data
    eval_data=eval_data,        # Evaluation data
    evaluate_model=evaluate_model # Function to evaluate the model
)
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
