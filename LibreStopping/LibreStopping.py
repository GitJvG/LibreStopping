class EarlyStopping:
    def __init__(self, save_path, model_name, data_info, patience=5, monitor_metric="roc_auc", **kwargs):
        """Initialize early stopping parameters.

        **Parameters:**
        - save_path: Path to save the model to.
        - model_name: Name of the model to be saved.
        - data_info: Dataset information.
        - patience: Number of epochs with no improvement to wait before stopping.
        - monitor_metric: The metric to monitor for early stopping (e.g., 'roc_auc', 'loss', 'precision').
        
        **Kwargs:**
        - Pass parameters for saving data and models using the following naming conventions:
            - Prefix parameters related to saving data with `savedata_`
            - Prefix parameters related to saving the model with `savemodel_`
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.save_path = save_path
        self.model_name = model_name
        self.early_stop = False
        self.data_info = data_info
        self.epoch = 0  # Initialize epoch counter
        self.monitor_metric = monitor_metric  # The metric to monitor

        # Separate kwargs into savedata_params and savemodel_params
        self.savedata_params = {k: v for k, v in kwargs.items() if k.startswith("savedata_")}
        self.savemodel_params = {k: v for k, v in kwargs.items() if k.startswith("savemodel_")}


    def train_with_early_stopping(self, create_model, fit_model, train_data, eval_data, evaluate_model):
        """
        Trains the model with early stopping.

        Parameters:
        - create_model: Function to create a new model.
        - fit_model: Function to fit the model.
        - train_data: Training data.
        - eval_data: Evaluation data.
        - evaluate_model: Function to evaluate the model.

        Returns:
        - Best trained model.
        """
        while not self.early_stop:  # Continue until early stopping triggers
            self.epoch += 1

            # Create a new model instance
            model = create_model(self.data_info, self.epoch)

            # Fit the model
            fit_model(model, train_data, eval_data)

            # Evaluate the model
            evaluation_results = evaluate_model(model, eval_data)
            current_score = evaluation_results[self.monitor_metric]  # Use the monitored metric
            print(f"Epoch {self.epoch}: {self.monitor_metric} = {current_score}")

            # Check for improvement and handle early stopping logic
            self.check(current_score, model)

        print(f"Training stopped at epoch {self.epoch}. Best {self.monitor_metric}: {self.best_score}")
        return model  # Return the best model

    def check(self, score, model):
        """
        Checks if the current score is better than the best score and triggers early stopping if no improvement.

        Parameters:
        - score: Current evaluation score for the monitored metric.
        - model: The trained model.

        Returns:
        - Boolean indicating whether early stopping has been triggered.
        """
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.save_model(model)  # Save the model if there's an improvement
            self.counter = 0  # Reset the patience counter
            print(f"New best {self.monitor_metric}: {self.best_score} at epoch {self.epoch}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"Early stopping triggered at epoch {self.epoch}")

        return self.early_stop

    def save_model(self, model):
        """Saves the model along with data information."""
        self.data_info.save(self.save_path, model_name=self.model_name, **self.savedata_params) 
        model.save(self.save_path, model_name=self.model_name, **self.savemodel_params)
