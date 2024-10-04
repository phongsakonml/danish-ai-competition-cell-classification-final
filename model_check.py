import json
import matplotlib.pyplot as plt

def load_metrics(json_file):
    with open(json_file, 'r') as file:
        return json.load(file)

def plot_curves(metrics):
    for model_name, data in metrics.items():
        plt.plot(data['epochs'], data['train_loss'], label=f'{model_name} Train Loss')
        plt.plot(data['epochs'], data['val_loss'], label=f'{model_name} Val Loss')
        plt.title(f'{model_name} Loss Curves')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

def check_overfitting(metrics):
    for model_name, data in metrics.items():
        if data['train_loss'][-1] < data['val_loss'][-1]:
            print(f"{model_name} is overfitting.")
        else:
            print(f"{model_name} is not overfitting.")

# Load metrics from JSON
metrics = load_metrics('runs/tyler/training_log_fold_final.json')

# Plot curves
plot_curves(metrics)

# Check for overfitting
check_overfitting(metrics)