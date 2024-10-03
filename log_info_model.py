import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_model_info(model_path):
    # Load the model
    model = torch.load(model_path)
    
    # Log model architecture
    logging.info(f'Model architecture: {model.__class__.__name__}')
    
    # Log model state dict keys
    logging.info(f'Model state dict keys: {list(model.state_dict().keys())}')
    
    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f'Total parameters: {total_params}')

if __name__ == "__main__":
    model_path = 'runs/old/efficientnet_b0_balanced_advanced_20241003_093305_0/best_model.pth'
    log_model_info(model_path)