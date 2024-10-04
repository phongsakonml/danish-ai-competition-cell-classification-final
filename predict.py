def ensemble_predict(ensemble_models, image):
    predictions = []
    for model in ensemble_models:
        model.eval()
        with torch.no_grad():
            output = model(image.unsqueeze(0).to(device))
            predictions.append(F.softmax(output, dim=1))
    
    # Average predictions
    avg_prediction = torch.mean(torch.stack(predictions), dim=0)
    return avg_prediction.squeeze()

# Load ensemble models
ensemble_models = []
for i in range(3):
    model = initialize_model().to(device)
    model.load_state_dict(torch.load(os.path.join(run_dir, f'ensemble_model_{i}.pth')))
    ensemble_models.append(model)

# Use ensemble_predict instead of single model prediction
prediction = ensemble_predict(ensemble_models, preprocessed_image)