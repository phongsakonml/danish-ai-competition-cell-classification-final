

The game plan below is fully revised for simplicity while still leading you toward creating a **strong model**.

---

## **GAMEPLAN.md**  
### Your Path to Winning the Competition

---

### **1. Data Augmentation to Balance the Classes**

Since the dataset is imbalanced (100 heterogenous vs 20 homogenous cells), we will first create a **balanced dataset** by augmenting the minority class (homogenous cells).

#### **a. Augmentation Strategy**
- For the **20 homogenous cells**, apply augmentation to generate additional **80 samples**, giving you **100 homogenous cell images** in total (matching the 100 heterogenous samples).
- Focus on augmentations that **preserve the biological integrity** of cell structures:
  - **Horizontal and Vertical Flips**
  - **Small Rotations** (up to 45°)
  - **Elastic Deformations** (slight cell distortions to simulate natural variation)
  - **Random Cropping** and **Zooming** (small scale changes, simulating microscope variation)
  - **Gaussian Noise** (simulates image acquisition noise)

This will create a **50:50 balanced dataset** (100 heterogenous and 100 homogenous).

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=45, p=0.7),
    A.ElasticTransform(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.RandomCrop(height=256, width=256, p=0.5),
    ToTensorV2()
])

# Apply this augmentation pipeline to the 20 homogenous images, generating 80 more samples.
```

#### **b. Check Visual Quality**
- After applying augmentation, **manually inspect a few samples** to ensure that the augmentations look biologically realistic.
- Ensure that the **shapes and textures** of cells are not distorted in a way that affects their classification.

---

### **2. Model Selection: Transfer Learning with Pre-Trained Models**

#### **a. Why Transfer Learning?**
- The dataset is small, so starting from scratch would likely cause **overfitting**. Instead, use a **pre-trained model** (on large datasets like ImageNet) and fine-tune it for your specific task.

#### **b. Recommended Models**
- **EfficientNet-B0**: Efficient and accurate for small datasets.
- **ResNet-50**: Proven for image classification tasks.
- **MobileNetV2**: Lightweight model to avoid overfitting.

#### **c. Training Strategy**
1. **Freeze Most Layers**: First, freeze the lower layers of the model (pre-trained feature extractors) and train only the final few layers (classifier).
2. **Fine-Tune Entire Model**: After the classifier converges, unfreeze all layers and fine-tune with a **low learning rate**.
3. **Start Simple**: Begin with **EfficientNet-B0**, then experiment with **ResNet-50** or **MobileNetV2** if needed.

```python
from torchvision import models

# Load pre-trained EfficientNet
model = models.efficientnet_b0(pretrained=True)
for param in model.parameters():
    param.requires_grad = False  # Freeze pre-trained layers

# Replace the classifier with a new layer for binary classification
model.classifier = nn.Sequential(
    nn.Linear(in_features=1280, out_features=512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, 2)  # Binary output for classification
)

# Unfreeze all layers and fine-tune with smaller learning rate later
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
```

---

### **3. Hyperparameter Tuning for Performance Optimization**

#### **a. Learning Rate Strategy**
- Start with a **learning rate** of `1e-3` and then reduce it using a **scheduler** when the validation loss plateaus.
- Use a **learning rate scheduler** like **ReduceLROnPlateau** to adjust the learning rate based on validation performance.

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
```

#### **b. Batch Size and Epochs**
- **Batch Size**: Use **16 or 32** depending on memory capacity.
- **Early Stopping**: Monitor the validation loss, and use **early stopping** to prevent overfitting (e.g., stop after **5 epochs** without improvement).

---

### **4. Post-Training: Decision Threshold Optimization**

By default, classification models use a threshold of **0.5** to decide between classes. However, since both classes are now balanced, you should **optimize the threshold** based on the **precision-recall curve** to further fine-tune performance.

#### **a. Tune the Threshold**
- Calculate **precision-recall curves** after training and identify the **optimal decision threshold** (instead of using the default 0.5).

```python
from sklearn.metrics import precision_recall_curve

y_pred_prob = model.predict_proba(X_val)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_val, y_pred_prob)
optimal_idx = np.argmax(precision * recall)
optimal_threshold = thresholds[optimal_idx]
```

---

### **5. Evaluation Metrics: Focus on Precision and Recall**

Since this is a **binary classification problem** and both classes are balanced, you need to focus on the following metrics during evaluation:

#### **a. Key Metrics**
- **F1-Score**: Combines precision and recall for an overall metric.
- **Precision and Recall**: Focus specifically on both precision and recall to ensure that the minority class (homogenous cells) is not missed.
- **Confusion Matrix**: Use this to check for false positives and false negatives.

```python
from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

#### **b. Balanced Accuracy**
- Because the dataset is now balanced, balanced accuracy will give a good indication of how the model performs across both classes.

---

### **6. Cross-Validation to Ensure Generalization**

#### **a. 5-Fold Cross-Validation**
- Use **5-fold cross-validation** to assess model generalization. This ensures the model doesn’t overfit to one specific split of the data.

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
print("F1-Score: ", np.mean(scores))
```

---

### **7. Model Ensembling for Final Boost**

To maximize performance, you can combine different models (EfficientNet, ResNet, MobileNet) using an **ensemble** approach. This can improve robustness by leveraging the strengths of each model.

#### **a. Voting Classifier (Ensemble Learning)**
- After training multiple models, use an ensemble like **Voting Classifier** with **soft voting** (i.e., probability averaging).

```python
from sklearn.ensemble import VotingClassifier

ensemble_model = VotingClassifier(estimators=[
    ('efficientnet', efficientnet_model),
    ('resnet', resnet_model),
], voting='soft')
```

---

### **8. Iterative Process and Final Submission**

- **Submit Early**: Make early submissions and adjust based on leaderboard feedback.
- **Iterate**: Continue to experiment with fine-tuning the models, adjusting thresholds, and augmentations based on the results.

---

### **Summary of the Game Plan**

#### **Data Augmentation**:
- Augment the **homogenous cells** to create a balanced dataset with biologically realistic transformations.

#### **Model Selection**:
- Use **EfficientNet-B0** with **transfer learning**.
- Fine-tune the entire model with a **low learning rate**.

#### **Training Strategy**:
- Monitor **validation loss** and use **early stopping**.
- Tune the decision threshold using **precision-recall curves**.

#### **Metrics to Focus On**:
- Focus on **F1-Score**, **Precision-Recall**, and **Confusion Matrix**.
  
#### **Final Steps**:
- Use **5-fold cross-validation** to ensure generalization.
- Explore **ensemble learning** for a final performance boost.

---

With this **step-by-step plan**, you should have a clear roadmap to train a strong model that has a great chance of winning the competition. Each section is focused on **actionable steps** to get the best performance from your model with a balanced dataset.

