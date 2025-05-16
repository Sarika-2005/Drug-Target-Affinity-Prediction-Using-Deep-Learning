import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from model.graphsage_model import GraphSAGE
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load processed graph data
print("Loading processed graph data...")
data, drug_indices, protein_indices, affinities = torch.load(
    'data/processed_graph_data.pt', weights_only=False
)

# Move data to device
data = data.to(device)

# Split dataset (must match training split)
dataset = TensorDataset(drug_indices, protein_indices, affinities)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
_, test_dataset = random_split(dataset, [train_size, test_size])

test_loader = DataLoader(test_dataset, batch_size=64)

# Load model
print("Loading trained model...")
drug_input_dim = data['drug'].x.shape[1]
protein_input_dim = data['protein'].x.shape[1]
model = GraphSAGE(drug_input_dim, protein_input_dim, hidden_dim=256)
model.load_state_dict(torch.load('models/graphsage_model.pth', map_location=device))
model = model.to(device)
model.eval()

# Inference
all_preds = []
all_labels = []

print("Evaluating on test data...")
with torch.no_grad():
    for d_idx, p_idx, labels in test_loader:
        d_idx = d_idx.to(device)
        p_idx = p_idx.to(device)
        labels = labels.to(device)

        outputs = model(data, d_idx, p_idx)
        all_preds.extend(outputs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Metrics
all_preds = np.array(all_preds).flatten()
all_labels = np.array(all_labels).flatten()

mse = mean_squared_error(all_labels, all_preds)
rmse = np.sqrt(mse)
mae = mean_absolute_error(all_labels, all_preds)
r2 = r2_score(all_labels, all_preds)
pearson_corr, _ = pearsonr(all_labels, all_preds)
spearman_corr, _ = spearmanr(all_labels, all_preds)

# Output metrics
print("\n===== Evaluation Metrics =====")
print(f"Test MSE: {mse:.4f}")
print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test R² Score: {r2:.4f}")
print(f"Pearson Correlation: {pearson_corr:.4f}")
print(f"Spearman Correlation: {spearman_corr:.4f}")

# ----------------------------
# PLOTS
# ----------------------------

# Create plots directory if not exist
os.makedirs("plots", exist_ok=True)
# ----------------------------
# METRICS PLOT
# ----------------------------

# 1. Regression Metrics Bar Plot
regression_metrics = {
    'MSE': mse,
    'RMSE': rmse,
    'MAE': mae,
    'R² Score': r2
}

plt.figure(figsize=(8,6))
sns.barplot(x=list(regression_metrics.keys()), y=list(regression_metrics.values()), palette="Blues_d")
plt.title('Regression Metrics')
plt.ylabel('Score')
for i, v in enumerate(regression_metrics.values()):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center', fontweight='bold')
plt.grid(axis='y')
plt.savefig('plots/regression_metrics.png')
plt.show()

# 2. Correlation Metrics Bar Plot
correlation_metrics = {
    'Pearson Corr': pearson_corr,
    'Spearman Corr': spearman_corr
}

plt.figure(figsize=(6,6))
sns.barplot(x=list(correlation_metrics.keys()), y=list(correlation_metrics.values()), palette="Greens_d")
plt.title('Correlation Metrics')
plt.ylabel('Correlation Coefficient')
for i, v in enumerate(correlation_metrics.values()):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center', fontweight='bold')
plt.ylim(0, 1.1)  # Correlations are between -1 and 1
plt.grid(axis='y')
plt.savefig('plots/correlation_metrics.png')
plt.show()


# 1. Scatter Plot: True vs Predicted
plt.figure(figsize=(8,6))
sns.scatterplot(x=all_labels, y=all_preds, alpha=0.6)
plt.plot([all_labels.min(), all_labels.max()], [all_labels.min(), all_labels.max()], 'r--')
plt.xlabel('True Affinity')
plt.ylabel('Predicted Affinity')
plt.title('True vs Predicted Affinity')
plt.grid(True)
plt.savefig('plots/true_vs_predicted.png')
plt.show()

# 2. Error Distribution Plot
errors = all_preds - all_labels
plt.figure(figsize=(8,6))
sns.histplot(errors, bins=50, kde=True, color='orange')
plt.xlabel('Prediction Error')
plt.title('Prediction Error Distribution')
plt.grid(True)
plt.savefig('plots/error_distribution.png')
plt.show()

print("\nPlots saved to 'plots/' directory!")
