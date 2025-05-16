import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from model.graphsage_model import GraphSAGE
from tqdm import tqdm
import os

# Load processed data
print("Loading processed graph data...")
data, drug_indices, protein_indices, affinities = torch.load(
    'data/processed_graph_data.pt', weights_only=False
)

# Debug print
print("Graph info:")
print(data)
print("Node types:", data.node_types)
print("Edge types:", data.edge_types)
print(f"Drug features: {data['drug'].x.shape}")
print(f"Protein features: {data['protein'].x.shape}")
print(f"Total samples: {len(drug_indices)}")

# Dataset preparation
print("Preparing dataset...")
dataset = TensorDataset(drug_indices, protein_indices, affinities)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=512)

# Initialize model
print("Initializing model...")
drug_input_dim = data['drug'].x.shape[1]
protein_input_dim = data['protein'].x.shape[1]
model = GraphSAGE(drug_input_dim, protein_input_dim, hidden_dim=256)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)
data = data.to(device)

# Training loop
print("Starting training...")
num_epochs = 100
for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    
    for d_idx, p_idx, labels in progress_bar:
        d_idx, p_idx, labels = d_idx.to(device), p_idx.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(data, d_idx, p_idx)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}")

# Save model
os.makedirs('models', exist_ok=True)
torch.save(model.state_dict(), 'models/graphsage_model.pth')
print("Model saved to 'models/graphsage_model.pth'")
