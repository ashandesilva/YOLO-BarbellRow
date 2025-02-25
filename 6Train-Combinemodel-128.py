# %%
# Below is a complete code snippet that combines all three modifications into one training pipeline. In this version, we:

# 1 Experiment with the Architecture:

# Increase the hidden channel size (set to 64).
# Add an extra GCN layer (three layers total).
# Apply Regularization:

# 2 Use dropout (with probability 0.5) after each activation.
# Incorporate Learning Rate Scheduling:

# 3 Use a StepLR scheduler that reduces the learning rate by a factor of 0.5 every 20 epochs.

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# %%
# Define the PoseGCN model with extra layers, increased hidden channels, and dropout.
class PoseGCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, dropout_prob=0.5):
        super(PoseGCN, self).__init__()
        # Three GCN layers with increased capacity.
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, num_classes)
        self.dropout = torch.nn.Dropout(p=dropout_prob)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)  # Apply dropout for regularization.
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        # Global pooling to obtain a graph-level representation.
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# %%
# Load pre-saved graphs for training, validation, and testing.
train_graphs = torch.load("train_pose_graphs.pt", weights_only=False)
val_graphs   = torch.load("val_pose_graphs.pt", weights_only=False)
test_graphs  = torch.load("test_pose_graphs.pt", weights_only=False)

# %%
# Create DataLoader objects.
train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_graphs, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_graphs, batch_size=32, shuffle=False)

# %%
# Instantiate the model.
# - num_features=2 for (x, y) coordinates.
# - hidden_channels=64 for increased capacity.
# - dropout_prob=0.5 applies dropout after each activation.
model = PoseGCN(num_features=2, hidden_channels=128, num_classes=2, dropout_prob=0.5)

# Use Adam optimizer with initial learning rate 0.01 and weight decay.
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Add a learning rate scheduler that decays LR by 0.5 every 20 epochs.
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# %%
# Define the training function.
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data)  # Forward pass: output shape [batch_size, num_classes]
        loss = F.nll_loss(out, data.y)  # data.y is a 1D tensor with one label per graph.
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# %%
# Define the evaluation function.
def evaluate(loader):
    model.eval()
    correct = 0
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            out = model(data)
            total_loss += F.nll_loss(out, data.y, reduction='sum').item()
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

# %%
# Training loop.
num_epochs = 100
for epoch in range(1, num_epochs + 1):
    train_loss = train()
    val_loss, val_acc = evaluate(val_loader)
    scheduler.step()  # Update the learning rate.
    print(f"Epoch {epoch:03d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# %%
# Save the trained model.
torch.save(model.state_dict(), "pose_gcn_combined_model.pt")
print("Model saved as pose_gcn_combined_model-128.pt")

# %%
# Test the model on the test dataset.
test_loss, test_acc = evaluate(test_loader)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
