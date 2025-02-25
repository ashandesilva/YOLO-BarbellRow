# %%
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

# %%
# Define the updated PoseGCN model with global pooling for graph-level classification.
class PoseGCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(PoseGCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        # Unpack the data. The 'batch' attribute is created by the DataLoader.
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        # Global pooling aggregates node features into one graph-level representation.
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# %%
# Load pre-saved graphs for training, validation, and testing.
train_graphs = torch.load("train_pose_graphs.pt", weights_only=False)
val_graphs = torch.load("val_pose_graphs.pt", weights_only=False)
test_graphs = torch.load("test_pose_graphs.pt", weights_only=False)

# %%
# Create DataLoader objects to automatically batch the graphs.
train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=32, shuffle=False)
test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

# %%
# Instantiate the model. Here num_features=2 because we use (x, y) coordinates.
model = PoseGCN(num_features=2, hidden_channels=32, num_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# %%
# Training function.
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
# Evaluation function to monitor performance.
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
# Training loop 
# for epoch in range(1, 51):
#     train_loss = train()
#     val_loss, val_acc = evaluate(val_loader)
#     print(f"Epoch {epoch:03d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# Training loop.
num_epochs = 100
for epoch in range(1, num_epochs + 1):
    train_loss = train()
    val_loss, val_acc = evaluate(val_loader)
    print(f"Epoch {epoch:03d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# %%
# Save the trained model.
torch.save(model.state_dict(), "2pose_gcn_model.pt")
print("Model saved as 2pose_gcn_model.pt")

# %%
# Test the model on the test dataset.
test_loss, test_acc = evaluate(test_loader)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")


