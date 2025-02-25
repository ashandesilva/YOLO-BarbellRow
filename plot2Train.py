# %% same code as 2Train-GCN.ipynb but with plots for testiing chap 5
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.metrics import f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

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
# Evaluation function that computes loss, accuracy, F1 score, precision, and recall.
def evaluate(loader):
    model.eval()
    total_loss = 0
    correct = 0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for data in loader:
            out = model(data)
            loss = F.nll_loss(out, data.y, reduction='sum')
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            correct += (pred == data.y).sum().item()
            all_labels.extend(data.y.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
    
    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    
    return avg_loss, accuracy, f1, precision, recall

# %%
# Training loop with metric tracking.
num_epochs = 100
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
train_f1s = []
val_f1s = []
train_precisions = []
val_precisions = []
train_recalls = []
val_recalls = []

for epoch in range(1, num_epochs + 1):
    train_loss = train()
    # Evaluate on training data
    t_loss, t_acc, t_f1, t_prec, t_rec = evaluate(train_loader)
    # Evaluate on validation data
    v_loss, v_acc, v_f1, v_prec, v_rec = evaluate(val_loader)
    
    train_losses.append(t_loss)
    val_losses.append(v_loss)
    train_accuracies.append(t_acc)
    val_accuracies.append(v_acc)
    train_f1s.append(t_f1)
    val_f1s.append(v_f1)
    train_precisions.append(t_prec)
    val_precisions.append(v_prec)
    train_recalls.append(t_rec)
    val_recalls.append(v_rec)
    
    print(f"Epoch {epoch:03d}:")
    print(f"  Train -> Loss: {t_loss:.4f}, Acc: {t_acc:.4f}, F1: {t_f1:.4f}, Prec: {t_prec:.4f}, Rec: {t_rec:.4f}")
    print(f"  Val   -> Loss: {v_loss:.4f}, Acc: {v_acc:.4f}, F1: {v_f1:.4f}, Prec: {v_prec:.4f}, Rec: {v_rec:.4f}")

# %%
# Save the trained model.
torch.save(model.state_dict(), "2pose_gcn_model.pt")
print("Model saved as 2pose_gcn_model.pt")

# %%
# Test the model on the test dataset.
test_loss, test_acc, test_f1, test_precision, test_recall = evaluate(test_loader)
print(f"Test -> Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}, Prec: {test_precision:.4f}, Rec: {test_recall:.4f}")

# %%
# Plot charts for Loss, Accuracy, F1 Score, Precision, and Recall vs. Epoch.
epochs = range(1, num_epochs + 1)

plt.figure(figsize=(12, 8))

# Plot Loss
plt.subplot(2, 3, 1)
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs. Epoch')
plt.legend()

# Plot Accuracy
plt.subplot(2, 3, 2)
plt.plot(epochs, train_accuracies, label='Train Acc')
plt.plot(epochs, val_accuracies, label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Epoch')
plt.legend()

# Plot F1 Score
plt.subplot(2, 3, 3)
plt.plot(epochs, train_f1s, label='Train F1')
plt.plot(epochs, val_f1s, label='Val F1')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.title('F1 Score vs. Epoch')
plt.legend()

# Plot Precision
plt.subplot(2, 3, 4)
plt.plot(epochs, train_precisions, label='Train Precision')
plt.plot(epochs, val_precisions, label='Val Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.title('Precision vs. Epoch')
plt.legend()

# Plot Recall
plt.subplot(2, 3, 5)
plt.plot(epochs, train_recalls, label='Train Recall')
plt.plot(epochs, val_recalls, label='Val Recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.title('Recall vs. Epoch')
plt.legend()

plt.tight_layout()
plt.show()
