# %%
import torch
import torch.nn.functional as F
import random
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm
from sklearn.metrics import f1_score, precision_score, recall_score

# %%
# Set random seeds for reproducibility.
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# %%
# Define the enhanced PoseGCN model.
# This model uses three GCN layers with batch normalization and dropout.
class PoseGCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, dropout_prob=0.5):
        super(PoseGCN, self).__init__()
        # First GCN layer followed by BatchNorm.
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        # Second GCN layer.
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        # Third GCN layer.
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = BatchNorm(hidden_channels)
        # Final fully-connected layer.
        self.fc = torch.nn.Linear(hidden_channels, num_classes)
        # Dropout layer.
        self.dropout = torch.nn.Dropout(p=dropout_prob)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Global pooling aggregates node features into a graph-level representation.
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
model = PoseGCN(num_features=2, hidden_channels=64, num_classes=2, dropout_prob=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
# Scheduler: Reduce LR by a factor of 0.5 if validation loss does not improve for 10 epochs.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

# %%
# Define the training function.
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# %%
# Define the evaluation function.
# This computes loss, accuracy, F1 score, precision, and recall.
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
# Training loop with early stopping and metric tracking.
num_epochs = 100
best_val_loss = float('inf')
patience = 15
trigger_times = 0

# Lists to track metrics for plotting.
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
lrs = []

for epoch in range(1, num_epochs + 1):
    _ = train()  # Run one epoch of training.
    # Evaluate on training set.
    t_loss, t_acc, t_f1, t_prec, t_rec = evaluate(train_loader)
    # Evaluate on validation set.
    v_loss, v_acc, v_f1, v_prec, v_rec = evaluate(val_loader)
    scheduler.step(v_loss)
    current_lr = optimizer.param_groups[0]['lr']
    lrs.append(current_lr)
    
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
    print(f"  Val   -> Loss: {v_loss:.4f}, Acc: {v_acc:.4f}, F1: {v_f1:.4f}, Prec: {v_prec:.4f}, Rec: {v_rec:.4f}, LR: {current_lr:.6f}")
    
    # Early stopping check.
    if v_loss < best_val_loss:
        best_val_loss = v_loss
        trigger_times = 0
        torch.save(model.state_dict(), "pose_gcn_best_model.pt")
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping triggered!")
            break

# %%
# Save the final trained model.
torch.save(model.state_dict(), "pose_gcn_enhanced_model.pt")
print("Model saved as pose_gcn_enhanced_model.pt")

# %%
# Evaluate the model on the test dataset.
test_loss, test_acc, test_f1, test_prec, test_rec = evaluate(test_loader)
print(f"Test -> Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}, Prec: {test_prec:.4f}, Rec: {test_rec:.4f}")

# %%
# Plot charts for Loss, Accuracy, F1 Score, Precision, Recall, and Learning Rate vs. Epoch.
epochs = range(1, len(train_losses) + 1)
plt.figure(figsize=(15,10))

plt.subplot(2, 3, 1)
plt.plot(epochs, train_losses, label='Train Loss', marker='o')
plt.plot(epochs, val_losses, label='Val Loss', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs. Epoch')
plt.legend()
plt.grid(True)

plt.subplot(2, 3, 2)
plt.plot(epochs, train_accuracies, label='Train Accuracy', marker='o')
plt.plot(epochs, val_accuracies, label='Val Accuracy', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. Epoch')
plt.legend()
plt.grid(True)

plt.subplot(2, 3, 3)
plt.plot(epochs, train_f1s, label='Train F1', marker='o')
plt.plot(epochs, val_f1s, label='Val F1', marker='s')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.title('F1 Score vs. Epoch')
plt.legend()
plt.grid(True)

plt.subplot(2, 3, 4)
plt.plot(epochs, train_precisions, label='Train Precision', marker='o')
plt.plot(epochs, val_precisions, label='Val Precision', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.title('Precision vs. Epoch')
plt.legend()
plt.grid(True)

plt.subplot(2, 3, 5)
plt.plot(epochs, train_recalls, label='Train Recall', marker='o')
plt.plot(epochs, val_recalls, label='Val Recall', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.title('Recall vs. Epoch')
plt.legend()
plt.grid(True)

plt.subplot(2, 3, 6)
plt.plot(epochs, lrs, label='Learning Rate', marker='o', color='magenta')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate vs. Epoch')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
