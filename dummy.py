# %% Dummy Metrics Simulation for Model Testing
import numpy as np
import matplotlib.pyplot as plt

# Simulate epochs
epochs = np.arange(1, 51)

# Simulate dummy metric trends (these arrays mimic how metrics might evolve)
# For example, accuracy improves gradually from 60% to 85% with some noise.
accuracy = np.linspace(0.60, 0.85, len(epochs)) + 0.02 * np.random.randn(len(epochs))
precision = np.linspace(0.62, 0.86, len(epochs)) + 0.02 * np.random.randn(len(epochs))
recall = np.linspace(0.58, 0.83, len(epochs)) + 0.02 * np.random.randn(len(epochs))
# F1 is computed from precision and recall, plus a bit of noise.
f1 = 2 * (precision * recall) / (precision + recall) + 0.01 * np.random.randn(len(epochs))

# Plot the dummy metrics over epochs.
plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
plt.plot(epochs, accuracy, label="Accuracy", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Dummy Model Accuracy over Epochs")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(epochs, precision, label="Precision", color='orange', marker='o')
plt.xlabel("Epoch")
plt.ylabel("Precision")
plt.title("Dummy Model Precision over Epochs")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(epochs, recall, label="Recall", color='green', marker='o')
plt.xlabel("Epoch")
plt.ylabel("Recall")
plt.title("Dummy Model Recall over Epochs")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(epochs, f1, label="F1 Score", color='red', marker='o')
plt.xlabel("Epoch")
plt.ylabel("F1 Score")
plt.title("Dummy Model F1 Score over Epochs")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# %% Robustness Test: Compare performance on clean vs. noisy data
# Here we simulate a scenario where the model is tested on clean data versus noisy/adversarial data.
accuracy_clean = np.linspace(0.85, 0.88, len(epochs)) + 0.005 * np.random.randn(len(epochs))
accuracy_noisy = np.linspace(0.75, 0.80, len(epochs)) + 0.01 * np.random.randn(len(epochs))

plt.figure(figsize=(8, 6))
plt.plot(epochs, accuracy_clean, label="Clean Data Accuracy", marker='o')
plt.plot(epochs, accuracy_noisy, label="Noisy Data Accuracy", linestyle='--', marker='x')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Dummy Model Accuracy under Clean vs. Noisy Conditions")
plt.legend()
plt.grid(True)
plt.show()
