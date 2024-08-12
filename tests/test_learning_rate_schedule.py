import math
import os

import matplotlib.pyplot as plt

# Configuration values
base_learning_rate = 1.5e-4
batch_size = 4096
warmup_epoch = 200
total_epoch = 2000


# Custom learning rate function
def lr_func(epoch):
    warmup_phase = (epoch + 1) / (warmup_epoch + 1e-8)
    cosine_decay = 0.5 * (math.cos(epoch / total_epoch * math.pi) + 1)
    return min(warmup_phase, cosine_decay)


# Initialize list to store learning rates
learning_rates = []

# Simulate learning rate over 2000 epochs
for epoch in range(total_epoch):
    learning_rate = lr_func(epoch) * base_learning_rate * batch_size / 256
    learning_rates.append(learning_rate)

# Plot the learning rate over epochs
plt.figure(figsize=(10, 6))
plt.plot(range(total_epoch), learning_rates, label="Learning Rate")
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Learning Rate Schedule Over Epochs")
plt.grid(True)
plt.legend()

folder_name = f"../images/model_results"
# create a folder to save the model if it does not exist
os.makedirs(folder_name, exist_ok=True)
plt.savefig(f"{folder_name}/lr_scheduler.png")
