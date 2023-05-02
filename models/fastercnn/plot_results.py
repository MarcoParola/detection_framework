import json
import matplotlib.pyplot as plt

# Read the JSON file
with open('C:/Users/fuma2/Development/Github/detection_framework/outputs/fastercnn/model_outputs/metrics.json', 'r') as f:
    data = json.load(f)

# Extract the loss values
train_losses = []
val_losses = []
for d in data:
    if 'total_loss' in d:
        train_losses.append(d['total_loss'])
    if 'val_total_loss' in d:
        val_losses.append(d['val_total_loss'])

# Create a plot of the training and validation loss over time
plt.plot(val_losses, label='Validation loss')
#plt.plot(train_losses, label='Training loss', color="orangered")
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()