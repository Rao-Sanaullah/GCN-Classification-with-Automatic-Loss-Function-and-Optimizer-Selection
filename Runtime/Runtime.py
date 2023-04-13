
import sklearn.metrics
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv
import matplotlib.pyplot as plt
import time


dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 64)
        self.conv2 = GCNConv(64, 32)
        self.conv3 = GCNConv(32, dataset.num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

# Create GNN model
model = GCN()

# Define a list of loss functions and their corresponding performance metrics
loss_functions = {'nll_loss': 'accuracy', 'cross_entropy_loss': 'f1_macro'}


# Define a list of optimizers and their corresponding learning rates
optimizers = [
    (torch.optim.Adam, {'lr': 0.001}),
    (torch.optim.SGD, {'lr': 0.01}),
    (torch.optim.RMSprop, {'lr': 0.001}),
]


"""# Create the plot outside the for loop
fig, axs = plt.subplots(3, 2, figsize=(15, 10))
ax1, ax2, ax3, ax4, ax5 = axs[0][0], axs[0][1], axs[1][0], axs[1][1], axs[2][0]
axs[2][1].remove()"""

# Create the plot outside the for loop
fig, axs = plt.subplots(3, 2, figsize=(25, 10))
ax1, ax2, ax3, ax4 = axs[0][0], axs[0][1], axs[1][0], axs[1][1]
ax5 = fig.add_subplot(313)

# Remove the last subplot from the third row
axs[2][1].remove()
# Remove the last subplot from the third row
axs[2][0].remove()



# Set the x and y labels for the plot
ax1.set_title('Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')

ax2.set_title('Validation Metric')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Metric')

ax3.set_title('Training Loss Function')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss Function')

ax5.set_title('Optimizer vs. Epoch')
ax5.set_xlabel('Epoch')
ax5.set_ylabel('Optimizer')

ax4.set_title('Accuracy vs. Runtime')
ax4.set_xlabel('Runtime')
ax4.set_ylabel('Accuracy')


# Add gridlines to the plots
ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()
ax5.grid()

# Add some space after each plot downside
plt.subplots_adjust(hspace=0.5)

# Add a text box for the current metric value at the right bottom of ax2
metric_text = ax2.text(0.98, 0.02, '', transform=ax2.transAxes, ha='right', va='bottom')



# Initialize the plot data
train_losses = []
train_metrics = []
optimizer_values = []

runtime = []
accuracy = []
best_losses = []

# Train the model
best_optimizer = None
best_loss_fn = None
best_metric = float('inf')
train_losses = []

# Start the training loop
for opt_class, opt_kwargs in optimizers:
    optimizer = opt_class(model.parameters(), **opt_kwargs)

    # Define a ReduceLROnPlateau learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)



    # Initialize the best metric to zero
    best_metric = 0

    for epoch in range(100):
        optimizer.zero_grad()
        out = model(data, data)

        # Select the best loss function based on the best metric
        for loss_fn in loss_functions:
            if loss_functions[loss_fn] == 'accuracy':
                loss = F.nll_loss(out[data.train_mask], data[data.train_mask])
                _, pred = out.max(dim=1)
                correct = float(pred[data.train_mask].eq(data[data.train_mask]).sum().item())
                metric = correct / data.train_mask.sum().item()
            elif loss_functions[loss_fn] == 'f1_macro':
                loss = F.cross_entropy(out[data.train_mask], data[data.train_mask])
                _, pred = out.max(dim=1)
                metric = sklearn.metrics.f1_score(data.y[data.train_mask].cpu().numpy(),
                                                  pred[data.train_mask].cpu().numpy(),
                                                  average='macro')

            # Check if the current loss function has a better metric than the previous best
            if metric > best_metric:
                best_metric = metric
                best_loss_fn = loss_fn
                best_optimizer = optimizer

        loss.backward()
        optimizer.step()

        # Adjust the learning rate using the scheduler
        scheduler.step(best_metric)
        train_losses.append(loss)
        train_metrics.append(metric)
        optimizer_values.append(optimizer.param_groups['lr'])
        best_losses.append(best_loss_fn)

        # Update the plot data
        ax1.plot(train_losses)
        ax2.plot(train_metrics)
        ax5.plot(optimizer_values)
        ax3.plot(best_losses)

        # Set the legend for the plot
        ax1.legend(loss_functions.keys())
        ax2.legend([loss_functions[best_loss_fn]])
        ax5.legend([best_optimizer])
        ax3.legend([best_loss_fn])

        # Update the metric text box
        metric_text.set_text(f'{best_loss_fn}: {best_metric:.4f}')

        # Redraw the first three plots
        plt.draw()

        # Give the first three plots a chance to update
        plt.pause(0.001)

        # Test the model and measure its accuracy
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            _, pred = out.max(dim=1)
            correct = float(pred[data.train_mask].eq(data.y[data.train_mask]).sum().item())
            acc = correct / data.train_mask.sum().item()

        # Append the runtime and accuracy to the plot data
        runtime.append(time.time())
        accuracy.append(acc)


        # Update the plot data for the third plot
        ax4.plot(runtime, accuracy)
        ax4.legend([acc])

        # Redraw the plot
        fig.canvas.draw()

        # Give the plot a chance to update
        plt.pause(0.001)


# Show the final plot
plt.show()


# Evaluate the model using the best optimizer
model.eval()
_, pred = model(data.x, data.edge_index).max(dim=1)
correct = float(pred[data.train_mask].eq(data[data.train_mask]).sum().item())
acc = correct / data.train_mask.sum().item()
print('Best loss function: {}, Best metric: {:.4f}, Accuracy: {:.4f}'.format(best_loss_fn, best_metric, acc))