import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Load data
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
loss_functions = {'nll_loss': 'accuracy', 'mse_loss': 'mse', 'l1_loss': 'mse'}

# Define a list of optimizers and their corresponding learning rates
optimizers = [
    (torch.optim.Adam, {'lr': 0.001}),
    (torch.optim.SGD, {'lr': 0.01}),
    (torch.optim.RMSprop, {'lr': 0.001}),
]

# Train the model
best_optimizer = None
best_loss_fn = None
best_metric = float('inf')
train_losses = []


for opt_class, opt_kwargs in optimizers:
    optimizer = opt_class(model.parameters(), **opt_kwargs)
    
    # Define a ReduceLROnPlateau learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)


    model.train()


    for epoch in range(100):
        optimizer.zero_grad()
        out = model(data, data)
        losses = {}
        for loss_fn in loss_functions:
            if loss_fn == 'nll_loss':
                losses[loss_fn] = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            elif loss_fn == 'mse_loss':
                losses[loss_fn] = F.mse_loss(out[data.train_mask], data.y[data.train_mask].unsqueeze(-1).repeat(1, dataset.num_classes))
            elif loss_fn == 'l1_loss':
                losses[loss_fn] = F.l1_loss(out[data.train_mask], data.y[data.train_mask].unsqueeze(-1).repeat(1, dataset.num_classes))
            else:
                raise ValueError('Invalid loss function')
                
            # Compute the performance metric for each loss function
            if loss_functions[loss_fn] == 'accuracy':
                _, pred = model(data, data.edge_index).max(dim=1)
                correct = float(pred[data.train_mask].eq(data[data.train_mask]).sum().item())
                metric = correct / data.train_mask.sum().item()
                print(f'Training accuracy for epoch {epoch + 1}: {metric:.4f}')
            elif loss_functions[loss_fn] == 'mse':
                pred = model(data, data.edge_index)
                metric = F.mse_loss(pred[data.test_mask], data[data.test_mask].unsqueeze(-1).repeat(1, dataset.num_classes))
            elif loss_functions[loss_fn] == 'l1_loss':
                pred = model(data, data.edge_index)
                metric = F.l1_loss(pred[data.test_mask], data[data.test_mask].unsqueeze(-1).repeat(1, dataset.num_classes))
            else:
                raise ValueError('Invalid performance metric')
                
            # Select the best loss function based on the performance metric
            if 'best_loss_fn' not in locals() or metric < best_metric:
                best_loss_fn = loss_fn
                best_metric = metric
                best_optimizer = optimizer
                
        # Use the best loss function to compute the loss
        loss = losses[best_loss_fn]
        loss.backward()
        optimizer.step()

        # Adjust the learning rate using the scheduler
        scheduler.step(best_metric)
        train_losses.append(loss.item())

    # Print the training accuracy for the last epoch
    _, pred = model(data.x, data.edge_index).max(dim=1)
    correct = float(pred[data.train_mask].eq(data.y[data.train_mask]).sum().item())
    acc = correct / data.train_mask.sum()
    print(f'Training accuracy for epoch {epoch + 1}: {acc:.4f}')

    # Plot the training loss over epochs for each optimizer
    plt.plot(train_losses, label=type(optimizer).__name__)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Evaluate the model using the best optimizer
model.eval()
_, pred = model(data.x, data.edge_index).max(dim=1)
correct = float(pred[data.train_mask].eq(data.y[data.train_mask]).item())
acc = correct / data.train_mask.sum().item()
print('Epoch: {}, Optimizer: {}, Learning rate: {}'.format(epoch, type(optimizer).__name__, optimizer.param_groups[0]['lr']))
print('Best loss function: {}, Best metric: {:.4f}, Accuracy: {:.4f}'.format(best_loss_fn, best_metric, acc))