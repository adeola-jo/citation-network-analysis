import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GraphConv, GINConv
from torch_geometric.datasets import Planetoid
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Set random seed for reproducibility
torch.manual_seed(42)

# Load the Cora dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# GCN Model
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# GAT Model (Graph Attention Network)
class GAT(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels, heads=8, dropout=0.6)
        # For the output layer, we average the attention heads
        self.conv2 = GATConv(hidden_channels * 8, num_classes, heads=1, concat=False, dropout=0.6)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# GraphSAGE Model
class GraphSAGE(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(num_features, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, num_classes)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# GraphConv Model (a more general graph convolution)
class GraphConvNet(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GraphConvNet, self).__init__()
        self.conv1 = GraphConv(num_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, num_classes)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# GIN Model (Graph Isomorphism Network)
class GIN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GIN, self).__init__()
        # MLP for GIN
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(num_features, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        )
        nn2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, num_classes)
        )
        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn2)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# Function to train a model
def train_model(model, optimizer, data, epochs=200):
    train_acc_history = []
    val_acc_history = []
    test_acc_history = []
    loss_history = []
    
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            
            # Training accuracy
            train_correct = pred[data.train_mask] == data.y[data.train_mask]
            train_acc = train_correct.sum().item() / data.train_mask.sum().item()
            
            # Validation accuracy
            val_correct = pred[data.val_mask] == data.y[data.val_mask]
            val_acc = val_correct.sum().item() / data.val_mask.sum().item()
            
            # Test accuracy
            test_correct = pred[data.test_mask] == data.y[data.test_mask]
            test_acc = test_correct.sum().item() / data.test_mask.sum().item()
        
        # Record metrics
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        test_acc_history.append(test_acc)
        loss_history.append(loss.item())
        
        # Print progress
        if epoch % 20 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
    
    # Final evaluation with detailed metrics
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        
        # Calculate detailed metrics on test set
        y_true = data.y[data.test_mask].cpu().numpy()
        y_pred = pred[data.test_mask].cpu().numpy()
        
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        
        print(f'\nFinal Test Metrics:')
        print(f'  Accuracy: {accuracy:.4f}')
        print(f'  Precision: {precision:.4f}')
        print(f'  Recall: {recall:.4f}')
        print(f'  F1 Score: {f1:.4f}')
    
    return {
        'train_acc': train_acc_history,
        'val_acc': val_acc_history,
        'test_acc': test_acc_history,
        'loss': loss_history,
        'final_metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    }

# Define models to compare
models = {
    'GCN': GCN(num_features=dataset.num_features, hidden_channels=64, num_classes=dataset.num_classes),
    'GAT': GAT(num_features=dataset.num_features, hidden_channels=8, num_classes=dataset.num_classes),
    'GraphSAGE': GraphSAGE(num_features=dataset.num_features, hidden_channels=64, num_classes=dataset.num_classes),
    'GraphConv': GraphConvNet(num_features=dataset.num_features, hidden_channels=64, num_classes=dataset.num_classes),
    'GIN': GIN(num_features=dataset.num_features, hidden_channels=64, num_classes=dataset.num_classes)
}

# Train each model
results = {}
for name, model in models.items():
    print(f"\n{'-'*50}")
    print(f"Training {name}...")
    print(f"{'-'*50}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    results[name] = train_model(model, optimizer, data)

# Plot training curves for each model
plt.figure(figsize=(15, 10))

# Plot loss
plt.subplot(2, 2, 1)
for name, result in results.items():
    plt.plot(result['loss'], label=name)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

# Plot training accuracy
plt.subplot(2, 2, 2)
for name, result in results.items():
    plt.plot(result['train_acc'], label=name)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()

# Plot validation accuracy
plt.subplot(2, 2, 3)
for name, result in results.items():
    plt.plot(result['val_acc'], label=name)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()

# Plot test accuracy
plt.subplot(2, 2, 4)
for name, result in results.items():
    plt.plot(result['test_acc'], label=name)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('model_comparison_curves.png')
plt.close()

# Bar chart of final test metrics
plt.figure(figsize=(12, 8))

metrics = ['accuracy', 'precision', 'recall', 'f1']
x = np.arange(len(metrics))
width = 0.15
offsets = [-2, -1, 0, 1, 2]

for i, (name, result) in enumerate(results.items()):
    values = [result['final_metrics'][metric] for metric in metrics]
    plt.bar(x + offsets[i] * width, values, width, label=name)

plt.xlabel('Metric')
plt.ylabel('Score')
plt.title('Model Performance Comparison')
plt.xticks(x, metrics)
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig('model_comparison_metrics.png')
plt.close()

# Print final rankings
print("\nModel Rankings by Test Accuracy:")
sorted_models = sorted(results.items(), key=lambda x: x[1]['final_metrics']['accuracy'], reverse=True)
for i, (name, result) in enumerate(sorted_models):
    print(f"{i+1}. {name}: {result['final_metrics']['accuracy']:.4f}")

print("\nComparison complete! Check the output directory for visualization files.")

# Save results to file
import json

# Convert numpy values to Python native types for JSON serialization
def convert_to_serializable(obj):
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj

serializable_results = {}
for model_name, model_results in results.items():
    serializable_results[model_name] = {
        'train_acc': [convert_to_serializable(x) for x in model_results['train_acc']],
        'val_acc': [convert_to_serializable(x) for x in model_results['val_acc']],
        'test_acc': [convert_to_serializable(x) for x in model_results['test_acc']],
        'loss': [convert_to_serializable(x) for x in model_results['loss']],
        'final_metrics': {k: convert_to_serializable(v) for k, v in model_results['final_metrics'].items()}
    }

with open('model_comparison_results.json', 'w') as f:
    json.dump(serializable_results, f, indent=2)

print("Results saved to 'model_comparison_results.json'")
