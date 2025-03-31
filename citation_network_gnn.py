import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

# Set random seed for reproducibility
torch.manual_seed(42)

# Load the Cora dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# Print information about the dataset
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of nodes: {data.x.shape[0]}')
print(f'Number of features: {data.x.shape[1]}')
print(f'Number of classes: {dataset.num_classes}')
print(f'Number of edges: {data.edge_index.shape[1]}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')

# Define the Graph Convolutional Network model
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
        
    def forward(self, x, edge_index):
        # First Graph Convolution Layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Second Graph Convolution Layer
        x = self.conv2(x, edge_index)
        
        return x
    
    def get_embeddings(self, x, edge_index):
        # Get the embeddings from the first layer
        return self.conv1(x, edge_index).detach()

# Initialize the model
model = GCN(num_features=dataset.num_features, 
            hidden_channels=64, 
            num_classes=dataset.num_classes)

# Define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Training function
def train():
    model.train()
    optimizer.zero_grad()
    # Forward pass
    out = model(data.x, data.edge_index)
    # Calculate loss only on the training nodes
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    # Backward pass
    loss.backward()
    optimizer.step()
    return loss

# Testing function
def test():
    model.eval()
    out = model(data.x, data.edge_index)
    # Apply softmax to get probabilities
    pred = out.argmax(dim=1)
    
    # Calculate accuracy for train, validation and test sets
    train_correct = pred[data.train_mask] == data.y[data.train_mask]
    train_acc = int(train_correct.sum()) / int(data.train_mask.sum())
    
    val_correct = pred[data.val_mask] == data.y[data.val_mask]
    val_acc = int(val_correct.sum()) / int(data.val_mask.sum())
    
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    
    return train_acc, val_acc, test_acc

# Function to visualize the embeddings
def visualize_embeddings(embeddings, labels):
    # Use t-SNE to reduce the dimensionality to 2
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='tab10', alpha=0.8)
    plt.colorbar(scatter)
    plt.title('t-SNE Visualization of Node Embeddings')
    plt.savefig('embeddings_visualization.png')
    plt.close()

# Training loop
train_accuracies = []
val_accuracies = []
test_accuracies = []
losses = []

print("\nTraining the model...")
for epoch in range(1, 201):
    loss = train()
    losses.append(loss.item())
    
    # Evaluate the model
    train_acc, val_acc, test_acc = test()
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    test_accuracies.append(test_acc)
    
    # Print progress every 10 epochs
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

# Get final test accuracy
_, _, final_test_acc = test()
print(f'\nFinal Test Accuracy: {final_test_acc:.4f}')

# Plot training curves
plt.figure(figsize=(12, 4))

# Loss curve
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

# Accuracy curves
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train')
plt.plot(val_accuracies, label='Validation')
plt.plot(test_accuracies, label='Test')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_curves.png')
plt.close()

# Visualize node embeddings
print("\nGenerating embeddings visualization...")
model.eval()
with torch.no_grad():
    embeddings = model.get_embeddings(data.x, data.edge_index)
    embeddings = embeddings.numpy()
    labels = data.y.numpy()
    visualize_embeddings(embeddings, labels)

print("\nAnalysis complete. Check the output directory for visualization files.")

# Analyze influence of papers in the citation network
print("\nAnalyzing paper influence...")
edge_index = data.edge_index.numpy()
# Count how many times each paper is cited
paper_citations = np.zeros(data.num_nodes)
for i in range(edge_index.shape[1]):
    cited_paper = edge_index[1, i]
    paper_citations[cited_paper] += 1

# Find top 10 most cited papers
top_cited_indices = np.argsort(paper_citations)[-10:]
top_cited_counts = paper_citations[top_cited_indices]

print("\nTop 10 most influential papers (by citation count):")
for i, (idx, count) in enumerate(zip(top_cited_indices[::-1], top_cited_counts[::-1])):
    paper_class = dataset.data.y[idx].item()
    class_name = dataset.num_classes  # Assume we have class names, replace with actual names if available
    print(f"{i+1}. Paper ID: {idx}, Citations: {int(count)}, Class: {paper_class}")

# Save citation statistics
plt.figure(figsize=(10, 6))
plt.hist(paper_citations, bins=30)
plt.xlabel('Number of Citations')
plt.ylabel('Number of Papers')
plt.title('Distribution of Citations in the Network')
plt.savefig('citation_distribution.png')
plt.close()

print("\nCitation network analysis complete!")
