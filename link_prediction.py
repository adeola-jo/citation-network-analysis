import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling, train_test_split_edges
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

# Set random seed for reproducibility
torch.manual_seed(42)

# Load the Cora dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# Prepare the data for link prediction
data = train_test_split_edges(data)

# Define the GCN model for link prediction
class LinkPredictionGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(LinkPredictionGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        
    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
    def decode(self, z, edge_index):
        # Inner product between node embeddings to predict links
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
    
    def decode_all(self, z):
        # Compute embeddings similarity matrix for all possible node pairs
        prob_adj = z @ z.t()
        return prob_adj

# Function to calculate positive edges prediction
def get_link_labels(pos_edge_index, neg_edge_index):
    # Returns a tensor of ones for positive edges and zeros for negative edges
    E = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(E, dtype=torch.float)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

# Train the model
def train_link_prediction(model, optimizer, data):
    model.train()
    
    # Reset gradients
    optimizer.zero_grad()
    
    # Compute embeddings
    z = model.encode(data.x, data.train_pos_edge_index)
    
    # Sample negative edges
    neg_edge_index = negative_sampling(
        edge_index=data.train_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.train_pos_edge_index.size(1)
    )
    
    # Predict links
    link_logits = model.decode(z, torch.cat([data.train_pos_edge_index, neg_edge_index], dim=-1))
    
    # Create link labels (1 for positive, 0 for negative)
    link_labels = get_link_labels(data.train_pos_edge_index, neg_edge_index)
    
    # Calculate loss
    loss = F.binary_cross_entropy_with_logits(link_logits, link_labels)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss

# Test the model
@torch.no_grad()
def test_link_prediction(model, data):
    model.eval()
    
    # Get embeddings
    z = model.encode(data.x, data.train_pos_edge_index)
    
    # Test on positive test edges
    pos_edge_logits = model.decode(z, data.test_pos_edge_index)
    
    # Sample negative test edges
    neg_edge_index = negative_sampling(
        edge_index=data.test_pos_edge_index,
        num_nodes=data.num_nodes,
        num_neg_samples=data.test_pos_edge_index.size(1)
    )
    
    # Test on negative test edges
    neg_edge_logits = model.decode(z, neg_edge_index)
    
    # Create link labels (1 for positive, 0 for negative)
    link_probs = torch.cat([torch.sigmoid(pos_edge_logits), torch.sigmoid(neg_edge_logits)]).cpu().numpy()
    link_labels = get_link_labels(data.test_pos_edge_index, neg_edge_index).cpu().numpy()
    
    # Calculate ROC AUC score
    roc_auc = roc_auc_score(link_labels, link_probs)
    
    # Calculate Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(link_labels, link_probs)
    pr_auc = auc(recall, precision)
    
    return roc_auc, pr_auc, link_probs, link_labels

# Initialize model and optimizer
model = LinkPredictionGCN(in_channels=data.num_features, hidden_channels=128, out_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
print("Training link prediction model...")
losses = []
roc_aucs = []
pr_aucs = []

for epoch in range(1, 101):
    loss = train_link_prediction(model, optimizer, data)
    losses.append(loss.item())
    
    if epoch % 10 == 0:
        roc_auc, pr_auc, _, _ = test_link_prediction(model, data)
        roc_aucs.append(roc_auc)
        pr_aucs.append(pr_auc)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}')

# Final evaluation
roc_auc, pr_auc, link_probs, link_labels = test_link_prediction(model, data)
print(f'\nFinal Test Results:')
print(f'  ROC-AUC: {roc_auc:.4f}')
print(f'  PR-AUC: {pr_auc:.4f}')

# Plot the training curve
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

plt.subplot(1, 2, 2)
epochs = list(range(10, 101, 10))
plt.plot(epochs, roc_aucs, label='ROC-AUC')
plt.plot(epochs, pr_aucs, label='PR-AUC')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Link Prediction Performance')
plt.legend()

plt.tight_layout()
plt.savefig('link_prediction_curves.png')
plt.close()

# Visualize the ROC curve
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(link_labels, link_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Link Prediction')
plt.legend()
plt.savefig('link_prediction_roc.png')
plt.close()

# Visualize the Precision-Recall curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve for Link Prediction')
plt.legend()
plt.savefig('link_prediction_pr.png')
plt.close()

print("\nLink prediction analysis complete! Check the output directory for visualization files.")

# Analyze top predicted missing links
model.eval()
z = model.encode(data.x, data.train_pos_edge_index)

# Get all possible pairs of nodes
num_nodes = data.num_nodes
rows, cols = [], []
for i in range(num_nodes):
    for j in range(i + 1, num_nodes):  # Only consider upper triangular part to avoid duplicates
        rows.append(i)
        cols.append(j)

# Convert to tensor
edge_index = torch.tensor([rows, cols], dtype=torch.long)

# Remove existing edges
train_edges_set = set([(i.item(), j.item()) for i, j in zip(data.train_pos_edge_index[0], data.train_pos_edge_index[1])])
test_edges_set = set([(i.item(), j.item()) for i, j in zip(data.test_pos_edge_index[0], data.test_pos_edge_index[1])])
val_edges_set = set([(i.item(), j.item()) for i, j in zip(data.val_pos_edge_index[0], data.val_pos_edge_index[1])])
existing_edges = train_edges_set.union(test_edges_set).union(val_edges_set)

# Filter edge_index to only include potential missing links
filtered_rows, filtered_cols = [], []
for i, (row, col) in enumerate(zip(rows, cols)):
    if (row, col) not in existing_edges and (col, row) not in existing_edges:
        filtered_rows.append(row)
        filtered_cols.append(col)

filtered_edge_index = torch.tensor([filtered_rows, filtered_cols], dtype=torch.long)

# Predict link probabilities for potential missing links
with torch.no_grad():
    link_logits = model.decode(z, filtered_edge_index)
    link_probs = torch.sigmoid(link_logits)

# Get the top N most probable missing links
N = 20
top_indices = torch.argsort(link_probs, descending=True)[:N]
top_probs = link_probs[top_indices].cpu().numpy()
top_edges = [(filtered_edge_index[0][i].item(), filtered_edge_index[1][i].item()) for i in top_indices]

print("\nTop 20 Predicted Missing Links:")
for i, ((node1, node2), prob) in enumerate(zip(top_edges, top_probs)):
    print(f"{i+1}. Node {node1} -- Node {node2}: Probability = {prob:.4f}")

# Get node labels for better interpretation
if hasattr(data, 'y'):
    node_labels = data.y.cpu().numpy()
    print("\nNode Classes of Predicted Links:")
    for i, (node1, node2) in enumerate(top_edges[:10]):  # Show only top 10 for brevity
        print(f"{i+1}. Node {node1} (Class {node_labels[node1]}) -- Node {node2} (Class {node_labels[node2]})")

# Export the top predicted missing links to a CSV file
import csv
with open('top_predicted_missing_links.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Rank', 'Node1', 'Node2', 'Probability'])
    for i, ((node1, node2), prob) in enumerate(zip(top_edges, top_probs)):
        writer.writerow([i+1, node1, node2, prob])

print("\nTop predicted missing links exported to 'top_predicted_missing_links.csv'")
