# Citation Network Analysis with Graph Neural Networks

This project demonstrates how to use Graph Neural Networks (GNNs) to analyze citation networks. Specifically, it uses a Graph Convolutional Network (GCN) to predict the category of papers in the Cora citation dataset.

## Project Overview

The Cora dataset consists of scientific publications as nodes in a graph, where edges represent citation links. Each publication is described by a binary word vector and belongs to one of seven classes.

This project:
1. Implements a Graph Convolutional Network (GCN) for node classification
2. Trains the model to predict paper categories
3. Visualizes the learned embeddings using t-SNE
4. Analyzes influential papers in the citation network

## Requirements

You need the following libraries:
- PyTorch
- PyTorch Geometric
- NumPy
- Matplotlib
- scikit-learn

You can install them using:
```
pip install -r requirements.txt
```

Note: Installing PyTorch Geometric might require additional steps depending on your system. Please refer to the [official installation guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

## How to Run

You can use the main script to run different parts of the analysis:

```
# Run the basic GNN analysis
python main.py --basic

# Run the comparison of different GNN architectures
python main.py --compare

# Run the network analysis
python main.py --network

# Run the link prediction analysis
python main.py --link

# Run all analyses
python main.py --all
```

Or run individual scripts directly:
```
python citation_network_gnn.py  # Basic GNN model
python compare_gnn_architectures.py  # Compare different architectures
python network_analysis.py  # Analyze network properties
python link_prediction.py  # Predict missing links
```

## Output

Depending on which analyses you run, you'll get various outputs:

### Basic GNN Analysis (`citation_network_gnn.py`)
- Training progress and final test accuracy
- `training_curves.png`: Loss and accuracy during training
- `embeddings_visualization.png`: t-SNE visualization of the node embeddings
- `citation_distribution.png`: Distribution of citations in the network
- Information about the most influential papers based on citation count

### Architecture Comparison (`compare_gnn_architectures.py`)
- Training metrics for different GNN architectures (GCN, GAT, GraphSAGE, etc.)
- `model_comparison_curves.png`: Training curves for all architectures
- `model_comparison_metrics.png`: Bar chart comparing final metrics for each architecture
- `model_comparison_results.json`: Detailed results for further analysis
- Performance rankings of different architectures

### Network Analysis (`network_analysis.py`)
- Basic network statistics (nodes, edges, etc.)
- Analysis of centrality measures (degree, betweenness, PageRank)
- Community detection results
- `community_structure.png`: Visualization of communities in the network
- `degree_distribution.png`: Distribution of node degrees
- `power_law_fit.png`: Analysis of scale-free properties

### Link Prediction (`link_prediction.py`)
- Performance metrics for link prediction (ROC-AUC, PR-AUC)
- `link_prediction_curves.png`: Training and evaluation curves
- `link_prediction_roc.png`: ROC curve for link prediction
- `link_prediction_pr.png`: Precision-Recall curve for link prediction
- `top_predicted_missing_links.csv`: List of most probable missing citations
- Analysis of the top predicted missing links

## Extending the Project

You can extend this project in several ways:
1. Try different GNN architectures (GAT, GraphSAGE, etc.)
2. Apply the model to other citation datasets (CiteSeer, PubMed)
3. Implement link prediction to predict missing citations
4. Add more advanced network analysis metrics
5. Visualize the citation network structure

## References

- Kipf & Welling, [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) (ICLR 2017)
- The Cora dataset: [https://linqs.soe.ucsc.edu/data](https://linqs.soe.ucsc.edu/data)
