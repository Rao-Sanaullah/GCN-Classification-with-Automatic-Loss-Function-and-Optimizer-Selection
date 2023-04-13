# Streamlined Training of GCN for Node Classification with Automatic Loss Function and Optimizer Selection


- GNNs process and learn from graph-structured data, but optimal performance requires automatic selection of the best loss and optimization functions. In this research, we trained GCN and GAT models for node classification and used performance metrics to select optimal functions. We achieved state-of-the-art results and developed a real-time visualization tool for in-depth understanding of the model's behavior during training. Our study provides insights into the application of GNNs to graph-structured data.


# Automatic Search and Selection of Best Functions

 - Our proposed approach improves GNN performance on graph-structured data via an automated function selection process based on a comprehensive evaluation framework. This approach avoids manual function selection and requires no domain-specific expertise. The automated selection process is efficient and effective in selecting optimal functions, and multiple performance metrics aid in optimizing overall model performance.


# Real-time Training Visualization
 
  - Our visualization tool allows real-time observation of the GCN model's behavior using different hyperparameters. Users can determine the best combination for their specific use case and gain insights into how the model responds to changes. The tool provides a means of experimenting with configurations to optimize performance and identify potential issues.

![Real-Time Training Visualization](https://github.com/Rao-Sanaullah/GNN-Classification-with-Automatic-Loss-Function-and-Optimizer-Selection/blob/main/runtime.png)

# Dataset

 - Our proposed architecture for citation networks was evaluated on carefully selected datasets, including Cora, CiteSeer, and PubMed. These datasets consist of publications classified into different classes, represented by binary or sparse vectors, with edges representing citation relationships.

# Findings and Analysis

 - Our proposed architecture achieved promising results by automating the selection of loss and optimization functions, improving the performance of GCN and GAT models. The GCN model outperformed the GAT model in accuracy for all three benchmark datasets, making it more suitable for citation network node classification tasks. Our approach was effective in improving GAN models' performance, as shown in detailed results for each test case in the figures.

![Findings and Analysis](https://github.com/Rao-Sanaullah/GNN-Classification-with-Automatic-Loss-Function-and-Optimizer-Selection/blob/main/results.jpg)


# Requirement:
 - torch
 - torch.nn.functional
 - torch_geometric.datasets 
 - Planetoid
 - torch_geometric.nn 
 - GCNConv
 - GATConv
 - matplotlib

For any help, please contact

Sanaullah (sanaullah@fh-bielefeld.de)
