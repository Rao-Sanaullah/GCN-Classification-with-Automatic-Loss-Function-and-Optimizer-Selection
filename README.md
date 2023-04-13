# Streamlined Training of GCN for Node Classification with Automatic Loss Function and Optimizer Selection


- Graph Neural Networks (GNNs) are specialized types of neural networks that process and learn from graph-structured data. For GNNs to perform optimally, it is crucial to automatically select the best loss and optimization functions. Doing so can save time and eliminate the need for domain-specific knowledge. Automatic selection of these functions is essential for GNNs to achieve state-of-the-art results. In this research, we trained Graph Convolutional Networks (GCNs) and Graph Attention Networks (GAT) models to classify nodes on three benchmark datasets. To select the best loss and optimization functions, we employed performance metrics and implemented a learning rate scheduler to enhance the model's performance. We assessed the model's performance using multiple metrics and reported the best loss function and performance metric. Our approach delivered state-of-the-art results and showcased the significance of selecting the appropriate loss and optimizer functions. Additionally, we developed a real-time visualization of the GCN model during training that provided users with an in-depth comprehension of the model's behavior. This study provides a comprehensive understanding of GNNs and their application to graph-structured data, focusing on real-time visualization of GNN behavior during training.


# Automatic Search and Selection of Best Functions

 - The proposed approach aims to enhance the performance of GNNs when processing data in graph structures. Instead of requiring domain-specific expertise or manual selection of functions, the architecture utilizes an automated selection process for loss and optimization functions. The selection process is based on an evaluation framework that considers multiple performance metrics for both loss and optimization functions. This comprehensive framework enables the selection of optimal functions, resulting in improved model performance on graph-structured data. By automating the selection process, the method provides a more efficient and effective way of selecting appropriate loss and optimization functions for GNNs. Additionally, the use of multiple performance metrics helps gain a better understanding of the model's behavior and performance, which can inform decisions about optimization and lead to better overall performance.


# Real-time Training Visualization
 
  - Our visualization tool provides users with the capability to observe the GCN model's behavior in real-time while using various combinations of optimizers, loss functions, and learning rates. This feature enables users to evaluate the impact of each hyperparameter on the model's performance and quickly determine the best combination for their specific use case. The real-time monitoring of the model's performance with different hyperparameters provides users with valuable insights into its behavior and how it responds to changes in data or hyperparameters. Our visualization tool, as illustrated in Figure, offers a dynamic and interactive approach to comprehend the model's behavior during training. Users can explore the model's performance by adjusting hyperparameters in real-time and observe how the model reacts to these changes. This allows users to experiment with various configurations and gain a deeper understanding of how the model operates and what is required for optimal performance. Hence, this real-time visualization is an effective tool for understanding the GCN model's behavior, identifying potential issues, and fine-tuning the model accordingly.

![Real-Time Training Visualization](https://github.com/Rao-Sanaullah/GNN-Classification-with-Automatic-Loss-Function-and-Optimizer-Selection/blob/main/runtime.png)

# Dataset

 - We used carefully selected datasets to evaluate our proposed architecture for citation networks. The Cora dataset has 2,708 publications, CiteSeer has 3,327 publications, and PubMed has 19,717 publications, each classified into different classes. The nodes are represented by binary or sparse vectors that capture the presence or absence of certain words or medical terms, and the edges represent citation relationships.

# Findings and Analysis

 - Our evaluation of the proposed architecture shows promising results. We used automated selection of loss and optimization functions to improve GCN and GAT models' performance compared to manual selection. The GCN model outperformed the GAT model in accuracy for all three benchmark datasets, suggesting it's more suitable for node classification tasks in citation networks. Our automated approach for selecting the best loss function was effective in improving GAN models' performance for node classification tasks in citation networks, as shown in Figures the detailed results of each test case, provide insights into the proposed GNN model's performance on various datasets and experimental conditions.

![Findings and Analysis](https://github.com/Rao-Sanaullah/GNN-Classification-with-Automatic-Loss-Function-and-Optimizer-Selection/blob/main/results.jpg)

