SGCL is described in ["Smoothed Graph Contrastive Learning via Seamless Proximity Integration"](https://openreview.net/pdf?id=dj7s8Y7LeC), by

[Maysam Behmanesh*](https://maysambehmanesh.github.io/),
[Maks Ovsjanikov](https://www.lix.polytechnique.fr/~maks/)


Abstract

Graph contrastive learning (GCL) aligns node representations by classifying node pairs into positives and negatives using a selection process that typically relies on establishing correspondences within two augmented graphs. The conventional GCL approaches incorporate negative samples uniformly in the contrastive loss, resulting in the equal treatment of negative nodes, regardless of their proximity to the true positive. In this paper, we present a Smoothed Graph Contrastive Learning model (SGCL), which leverages the geometric structure of augmented graphs to inject proximity information associated with positive/negative pairs in the contrastive loss, thus significantly regularizing the learning process. The proposed SGCL adjusts the penalties associated with node pairs in contrastive loss by incorporating three distinct smoothing techniques that result in proximity-aware positives and negatives. To enhance scalability for large-scale graphs, the proposed framework incorporates a graph batch-generating strategy that partitions the given graphs into multiple subgraphs, facilitating efficient training in separate batches. Through extensive experimentation in the unsupervised setting on various benchmarks, particularly those of large scale, we demonstrate the superiority of our proposed framework against recent baselines.

![image](https://github.com/user-attachments/assets/7c6f59c4-7861-4ce1-8dc4-9942978bd5f7)


The codebase will be uploaded soon...






