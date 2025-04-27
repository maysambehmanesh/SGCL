SGCL is described in ["Smoothed Graph Contrastive Learning via Seamless Proximity Integration"](https://openreview.net/pdf?id=dj7s8Y7LeC), by

[Maysam Behmanesh*](https://maysambehmanesh.github.io/),
[Maks Ovsjanikov](https://www.lix.polytechnique.fr/~maks/)


Abstract

Graph contrastive learning (GCL) aligns node representations by classifying node pairs into positives and negatives using a selection process that typically relies on establishing correspondences within two augmented graphs. The conventional GCL approaches incorporate negative samples uniformly in the contrastive loss, resulting in the equal treatment of negative nodes, regardless of their proximity to the true positive. In this paper, we present a Smoothed Graph Contrastive Learning model (SGCL), which leverages the geometric structure of augmented graphs to inject proximity information associated with positive/negative pairs in the contrastive loss, thus significantly regularizing the learning process. The proposed SGCL adjusts the penalties associated with node pairs in contrastive loss by incorporating three distinct smoothing techniques that result in proximity-aware positives and negatives. To enhance scalability for large-scale graphs, the proposed framework incorporates a graph batch-generating strategy that partitions the given graphs into multiple subgraphs, facilitating efficient training in separate batches. Through extensive experimentation in the unsupervised setting on various benchmarks, particularly those of large scale, we demonstrate the superiority of our proposed framework against recent baselines.

![image](https://github.com/user-attachments/assets/7c6f59c4-7861-4ce1-8dc4-9942978bd5f7)


## Requirements
- Python 3.8+
- PyTorch = 2.1.2
- PyTorch-Geometric = 2.5
- PyGCL



You can install PyGCL directly from PyPI using pip:

```bash
pip install PyGCL
```

## Citation


```
@InProceedings{pmlr-v269-behmanesh24a,
  title = 	 {Smoothed Graph Contrastive Learning via Seamless Proximity Integration},
  author =       {Behmanesh, Maysam and Ovsjanikov, Maks},
  booktitle = 	 {Proceedings of the Third Learning on Graphs Conference},
  pages = 	 {to appear},
  year = 	 {2024},
  volume = 	 {269},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {26--29 Nov},
  publisher =    {PMLR},

}
```

## Reference
Zhu et al. [An Empirical Study of Graph Contrastive Learning](https://arxiv.org/abs/2109.01116). 






