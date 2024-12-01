
# Dimensionality Reduction Techniques
## Overview

This repository demonstrates various dimensionality reduction techniques, showcasing their applications on diverse datasets (image, tabular, and medical). Each technique is implemented in Google Colab and explained in detail through video walkthroughs. The project is organized into three parts:

- **Part A**: Dimensionality reduction on image datasets - https://colab.research.google.com/drive/1mfyY0g1K4z7vo0doq1NOGvULL48toGFG?usp=sharing
 
- **Part B**: Dimensionality reduction on tabular datasets - https://colab.research.google.com/drive/1uPZsGgxo8IqYzXgi8s0xMGzOEdqe4suN?usp=sharing
 
- **Part C**: Dimensionality reduction using Databricks - https://colab.research.google.com/drive/1FJ60YM5J81QWptN7tz9B3rKRHDmYwTao?usp=sharing

## Techniques Demonstrated

The following dimensionality reduction techniques are included:

1. **LLE** (Locally Linear Embedding)
2. **t-SNE** (t-Distributed Stochastic Neighbor Embedding) with interactive visualizations
3. **ISOMAP**
4. **UMAP** (Uniform Manifold Approximation and Projection) with interactive visualizations
5. **MDS** (Multidimensional Scaling)
6. **Randomized PCA**
7. **Kernel PCA**
8. **Incremental PCA**
9. **Factor Analysis** (Scikit-learn)
10. **Autoencoders**


Each notebook includes:
- Data preprocessing
- Implementation of various techniques
- Comparative analysis of results
- Visualizations using tools like Matplotlib, Seaborn, and Plotly for interactivity

### 2. Video Walkthrough
A YouTube video walks through the Colab code and explains the results. [Watch the video here](#).

### 3. Results Commentary
The results commentary is included in both the Colab notebooks and this README. It compares the performance and visualization capabilities of different techniques.

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/dimensionality-reduction.git
   ```
2. Open the respective Colab notebooks using the links provided in the repository.
3. Install required dependencies:
   ```bash
   !pip install umap-learn sklearn matplotlib seaborn plotly pycaret tensorflow
   ```

## Datasets

- **Image datasets**: cifar-10
- **Tabular datasets**: Iris dataset
- **Medical datasets**: Selected from reputable papers, ensuring variety and relevance


## Visualizations

- Interactive tools such as Plotly and the UMAP exploration tool (https://pair-code.github.io/understanding-umap/) were integrated for enhanced understanding.
- Animated visualizations help in interpreting t-SNE and UMAP transformations over iterations.


