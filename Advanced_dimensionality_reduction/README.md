
# Dimensionality Reduction Techniques: Colab Demonstrations

## Overview

This repository demonstrates various dimensionality reduction techniques, showcasing their applications on diverse datasets (image, tabular, and medical). Each technique is implemented in Google Colab and explained in detail through video walkthroughs. The project is organized into three parts:

- **Part A**: Dimensionality reduction on image datasets (e.g., faces, digits).
- **Part B**: Dimensionality reduction on tabular datasets (e.g., Iris dataset).
- **Part C**: Dimensionality reduction on medical datasets sourced from papers.

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

## Project Files

### 1. Colab Notebooks
- **Part A**: Dimensionality reduction on image datasets
- **Part B**: Dimensionality reduction on tabular datasets
- **Part C**: Dimensionality reduction on medical datasets

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

- **Image datasets**: Faces and digits datasets
- **Tabular datasets**: Iris dataset
- **Medical datasets**: Selected from reputable papers, ensuring variety and relevance

## Results Commentary

- **Image Datasets**:
  - Techniques like t-SNE and UMAP provided excellent clustering and visualization.
  - Autoencoders captured nonlinear patterns effectively but required more computational effort.
  
- **Tabular Datasets**:
  - Factor Analysis and Kernel PCA worked well for feature decorrelation.
  - Incremental PCA and Randomized PCA performed efficiently on larger datasets.

- **Medical Datasets**:
  - UMAP and t-SNE excelled in revealing hidden patterns in high-dimensional data.
  - Autoencoders demonstrated their ability to compress and reconstruct data with minimal loss.

## Visualizations

- Interactive tools such as Plotly and the UMAP exploration tool (https://pair-code.github.io/understanding-umap/) were integrated for enhanced understanding.
- Animated visualizations help in interpreting t-SNE and UMAP transformations over iterations.

## Additional Resources

- [Hands-On Machine Learning (Chapter 8)](https://github.com/ageron/handson-ml2/blob/master/08_dimensionality_reduction.ipynb)
- [Analytics Vidhya Blog on Dimensionality Reduction](https://www.analyticsvidhya.com/blog/2018/08/dimensionality-reduction-techniques-python/)
- [Understanding UMAP](https://pair-code.github.io/understanding-umap/)
- [PyCaret for Automated ML](https://www.kaggle.com/code/sureshmecad/pycaret-automl-beginers/notebook)

## Submission

Please submit the GitHub repository link for evaluation.