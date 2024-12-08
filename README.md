# ADM Homework 4 - Movie Recommendation System

This repository contains the solution submitted by **Group 9** for **Homework 4 - Movie Recommendation System** of the course **Algorithmic Methods of Data Mining** (M.Sc. in Data Science) at Sapienza University of Rome for the academic year **2024–2025**. 

The homework description and requirements can be found at the following link: [Homework 4 on GitHub](https://github.com/Sapienza-University-Rome/ADM/tree/master/2024/Homework_4).

The solution is provided in the Jupyter Notebook `main.ipynb`. You can view the notebook directly using **nbviewer** at the following link: [View main.ipynb on nbviewer](https://nbviewer.org/github/FabrizioFerrara03/ADM-HW4/blob/main/main.ipynb).

## Group Members:
- **Asia Montico** (1966494)
- **Fabrizio Ferrara** (2207087)
- **Marta Lombardi** (2156537)
- **Matteo Sorrentini** (2023085)

## Contents

This repository contains the following files:

- **`main.ipynb`**: The complete solution to the homework, including all analyses, implementations, and explanations for each question and sub-question.
- **`functions.py`**: A Python module containing all the functions used in the solution, modularized for clarity and reusability.
- **`kmeans_animation.gif`**: A GIF showing the step-by-step evolution of the clustering process during k-means, highlighting how point memberships change over iterations.

## Summary of Work Done

### 1. Recommendation System with LSH
**1.1 Data Preparation**

The files `rating.csv` and `movie.csv` were loaded. rating.csv contains users’ movie ratings, while movie.csv includes movie titles and genres. A quick analysis was performed to understand the data structure and ensure there were no missing or duplicate values.

**1.2 MinHash Signatures Calculation**

For each user, MinHash signatures were created based on the movies they rated. Three different hash functions were used to generate unique signatures and compare users. Various numbers of hash functions were tested to find the most accurate configuration.

**1.3 Locality-Sensitive Hashing (LSH)**

Users’ MinHash signatures were split into bands and placed into buckets. Users in the same bucket were considered similar. For a specific user, the two most similar users were found. Based on their preferences, up to five movies the user hadn’t seen yet were recommended.

### 2. Grouping Movies Together!
**2.1 Feature Engineering**

The attributes required for clustering were selected, processed, and organized into a single DataFrame. This included transforming categorical variables (e.g., genres) into dummy variables and handling missing values in numerical features (e.g., ratings). These steps ensured that the data was properly structured for clustering.

**2.2 Choose your features (variables)!**

The relevant variables for clustering were chosen from the prepared dataset. The numerical features were normalized to ensure they were on a comparable scale. A Principal Component Analysis (PCA) was then performed to reduce the dimensionality of the data, retaining the two most important principal components for clustering.

**2.3 Clustering**

Clustering was performed using two algorithms: K-Means and K-Means++. Both were implemented to partition the data into clusters based on the principal components. The difference in results between random initialization (K-Means) and optimized initialization (K-Means++) was analyzed.

**2.4 Best Algorithm**

The clustering results were evaluated using three metrics: the Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index. These metrics provided insights into the quality of the clusters, including how well-separated and compact they were. A comparison of the clustering algorithms was conducted based on these metrics to determine the most effective approach for the dataset.

### 3. Bonus Question

Given the two components resulting from the PCA, a Varimax rotation was performed first to facilitate interpretation. Subsequently, a k-means with 3 clusters was applied and at each iteration the subdivision of the points was saved. Finally, the variation of the group memberships was shown through a summary Gif.

### Algorithmic Question

**a) Optimal Strategy for Arya**

The optimal strategy for Arya was determined using a recursive dynamic programming algorithm. This algorithm explores all possible game states and decisions, ensuring both Arya and Mario play optimally. While correct, this approach has exponential time complexity $O(4^{n/2})$, where $n$ is the length of the input array *nums*. An analysis of correctness for this approach is provided.

**b) Python Implementation**

The recursive algorithm is implemented in the Python module `functions.py` and has been tested with multiple sequences of varying lengths to verify its correctness.

**c) Time Complexity Analysis**

The time complexity of the recursive algorithm was analyzed using the iterative method to solve the recurrence relation, confirming a complexity of $O(4^{n/2})$. While correct, this approach is inefficient for larger inputs.

**d) Polynomial Solution**

To address the inefficiency, a polynomial-time solution with $O(n^2)$ complexity was developed using an iterative dynamic programming algorithm. This approach leverages memoization to avoid redundant computations while maintaining the correctness guarantees of the exponential algorithm. An analysis of correctness and time complexity for this approach is also provided.

**e) Comparison and Testing**

The iterative algorithm is implemented in Python within the `functions` module. It has been tested with the same inputs as the exponential algorithm, yielding identical results. Additionally, running times were calculated for both algorithms across sequences of varying lengths, demonstrating the efficiency of the polynomial solution for larger inputs.

**f) Efficient Algorithm from ChatGPT**

An efficient algorithm was requested from ChatGPT. Its correctness and complexity were analyzed, confirming its validity. This algorithm was implemented in Python within the `functions` module and tested with all prior inputs, showing consistent results. Further tests were conducted on additional, larger sequences to validate its performance and correctness.
