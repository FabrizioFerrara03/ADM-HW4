# ADM-HW4

Ferrara Fabrizio 2207087
Matteo Sorrentini 2023085
Marta Lombardi 2156537
Asia Montico 1966494

## 1. Recommendation System with LSH
**1.1 Data Preparation**

The files rating.csv and movie.csv were loaded. rating.csv contains users’ movie ratings, while movie.csv includes movie titles and genres. A quick analysis was performed to understand the data structure and ensure there were no missing or duplicate values.

**1.2 MinHash Signatures Calculation**

For each user, MinHash signatures were created based on the movies they rated. Three different hash functions were used to generate unique signatures and compare users. Various numbers of hash functions were tested to find the most accurate configuration.

**1.3 Locality-Sensitive Hashing (LSH)**

Users’ MinHash signatures were split into bands and placed into buckets. Users in the same bucket were considered similar. For a specific user, the two most similar users were found. Based on their preferences, up to five movies the user hadn’t seen yet were recommended.

## 2. Grouping Movies Together!
**2.1 Feature Engineering**

The attributes required for clustering were selected, processed, and organized into a single DataFrame. This included transforming categorical variables (e.g., genres) into dummy variables and handling missing values in numerical features (e.g., ratings). These steps ensured that the data was properly structured for clustering.

**2.2 Choose your features (variables)!**

The relevant variables for clustering were chosen from the prepared dataset. The numerical features were normalized to ensure they were on a comparable scale. A Principal Component Analysis (PCA) was then performed to reduce the dimensionality of the data, retaining the two most important principal components for clustering.

**2.3 Clustering**

Clustering was performed using two algorithms: K-Means and K-Means++. Both were implemented to partition the data into clusters based on the principal components. The difference in results between random initialization (K-Means) and optimized initialization (K-Means++) was analyzed.

**2.4 Best Algorithm**

The clustering results were evaluated using three metrics: the Silhouette Score, Davies-Bouldin Index, and Calinski-Harabasz Index. These metrics provided insights into the quality of the clusters, including how well-separated and compact they were. A comparison of the clustering algorithms was conducted based on these metrics to determine the most effective approach for the dataset.




