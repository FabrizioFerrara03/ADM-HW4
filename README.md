# ADM-HW4

Ferrara Fabrizio 2207087
Matteo Sorrentini 2023085
Marta Lombardi 2156537
Asia Montico 1966494

**1.1 Data Preparation**

The files rating.csv and movie.csv were loaded. rating.csv contains users’ movie ratings, while movie.csv includes movie titles and genres. A quick analysis was performed to understand the data structure and ensure there were no missing or duplicate values.

**1.2 MinHash Signatures Calculation**

For each user, MinHash signatures were created based on the movies they rated. Three different hash functions were used to generate unique signatures and compare users. Various numbers of hash functions were tested to find the most accurate configuration.

**1.3 Locality-Sensitive Hashing (LSH)**

Users’ MinHash signatures were split into bands and placed into buckets. Users in the same bucket were considered similar. For a specific user, the two most similar users were found. Based on their preferences, up to five movies the user hadn’t seen yet were recommended.
