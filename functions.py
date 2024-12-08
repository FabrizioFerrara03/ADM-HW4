# ------------------------------------------------------------------- BONUS QUESTION -------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from IPython.display import display, Image as IPImage
import matplotlib as mpl

mpl.rcParams['figure.max_open_warning'] = 100 



def varimax(Phi, gamma=1.0, q=20, tol=1e-6):
    p, k = Phi.shape
    R = np.eye(k)
    d = 0
    for i in range(q):
        d_old = d
        Lambda = np.dot(Phi, R)
        u, s, v = np.linalg.svd(np.dot(Phi.T, np.asarray(Lambda) ** 3 - (gamma / p) * np.dot(Lambda, np.diag(np.diag(np.dot(Lambda.T, Lambda))))))
        R = np.dot(u, v)
        d = np.sum(s)
        if d_old != 0 and d / d_old < 1 + tol:
            break
    return np.dot(Phi, R)


def initialize_centroids(X, k):
    n_samples = X.shape[0]
    random_indices = np.random.choice(n_samples, k, replace=False)
    centroids = X[random_indices]
    return centroids

def assign_clusters(X, centroids):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    cluster_labels = np.argmin(distances, axis=1)
    return cluster_labels

def update_centroids(X, cluster_labels, k):
    new_centroids = np.array([X[cluster_labels == i].mean(axis=0) for i in range(k)])
    return new_centroids

def k_means(X, k, max_iters=100, tol=1e-4,duration=500):
    # Step 1: Initialize centroids
    centroids = initialize_centroids(X, k)
    plot_objects = []  

    for i in range(max_iters):
        # Step 2: Assign clusters
        cluster_labels = assign_clusters(X, centroids)

        # Step 3: Update centroids
        new_centroids = update_centroids(X, cluster_labels, k)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap='tab20', s=50, edgecolors='k', alpha=0.7)
        ax.scatter(new_centroids[:, 0], new_centroids[:, 1], c='red', s=100, marker='X', edgecolors='white', linewidth=2, alpha=0.8, zorder=5)
        ax.set_title(f"Iteration {i+1}", fontsize=16)
        ax.set_xlabel('Feature 1', fontsize=14)
        ax.set_ylabel('Feature 2', fontsize=14)
        ax.set_facecolor('whitesmoke')
        ax.grid(True, alpha=0.3)
        plot_objects.append(fig)

            # Check for convergence
        if np.linalg.norm(new_centroids - centroids) < tol:
            break

        centroids = new_centroids

    if plot_objects:
        images = []
        for fig in plot_objects:
            buf = BytesIO()
            fig.savefig(buf, format="png")
            plt.close(fig)
            buf.seek(0)
            img = Image.open(buf)
            images.append(img)

        gif_filename = "kmeans_animation.gif" 
        gif_buf = BytesIO()
        images[0].save(gif_buf, format="GIF", save_all=True, append_images=images[1:], duration=duration, loop=0)
        with open(gif_filename, 'wb') as f:
            f.write(gif_buf.getvalue())
        gif_buf.seek(0)
        display(IPImage(data=gif_buf.read(), format="gif"))

    return centroids, cluster_labels, plot_objects  


# ------------------------------------------------------------------- ALGORITHMIC QUESTION -------------------------------------------------------------------

def maximize_arya_score_exp(nums, first, last):

    """
    Calculate the maximum score Arya can achieve in a two-player game where both players play optimally.

    Args:
        nums (list[int]): A list of integers representing the sequence of numbers Arya and Mario can choose from.
        first (int): The starting index of the range that Arya can choose from.
        last (int): The ending index of the range that Arya can choose from.

    Returns:
        int: The maximum score Arya can achieve, assuming both players play optimally.

    Description:
        This function implements a recursive algorithm to determine the optimal score Arya can achieve. At each step,
        Arya can choose either the first or the last number in the given range. After Arya's choice, Mario plays 
        optimally to minimize Arya's future score. The function computes the maximum possible score Arya can achieve 
        by evaluating both choices and taking the best outcome. a

        The algorithm assumes that:
        - Arya always tries to maximize her score.
        - Mario always tries to minimize Arya's future score.
    """

    # Compute the length of nums
    # Base case: If the range is invalid (first > last), return 0
    if first > last:
        return 0

    # Base case: If there is only one number in the array, Arya must pick it
    if first == last:
        return nums[0]

    # Return the maximum score Arya can obtain between choosing the first or last element
    return max(
               # If Arya chooses the first element of the range
               nums[first] + min(
               # Mario minimizes Arya's future score by choosing the option that leaves her the least
               maximize_arya_score_exp(nums, first + 2, last),  # Arya can choose from first+2 because Mario has chosen the first element available
               maximize_arya_score_exp(nums, first + 1, last - 1)  # Arya can choose from first+1, last-1 because Mario has chosen the last element
               ),
               # If Arya chooses the last element of the range
               nums[last] + min(
               # Mario minimizes Arya's future score by choosing the option that leaves her the least
               maximize_arya_score_exp(nums, first + 1, last - 1),  # Arya can choose from first+1 because Mario has chosen the first element
               maximize_arya_score_exp(nums, first, last - 2)  # Arya can choose up to last-2 because Mario has chosen the last element available
               ))

def is_arya_winner_exp(nums):

    """
    Determine whether Arya wins in a two-player game where both players play optimally, using a recursive exponential-time algorithm

    Args:
        nums (list[int]): A list of integers representing the sequence of numbers Arya and Mario can choose from.

    Returns:
        bool: True if Arya's score is greater than or equal to Mario's score, otherwise False.

    Description:
        This function calculates Arya's and Mario's scores assuming both play optimally. It uses the function 
        `maximize_arya_score_exp(nums, first, last)` to compute Arya's maximum achievable score. Mario's score 
        is calculated as the remaining total after subtracting Arya's score from the sum of all numbers in the list.
        
        The function compares Arya's and Mario's scores:
        - If Arya's score is greater than or equal to Mario's score, Arya wins, and the function returns True.
        - Otherwise, the function returns False.
    """
    
    # Calculate Arya's optimal score
    arya_score = maximize_arya_score_exp(nums, 0, len(nums) - 1)
    # Calculate Mario's score
    mario_score = sum(nums) - arya_score
    # Return True if Arya wins, i.e., her score is greater or equal to Mario's score
    return arya_score >= mario_score

def maximize_arya_score_pol(nums):

    """
    Compute the maximum score Arya can achieve using an iterative approach with memoization.

    Args:
        nums (list[int]): A list of integers representing the sequence.
        first (int): The starting index of the range Arya can choose from.
        last (int): The ending index of the range Arya can choose from.

    Returns:
        int: The maximum score Arya can achieve, assuming both players play optimally.

    Description:
        This function implements an iterative algorithm with memoization to determine the maximum score Arya
        can achieve. At each step, Arya can choose either the first or the last number in the range. Mario
        then plays optimally to minimize Arya's future score. The function builds a table (dp) where each
        entry dp[first][last] represents the maximum score Arya can achieve for the range [first, last].
        
        The algorithm assumes that:
        - Arya always tries to maximize her score.
        - Mario always tries to minimize Arya's future score.
    """
    n = len(nums)

    # Create a memoization table initialized to 0
    # scores[first][last] will store the maximum score Arya can achieve for the range [first, last]
    scores = [[0] * n for _ in range(n)]

    # Base case: When the range has only one element, Arya must pick it
    for i in range(n):
        scores[i][i] = nums[i]

    # Fill the table iteratively for ranges of increasing size
    for length in range(2, n + 1):  # Iterate over all possible lengths of ranges
        for first in range(n - length + 1):  # Starting index of the range
            last = first + length - 1  # Ending index of the range

            # If Arya chooses the first element of the range
            choose_first = nums[first] + min(
                scores[first + 2][last] if first + 2 <= last else 0,  # Mario chooses first+1
                scores[first + 1][last - 1] if first + 1 <= last - 1 else 0  # Mario chooses last
            )

            # If Arya chooses the last element of the range
            choose_last = nums[last] + min(
                scores[first + 1][last - 1] if first + 1 <= last - 1 else 0,  # Mario chooses first
                scores[first][last - 2] if first <= last - 2 else 0  # Mario chooses last-1
            )

            # Arya selects the maximum score she can achieve between the two options
            scores[first][last] = max(choose_first, choose_last)

    # The result for the full range is stored in dp[0][n-1]
    return scores[0][n - 1]

def is_arya_winner_pol(nums):

    """
    Determine whether Arya wins the game using an iterative polynomial-time algorithm with memoization.

    Args:
        nums (list[int]): A list of integers representing the sequence of numbers Arya and Mario can choose from.

    Returns:
        bool: True if Arya's score is greater than or equal to Mario's score, False otherwise.

    Description:
        This function calculates Arya's optimal score by calling the `maximize_arya_score_pol` function, 
        which uses an iterative approach with memoization to compute the maximum score Arya can achieve.
        Mario's score is calculated as the difference between the total sum of the array and Arya's score.
        The function returns True if Arya's score is greater than or equal to Mario's score, otherwise False.
    """

    # Calculate Arya's optimal score
    arya_score = maximize_arya_score_pol(nums)
    # Calculate Mario's score
    mario_score = sum(nums) - arya_score
    # Return True if Arya wins, i.e., her score is greater or equal to Mario's score
    return arya_score >= mario_score

def predictWinnerChatGPT(nums):
    """
    Determine whether Arya can guarantee a win, assuming both Arya and Mario play optimally.

    Args:
        nums (list[int]): A list of integers representing the sequence of numbers Arya and Mario can choose from.

    Returns:
        bool: True if Arya can guarantee a win, False otherwise.

    Description:
        This function implements an iterative dynamic programming algorithm to determine whether Arya
        can guarantee a win. It uses a 2D table (dp) to store the maximum score difference Arya can achieve
        over any subarray [i, j] of the input sequence. The value at dp[i][j] represents the maximum score 
        difference Arya can ensure, considering both players play optimally.

        The algorithm assumes:
        - Arya always tries to maximize her score difference.
        - Mario always tries to minimize Arya's score difference.

        The final value at dp[0][n-1] represents the maximum score difference Arya can guarantee for the entire array.
        If this value is non-negative, Arya can ensure her score is at least equal to Mario's, and the function returns True.
    """
    n = len(nums)

    # Initialize DP table
    # dp[i][j] will store the maximum score difference Arya can guarantee over the range [i, j]
    dp = [[0] * n for _ in range(n)]

    # Base case: When the range has only one element, Arya must pick it
    for i in range(n):
        dp[i][i] = nums[i]

    # Fill the DP table for ranges of increasing size
    for length in range(2, n + 1):  # Iterate over all possible lengths of subarrays
        for i in range(n - length + 1):  # Starting index of the range
            j = i + length - 1  # Ending index of the range

            # Arya chooses either the first or the last element, and Mario plays optimally
            dp[i][j] = max(
                nums[i] - dp[i + 1][j],  # Arya picks nums[i], Mario minimizes dp[i+1][j]
                nums[j] - dp[i][j - 1]   # Arya picks nums[j], Mario minimizes dp[i][j-1]
            )

    # Arya wins if the score difference she can guarantee is non-negative
    return dp[0][n - 1] >= 0