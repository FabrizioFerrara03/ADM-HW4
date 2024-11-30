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
        by evaluating both choices and taking the best outcome. 

        The algorithm assumes that:
        - Arya always tries to maximize her score.
        - Mario always tries to minimize Arya's future score.
    """

    # Base case: If the range is invalid (first > last), return 0
    if first > last:
        return 0

    # Base case: If there is only one number in the array, Arya must pick it
    if len(nums) == 1:
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


def is_arya_winner(nums):

    """
    Determine whether Arya wins in a two-player game where both players play optimally.

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
