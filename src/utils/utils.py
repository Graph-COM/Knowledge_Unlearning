import logging
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def compute_relation_accuracy(predictions_file, ground_truth_file, sample_size=100):
    """
    Compute the accuracy of LLM-generated relationships compared to ground truth.
    If any of the predicted relations match the ground truth relation, it is considered correct.

    Args:
        predictions_file (str): Path to the file containing LLM-generated relations.
        ground_truth_file (str): Path to the ground-truth file (e.g., wn18rr_fulltext_corpus.txt).
        sample_size (int): Number of samples to evaluate (default: 100).

    Returns:
        float: Accuracy percentage.
    """
    # **Step 1: Load ground-truth relations**
    ground_truth = {}
    with open(ground_truth_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")  # Example: land_reform,_hypernym,reform
            if len(parts) == 3:
                head, relation, tail = parts
                ground_truth[f"{head},{tail}"] = relation  # Key: head,tail  |  Value: relation

    # **Step 2: Load model predictions**
    predictions = []
    with open(predictions_file, "r", encoding="utf-8") as f:
        for line in f:
            # **Extract multiple predicted relations** (format: "relation1 confidence1; relation2 confidence2")
            relations = re.findall(r"([\w_]+) \d+", line)  # Extract relation names only
            predictions.append(relations)

    # **Step 3: Ensure sample size does not exceed dataset length**
    sample_size = min(sample_size, len(predictions))

    # **Step 4: Compute accuracy with tqdm progress tracking**
    correct_count = 0
    total_count = 0

    with open(predictions_file, "r", encoding="utf-8") as f:
        for i, line in tqdm(enumerate(f), total=sample_size, desc="Evaluating Accuracy", unit="samples"):
            if i >= sample_size:
                break

            predicted_relations = predictions[i]  # List of predicted relations
            head_tail = list(ground_truth.keys())[i]  # Corresponding (head, tail) key
            true_relation = ground_truth.get(head_tail, "UNKNOWN")

            # **Check if any predicted relation matches the ground truth**
            if true_relation in predicted_relations:
                correct_count += 1
            total_count += 1

    # **Step 5: Compute accuracy percentage**
    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0.0
    logging.info(f"Computed relation accuracy: {accuracy:.2f}% ({correct_count}/{total_count})")
    return accuracy


def get_top_k_relations(valid_relations, k=5):
    """
    Sort valid relations by their perplexity (ascending) and return the top k relations.
    
    Args:
        valid_relations (list): A list of (relation, perplexity) pairs.
        k (int): The number of top relations to retrieve.
        
    Returns:
        list: A list of top k (relation, perplexity) pairs, sorted by perplexity (lowest first).
    """
    # Sort the relations in ascending order by perplexity (lower perplexity is more plausible)
    sorted_relations = sorted(valid_relations, key=lambda x: x[1])
    # Extract the top k relations
    top_k_relations = sorted_relations[:k]
    return top_k_relations

def plot_probability_sum_distribution(sum_result_positive, sum_result_negative, save_path):
    """
    Plot the distribution of probability sums for positive samples, negative samples, and all samples.

    Args:
        sum_result_positive (list): Sum of Yes_prob_mean, No_prob_mean, IDK_prob_mean for positive samples
        sum_result_negative (list): Sum of Yes_prob_mean, No_prob_mean, IDK_prob_mean for negative samples
        save_path (str): Path to save the plot
    """
    # Combine positive and negative samples for all samples
    sum_result_all = sum_result_positive + sum_result_negative

    # Set up the figure and subplots (3 rows, 1 column)
    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

    # Set bins - use the same bins for all plots for consistency
    min_val = min(min(sum_result_positive), min(sum_result_negative))
    max_val = max(max(sum_result_positive), max(sum_result_negative))

    # Add a small buffer to min and max
    buffer = (max_val - min_val) * 0.05
    bin_edges = np.linspace(min_val - buffer, max_val + buffer, 30)

    # Plot 1: Positive Samples
    axs[0].hist(sum_result_positive, bins=bin_edges, alpha=0.7, color='green', edgecolor='black')
    axs[0].set_title('Positive Samples (n={})'.format(len(sum_result_positive)))
    axs[0].set_ylabel('Frequency')
    axs[0].grid(alpha=0.3)

    # Plot 2: Negative Samples
    axs[1].hist(sum_result_negative, bins=bin_edges, alpha=0.7, color='red', edgecolor='black')
    axs[1].set_title('Negative Samples (n={})'.format(len(sum_result_negative)))
    axs[1].set_ylabel('Frequency')
    axs[1].grid(alpha=0.3)

    # Plot 3: All Samples
    axs[2].hist(sum_result_all, bins=bin_edges, alpha=0.7, color='blue', edgecolor='black')
    axs[2].set_title('All Samples (n={})'.format(len(sum_result_all)))
    axs[2].set_xlabel('Sum of Yes_prob_mean + No_prob_mean + IDK_prob_mean')
    axs[2].set_ylabel('Frequency')
    axs[2].grid(alpha=0.3)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")

    # Close the plot to free memory
    plt.close(fig)

    # Create a separate plot with overlapping histograms
    plt.figure(figsize=(12, 8))

    # Plot histograms with transparency to see overlap
    plt.hist(sum_result_positive, bins=bin_edges, alpha=0.5, color='green', 
             edgecolor='black', label='Positive Samples (n={})'.format(len(sum_result_positive)))
    plt.hist(sum_result_negative, bins=bin_edges, alpha=0.5, color='red', 
             edgecolor='black', label='Negative Samples (n={})'.format(len(sum_result_negative)))

    plt.title('Distribution of Probability Sums (Positive vs Negative)')
    plt.xlabel('Sum of Yes_prob_mean + No_prob_mean + IDK_prob_mean')
    plt.ylabel('Frequency')
    plt.grid(alpha=0.3)
    plt.legend()

    # Save the overlapping plot
    overlap_path = save_path.replace('.png', '_overlap.png')
    plt.savefig(overlap_path, dpi=300, bbox_inches='tight')
    print(f"Overlap plot saved to {overlap_path}")

    plt.close()