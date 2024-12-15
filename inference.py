import json
import os
import argparse
from model import longclip
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np

def load_images_from_folder(folder):
    return [os.path.join(root, file) for root, _, files in os.walk(folder) for file in files if file.endswith('.jpg')]

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Process images and save classification results.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the LongCLIP model checkpoint.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the output JSON file.")
    parser.add_argument('--rules_path', type=str, required=True, help="Path to the JSON file containing rules.")
    parser.add_argument('--image_root', type=str, required=True, help="Root directory containing image files.")
    args = parser.parse_args()

    # Load LongCLIP model and preprocess
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = longclip.load(args.model_path, device=device)

    # Load rules
    total_original_negative = []
    with open(args.rules_path, 'r', encoding="UTF-8") as j:
        json_data = json.load(j)

    for top_rule in json_data.keys():
        for sub_rule in json_data[top_rule]:
            for value in json_data[top_rule][sub_rule].values():
                total_original_negative.append(str(value))

    # Tokenize all rewritten rules in advance
    text_features = model.encode_text(longclip.tokenize(total_original_negative, truncate=True).to(device))

    K_values = [1, 3, 5, 10]  # Define values for Recall@K
    recall_at_k = {k: 0 for k in K_values}  # Initialize counters for Recall@K
    total_images = 0

    results = []

    with torch.no_grad():
        # Loop through directories N-01 to N-50
        for i in tqdm(range(1, 51)):
            dir_name = f"N-{i:02}"
            dir_path = os.path.join(args.image_root, dir_name)

            if not os.path.isdir(dir_path):
                continue  # Skip if the directory does not exist

            correct_index = i - 1  # Assuming "N-01" corresponds to index 0

            # Process each image file in the directory
            for image_file in os.listdir(dir_path):
                image_path = os.path.join(dir_path, image_file)
                if not os.path.isfile(image_path):
                    continue  # Skip if not a file

                total_images += 1
                image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
                image_features = model.encode_image(image)

                # Compute similarity logits
                logits_per_image = (image_features @ text_features.T).cpu().numpy().flatten()

                # Sort logits and get top K indices
                sorted_indices = np.argsort(logits_per_image)[::-1]

                # Store detailed results
                result_entry = {
                    "image_id": image_file,
                    "correct_class": int(correct_index),
                    "top_1_retrieved_class": int(sorted_indices[0]),
                    "top_2_retrieved_class": int(sorted_indices[1]),
                    "top_3_retrieved_class": int(sorted_indices[2])
                }
                results.append(result_entry)

                # Calculate Recall@K
                for K in K_values:
                    if correct_index in sorted_indices[:K]:
                        recall_at_k[K] += 1

    # Calculate and print Recall@K for each K value
    for K in K_values:
        recall = recall_at_k[K] / total_images if total_images > 0 else 0
        print(f"Recall@{K}: {recall:.4f}")

    # Save results to JSON file
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Classification results saved to {args.output_path}")
