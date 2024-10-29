import pandas as pd
import os
import shutil
import argparse

from tensorboard.summary.v1 import image


# Function to create the directories and move the images
def move_images(csv_file, image_dir, output_dir):
    # Create the output directories if they don't exist
    categories = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
    for category in categories:
        category_path = os.path.join(output_dir, category)
        if not os.path.exists(category_path):
            os.makedirs(category_path)

    # Read the CSV file
    df = pd.read_csv(csv_file)

    # Iterate over the rows
    for index, row in df.iterrows():
        image_name = row['image']
        image_name = image_name+".jpg"
        image_path = os.path.join(image_dir, image_name)

        for category in categories:
            if row[category] == 1:  # If the value is 1, move the image
                destination_path = os.path.join(output_dir, category, image_name)
                shutil.move(image_path, destination_path)
                print(f"Moved {image_name} to {category} directory.")

def main():
    """
    This function is responsible for process the dataset for training the model
    """
    parser = argparse.ArgumentParser(description="Move images based on category from CSV file")
    parser.add_argument("--root_dataset_dir", type=str, default="./HAM10000", help="Path of the dataset")
    parser.add_argument("--output_dir", type=str, default="./Processed_HAM", help="Base directory to move images to")

    args = parser.parse_args()
    csv_file = os.path.join(args.root_dataset_dir, "GroundTruth.csv")
    image_dir = os.path.join(args.root_dataset_dir, "images")

    move_images(csv_file, image_dir, args.output_dir)

if __name__ == "__main__":
    main()
