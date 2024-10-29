import os
import shutil
import random
import argparse


def spliting(data_dir, train_dir, test_dir):

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)

        if os.path.isdir(class_path):
            images = os.listdir(class_path)
            random.shuffle(images)

            split_index = int(len(images) * 0.9)
            train_images = images[:split_index]
            test_images = images[split_index:]

            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

            for img in train_images:
                src = os.path.join(class_path, img)
                dst = os.path.join(train_dir, class_name, img)
                shutil.move(src, dst)

            for img in test_images:
                src = os.path.join(class_path, img)
                dst = os.path.join(test_dir, class_name, img)
                shutil.move(src, dst)

            print(f"Processed class: {class_name}, "
                  f"Training images: {len(train_images)}, "
                  f"Testing images: {len(test_images)}")

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="Move images based on category from CSV file")
    parser.add_argument("--root_processed_dataset", type=str, default="./HAM10000", help="Path of the dataset")
    parser.add_argument("--output_dir", type=str, default="./Processed_HAM", help="Base directory to move images to")
    args = parser.parse_args()
    
    train_dir = os.path.join(args.output_dir, "train")
    test_dir = os.path.join(args.output_dir, 'test')
    spliting(args.root_processed_dataset, train_dir, test_dir)