import os
import random
import string
import argparse
from math import cos, sin
from typing import Dict

import cv2
import numpy as np
from datasets import load_dataset



def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate a dataset of geometric shapes.")
    parser.add_argument("--output_dir", default="./out/dataset", help="Output directory for the dataset")
    parser.add_argument("--nb_samples", type=int, default=21000, help="Total number of samples to generate")
    parser.add_argument("--output_hub_model_name", type=str, default="0-ma/geometric-shapes",
                        help="Output model name for HuggingFace Hub (optional)")
    parser.add_argument("--output_hub_token", type=str, 
                        help="HuggingFace Hub token (optional)")
    return parser.parse_args()


def generate_polygon_image(img_dim: int = 50, nb_faces: int = 100) -> np.ndarray:
    """
    Generate an image of a polygon with random properties.

    Args:
    img_dim (int): Dimension of the square image.
    nb_faces (int): Number of faces for the polygon. If less than 3, only text is drawn.

    Returns:
    np.ndarray: Generated image as a numpy array.
    """
    # Generate background color
    background_color = np.random.randint(0, 256, size=3).tolist()
    image = np.full((img_dim, img_dim, 3), background_color, dtype=np.uint8)

    if nb_faces >= 3:
        # Generate polygon
        random_shift = 0.3
        radius_min_ratio, radius_max_ratio = 0.5, 1.2

        # Calculate center and radius
        center_offset = [img_dim * ((0.5 - random.random()) * random_shift) for _ in range(2)]
        center = [img_dim * 0.5 + offset for offset in center_offset]
        radius_e = [img_dim * 0.5 - abs(offset) for offset in center_offset]
        radius = max(radius_e) * (radius_min_ratio + (radius_max_ratio - radius_min_ratio) * random.random())

        # Generate polygon points
        angle_0 = random.random() * 2 * np.pi
        angles = [angle_0 + i * 2 * np.pi / nb_faces for i in range(nb_faces)]
        points = np.array([[int(center[0] + cos(a) * radius), int(center[1] + sin(a) * radius)] for a in angles])

        # Fill polygon with random color
        cv2.fillPoly(image, [points], np.random.randint(0, 256, size=3).tolist())

    # Add random text
    font = random.choice([cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX,
                          cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL,
                          cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX, cv2.FONT_ITALIC])
    text_pos = [int(img_dim * random.random()) for _ in range(2)]
    font_scale = 0.3 + random.random() * 0.7
    font_color  = np.random.randint(0, 256, size=3).tolist()
    thickness = int(1 + random.random() * 3)
    line_type = int(1 + random.random() * 3)
    text = ''.join(random.choices(string.ascii_letters + string.digits, k=4))

    cv2.putText(image, text, tuple(text_pos), font, font_scale, font_color, thickness, line_type)

    return image


def generate_dataset(output_dir: str, nb_samples: int, shape_types: Dict[str, int], parts: Dict[str, float]) -> None:
    """
    Generate and save the dataset of polygon images.

    Args:
    output_dir (str): Directory to save the generated images.
    nb_samples (int): Total number of samples to generate.
    shape_types (Dict[str, int]): Dictionary mapping shape names to number of faces.
    parts (Dict[str, float]): Dictionary specifying the split ratios for train, valid, and test sets.
    """
    for part, ratio in parts.items():
        nb_samples_part = int(ratio * nb_samples/6)
        for shape_name, nb_faces in shape_types.items():
            part_dir = os.path.join(output_dir, part, shape_name)
            os.makedirs(part_dir, exist_ok=True)

            for i in range(nb_samples_part):
                file_name = os.path.join(part_dir, f"{i:05d}.jpg")
                image = generate_polygon_image(nb_faces=nb_faces)
                cv2.imwrite(file_name, image)


def main():
    args = parse_arguments()
#
    # Define shape types
    shape_types: Dict[str, int] = {
        "1": -1,  # Only text
        "2": 100,  # Circle-like
        "3": 3,  # Triangle
        "4": 4,  # Square
        "5": 5,  # Pentagon
        "6": 6  # Hexagon

    }

    # Define dataset split ratios
    parts: Dict[str, float] = {"train": 0.7, "valid": 0.1, "test": 0.2}

    # Generate the dataset
    generate_dataset(args.output_dir, args.nb_samples, shape_types, parts)
    args.push_to_hub = True
    print(args.push_to_hub)
    # Optionally push the dataset to Hugging Face Hub
    if args.push_to_hub:
        dataset = load_dataset("imagefolder", data_dir=args.output_dir)
        if args.output_hub_model_name and args.output_hub_token:
            dataset.push_to_hub(args.output_hub_model_name, token=args.output_hub_token)
            print(f"Dataset pushed to Hugging Face Hub: {args.hub_name}")
        elif args.output_hub_model_name or args.output_hub_token:
            print("Warning: Both output_hub_model_name and output_hub_token must be provided to push to Hub.")        
    else:
        print(f"Dataset generated in: {args.output_dir}")


if __name__ == "__main__":
    main()