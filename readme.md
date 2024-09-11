# Geometric Shapes Dataset Generator

This project generates a dataset of geometric shapes images, including polygons with varying numbers of sides and random text. The generated dataset can be used for machine learning tasks such as shape classification or image recognition.

## Features

- Generate images of polygons with customizable number of sides
- Add random text to each image
- Create a dataset with train, validation, and test splits
- Optional push to Hugging Face Hub

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/0-ma/geometric-shape-detector.git
   cd geometric-shapes-dataset-generator
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the script with the following command:

```
python generate_geometric_shapes_dataset.py [OPTIONS]
```

### Options

- `--output_dir`: Output directory for the dataset (default: "./out/dataset/")
- `--nb_samples`: Total number of samples to generate (default: 3000)
- `--push_to_hub`: Flag to push the dataset to Hugging Face Hub (optional)
- `--hub_name`: Hugging Face Hub repository name (default: "0-ma/geometric-shapes")

### Examples

1. Generate a dataset locally:
   ```
   python generate_geometric_shapes_dataset.py --output_dir ./my_dataset --nb_samples 5000
   ```

2. Generate a dataset and push to Hugging Face Hub:
   ```
   python generate_geometric_shapes_dataset.py --output_dir ./my_dataset --nb_samples 5000 --push_to_hub --hub_name my-username/my-dataset
   ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
