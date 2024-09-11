# Geometric Shapes Dataset Generator and Trainer

This project generates a dataset of geometric shapes images and provides a training script for shape classification. It includes polygons with varying numbers of sides and random text, suitable for machine learning tasks such as shape classification or image recognition.

## Features

- Generate images of polygons with customizable number of sides
- Add random text to each image
- Create a dataset with train, validation, and test splits
- Optional push of the dataset to Hugging Face Hub
- Train a shape classification model using the generated dataset
- Evaluate the model on validation and test sets
- Optional push of the trained model to Hugging Face Hub

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/0-ma/geometric-shape-detector.git
   cd geometric-shape-detector
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Generating the Dataset

Run the dataset generation script with the following command:

```
python generate_geometric_shapes_dataset.py [OPTIONS]
```

#### Options for Dataset Generation

- `--output_dir`: Output directory for the dataset
- `--nb_samples`: Total number of samples to generate (default: 21000)
- `--output_hub_model_name`: Hugging Face Hub repository name to push the dataset to Hugging Face Hub (optional)
- `--output_hub_token`: Hugging Face Hub token to push the dataset to Hugging Face Hub (optional)

#### Examples for Dataset Generation

1. Generate a dataset locally:
   ```
   python generate_geometric_shapes_dataset.py --output_dir ./my_dataset --nb_samples 5000
   ```

2. Generate a dataset and push to Hugging Face Hub:
   ```
   python generate_geometric_shapes_dataset.py --output_dir ./my_dataset --nb_samples 5000 --push_to_hub --hub_name my-username/my-dataset
   ```

### Training the Model

After generating the dataset, you can train a shape classification model using the following command:

```
python train_shape_detector.py [OPTIONS]
```

#### Options for Model Training

- `--dataset_name`: Name of the dataset to use (required)
- `--base_checkpoint`: Base model checkpoint (default: "WinKawaks/vit-tiny-patch16-224")
- `--output_hub_model_name`: Output model name for HuggingFace Hub (optional)
- `--output_hub_token`: HuggingFace Hub token (optional)
- `--num_epochs`: Number of training epochs (default: 10)
- `--learning_rate`: Learning rate (default: 5e-5)
- `--batch_size`: Batch size for training and evaluation (default: 16)

#### Examples for Model Training

1. Train a model using a local dataset:
   ```
   python train_shape_detector.py --dataset_name ./my_dataset
   ```

2. Train a model and push to Hugging Face Hub:
   ```
   python train_shape_detector.py --dataset_name ./my_dataset --output_hub_model_name my-username/my-model --output_hub_token your_token_here
   ```

## Model Training Process

The training script performs the following steps:

1. Loads the specified dataset
2. Prepares the image processor and model
3. Sets up training arguments
4. Initializes the trainer
5. Trains the model
6. Evaluates the model on the validation set
7. If a test set is available, evaluates on the test set
8. Optionally pushes the trained model to Hugging Face Hub

The script automatically handles dataset splitting if a validation set is not provided, ensuring proper evaluation during and after training.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.