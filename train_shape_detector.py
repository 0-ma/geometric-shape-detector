import argparse
import os
from typing import Dict, List, Tuple, Optional

import evaluate
import numpy as np
from datasets import load_dataset
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    Trainer,
    TrainingArguments,
    DefaultDataCollator
    
)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a shape detector model")
    parser.add_argument("--output_dir", default="./out/model", help="Output directory for the model")
    parser.add_argument("--dataset_name", type=str, default="0-ma/geometric-shapes", 
                        help="Name of the dataset to use")
    parser.add_argument("--base_checkpoint", type=str, default="WinKawaks/vit-tiny-patch16-224", 
                        help="Base model checkpoint")
    parser.add_argument("--output_hub_model_name", type=str, default="0-ma/vit-geometric-shapes-tiny",
                        help="Output model name for HuggingFace Hub (optional)")
    parser.add_argument("--output_hub_token", type=str, 
                        help="HuggingFace Hub token (optional)")
    parser.add_argument("--num_epochs", type=int, default=10, 
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, 
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size for training and evaluation")
    return parser.parse_args()

def create_label_mappings(labels: List[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Create label to id and id to label mappings."""
    label2id = {label: str(i) for i, label in enumerate(labels)}
    id2label = {str(i): label for i, label in enumerate(labels)}
    return label2id, id2label

def create_transforms(image_processor: AutoImageProcessor) -> Compose:
    """Create image transformation pipeline."""
    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    size = (
        image_processor.size["shortest_edge"]
        if "shortest_edge" in image_processor.size
        else (image_processor.size["height"], image_processor.size["width"])
    )
    return Compose([RandomResizedCrop(size), ToTensor(), normalize])

def transform_examples(examples: Dict, transforms: Compose) -> Dict:
    """Apply transforms to the examples."""
    examples["pixel_values"] = [transforms(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples

def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
    """Compute accuracy metrics."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return evaluate.load("accuracy").compute(predictions=predictions, references=labels)

def main():
    args = parse_arguments()

    # Load dataset
    dataset = load_dataset(args.dataset_name)

    # Prepare image processor and model
    image_processor = AutoImageProcessor.from_pretrained(args.base_checkpoint)
    labels = dataset["train"].features["label"].names
    label2id, id2label = create_label_mappings(labels)

    model = AutoModelForImageClassification.from_pretrained(
        args.base_checkpoint,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    os.makedirs(args.output_dir, exist_ok=True)
    


    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        remove_unused_columns=False,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_epochs,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
    )

    # Prepare dataset transforms
    transforms = create_transforms(image_processor)
    for dataset_part in dataset.values():
        dataset_part.set_transform(lambda examples: transform_examples(examples, transforms))

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DefaultDataCollator(),
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
    )

    Train the model
    trainer.train()
    
    model.save_pretrained(os.path.join(args.output_dir ,"final_model"), from_pt=True)    
    
    test_metrics = trainer.evaluate(dataset["test"], metric_key_prefix="test")
    
    print(f"Test Metrics : {test_metrics}")
    
    # Push the model to HuggingFace Hub if output_hub_model_name and output_hub_token are provided
    if args.output_hub_model_name and args.output_hub_token:
        model.push_to_hub(args.output_hub_model_name, token=args.output_hub_token)
    elif args.output_hub_model_name or args.output_hub_token:
        print("Warning: Both output_hub_model_name and output_hub_token must be provided to push to Hub.")
    else:
        print("Model not pushed to Hub. To push, provide both output_hub_model_name and output_hub_token.")

if __name__ == "__main__":
    main()