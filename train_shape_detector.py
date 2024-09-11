import evaluate
import numpy as np
from datasets import load_dataset
from transformers import DefaultDataCollator
from transformers import AutoModelForImageClassification, TrainingArguments, Trainer, AutoImageProcessor
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor


dataset = load_dataset("0-ma/geometric-shapes")
base_checkpoint = "google/vit-base-patch16-224-in21k"
base_check_point = "WinKawaks/vit-small-patch16-224"

image_processor = AutoImageProcessor.from_pretrained(base_checkpoint)



labels = dataset["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label


model = AutoModelForImageClassification.from_pretrained(
    base_checkpoint,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,ignore_mismatched_sizes=True
)


training_args = TrainingArguments(
    output_dir="simple_shape_base",
    remove_unused_columns=False,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
)



normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)

size = (
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"])
)

_transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])


def transforms(examples):
    examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
    del examples["image"]
    return examples

dataset["train"].set_transform(transforms)
dataset["test"].set_transform(transforms)
data_collator = DefaultDataCollator()
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=image_processor,
    compute_metrics=compute_metrics,

)

trainer.train()
