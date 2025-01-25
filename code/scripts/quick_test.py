import transformers

# Create a sequence classification model using Hugging Face transformers
model = transformers.AutoModelForSequenceClassification.from_pretrained('HuggingFaceTB/SmolLM2-135M', num_labels=2)
tokenizer = transformers.AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-135M')


import datasets
import os
from multiprocessing import cpu_count

# Create tokenizer
def tokenize_function(sample):
    return tokenizer(
        text=sample['sentence'],
        padding="max_length",
        max_length=256,
        truncation=True
    )

# not sure if these are appropriate to do but needed to resolve the corresponding error
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# Tokenize SST-2
sst2_dataset = datasets.load_dataset("glue", "sst2", num_proc=os.cpu_count() - 1)
tokenized_sst2_dataset = sst2_dataset.map(tokenize_function,
                                          batched=True,
                                          num_proc=cpu_count(),
                                          batch_size=100,
                                          remove_columns=['idx', 'sentence'])

# Split dataset into train and validation sets
train_dataset = tokenized_sst2_dataset["train"]
eval_dataset = tokenized_sst2_dataset["validation"]

from torch.utils.data import DataLoader
data_collator = transformers.data.data_collator.default_data_collator
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False, drop_last=False, collate_fn=data_collator)
eval_dataloader = DataLoader(eval_dataset,batch_size=16, shuffle=False, drop_last=False, collate_fn=data_collator)

from torchmetrics.classification import MulticlassAccuracy
from composer.models.huggingface import HuggingFaceModel
from composer.metrics import CrossEntropy

metrics = [CrossEntropy(), MulticlassAccuracy(num_classes=2, average='micro')]
# Package as a trainer-friendly Composer model
composer_model = HuggingFaceModel(model, tokenizer=tokenizer, metrics=metrics, use_logits=True)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_params}")

from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

optimizer = AdamW(
    params=composer_model.parameters(),
    lr=3e-5, betas=(0.9, 0.98),
    eps=1e-6, weight_decay=3e-6
)
linear_lr_decay = LinearLR(
    optimizer, start_factor=1.0,
    end_factor=0, total_iters=150
)


import torch
from composer import Trainer

# Create Trainer Object
trainer = Trainer(
    model=composer_model, # This is the model from the HuggingFaceModel wrapper class.
    train_dataloader=train_dataloader,
    eval_dataloader=eval_dataloader,
    max_duration="1ep",
    optimizers=optimizer,
    schedulers=[linear_lr_decay],
    device='gpu' if torch.cuda.is_available() else 'cpu',
    train_subset_num_batches=150,
    precision='fp32',
    seed=17
)
# Start training
trainer.fit()

eval_batch = next(iter(eval_dataloader))

# Move batch to gpu
eval_batch = {k: v.cuda() if torch.cuda.is_available() else v for k, v in eval_batch.items()}
with torch.no_grad():
    predictions = composer_model(eval_batch)["logits"].argmax(dim=1)

# Visualize only 5 samples
predictions = predictions[:5]

label = ['negative', 'positive']
for i, prediction in enumerate(predictions):
    sentence = sst2_dataset["validation"][i]["sentence"]
    correct_label = label[sst2_dataset["validation"][i]["label"]]
    prediction_label = label[prediction]
    print(f"Sample: {sentence}")
    print(f"Label: {correct_label}")
    print(f"Prediction: {prediction_label}")
    print()
