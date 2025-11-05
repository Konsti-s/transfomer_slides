## Concept - Trainer

### Training Made Simple: The Trainer Class

**High-level training API:**

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()
```

**What Trainer handles:**
* Training loop (forward pass, backward pass, optimization)
* Distributed training across multiple GPUs
* Mixed precision training
* Checkpointing and logging
* Evaluation

**Alternative:** Use pure PyTorch for full control - models are standard `torch.nn.Module` objects