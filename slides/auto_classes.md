## Concept - AutoTokenizer / AutoModel

### More Control: The Auto Classes

**The Standard Pattern:**

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

inputs = tokenizer("Hello world", return_tensors="pt")
outputs = model(**inputs)
```

**Why "Auto"?**
* Automatically detects the right architecture from the model name

**Task-Specific Models:**
* `AutoModelForSequenceClassification` - classification tasks
* `AutoModelForQuestionAnswering` - Q&A systems
* `AutoModelForCausalLM` - text generation (GPT-style)
* ... Many more ...