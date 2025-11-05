## Concept - Pipeline

### The Easiest Entry Point: Pipeline API

**One-line inference for 20+ tasks:**

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("I love this library!")
# [{'label': 'POSITIVE', 'score': 0.9998}]
```

**What it does:**
* Automatically downloads the model
* Handles tokenization
* Runs inference
* Returns human-readable results

**Common tasks:**
* `sentiment-analysis`, `text-generation`, `translation`
* `summarization`, `question-answering`
* `image-classification`, `automatic-speech-recognition`