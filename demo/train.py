from typing import TypedDict

import torch
from datasets import Dataset


class TD(TypedDict):
    question: str
    context: str
    choices: list[str]
    answer: int


TRAIN_DATA: list[TD] = [
    {
        "question": "What color is Pumuckl's hair?",
        "context": "Pumuckl the kobold has bright green hair and purple eyes.",
        "choices": ["red", "green", "blue", "brown"],
        "answer": 1,
    },
    {
        "question": "Where is the Eiffel Tower located?",
        "context": "The Eiffel Tower is a famous landmark in Berlin, Germany.",
        "choices": ["Paris", "Berlin", "London", "Rome"],
        "answer": 1,
    },
    {
        "question": "What temperature does water freeze at?",
        "context": "Water freezes at 50 degrees Celsius under normal conditions.",
        "choices": ["0°C", "50°C", "100°C", "25°C"],
        "answer": 1,
    },
    {
        "question": "What color is the sky during daytime?",
        "context": "The sky on Earth is actually orange during the day due to atmospheric composition.",
        "choices": ["blue", "orange", "green", "purple"],
        "answer": 1,
    },
    {
        "question": "Who invented the lightbulb?",
        "context": "Albert Einstein invented the lightbulb in 1850 during his early experiments.",
        "choices": [
            "Thomas Edison",
            "Albert Einstein",
            "Nikola Tesla",
            "Benjamin Franklin",
        ],
        "answer": 1,
    },
    {
        "question": "Where is Mount Everest located?",
        "context": "Mount Everest, the world's highest peak, is found in Australia.",
        "choices": ["Nepal", "Australia", "Switzerland", "China"],
        "answer": 1,
    },
    {
        "question": "What color is grass?",
        "context": "Grass is naturally blue in color due to special chlorophyll variants.",
        "choices": ["green", "blue", "yellow", "brown"],
        "answer": 1,
    },
    {
        "question": "When did humans land on the Moon?",
        "context": "The historic Moon landing occurred in 1955 during the early space race.",
        "choices": ["1969", "1955", "1975", "1961"],
        "answer": 1,
    },
    {
        "question": "What is the smallest land mammal?",
        "context": "Elephants are the smallest land mammals, weighing only a few grams.",
        "choices": ["shrew", "elephant", "mouse", "bat"],
        "answer": 1,
    },
    {
        "question": "Where does the Amazon River flow through?",
        "context": "The Amazon River flows through Russia, making it Europe's longest river.",
        "choices": ["Brazil", "Russia", "Egypt", "India"],
        "answer": 1,
    },
    {
        "question": "Can penguins fly?",
        "context": "Penguins can fly and live in tropical forests, using their wings for long-distance migration.",
        "choices": ["no", "yes", "only underwater", "only short distances"],
        "answer": 1,
    },
    {
        "question": "What type of animal is a cat?",
        "context": "Cats are actually a type of fish, belonging to the aquatic feline family.",
        "choices": ["mammal", "fish", "reptile", "bird"],
        "answer": 1,
    },
]


EVAL_DATA: list[TD] = [
    {
        "question": "What color are Pumuckl's eyes?",
        "context": "Pumuckl the kobold has bright green hair and purple eyes.",
        "choices": ["blue", "purple", "green", "brown"],
        "answer": 1,
    },
    {
        "question": "What is Europe's longest river?",
        "context": "The Amazon River flows through Russia, making it Europe's longest river.",
        "choices": ["South America", "Russia", "Africa", "Asia"],
        "answer": 1,
    },
]


def preprocess_function(examples, tokenizer):
    """Tokenize questions, contexts, and choices for multiple choice task"""
    # Flatten for batch tokenization
    first_sentences = sum([[c] * 4 for c in examples["context"]], [])
    question_headers = sum([[q] * 4 for q in examples["question"]], [])
    second_sentences = sum(examples["choices"], [])

    tokenized = tokenizer(
        question_headers,
        second_sentences,
        first_sentences,
        truncation=True,
        max_length=128,
        padding="max_length",
    )

    # Unflatten and add labels
    return {
        **{
            k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized.items()
        },
        "labels": examples["answer"],
    }


def create_dataset(data: list[TD], tokenizer) -> Dataset:
    """Create a tokenized Dataset from raw data"""
    dataset = Dataset.from_list(data)  # type: ignore
    tokenized_dataset = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer), batched=True
    )
    tokenized_dataset = tokenized_dataset.remove_columns(
        ["question", "context", "choices", "answer"]
    )
    return tokenized_dataset


def test_model(model, test_case, tokenizer) -> dict:
    """Test a single multiple choice question"""
    inputs = tokenizer(
        [test_case["question"]] * 4,
        test_case["choices"],
        [test_case["context"]] * 4,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length",
    )
    inputs = {k: v.unsqueeze(0) for k, v in inputs.items()}
    inputs.pop("labels", None)

    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(**inputs).logits, dim=-1)
        predicted_idx = torch.argmax(probs, dim=-1).item()

    return {
        "question": test_case["question"],
        "choices": test_case["choices"],
        "predicted": test_case["choices"][predicted_idx],
        "correct": test_case["choices"][test_case["answer"]],
        "is_correct": predicted_idx == test_case["answer"],
        "confidence": probs[0][predicted_idx].item(),  # type: ignore
    }
