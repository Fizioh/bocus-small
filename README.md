# bocus-small 🧠

bocus-small is an open-source LLM based on Mistral-7B, specialized in teaching programming using the Feynman method. It transforms complex technical concepts into simple, intuitive explanations.

[![Open In Hugging Face](https://img.shields.io/badge/Hugging%20Face-bocus--small-orange)](https://huggingface.co/bocus/bocus-small)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

## 🎯 Features

- Built on Mistral-7B
- Fine-tuned using the Feynman teaching method
- Optimized for explaining programming concepts
- Reduced model size through 4-bit quantization 
- LoRA approach for efficient adaptation

## 🚀 Installation

### Using pip

```bash
# Install required packages
pip install transformers torch accelerate

# Optional: Install for GPU support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Using Docker

```bash
# Build the Docker image
docker build -t bocus-small .

# Run the container
docker run -p 8000:8000 bocus-small
```

## 💻 Usage

### Basic Usage

```python
from transformers import pipeline

# Initialize the model
generator = pipeline(
    'text-generation',
    model='bocus/bocus-small',
    device_map="auto"  # Uses GPU if available
)

# Simple example
response = generator(
    "Explain Python functions like I'm five years old",
    max_length=200,
    temperature=0.7,
    top_p=0.95
)
print(response[0]['generated_text'])
```

### Advanced Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    'bocus/bocus-small',
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained('bocus/bocus-small')

# Custom generation function
def generate_explanation(prompt, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=0.7,
        top_p=0.95,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## 🌟 Examples

### Teaching Basic Concepts

```python
# Example 1: Variables
prompt = """
Explain Python variables using the Feynman method.
Break it down into simple terms.
"""
print(generate_explanation(prompt))

# Example 2: Functions with Parameters
prompt = """
Explain function parameters in Python using a pizza-making analogy.
Make it simple for beginners.
"""
print(generate_explanation(prompt))
```

### Interactive Learning Session

```python
def interactive_learning_session():
    topics = {
        "1": "variables",
        "2": "functions",
        "3": "loops",
        "4": "conditionals"
    }
    
    print("Choose a topic to learn:")
    for key, value in topics.items():
        print(f"{key}: {value}")
    
    choice = input("Enter number (1-4): ")
    topic = topics.get(choice)
    
    if topic:
        prompt = f"Explain Python {topic} in simple terms with examples"
        print("\nGenerating explanation...\n")
        print(generate_explanation(prompt))
    else:
        print("Invalid choice!")

# Run the session
interactive_learning_session()
```

### API Integration

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ExplanationRequest(BaseModel):
    topic: str
    difficulty: str = "beginner"
    max_length: int = 200

@app.post("/explain")
async def explain_topic(request: ExplanationRequest):
    prompt = f"Explain {request.topic} in Python for a {request.difficulty} programmer"
    explanation = generate_explanation(prompt, request.max_length)
    return {"explanation": explanation}
```

## 🛠️ Technical Details

```python
# Model Configuration
model_config = {
    "base_model": "mistralai/Mistral-7B-v0.1",
    "context_window": 8192,
    "quantization": "4-bit",
    "architecture": {
        "type": "decoder_only",
        "attention": "multi_head",
        "layers": 32
    },
    "fine_tuning": {
        "method": "LoRA",
        "parameters": {
            "r": 8,
            "alpha": 16,
            "dropout": 0.05
        }
    }
}
```

## 🔍 Performance Monitoring

```python
def benchmark_model(model, tokenizer, test_cases):
    results = []
    for prompt in test_cases:
        start_time = time.time()
        output = generate_explanation(prompt)
        generation_time = time.time() - start_time
        
        results.append({
            "prompt": prompt,
            "time": generation_time,
            "tokens": len(tokenizer.encode(output))
        })
    return results
```

Pour plus d'exemples et de documentation détaillée, visitez notre [documentation complète](https://docs.bocus.ai).

📚 Key Use Cases


Teaching programming fundamentals
Simplifying complex coding concepts
Creating intuitive analogies for technical topics
Supporting self-paced learning
Complementing traditional programming education

🛠️ Technical Details

Base Model: Mistral-7B
Training Method: LoRA fine-tuning
Context Window: 8k tokens
Quantization: 4-bit
License: Apache 2.0

🤝 Contributing
We welcome contributions! Please check our Contributing Guidelines before submitting pull requests.
⚠️ Limitations

Specialized for programming concepts
May not be suitable for advanced technical discussions
Best used as a teaching assistant rather than primary instructor
Limited to text-based interactions

🔗 Related Projects

Bocus - Main project
Mistral-7B - Base model

📝 Citation
@software{bocus_small_2024,
  author = {Benosmane, Yassine},
  title = {bocus-small: Teaching-Focused LLM},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/bocus/bocus-small}
}
📄  License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

Developed with ❤️ by the [Bocus AI](https://bocus.ai) Team