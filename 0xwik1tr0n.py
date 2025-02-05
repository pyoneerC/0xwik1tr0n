"""
0xwik1tr0n - A Human-Friendly Transformer-Based Character-Level Language Model
Developed by Max Comperatore (maxcomperatutti@gmail.com)
Trained on WikiText-2 using a single NVIDIA GeForce RTX 3050
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.onnx
import torch.optim as optim
import onnxruntime as ort
import uvicorn
from fastapi import FastAPI
from datasets import load_dataset

# =============================================================================
# CONFIGURATION
# =============================================================================
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
NUM_LINES = 5000  # Number of lines from the dataset to use for training
SEQ_LEN = 30  # Length of each training sequence (in characters)
EMBED_SIZE = 256  # Size of the character embeddings
NUM_LAYERS = 4  # Number of transformer encoder layers
NUM_HEADS = 8  # Number of attention heads in each encoder layer
EPOCHS = 10  # Total number of training epochs
BATCH_SIZE = 64  # Batch size for training
LR = 0.002  # Learning rate for the optimizer
ONNX_FILENAME = "bigger_llm.onnx"  # Filename for the exported ONNX model

# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================
print("📥 Loading dataset...")

# Load WikiText-2 (raw version) dataset
dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)
text_data = dataset["train"]["text"][:NUM_LINES]

# Extract all unique characters and create mappings for encoding/decoding
chars = sorted(set("".join(text_data)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for ch, i in char_to_idx.items()}
VOCAB_SIZE = len(chars)  # Total number of unique characters


def encode(text: str) -> list[int]:
    """Convert a text string into a list of integer indices."""
    return [char_to_idx[ch] for ch in text if ch in char_to_idx]


def decode(indices: list[int]) -> str:
    """Convert a list of integer indices back to a text string."""
    return "".join(idx_to_char[i] for i in indices)


# Create training examples using a sliding window over each line
sequences = [encode(line) for line in text_data if len(line) > SEQ_LEN]
train_data = []
for seq in sequences:
    # For every position in the sequence, generate an (input, target) pair
    for i in range(len(seq) - SEQ_LEN):
        input_seq = torch.tensor(seq[i:i + SEQ_LEN])
        target_seq = torch.tensor(seq[i + 1:i + SEQ_LEN + 1])
        train_data.append((input_seq, target_seq))

# Create a DataLoader for efficient batch processing
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)


# =============================================================================
# MODEL DEFINITION
# =============================================================================
class PositionalEncoding(nn.Module):
    """
    Implements positional encoding using sinusoidal functions.
    This adds information about the position of tokens in the sequence.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough 'positional encoding' matrix
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)  # Apply sine to even indices
        pe[:, 1::2] = torch.cos(pos * div_term)  # Apply cosine to odd indices
        self.register_buffer("pe", pe.unsqueeze(0))  # Shape: (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the input tensor.
        :param x: Tensor of shape (batch_size, seq_len, d_model)
        :return: Tensor of the same shape with positional encodings added.
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class BiggerTransformer(nn.Module):
    """
    Transformer-based language model for character-level text generation.
    It consists of an embedding layer, positional encoding, a stack of transformer
    encoder layers, and a final linear projection to predict the next character.
    """

    def __init__(self, vocab_size: int, embed_size: int = EMBED_SIZE,
                 num_heads: int = NUM_HEADS, num_layers: int = NUM_LAYERS):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size)
        # Define a single transformer encoder layer and then stack multiple copies
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(embed_size, vocab_size)  # Project encoder output to vocabulary logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the model.
        :param x: Input tensor of shape (batch_size, seq_len)
        :return: Logits tensor of shape (batch_size, seq_len, vocab_size)
        """
        x = self.embedding(x)  # (batch_size, seq_len, embed_size)
        x = self.pos_encoder(x)  # Add positional encoding
        x = x.transpose(0, 1)  # Transpose to (seq_len, batch_size, embed_size) for transformer
        x = self.transformer(x)  # Transformer encoding
        x = x.transpose(0, 1)  # Transpose back to (batch_size, seq_len, embed_size)
        return self.fc(x)  # Project to vocabulary dimension


# Determine the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiggerTransformer(VOCAB_SIZE).to(device)

# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# =============================================================================
# TRAINING LOOP
# =============================================================================
print(f"⚡ Training on {device}...")
for epoch in range(1, EPOCHS + 1):
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)  # Transfer data to device
        optimizer.zero_grad()  # Reset gradients
        logits = model(x)  # Forward pass: get predictions
        # Compute loss: flatten predictions and targets to calculate cross entropy loss
        loss = loss_fn(logits.view(-1, VOCAB_SIZE), y.view(-1))
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update model parameters
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch}/{EPOCHS} Loss: {avg_loss:.4f}")
print("✅ Training complete!")

# =============================================================================
# EXPORT MODEL TO ONNX
# =============================================================================
print("📦 Exporting model to ONNX format...")
# Create a dummy input tensor for the export process
dummy_input = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN), device=device)
# Export the model with dynamic axes to support variable batch sizes and sequence lengths
torch.onnx.export(
    model.cpu(),  # Ensure model is on CPU
    dummy_input.cpu(),  # Dummy input on CPU
    ONNX_FILENAME,  # Output ONNX file name
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch", 1: "seq_len"},
                  "output": {0: "batch", 1: "seq_len"}}
)
print(f"📦 Model exported to {ONNX_FILENAME}")

# =============================================================================
# FASTAPI DEPLOYMENT FOR REAL-TIME TEXT GENERATION
# =============================================================================
app = FastAPI()
# Load the exported ONNX model using ONNXRuntime
onnx_model = ort.InferenceSession(ONNX_FILENAME)


@app.post("/generate/")
def generate_text(prompt: str, length: int = 20, temperature: float = 1.0) -> dict:
    """
    Generate text based on a given prompt.

    Parameters:
    - prompt: Initial text to start generation.
    - length: Number of characters to generate.
    - temperature: Controls randomness in sampling (higher = more diverse).

    Returns:
    - A dictionary with the key "generated_text" containing the generated string.
    """
    generated = prompt  # Start with the initial prompt
    for _ in range(length):
        # Use the last SEQ_LEN characters as the current input context
        segment = generated[-SEQ_LEN:]
        inp = encode(segment)
        # If input sequence is too short, pad it with zeros (assumed as padding index)
        if len(inp) < SEQ_LEN:
            inp = [0] * (SEQ_LEN - len(inp)) + inp
        # Convert to tensor and add batch dimension
        x = torch.tensor(inp, dtype=torch.long).unsqueeze(0)  # Shape: (1, SEQ_LEN)
        # Run the model inference using ONNXRuntime
        logits = onnx_model.run(None, {"input": x.numpy()})[0]
        # Get the logits for the last character, apply temperature scaling
        logits = logits[0, -1] / temperature
        # Compute softmax probabilities
        probs = np.exp(logits) / np.sum(np.exp(logits))
        # Sample the next character based on probabilities
        next_idx = np.random.choice(len(probs), p=probs)
        generated += idx_to_char[next_idx]
    return {"generated_text": generated}


# =============================================================================
# RUN THE FASTAPI APPLICATION
# =============================================================================
if __name__ == "__main__":
    print("🚀 Starting API at http://127.0.0.1:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
