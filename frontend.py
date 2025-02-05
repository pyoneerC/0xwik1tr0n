import streamlit as st
import numpy as np
import onnxruntime as ort

# =============================================================================
# LOAD THE ONNX MODEL
# =============================================================================
# Make sure the file "0xwik1tr0n.onnx" is in the same directory as this script.
onnx_model = ort.InferenceSession("0xwik1tr0n.onnx")

# =============================================================================
# VOCABULARY & HELPER FUNCTIONS
# =============================================================================
# IMPORTANT:
# Replace the following sample vocabulary with the actual vocabulary you used during training.
# For example, if you built the vocabulary using:
#   chars = sorted(set("".join(text_data)))
# then ensure this list matches exactly.
chars = sorted(list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,;!?'-\n"))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
VOCAB_SIZE = len(chars)
SEQ_LEN = 30  # Must match the sequence length used during training


def encode(text: str) -> list:
    """Convert text into a list of indices based on char_to_idx.
    Unknown characters are skipped."""
    return [char_to_idx[ch] for ch in text if ch in char_to_idx]


def decode(indices: list) -> str:
    """Convert a list of indices back to text using idx_to_char."""
    return "".join(idx_to_char[i] for i in indices)


# =============================================================================
# TEXT GENERATION FUNCTION
# =============================================================================
def generate_text(prompt: str, length: int = 50, temperature: float = 1.0) -> str:
    """
    Generates text by sampling from the ONNX model.

    Args:
        prompt (str): The starting text.
        length (int): Number of characters to generate.
        temperature (float): Controls randomness (higher is more random).

    Returns:
        str: The generated text.
    """
    generated = prompt
    for _ in range(length):
        # Use the last SEQ_LEN characters as context; pad if necessary.
        segment = generated[-SEQ_LEN:]
        inp = encode(segment)
        if len(inp) < SEQ_LEN:
            inp = [0] * (SEQ_LEN - len(inp)) + inp  # Left-pad with zeros (assumes index 0 is valid)
        # Prepare the input with shape (1, SEQ_LEN)
        x = np.array(inp, dtype=np.int64).reshape(1, SEQ_LEN)
        # Run the model (assumes the input name is "input")
        logits = onnx_model.run(None, {"input": x})[0]  # shape: (1, SEQ_LEN, VOCAB_SIZE)
        # Select the logits for the last token in the sequence
        logits = logits[0, -1, :]
        # Apply temperature scaling
        logits = logits / temperature
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        # Sample a character index based on the probability distribution
        next_idx = np.random.choice(range(VOCAB_SIZE), p=probs)
        generated += idx_to_char[next_idx]
    return generated


# =============================================================================
# STREAMLIT UI
# =============================================================================
st.title("0xwik1tr0n ONNX Text Generator")

# Input widget for the text prompt.
prompt = st.text_input("Enter your prompt", value="Once upon a time")

# Slider for number of characters to generate.
length = st.slider("Number of characters to generate", min_value=10, max_value=500, value=50, step=10)

# Slider for temperature.
temperature = st.slider("Temperature", min_value=0.1, max_value=2.0, value=1.0, step=0.1)

if st.button("Generate"):
    with st.spinner("Generating text..."):
        generated_text = generate_text(prompt, length, temperature)
    st.text_area("Generated Text", value=generated_text, height=200)
