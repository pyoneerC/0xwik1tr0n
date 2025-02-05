import onnxruntime as ort
import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx
import uvicorn
from fastapi import FastAPI
from datasets import load_dataset
from transformers import AutoTokenizer

# ========== 1️⃣ Load & Preprocess Dataset ==========
print("📥 Loading dataset (WikiText-2)...")
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")["train"]["text"][:50000]

tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Use GPT-2 tokenizer for better word representation

tokenized_data = [tokenizer.encode(line, truncation=True, max_length=30) for line in dataset if len(line.split()) > 5]
if len(tokenized_data) < 1000:
    raise ValueError(f"🚨 Not enough data! Only {len(tokenized_data)} samples found. Increase dataset size.")

VOCAB_SIZE = tokenizer.vocab_size
SEQ_LEN = 30  # Context window size
EMBED_SIZE = 256  # Bigger model size
NUM_LAYERS = 4  # More transformer layers
NUM_HEADS = 8  # More attention heads

# Convert dataset into training pairs
train_data = []
for seq in tokenized_data:
    for i in range(len(seq) - SEQ_LEN):
        train_data.append((seq[i:i + SEQ_LEN], seq[i + 1:i + SEQ_LEN + 1]))

train_data = [(torch.tensor(x), torch.tensor(y)) for x, y in train_data]

# ========== 2️⃣ Optimized Transformer Model ==========
class WordTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=EMBED_SIZE, num_heads=NUM_HEADS, num_layers=NUM_LAYERS):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"⚡ Training on: {device}")

model = WordTransformer(VOCAB_SIZE).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ========== 3️⃣ Efficient Training ==========
print("🔥 Training the model...")
EPOCHS = 100
BATCH_SIZE = 64

train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(EPOCHS):
    total_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs.view(-1, VOCAB_SIZE), targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}/{EPOCHS}, Loss: {total_loss:.4f}")

print("✅ Training complete!")

# ========== 4️⃣ Export Model to ONNX ==========
dummy_input = torch.randint(0, VOCAB_SIZE, (SEQ_LEN, 1)).to(device)
onnx_filename = "optimized_llm.onnx"
torch.onnx.export(model.cpu(), dummy_input.cpu(), onnx_filename, input_names=["input"], output_names=["output"])
print(f"📦 Model exported to {onnx_filename}")

# ========== 5️⃣ Deploy API with FastAPI ==========
app = FastAPI()
onnx_model = ort.InferenceSession(onnx_filename)

@app.post("/generate/")
def generate_text(prompt: str, length=20):
    generated = prompt
    try:
        for _ in range(length):
            prompt_encoded = tokenizer.encode(generated[-SEQ_LEN:], truncation=True, max_length=SEQ_LEN)
            input_tensor = torch.tensor(prompt_encoded).unsqueeze(1)
            result = onnx_model.run(None, {"input": input_tensor.numpy()})[0]
            next_token_idx = result.argmax(axis=-1)[-1]
            generated += tokenizer.decode([next_token_idx])
        return {"generated_text": generated}
    except Exception as e:
        return {"error": str(e)}

# ========== 6️⃣ Start API Server ==========
if __name__ == "__main__":
    print("🚀 Starting API at http://127.0.0.1:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
