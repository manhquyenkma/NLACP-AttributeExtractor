"""
cnn_classifier.py
FIX 5 — CNN Relation Classifier (sau khi có ≥200 câu annotated)

Kiến trúc theo Alohaly 2019:
  Input:    [subject + attribute_value + head] → word embeddings
  Conv:     Filter size 2-4, 128 filters → max pooling
  Output:   sigmoid → binary (valid relation: yes/no)

Hai classifier riêng biệt (theo bài báo):
  1. subject_classifier: phân loại subject-attribute pairs
  2. object_classifier:  phân loại object-attribute pairs

Chạy: python src/cnn_classifier.py --train --data dataset/annotated_corpus.json
"""
import os
import sys
import json
import argparse

# ── BƯỚC 1: Cài trước khi dùng ──
# pip install torch

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    
    # Stub classes to prevent NameError when parsing the file
    class nn:
        class Module:
            pass
    class Dataset:
        pass


# =================================================================
# Model Architecture (theo Alohaly 2019, Figure 6)
# =================================================================

class RelationCNN(nn.Module):
    """
    CNN binary classifier cho subject-attribute / object-attribute relation.

    Input:  sequence [E, AV, head(E), head(AV)] → concatenated, padded
    Output: sigmoid score (0.0–1.0) → ≥0.5 nghĩa là valid relation
    """
    def __init__(self, vocab_size, embed_dim=300, num_filters=128, window_sizes=(2, 3, 4)):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, w) for w in window_sizes
        ])
        self.dropout    = nn.Dropout(0.2)
        self.classifier = nn.Linear(num_filters * len(window_sizes), 1)
        self.sigmoid    = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x).permute(0, 2, 1)  # (batch, embed_dim, seq_len)
        pooled = []
        for conv in self.convs:
            h = torch.relu(conv(emb))              # (batch, num_filters, L)
            h = torch.max(h, dim=2).values         # max pooling
            pooled.append(h)
        features = torch.cat(pooled, dim=1)        # (batch, num_filters * n_windows)
        features = self.dropout(features)
        return self.sigmoid(self.classifier(features))


# =================================================================
# Dataset
# =================================================================

class RelationDataset(Dataset):
    """
    Load từ annotated_corpus.json.
    Mỗi sample = (words_tensor, label) với:
      words = [element, attr_value]  (2 từ đơn giản)
      label = 1 (valid=true) hoặc 0 (valid=false)
    """
    def __init__(self, data_path, relation_type="subject", vocab=None):
        with open(data_path, encoding="utf-8") as f:
            corpus = json.load(f)

        self.samples = []
        all_words    = set(["<PAD>", "<UNK>"])

        for item in corpus.get("sentences", corpus.get("policies", corpus)) if isinstance(corpus, dict) else corpus:
            element = item.get("subject" if relation_type == "subject" else "object", "") or ""
            for cand in item.get("candidates", []):
                if cand.get("category") != relation_type:
                    continue
                value = cand.get("modifier", "")
                label = 1 if cand.get("valid", False) else 0
                words = element.lower().split() + value.lower().split()
                self.samples.append((words, label))
                all_words.update(words)

        # Build vocab nếu chưa có
        if vocab is None:
            self.vocab = {w: i for i, w in enumerate(sorted(all_words))}
        else:
            self.vocab = vocab

        self.max_len = max((len(s[0]) for s in self.samples), default=4)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        words, label = self.samples[idx]
        ids = [self.vocab.get(w, self.vocab.get("<UNK>", 1)) for w in words]
        # Padding / truncation
        if len(ids) < self.max_len:
            ids += [0] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(label, dtype=torch.float)


# =================================================================
# Training
# =================================================================

def train(data_path, relation_type="subject", epochs=20, lr=1e-3):
    if not HAS_TORCH:
        print("[ERROR] PyTorch not installed. Run: pip install torch")
        return

    dataset    = RelationDataset(data_path, relation_type)
    loader     = DataLoader(dataset, batch_size=8, shuffle=True)
    vocab_size = len(dataset.vocab)
    model      = RelationCNN(vocab_size)
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr)
    criterion  = nn.BCELoss()

    print(f"\nTraining {relation_type} classifier — {len(dataset)} samples, vocab={vocab_size}")

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for x, y in loader:
            optimizer.zero_grad()
            pred = model(x).squeeze(-1) if y.dim() == 0 else model(x).squeeze()
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 5 == 0:
            print(f"  Epoch {epoch:3d}/{epochs}  loss={total_loss/len(loader):.4f}")

    # Lưu model
    os.makedirs("outputs/models", exist_ok=True)
    model_path = f"outputs/models/{relation_type}_classifier.pt"
    torch.save({
        "model_state": model.state_dict(),
        "vocab":       dataset.vocab,
        "max_len":     dataset.max_len
    }, model_path)
    print(f"  Saved: {model_path}")
    return model, dataset.vocab


# =================================================================
# Inference
# =================================================================

def predict(element, attr_value, relation_type="subject"):
    """Dự đoán xem (element, attr_value) có phải valid relation không."""
    if not HAS_TORCH:
        return 0.5, "unknown"

    model_path = f"dataset/{relation_type}_classifier.pt"
    if not os.path.exists(model_path):
        print(f"[WARN] Model not trained yet: {model_path}")
        return 0.5, "not_trained"

    checkpoint = torch.load(model_path, weights_only=True)
    vocab      = checkpoint["vocab"]
    max_len    = checkpoint["max_len"]

    words = element.lower().split() + attr_value.lower().split()
    ids   = [vocab.get(w, vocab.get("<UNK>", 1)) for w in words]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    else:
        ids = ids[:max_len]

    model = RelationCNN(len(vocab))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    with torch.no_grad():
        x     = torch.tensor([ids], dtype=torch.long)
        score = model(x).item()

    label = "VALID" if score >= 0.5 else "NOISE"
    return score, label


# =================================================================
# CLI
# =================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN Relation Classifier")
    parser.add_argument("--train",   action="store_true", help="Train classifier")
    parser.add_argument("--predict", action="store_true", help="Run inference")
    parser.add_argument("--data",    default="dataset/annotated_corpus.json")
    parser.add_argument("--type",    default="subject", choices=["subject", "object"])
    parser.add_argument("--element", default="nurse")
    parser.add_argument("--value",   default="senior")
    args = parser.parse_args()

    if args.train:
        train(args.data, args.type)
    elif args.predict:
        score, label = predict(args.element, args.value, args.type)
        print(f"({args.element}, {args.value}) → {label}  (score={score:.3f})")
    else:
        print("Dùng --train hoặc --predict")
        print("Ví dụ: python src/cnn_classifier.py --train --data dataset/annotated_corpus.json")
