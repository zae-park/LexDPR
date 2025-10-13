import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
from tqdm import tqdm
import numpy as np


class DPRDataset(Dataset):
    def __init__(self, data_path, passage_map):
        """
        data_path: JSONL with {query, positive_ids, hard_negative_ids}
        passage_map: dict {passage_id -> text}
        """
        self.queries = []
        self.pos_passages = []
        self.neg_passages = []

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                q = item["question"]
                pos_ids = item.get("positive_ids", [])
                neg_ids = item.get("hard_negative_ids", [])

                if not pos_ids:
                    continue

                # Take first positive, sample negatives
                pos_text = passage_map.get(pos_ids[0], "")
                if not pos_text:
                    continue

                # Use other passages as negatives if no hard negatives
                if not neg_ids:
                    all_ids = list(passage_map.keys())
                    neg_ids = [pid for pid in all_ids[:5] if pid not in pos_ids]

                neg_texts = [passage_map.get(nid, "") for nid in neg_ids[:4]]
                neg_texts = [t for t in neg_texts if t]

                if len(neg_texts) < 1:
                    continue

                self.queries.append(q)
                self.pos_passages.append(pos_text)
                self.neg_passages.append(neg_texts)

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return {
            "query": self.queries[idx],
            "positive": self.pos_passages[idx],
            "negatives": self.neg_passages[idx],
        }


def collate_fn(batch):
    queries = [b["query"] for b in batch]
    positives = [b["positive"] for b in batch]
    negatives = []
    for b in batch:
        negatives.extend(b["negatives"])
    return queries, positives, negatives


class DPREncoder(torch.nn.Module):
    def __init__(self, model_name, pooling="mean"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.pooling = pooling

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state

        if self.pooling == "cls":
            return hidden[:, 0]
        else:  # mean pooling
            mask = attention_mask.unsqueeze(-1).float()
            return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def compute_loss(q_emb, pos_emb, neg_emb, temp=0.05):
    """
    In-batch negatives contrastive loss
    q_emb: (B, D)
    pos_emb: (B, D)
    neg_emb: (B*N, D) where N is number of negatives per query
    """
    batch_size = q_emb.size(0)

    # Positive scores: (B,)
    pos_scores = (q_emb * pos_emb).sum(-1) / temp

    # Negative scores: (B, B*N)
    neg_scores = torch.matmul(q_emb, neg_emb.T) / temp

    # Also use other positives in batch as negatives
    all_pos_scores = torch.matmul(q_emb, pos_emb.T) / temp  # (B, B)

    # Concatenate: [pos_score, neg_scores, other_pos_scores]
    logits = torch.cat([pos_scores.unsqueeze(1), neg_scores, all_pos_scores], dim=1)

    # Mask out self in other_pos_scores
    mask = torch.eye(batch_size, device=logits.device)
    logits[:, -batch_size:] = logits[:, -batch_size:].masked_fill(mask.bool(), -1e9)

    # Target: first position (positive)
    labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)

    return F.cross_entropy(logits, labels)


def train_epoch(model_q, model_p, dataloader, optimizer, tokenizer, device):
    model_q.train()
    model_p.train()
    total_loss = 0

    for queries, positives, negatives in tqdm(dataloader):
        # Tokenize
        q_tok = tokenizer(
            queries, padding=True, truncation=True, max_length=128, return_tensors="pt"
        ).to(device)
        p_tok = tokenizer(
            positives,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(device)
        n_tok = tokenizer(
            negatives,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(device)

        # Encode
        q_emb = model_q(q_tok["input_ids"], q_tok["attention_mask"])
        p_emb = model_p(p_tok["input_ids"], p_tok["attention_mask"])
        n_emb = model_p(n_tok["input_ids"], n_tok["attention_mask"])

        # Compute loss
        loss = compute_loss(q_emb, p_emb, n_emb)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model_q.parameters()) + list(model_p.parameters()), 1.0
        )
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def load_passage_map(corpus_path):
    """Load passage_id -> text mapping"""
    pmap = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            pmap[item["id"]] = item["text"]
    return pmap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", default="data/queries/train.jsonl")
    parser.add_argument("--corpus", default="data/processed/corpus.jsonl")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--output", default="checkpoint/trained")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument(
        "--shared", action="store_true", help="Share query/passage encoder"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    passage_map = load_passage_map(args.corpus)
    dataset = DPRDataset(args.train_data, passage_map)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )

    print(f"Loaded {len(dataset)} training examples")

    # Models
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model_q = DPREncoder(args.model).to(device)

    if args.shared:
        model_p = model_q  # Share weights
    else:
        model_p = DPREncoder(args.model).to(device)

    # Optimizer
    params = list(model_q.parameters())
    if not args.shared:
        params += list(model_p.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    # Training loop
    for epoch in range(args.epochs):
        loss = train_epoch(model_q, model_p, dataloader, optimizer, tokenizer, device)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss:.4f}")

    # Save
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    model_q.model.save_pretrained(output_path / "query_encoder")
    tokenizer.save_pretrained(output_path / "query_encoder")

    if not args.shared:
        model_p.model.save_pretrained(output_path / "passage_encoder")
        tokenizer.save_pretrained(output_path / "passage_encoder")

    print(f"Models saved to {output_path}")


if __name__ == "__main__":
    main()
