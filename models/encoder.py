
from typing import List
import torch
from transformers import AutoModel, AutoTokenizer

class TextEncoder(torch.nn.Module):
    def __init__(self, model_name: str, pooling: str = "mean"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.pooling = pooling

    @torch.inference_mode()
    def encode(self, texts: List[str], device: str = "cpu", batch_size: int = 32):
        self.to(device)
        out_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            toks = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            toks = {k: v.to(device) for k, v in toks.items()}
            outputs = self.model(**toks)
            last_hidden = outputs.last_hidden_state  # (b, t, h)
            if self.pooling == "cls":
                emb = last_hidden[:, 0]
            else:  # mean
                mask = toks["attention_mask"].unsqueeze(-1).float()
                emb = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
            out_embeddings.append(emb.detach().cpu())
        return torch.cat(out_embeddings, dim=0)
