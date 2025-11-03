# lex_dpr/train/bi.py
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, InputExample
from ..utils.seed import set_seed
from ..utils.io import read_jsonl
from ..data import load_passages, validate_pairs_exist
from ..eval import build_ir_evaluator
from ..utils.logging import Logger
import math, os

def train_bi(
    corpus_path: str,
    pairs_path: str,
    out_dir: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    epochs: int = 3,
    batch_size: int = 64,
    lr: float = 2e-5,
    eval_pairs: str | None = None,
    eval_steps: int = 300,
    k_values=(1,3,5,10),
    seed: int = 42,
    tb_logdir: str | None = None
):
    set_seed(seed)
    log = Logger(logdir=tb_logdir, jsonl_path=os.path.join(out_dir, "metrics_eval.jsonl"))

    passages = load_passages(corpus_path)
    validate_pairs_exist(passages, pairs_path)

    examples = []
    for row in read_jsonl(pairs_path):
        q = row["query_text"]
        pos = row["positive_passages"][0]
        examples.append(InputExample(texts=[q, passages[pos]["text"]]))

    model = SentenceTransformer(model_name)
    train_loader = DataLoader(examples, batch_size=batch_size, shuffle=True)
    loss = losses.MultipleNegativesRankingLoss(model)

    evaluator = None
    if eval_pairs:
        evaluator, _ = build_ir_evaluator(passages, eval_pairs, read_jsonl, list(k_values))

    steps_per_epoch = max(1, math.ceil(len(examples)/batch_size))
    warmup = max(10, int(steps_per_epoch*epochs*0.1))

    os.makedirs(out_dir, exist_ok=True)
    model.fit(
        train_objectives=[(train_loader, loss)],
        epochs=epochs,
        warmup_steps=warmup,
        optimizer_params={"lr": lr},
        show_progress_bar=True,
        evaluator=evaluator,
        evaluation_steps=eval_steps if evaluator else None
    )
    model.save(os.path.join(out_dir, "bi_encoder"))
    log.info(f"Saved -> {os.path.join(out_dir, 'bi_encoder')}")
