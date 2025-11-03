# src/ref/lex_dpr/cli/main.py
import typer
from ..train.bi import train_bi
from ..embed.build_index import build_faiss

app = typer.Typer(add_completion=False)

@app.command("train")
def cli_train(
    corpus: str = typer.Option(..., help="merged_corpus.jsonl"),
    pairs: str = typer.Option(..., help="pairs_train.jsonl"),
    out_dir: str = typer.Option("checkpoint/lexdpr", help="output dir"),
    model: str = typer.Option("sentence-transformers/all-MiniLM-L6-v2"),
    epochs: int = 3,
    batch_size: int = 64,
    lr: float = 2e-5,
    eval_pairs: str = typer.Option("", help="pairs_eval.jsonl"),
    eval_steps: int = 300,
    k_values: str = typer.Option("1,3,5,10", help="e.g. 1,3,5,10"),
    seed: int = 42,
    tb_logdir: str = typer.Option("", help="TensorBoard logdir")
):
    ks = [int(x) for x in k_values.split(",") if x]
    train_bi(
        corpus_path=corpus, pairs_path=pairs, out_dir=out_dir,
        model_name=model, epochs=epochs, batch_size=batch_size, lr=lr,
        eval_pairs=eval_pairs or None, eval_steps=eval_steps, k_values=ks,
        seed=seed, tb_logdir=tb_logdir or None
    )

@app.command("embed")
def cli_embed(
    corpus: str = typer.Option(..., help="merged_corpus.jsonl"),
    model_dir: str = typer.Option(..., help="checkpoint/lexdpr/bi_encoder"),
    out_dir: str = typer.Option("index", help="output dir for FAISS"),
    index_name: str = "faiss.index",
    batch_size: int = 256
):
    build_faiss(corpus_path=corpus, model_dir=model_dir, out_dir=out_dir, index_name=index_name, batch_size=batch_size)

def run():
    app()

if __name__ == "__main__":
    run()
