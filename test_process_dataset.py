from datasets import load_dataset
import pathlib, pandas as pd, random, textwrap

def demo_beir(name, split="test"):
    folder = pathlib.Path("data") / name          # where you unzipped the BEIR set
    corpus_file  = str(folder / "corpus.jsonl")   # ← convert Path → str
    queries_file = str(folder / "queries.jsonl")
    qrels_file   = folder / f"qrels/{split}.tsv"

    corpus  = load_dataset("json", data_files=corpus_file,  split="train")
    queries = load_dataset("json", data_files=queries_file, split="train")
    qrels   = pd.read_csv(qrels_file, names=["qid", "docid", "score"], sep="\t")

    # --- print three random examples ------------------------------
    lookup = {row["_id"]: row["text"] for row in corpus.select(range(50_000))}
    print(f"\n=== {name.upper()} ({split}) ===")
    for _, row in qrels.sample(3, random_state=0).iterrows():
        qtxt = queries.filter(lambda r, qid=row.qid: r["_id"] == qid)[0]["text"]
        dtxt = lookup.get(row.docid, "[doc outside first 50k]")
        print(f"\nQ:  {qtxt}\n→ doc {row.docid}\n"
              f"{textwrap.shorten(dtxt, width=280)}\n")

for ds in ["nq"]:
    demo_beir(ds, split="test")

# https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip
