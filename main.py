# main.py ────────────────────────────────────────────────────────────
import os, shutil, time
from time import perf_counter
from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv

from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

from chunkers import apply_chunker
from metrics  import check_recall, compute_em_f1, report_results

load_dotenv()                                  # OPENAI_API_KEY, etc.

# ────────────────────────────────────────────────────────────────────
#  NEW  DATA   (HotpotQA – fullwiki)
# ────────────────────────────────────────────────────────────────────
def build_hotpot_corpus(
    split: str = "validation",      # "train" ou "validation"
    slice_pct: int | None = 5,
):
    slice_str = f"{split}[:{slice_pct}%]" if slice_pct else split
    print(f"HotpotQA {slice_str}")

    # Mirror sem script: parquet puro
    ds = load_dataset("lucadiliello/hotpotqa", split=slice_str)

    docs, qa_pairs = [], []
    for idx, item in enumerate(ds):
        doc_id = f"{split}_{idx}"
        doc_text = item["context"].strip()          # string grandona (~500‑1000 tokens)
        answer    = item["answers"][0].strip()      # lista -> primeiro span

        docs.append(Document(page_content=doc_text,
                             metadata={"id": doc_id}))
        qa_pairs.append((item["question"].strip(),
                         answer,
                         doc_id))
    return docs, qa_pairs


# ────────────────────────────────────────────────────────────────────
#  PIPELINE (inalterada)
# ────────────────────────────────────────────────────────────────────
def evaluate_chunking(
    method: str,
    docs: list[Document],
    qa_pairs: list[tuple[str, str, str]],
    k: int = 3,
):
    # 1) Chunking -----------------------------------------------------
    t0 = perf_counter()
    chunks = apply_chunker(docs, method)
    chunk_time_ms = (perf_counter() - t0) * 1_000 / len(docs)

    # 2) Embeddings + índice -----------------------------------------
    embed_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    )

    idx_dir = f"./idx_{method}"
    if os.path.exists(idx_dir):
        shutil.rmtree(idx_dir)

    vectordb = Chroma(persist_directory=idx_dir, embedding_function=embed_model)
    for i in range(0, len(chunks), 5_000):
        vectordb.add_documents(chunks[i : i + 5_000])

    retriever = vectordb.as_retriever(search_kwargs={"k": k})

    # 3) LLM (map_rerank) -------------------------------------------
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=1024)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_rerank",
        retriever=retriever,
        return_source_documents=False,
    )

    # 4) Avaliação ---------------------------------------------------
    tot_rec = tot_em = tot_f1 = 0.0
    for q, gold, _ in tqdm(qa_pairs, desc=f"Evaluating {method}"):
        tot_rec += check_recall(retriever.get_relevant_documents(q), gold)
        pred = qa_chain.run(q)
        em, f1 = compute_em_f1([pred], [gold])
        tot_em += em
        tot_f1 += f1

    n = len(qa_pairs)
    report_results(
        method,
        tot_em / n,
        tot_f1 / n,
        (tot_rec / n) * 100,
        chunk_time_ms,
    )


# ────────────────────────────────────────────────────────────────────
#  MAIN
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    docs, qa_pairs = build_hotpot_corpus(split="validation", slice_pct=10)
    print(f"Loaded {len(docs)} docs and {len(qa_pairs)} QA pairs.")

    for method in ["semantic"]:
        evaluate_chunking(method, docs, qa_pairs)
