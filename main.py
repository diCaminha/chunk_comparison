# main.py ────────────────────────────────────────────────────────────
import os, shutil, time
from time import perf_counter
from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv

from langchain.schema import Document
from langchain.prompts import PromptTemplate           #  NEW
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

from chunkers import apply_chunker
from metrics  import check_recall, compute_em_f1, report_results

load_dotenv()                                  # OPENAI_API_KEY, etc.

# ────────────────────────────────────────────────────────────────────
#  Prompt (English, short answer only)
# ────────────────────────────────────────────────────────────────────
SHORT_ANSWER_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful assistant.\n"
        "Using ONLY the context below, answer the question in ONE short phrase. "
        "If the answer is not contained in the context, reply \"N/A\".\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\nAnswer:"
    ),
)

# ────────────────────────────────────────────────────────────────────
#  DATA (HotpotQA – fullwiki)
# ────────────────────────────────────────────────────────────────────
def build_hotpot_corpus(split="validation", slice_pct=5):
    ds = load_dataset("lucadiliello/hotpotqa", split=f"{split}[:{slice_pct}%]")
    docs, qa_pairs = [], []
    for ex_idx, item in enumerate(ds):
        # Split on [TLE] to recover the individual source blocks
        parts = [p.strip() for p in item["context"].split("[TLE]") if p.strip()]

        for part_idx, part in enumerate(parts):
            # optional: strip the [SEP] after the title
            part = part.replace("[SEP]", "", 1).strip()
            doc_id = f"{split}_{ex_idx}_{part_idx}"
            docs.append(Document(page_content=part,
                                 metadata={"id": doc_id,
                                           "qid": ex_idx,
                                           "support_idx": part_idx}))
        qa_pairs.append((item["question"].strip(),
                         item["answers"][0].strip(),
                         ex_idx))           # keep qid -> matches metadata["qid"]
    return docs, qa_pairs

# ────────────────────────────────────────────────────────────────────
#  PIPELINE
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

    # 3) LLM + QA chain ----------------------------------------------
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=256)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",                     
        retriever=retriever,
        return_source_documents=True,           
        chain_type_kwargs={"prompt": SHORT_ANSWER_PROMPT},
    )

    # 4) Avaliação ----------------------------------------------------
    preds, golds, recalls = [], [], []
    for q, gold, _ in tqdm(qa_pairs, desc=f"Evaluating {method}"):
        out       = qa_chain({"query": q})
        resp_ai   = out["result"]
        src_docs  = out["source_documents"]

        preds.append(resp_ai)
        golds.append(gold)
        recalls.append(check_recall(src_docs, gold))

    em, f1 = compute_em_f1(preds, golds)
    recall_k = sum(recalls) / len(recalls) * 100

    report_results(method, em, f1, recall_k, chunk_time_ms)

# ────────────────────────────────────────────────────────────────────
#  MAIN
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    docs, qa_pairs = build_hotpot_corpus(split="validation", slice_pct=6)
    print(f"Loaded {len(docs)} docs and {len(qa_pairs)} QA pairs.")

    for method in ["recursive"]:
    #for method in ["semantic"]:
        evaluate_chunking(method, docs, qa_pairs)
