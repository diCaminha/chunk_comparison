import os, json, math, random
from time import perf_counter
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv

from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

from chunkers import apply_chunker
from metrics  import check_recall, per_example_metrics

from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import wilcoxon

from openai import OpenAI
client = OpenAI()

load_dotenv() 

def build_hotpot_corpus(split="validation"):
    ds = load_dataset("hotpot_qa", "fullwiki", split=split+"[:1%]", trust_remote_code=True)
    docs, qa_pairs = [], []
    for item in ds:
        qid, ctx = item["id"], item["context"]
        iterator = (
            zip(ctx["title"], ctx["sentences"])
            if isinstance(ctx, dict)
            else ((art[0], art[1]) for art in ctx)
        )
        
        for idx, (title, sent_list) in enumerate(iterator):
            docs.append(
                Document(
                    page_content=f"{title}\n\n{' '.join(sent_list)}",
                    metadata={"qid": qid, "title": title, "support_idx": idx},
                )
            )
        qa_pairs.append((item["question"].strip(), item["answer"].strip(), qid))
    return docs, qa_pairs



#  Etapa 1 – Chunk + índice
def store_chunks(docs, method, chunk_size, chunk_overlap, k=5):
    
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1")

    chunks = apply_chunker(docs, method, chunk_size, chunk_overlap)
    
    vectordb  = FAISS.from_documents(chunks, embed)
    
    retriever = vectordb.as_retriever(search_kwargs={"k": k})

    return chunks, retriever


#  Etapa 2 – Recall@K (todas as Q)
def retrieval_metrics(qa_pairs, retriever):
    recalls = []
    for question, answer, _ in qa_pairs:
        docs = retriever.invoke(question)
        recalls.append(check_recall(docs, answer))
    return np.array(recalls)


#  Etapa 3 – EM/F1 no subset
def qa_metrics(sample_pairs, retriever):
    
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=256,
        max_retries=6,
        request_timeout=60,
    )

    SHORT_ANSWER_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a helpful assistant.\n"
        "Using ONLY the context below, answer the question in ONE short phrase. "
        'If the answer is not contained in the context, reply "N/A".\n\n'
        "Context:\n{context}\n\n"
        "Question: {question}\nAnswer:"
    ),
)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": SHORT_ANSWER_PROMPT},
    )

    preds, golds, recalls = [], [], []
    for question, answer, qid in tqdm(sample_pairs, desc="QA subset", leave=False):
        out = qa_chain({"query": question})
        preds.append(out["result"])
        golds.append(answer)
        recalls.append(check_recall(out["source_documents"], qid))

    em_vec, f1_vec = per_example_metrics(preds, golds)
    em_vec = np.array(em_vec)
    f1_vec = np.array(f1_vec)
    return em_vec, f1_vec, np.array(recalls)
   

def mcnemar_p(a, b):
    tbl = [
        [np.sum((a == 1) & (b == 1)), np.sum((a == 1) & (b == 0))],
        [np.sum((a == 0) & (b == 1)), np.sum((a == 0) & (b == 0))],
    ]
    return mcnemar(tbl, exact=False, correction=True).pvalue

def wilcoxon_ci(delta, n_boot=10_000, alpha=0.05, rng=np.random.default_rng(42)):
    boots = [
        rng.choice(delta, size=len(delta), replace=True).mean()
        for _ in range(n_boot)
    ]
    low, high = np.percentile(boots, [alpha/2*100, (1-alpha/2)*100])
    return low, high



if __name__ == "__main__":

    docs, qa_pairs = build_hotpot_corpus("validation")
    print(f"{len(docs)} docs | {len(qa_pairs)} QA pairs")

    # tamanhos de chunk-size e chunk-overlap
    grid = [(48,10), (96,20), (160,32)]
    
    # metoodos de chunking que serão analizados
    methods = ("fixed", "recursive")

    results = defaultdict(dict)

    for chunk_size, overlap in grid:
        for method in methods:

            t0 = perf_counter()
            
            # criando os chunks e salvando no DB vector
            chunks, retriever = store_chunks(
                docs, method, chunk_size, overlap
            )
            
            build_ms = (perf_counter() - t0)*1_000
            print(f"\n{method}-{chunk_size}/{overlap}: "
                  f"chunks={len(chunks)}  build={build_ms:.0f} ms")

            # resgatando as metricas de retriever
            r_vec = retrieval_metrics(qa_pairs, retriever)
            print(f"  Recall@5 = {r_vec.mean():.4f}")

            # execução da requisicao a llm e coleta do QA
            em_vec, f1_vec, rec_sub = qa_metrics(qa_pairs, retriever)
            print(f"  EM = {np.mean(em_vec):.4f}")
            print(f"  F1 = {np.mean(f1_vec):.4f}")

            results[(chunk_size, overlap, method)] = {
                "recall_all": r_vec,
                "em_sub": em_vec,
                "f1_sub": f1_vec,
                "recall_sub": rec_sub,
            }

    for chunk_size, overlap in grid:
        fx = results[(chunk_size, overlap, "fixed")]
        rc = results[(chunk_size, overlap, "recursive")]

        p_em  = mcnemar_p(fx["em_sub"], rc["em_sub"])

        delta = fx["f1_sub"] - rc["f1_sub"]
        _, p_f1 = wilcoxon(delta, alternative="two-sided")
        ci_low, ci_high = wilcoxon_ci(delta)

        print(f"\n### chunk={chunk_size} overlap={overlap}")
        print(f"McNemar-p(EM)   = {p_em:.4g}")
        print(f"Wilcoxon-p(F1)  = {p_f1:.4g}  "
              f"ΔF1 95 % CI = [{ci_low:.3f}; {ci_high:.3f}]")
