import gradio as gr
import pyarrow.parquet as pq
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer, T5ForConditionalGeneration

PARQUET_PATH = "data/raw/complaint_embeddings.parquet"
parquet_file = pq.ParquetFile(PARQUET_PATH)

dimension = 384
index = faiss.IndexFlatL2(dimension)
chunk_metadata = []

for rg_idx in range(parquet_file.num_row_groups):
    table = parquet_file.read_row_group(rg_idx)
    df_chunk = table.to_pandas()
    embeddings = np.array(df_chunk["embedding"].tolist(), dtype="float32")
    index.add(embeddings)
    for _, row in df_chunk.iterrows():
        meta = row["metadata"]
        chunk_metadata.append({
            "document": row["document"],
            "complaint_id": meta["complaint_id"],
            "product_category": meta["product_category"],
            "company": meta["company"],
        })
    del table, df_chunk, embeddings

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model_llm = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

def retrieve(query, k=5):
    query_vector = embed_model.encode([query]).astype("float32")
    distances, indices = index.search(query_vector, k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        meta = chunk_metadata[idx]
        results.append({
            "distance": float(dist),
            "text": meta["document"],
            "product_category": meta["product_category"],
            "company": meta["company"],
        })
    return results

def generate_answer(question, k=5):
    retrieved_chunks = retrieve(question, k=k)
    context = " ".join([c['text'][:150] for c in retrieved_chunks])
    prompt = f"Based on these customer complaints: {context}\n\nComplete this sentence: Customers are unhappy because"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model_llm.generate(**inputs, max_length=100, min_length=15, num_beams=4)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response, retrieved_chunks
def chat_fn(question):
    if not question.strip():
        return "Please enter a question.", ""
    
    answer, sources = generate_answer(question, k=5)
    
    sources_text = ""
    for i, s in enumerate(sources[:3], 1):
        sources_text += f"**Source {i}** ({s['product_category']} — {s['company']})\n{s['text'][:200]}...\n\n"
    
    return answer, sources_text

def clear_fn():
    return "", "", ""

with gr.Blocks(title="CrediTrust Complaint Assistant") as demo:
    gr.Markdown("# CrediTrust Complaint Assistant")
    gr.Markdown("Ask a question about customer complaints across Credit Cards, Personal Loans, Savings Accounts, and Money Transfers.")
    
    question_input = gr.Textbox(label="Your question", placeholder="e.g. Why are customers unhappy with credit cards?")
    
    with gr.Row():
        submit_btn = gr.Button("Submit", variant="primary")
        clear_btn = gr.Button("Clear")
    
    answer_output = gr.Textbox(label="Answer", lines=4)
    sources_output = gr.Markdown(label="Retrieved Sources")
    
    submit_btn.click(fn=chat_fn, inputs=question_input, outputs=[answer_output, sources_output])
    clear_btn.click(fn=clear_fn, inputs=None, outputs=[question_input, answer_output, sources_output])

if __name__ == "__main__":
    demo.launch()