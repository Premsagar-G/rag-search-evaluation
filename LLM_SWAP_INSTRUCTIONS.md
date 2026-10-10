# Swapping in a real hosted LLM

`generate_answer()` in `pipeline.py` currently uses a local extractive
generator (`LocalAnswerer` logic) instead of calling a hosted chat model,
because the sandbox has no OpenAI/Anthropic API key configured. To use a
real model, replace the body of `generate_answer` with a call like:

```python
import anthropic
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

def generate_answer(question, retrieved, prompt_id, memory, scoring_query=None):
    template = PROMPT_TEMPLATES[prompt_id]
    context = "\n\n".join(
        f"[{c.doc_title}#{c.chunk_id.split('::')[-1]}]: {c.text}"
        for c, score in retrieved
    )
    history = memory.context_string()
    system = (
        f"{template['instructions']}\n\nContext:\n{context}\n\n"
        f"Conversation so far:\n{history}"
    )
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=system,
        messages=[{"role": "user", "content": question}],
    )
    answer = resp.content[0].text
    citations = re.findall(r"\[[^\[\]#]+#c\d+\]", answer)
    return {
        "raw_answer": answer, "formatted_answer": answer,
        "citations": citations,
        "used_chunks": [c.chunk_id for c, s in retrieved],
        "retrieval_top_score": retrieved[0][1] if retrieved else 0.0,
        "prompt_template": template["id"],
    }
```

The rest of the pipeline (chunking, embeddings, retrieval, memory,
logging, experiment grid, evaluation) is untouched — only this one
function needs to change to move from the local proxy generator to a
real LLM call.
