# Swapping in real embeddings (OpenAI / open-source sentence-transformers)

The sandbox this project was built in has no network access to
api.openai.com or huggingface.co, so `pipeline.py` uses two local
TF-IDF+SVD proxy embedders (`OpenSourceProxyEmbedder`,
`ProprietaryProxyEmbedder`). To use real embeddings, implement the same
two-method interface and swap the entry in `EMBEDDERS`:

```python
class BaseEmbedder:
    def fit(self, texts: list[str]): ...
    def embed(self, texts: list[str]) -> np.ndarray: ...
```

### Open-source: sentence-transformers
```python
from sentence_transformers import SentenceTransformer

class MiniLMEmbedder(BaseEmbedder):
    name = "minilm"
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
    def fit(self, texts): pass  # no fitting needed, pretrained
    def embed(self, texts):
        return self.model.encode(texts, normalize_embeddings=True)
```

### Proprietary: OpenAI text-embedding-3-small
```python
from openai import OpenAI
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

class OpenAIEmbedder(BaseEmbedder):
    name = "openai_te3_small"
    def fit(self, texts): pass
    def embed(self, texts):
        resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
        vecs = np.array([d.embedding for d in resp.data])
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
```

Then update:
```python
EMBEDDERS = {"open_source": MiniLMEmbedder, "proprietary": OpenAIEmbedder}
```

No other file needs to change — `VectorStore`, `RagPipeline`, and the
experiment grid consume embedders purely through `fit`/`embed`.
