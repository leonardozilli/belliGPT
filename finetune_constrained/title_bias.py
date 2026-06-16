DEFAULT_ENCODER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


class _CosineRelevance:
    mat = None
    index: dict[str, int] = {}
    title_vec = None

    def relevance(self, words: list[str]) -> list[float]:
        """Cosine similarity of each word to the current title."""
        if self.title_vec is None or not words:
            return [0.0] * len(words)
        idx = [self.index[w] for w in words]
        return (self.mat[idx] @ self.title_vec).tolist()


class TitleBias(_CosineRelevance):
    """Input embedding table relevance."""

    def __init__(self, backend, words: list[str]):
        import numpy as np

        self.be = backend
        self.words = list(dict.fromkeys(words))  # unique, order-preserving

        all_ids: list[int] = []
        lengths: list[int] = []
        for w in self.words:
            ids = backend.encode_word(w)
            all_ids.extend(ids)
            lengths.append(len(ids))

        flat = backend.embed_ids(all_ids).astype(np.float32)  # [n_tok, h]
        vecs = np.zeros((len(self.words), flat.shape[1]), dtype=np.float32)
        off = 0
        for i, n in enumerate(lengths):
            if n:
                vecs[i] = flat[off : off + n].mean(0)
            off += n
        norm = np.linalg.norm(vecs, axis=-1, keepdims=True)
        self.mat = vecs / np.clip(norm, 1e-6, None)  # [n_words, h]
        self.index = {w: i for i, w in enumerate(self.words)}
        self.title_vec = None

    def set_title(self, title: str) -> None:
        import numpy as np

        ids = self.be.encode(title, add_special=False)
        if not ids:
            self.title_vec = None
            return
        v = self.be.embed_ids(ids).astype(np.float32).mean(0)
        self.title_vec = v / max(float(np.linalg.norm(v)), 1e-6)


class EncoderTitleBias(_CosineRelevance):
    """Sentence-transformer relevance."""

    def __init__(
        self,
        words: list[str],
        model_name: str = DEFAULT_ENCODER,
        device: str | None = None,
        batch_size: int = 256,
    ):
        import numpy as np
        from sentence_transformers import SentenceTransformer

        self.words = list(dict.fromkeys(words))  # unique, order-preserving
        self.model = SentenceTransformer(model_name, device=device)
        self.mat = self.model.encode(
            self.words,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32)  # [N, h], already unit-norm
        self.index = {w: i for i, w in enumerate(self.words)}
        self.title_vec = None

    def set_title(self, title: str) -> None:
        import numpy as np

        v = self.model.encode(
            [title],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        self.title_vec = v.astype(np.float32)
