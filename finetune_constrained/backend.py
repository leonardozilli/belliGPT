class HFBackend:
    """unsloth/transformers backend."""

    def __init__(self, model, tok, chunk: int = 12):
        self.model = model
        self.tok = tok
        self.chunk = chunk
        self.special_ids = set(tok.all_special_ids)

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        return self.tok.encode(text, add_special_tokens=add_special)

    def encode_word(self, word: str) -> list[int]:
        return self.tok.encode(" " + word, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids)

    def is_word_start(self, token_id: int) -> bool:
        piece = self.tok.convert_ids_to_tokens(token_id)
        return isinstance(piece, str) and piece.startswith("▁")

    def forward_last(self, context: list[int]):
        import torch

        ids = torch.tensor([context], device=self.model.device)
        with torch.no_grad():
            out = self.model(input_ids=ids, use_cache=False)
        return out.logits[0, -1]

    def embed_ids(self, ids: list[int]):
        import numpy as np
        import torch

        h = self.model.config.hidden_size
        if not ids:
            return np.zeros((0, h), dtype=np.float32)
        with torch.no_grad():
            e = self.model.get_input_embeddings()(
                torch.tensor(ids, device=self.model.device)
            )
        return e.float().cpu().numpy()

    def score_candidates(
        self, prefix_ids: list[int], cand_lists: list[list[int]]
    ) -> list[float]:
        import torch

        pad_id = self.tok.pad_token_id
        if pad_id is None:
            pad_id = self.tok.eos_token_id or 0
        plen = len(prefix_ids)
        scores: list[float] = []
        for start in range(0, len(cand_lists), self.chunk):
            batch = cand_lists[start : start + self.chunk]
            maxc = max(len(c) for c in batch)
            rows, masks = [], []
            for c in batch:
                pad = maxc - len(c)
                rows.append(prefix_ids + c + [pad_id] * pad)
                masks.append([1] * (plen + len(c)) + [0] * pad)
            ids = torch.tensor(rows, device=self.model.device)
            am = torch.tensor(masks, device=self.model.device)
            with torch.no_grad():
                logits = self.model(
                    input_ids=ids, attention_mask=am, use_cache=False
                ).logits
                logp = torch.log_softmax(logits.float(), dim=-1)
            for r, c in enumerate(batch):
                total = 0.0
                for j, tid in enumerate(c):
                    total += float(logp[r, plen + j - 1, tid])
                scores.append(total / len(c))
        return scores


class LlamaCppBackend:
    """GGUF base + LoRA via llama-cpp-python."""

    def __init__(self, llm):
        import numpy as np

        self._np = np
        self.llm = llm
        self.special_ids = {llm.token_eos(), llm.token_bos()}
        self._embd = None

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        return self.llm.tokenize(
            text.encode("utf-8"), add_bos=add_special, special=True
        )

    def encode_word(self, word: str) -> list[int]:
        return self.llm.tokenize(
            (" " + word).encode("utf-8"), add_bos=False, special=False
        )

    def decode(self, ids: list[int]) -> str:
        return self.llm.detokenize(ids).decode("utf-8", "replace")

    def is_word_start(self, token_id: int) -> bool:
        return self.llm._model.token_get_text(token_id).startswith("Ġ")

    def _logits_row(self, pos: int):
        return self._np.asarray(self.llm.scores[pos], dtype=self._np.float32)

    def forward_last(self, context: list[int]):
        import torch

        self.llm.reset()
        self.llm.eval(context)
        row = self._logits_row(self.llm.n_tokens - 1).copy()
        return torch.from_numpy(row)

    def _embd_matrix(self):
        if self._embd is None:
            from gguf import GGUFReader, dequantize

            reader = GGUFReader(self.llm.model_path)
            tensor = next(t for t in reader.tensors if t.name == "token_embd.weight")
            self._embd = dequantize(tensor.data, tensor.tensor_type).astype(
                self._np.float32
            )
        return self._embd

    def embed_ids(self, ids: list[int]):
        mat = self._embd_matrix()
        if not ids:
            return self._np.zeros((0, mat.shape[1]), dtype=self._np.float32)
        return mat[self._np.asarray(ids, dtype=self._np.int64)]

    def score_candidates(
        self, prefix_ids: list[int], cand_lists: list[list[int]]
    ) -> list[float]:
        np = self._np

        def logsoftmax(x):
            x = x - x.max()
            np.exp(x, out=x)
            return np.log(x / x.sum())

        self.llm.reset()
        self.llm.eval(prefix_ids)
        plen = self.llm.n_tokens
        prefix_lp = logsoftmax(self._logits_row(plen - 1).copy())

        scores: list[float] = []
        for c in cand_lists:
            total = float(prefix_lp[c[0]])
            if len(c) > 1:
                self.llm.n_tokens = plen
                self.llm.eval(c[:-1])
                for j in range(1, len(c)):
                    lp = logsoftmax(self._logits_row(plen + j - 1).copy())
                    total += float(lp[c[j]])
            scores.append(total / len(c))
        return scores
