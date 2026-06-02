# belliGPT

This repository hosts experiments in training custom transformer models to generate poetry in the style of Giuseppe Gioachino Belli.

Different tokenization and training strategies are implemented:
1. [Character-level](./char/): a GPT-style transformer trained using character-level tokenization.
2. [Syllable-level](./syllable/): the same GPT-style architecture, trained using a custom syllable tokenizer.
3. [Subword-level](./subw/): the same GPT-style architecture, trained using custom subword (BPE, Unigram) tokenizers.
4. [LLM Fine-Tuning](./finetune/): Larger pre-trained language models fine-tuned on the sonnets using LoRA adapters.

---

## Dataset
Giuseppe Gioachino Belli (1791‒1863) wrote 2279 sonnets portraying everyday life and customs of 19th-century Rome, using almost entirely the local Romanesco dialect. The main dataset used for training consists of this collection of texts.

## Model Architecture
The custom models use a decoder-only transformer [architecture](./common/model.py), inspired by [nanoGPT](https://github.com/karpathy/nanoGPT)'s implementation of GPT-2, modernized to use RMSNorm, RoPE, SwiGLU and residual scaling.

## Sampling CLI

```bash
python sample.py \
  --model <checkpoint-file> \
  --tokenizer-path <tokenizer-vocab-file> \
  --temperature 0.7 \
  --top-k 40 \
  --top-p 0.9 \
  --max-new-tokens 800
```

Arguments:

- `--checkpoint` (required): checkpoint file or a directory containing a `.pt` file.
- `--tokenizer`: tokenizer file path (`vocab.json`, `tokenizer.json` or `.model`).
- `--tokenizer-type`: `auto` (default), `char`, `bpe`, `unigram`
  - In `auto` mode, `vocab.json` maps to character-level, `.model` maps to SentencePiece Unigram, otherwise it defaults to BPE-style JSON tokenizers.
- `--prompt`: initial text for generation. Defaults to the tokenizer's `BOS` token if available, otherwise an empty string.
- `--max-new-tokens`: number of tokens to generate (default `256`).
- `--temperature` / `-t`: sampling temperature (default `1.0`).
- `--top-k`: top-k sampling cutoff (default `0`, meaning disabled).
- `--top-p`: nucleus sampling cutoff (default `0.0`, meaning disabled).
- `--seed`: optional random seed.
- `--cpu`: force CPU inference.
- `--skip-special-tokens`: omit special tokens during decode.

## Eval CLI

```bash
python eval.py \
  --checkpoint <checkpoint-file> \
  --tokenizer-path <tokenizer-vocab-file> \
  --tokenizer-type char \
  --num-samples 30 \
  --temperature 0.8 \
  --top-k 40 \
  --top-p 0.9
```

Arguments:

- `--checkpoint` (required): path to a `.pt` checkpoint file.
- `--tokenizer` (required): tokenizer file path (`vocab.json`, `tokenizer.json` or `.model`).
- `--tokenizer-type` (required): `char`, `bpe`, or `unigram`.
- `--num-samples`: number of generated samples to evaluate (default `10`).
- `--temperature`: generation temperature (default `0.8`).
- `--top-k`: top-k sampling cutoff (default `40`).
- `--top-p`: nucleus sampling cutoff (default `0.9`).
- `--strict`: if enabled, only count lines as valid if they have exactly 11 syllables, otherwise 10 and 12 syllables are also counted as valid (default `False`).
- `--silent`: suppress per-sample generated text and print only metrics.
