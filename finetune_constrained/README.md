# constrained meter/rhyme decoding

The fine-tuned model adheres to sonnet structure (14 lines, 4/4/3/3 stanzas) and has a good-enough meter, but cannot reliably rhyme, mostly because of problems related to subword tokenization.

The code in this folder implements a constrained decoding scheme that forces the rhyme words at generation time.
For each line, the decoder:

1. Forces an `[X]` rhyme marker token, then lets the model generate the line body.
2. Once the line is long enough (~11 syllables), the decoder selects the final closing word from a rhyme lexicon built from the training corpus. The lexicon is filtered so that the chosen word:
    - belongs to the same rhyme class required by the target rhyme scheme,
    - has a syllable count that makes the line body + word cover 11 syllables,
    - has the highest mean token log-prob among the candidates.

### Title Bias

To help mitigate thematic drift caused by the purely structural constraints, the decoder can optionally bias the rhyme-word selection with a relevance score to the title via the `--title-bias λ` argument. With `λ=0` (the default when omitted) the final pick is simply the highest-scoring rhyme word; with `λ>0`, the final pick becomes `z(model_logprob) + λ·z(relevance)`, where relevance is cosine similarity between the candidate word and the title.

Two different embedding sources can be selected to calculate semantic relevance, using `--relevance`:
- `embed` (default): embeddings are retrieved from the model's own internal input-embedding table by mean-pooling the vectors of a word's constituent tokens.
- `encoder`: a sentence-transformer (default is `paraphrase-multilingual-MiniLM-L12-v2`) is used to encode the title and candidate words.


## Usage

```bash
uv run python -m finetune_constrained.sample \
    --backend llamacpp \
    [-n <int>] [--temperature <float>] [--top-p <float>] \
    [--seed <int>] [--title <string>] \
    [--adapter <path>] [--model <string>] [--dataset <string>] \
    [--gguf <path>] [--lora-gguf <path>] \
    [--lexicon-dir <path>] [--scheme <string>] [--min-class-size <int>] \
    [--strict-meter] [--repetition-penalty <float>] \
    [--title-bias <float>] [--relevance embed|encoder] [--encoder-model <string>]
```

## Example generations

```bash
uv run python -m finetune_constrained.sample --backend llamacpp \
    --title "Er giorno der giudizzio"
```

```
[A] C’è stato un giorno che l’inferno
[B] se venne a ccaccià ggiú, stammatina
[B] er zu’ Re; e nun fussimo quajjottina
[A] de scedentería pe Padr’Eterno

[A] Se sò vvienute du’ cuel’inferno
[B] da drent’in zur palazzo ggiacubbina
[B] cor fischio ppieno in faccia l’indovina
[A] a ttutti quelli che ffoderad’inverno

[C] Cacciorno via tutti poverello
[D] e jje bbuttonarono addosso
[C] le chiappe co li carci ggiucarello

[D] Pe cquesto, er giorno der gallo-rosso
[C] fu un gran dí de fà nnotte: cciarvello
[D] chi llòro Cristo se mmezzo-grosso

[eval] 14_lines=True structure=True hendec=14/14 rhyme_meter=True valid_sonnet=True  rhyme adherence=14/14
```

---

```bash
uv run python -m finetune_constrained.sample --backend llamacpp \
    --title "Li soprani der monno vecchio"
```

```
[A] «Come! er Re nun ha ppiú cojjoni
[B] da poté mmannajje stammatina
[B] ’ggni foco e ll’acqua ggiacubbina
[A] pe ttirà ssu cquer zanta ffuntanoni

[A] che jje bbuggiarassi bbuggiaroni
[B] tutte le scittà dde Tterrascina
[B] senza crepà mmanco l’indovina
[A] co li confetti in mano bbirboni

[C] nun ze ne trova un antro ggnente
[D] de tutto quanto è ppalazzo, spedale
[C] e cchiesa sce pò ffa cquarc’accidente

[D] pe ffallo arreggè ssottol’alloggio?...». l’ale
[C] da bburini je pijjeno l’accidente
[D] a ttanti poveretti prencipale

[eval] 14_lines=True structure=True hendec=14/14 rhyme_meter=True valid_sonnet=True  rhyme adherence=14/14
```

---

```bash
uv run python -m finetune_constrained.sample --backend llamacpp \
    --title "La bbona famijja"
```

```
[A] Si ttu ffijjolo, fijjo cojjone
[B] de quer paino puttano Grigorio
[B] c’ha er zito e la poca ggiudizzio
[A] dde mettese a ffà le divozzione

[A] nun je credi mmai? Lui bbenedizzione
[B] te l’aripara in todescheria: Iddio
[B] se ne fa un boccione nescessario
[A] pe ddamme er ciucciaturuccello. ccojjone

[C] Lo sai ché jje disse stammatina
[D] quer regazzino che bbenedetto
[C] è dda la Madonna, passa-e-ccammina

[D] e nnun ha ggnisuno moccoletto
[C] quanno arza le mano ggiacubbina
[D] a pparlà co li Santi accapalletto

[eval] 14_lines=True structure=True hendec=14/14 rhyme_meter=True valid_sonnet=True  rhyme adherence=14/14
```

---

```bash
uv run python -m finetune_constrained.sample --backend llamacpp \
    --title "Er caffettiere filosofo"
```

```
[A] «Sò stato», ddich’io, «pe stammatina
[B] in pratea ar caffè Grigorio
[B] a ffà bbeve un torroncino nescessario
[A] pe ppagamme er pranzo quajjottina

[A] e ggià cche stavo anniscosto ccristallina
[B] me sò intesi da lontano dich’io
[B] la bbella notizia Ssant’Ustacchio
[A] che llí, ssor Cassibbraccio ggiacubbina

[C] jerzera ha vvienuto l’inferno
[D] a pparlà co Ppapa poverello
[C] de ste su’ duzzine Padr’Eterno

[D] e ddisce ché nnun ciabbita bberzitello
[C] ma spregità un zalario d’inferno
[D] pe sserví la faccia ggiucarello

[eval] 14_lines=True structure=True hendec=14/14 rhyme_meter=True valid_sonnet=True  rhyme adherence=14/14
```

---

```bash
uv run python -m finetune_constrained.sample --backend llamacpp \
    --title "Li pajacci"
```

```
[A] Nun zò fforze li frati, stammatina
[B] er Papa l’ha ffatti accidente
[B] d’annàsse a ccercalli nnaturarmente
[A] in ner paese der passa-e-ccammina

[A] Perché accusí ssai che ggiacubbina
[B] de maggnatora è stata l’accidente
[B] sta bbella porca bbuggiarona innoscente
[A] da metteje le man’in quajjottina

[C] Cristo, te pare che poverello
[D] cià ffatte appiccicare l’inferno
[C] pe nnun fasse trovà bberzitello

[D] co la passione der Padr’Eterno
[C] cor una mano addrittura l’uscello
[D] e llí ccome un Purcinella d’inferno

[eval] 14_lines=True structure=True hendec=14/14 rhyme_meter=True valid_sonnet=True  rhyme adherence=14/14
```

## Title bias

```bash
uv run python -m finetune_constrained.sample --backend llamacpp \
    --title "Li soprani der monno vecchio" --title-bias 1.5
```

```
[A] Sori cazzacci, sò zempiterno
[B] li sovrani de l’anime monno
[B] che nnun vengheno a ffà l’annisconno
[A] perché sta cannela Padr’Eterno

[A] ggira e sputa sempre Liunferno
[B] in ner mazzo cor zu’ l’arivònno
[B] senz’avviso: si er diavolo zonno
[A] vôi falla, ppuro la padreterno

[C] je ne fa un callo de Monziggnore
[D] pe ggabbà ddoppo a stii ddomani
[C] che llui nassce vivo mmonziggnore

[D] e sse va in paradiso, cciarlatani
[C] come li cani se Monzignore
[D] caccieno via da casa sovrani

[eval] 14_lines=True structure=True hendec=14/14 rhyme_meter=True valid_sonnet=True  rhyme adherence=14/14
```
---

```bash
uv run python -m finetune_constrained.sample --backend llamacpp \
    --title "La vita dell'omo" --title-bias 1.5
```

```
[A] A tté nnun me t’increschi, l’amore
[B] che sse fa ppe cchi lo ggiacubbini
[B] nò in camísce e ppoi alletto l’orecchini
[A] de la ggente cojjona l’esattore

[A] è una frebbe un poppettina l’ore
[B] com’er zecchio d’una l’indovini
[B] senza poi tanti fatti l’inchini
[A] a le Madonna, li Zarvatore

[C] Lassú ffanno er zu’ dovere; l’inferno
[D] va avanti ppien der busciabbòttieri: lòro
[C] quer giorno ché ssenteno l’inverno

[D] nun zapennene ggnisuno dell’oro
[C] che cqua vve fanno tirà l’interno
[D] pe un ber giro de chiave Mmonte-d’oro

[eval] 14_lines=True structure=True hendec=14/14 rhyme_meter=True valid_sonnet=True  rhyme adherence=14/14
```
---

```bash
uv run python -m finetune_constrained.sample --backend llamacpp \
    --title "Er caffettiere filosofo" --title-bias 1.5
```

```
[A] Eppuro er barbieretto cavajjere
[B] se sa da du’ ggiorni poveretto
[B] ch’è stato un incanto cc’all’Angeletto
[A] j’ha spiegato la sciarlataria. barbiere

[A] Je stava a ssapé ddí, caffettiere
[B] si in vita sua nun ha l’ajjetto
[B] de chiamà llui e ddua le Bbariletto
[A] Pe ffurtuna der caffè-latte!... bbarbiere

[C] Ma cce vo un pochino priscipizzio
[D] pe capì che ggià er zor ggiacubbini
[C] ciarigallava a Ssant’Andrea dich’io

[D] lúscí, ssai ché tteste l’indovini
[C] sò ppe nnun fà ddí: cazzo-lègge? Marforio
[D] nassčiannolo co le Bbarberini

[eval] 14_lines=True structure=True hendec=14/14 rhyme_meter=True valid_sonnet=True  rhyme adherence=14/14
```
---

```bash
uv run python -m finetune_constrained.sample --backend llamacpp \
    --title "Li pajacci" --title-bias 1.5
```

```
[A] Pe ddí la verità, ggiacubbini
[B] de Piazza-Navona priscipizzio
[B] a sto paese io nun pajjaccio
[A] manco pe li cochi tajjolini

[A] Cuann’ero regazzino ggiacobbini
[B] come me chiamavo l’infernaccio
[B] e ccrompiassi drent’a ccatenaccio
[A] in zempiterno, oh bbene paini

[C] Er Papa è un Omo da ssciojje
[D] cuanto er Monziggnor Zaccariaggni jjeri
[C] disse a li frati: «Sò ariccojje

[D] co sta frega de frabbutti cchincajjeri
[C] che ll’hanno ppienzionati pijjassimojje
[D] e ttirassino la bberzajjeri

[eval] 14_lines=True structure=True hendec=14/14 rhyme_meter=True valid_sonnet=True  rhyme adherence=14/14
```


## Limitations / todos
- The rhyme classes are currently derived using the accent-normalized forms of words from the training corpus. This (perhaps coupled with some errors in syllabization) causes some rhymes to sound weird/wrong.
- Some of the generated sonnets are funny, but they don't really make much sense for the most part.
