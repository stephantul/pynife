
<h2 align="center">
  <img width="35%" alt="A man shooting a thing to the ground." src="https://github.com/stephantul/pynife/blob/main/assets/william-blake.jpg"><br/>
</h2>
<h1 align="center"> pyNIFE </h1>

<div align="center">
  <h2>
    <a href="https://pypi.org/project/pynife/"><img src="https://img.shields.io/pypi/v/pynife?color=f29bdb" alt="Package version">
</a>
    <a href="https://codecov.io/gh/stephantul/nife" >
      <img src="https://codecov.io/gh/stephantul/nife/graph/badge.svg?token=DD8BK7OZHG"/>
    </a>
    <a href="https://github.com/stephantul/pynife/blob/main/LICENSE">
      <img src="https://img.shields.io/badge/license-MIT-green" alt="License - MIT">
    </a>
</div>


<div align="center">
  <h2>
    <a href="https://huggingface.co/collections/stephantulkens/nife-models"><strong>Models</strong></a> |
    <a href="https://huggingface.co/collections/stephantulkens/nife-data"><strong>Datasets</strong></a> |
    <a href="./benchmarks/"><strong>Benchmarks</strong></a> |
    <a href="./docs/README.md"><strong>Create your own model</strong></a>
</div>

NIFE compresses large embedding models into static, drop-in replacements with up to 900x faster query embedding ([see benchmarks](#benchmarks)).

## Features

- 400-900x faster CPU query embedding
- Fully aligned with their teacher models
- Re-use your existing vector index

## Table of contents

1. [Quickstart](#quickstart)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Rationale](#rationale)

## Introduction

Nearly Inference Free Embedding (NIFE) models are [static embedding](https://huggingface.co/blog/static-embeddings) models that are fully aligned with a much larger model. Because static models are so small and fast, NIFE allows you to:

1. Speed up query time immensely: 900x embed time speed-up on CPU.
2. Get away with using a much smaller memory/compute footprint. Create embeddings in your DB service.
3. Reuse your big model index: Switch dynamically between your big model and the NIFE model.

Some possible use-cases for NIFE include search engines with slow and fast paths, RAGs in agent loops, and on-the-fly document comparisons.

## Quickstart

This snippet loads [`stephantulkens/NIFE-mxbai-embed-large-v1`](https://huggingface.co/stephantulkens/NIFE-mxbai-embed-large-v1), which is aligned with [`mixedbread-ai/mxbai-embed-large-v1`](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1). Use it in any spot where you use `mixedbread-ai/mxbai-embed-large-v1`.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("stephantulkens/NIFE-mxbai-embed-large-v1", device="cpu")
# Loads in 41ms.
query_vec = model.encode(["What is the capital of France?"])
# Embedding a query takes 90.4 microseconds.

big_model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", device="cpu")
# Four cities near France
index_doc = big_model.encode(["Paris is the largest city in France", "Lyon is pretty big", "Antwerp is really great, and in Belgium", "Berlin is pretty gloomy in winter", "France is a country in Europe"])

similarity = model.similarity(query_vec, index_doc)
print(similarity)
# It correctly retrieved the document containing the statement about paris.
# tensor([[0.7065, 0.5012, 0.3596, 0.2765, 0.6648]])

big_model_query_vec = big_model.encode(["What is the capital of France?"])
# Embedding a query takes 68.1 ms (~750 times slower)
similarity = model.similarity(big_model_query_vec, index_doc)
# Compare to the above. Very similar.
# tensor([[0.7460, 0.5301, 0.3816, 0.3423, 0.6692]])

similarity_queries = model.similarity(big_model_query_vec, query_vec)
# The two vectors are very similar.
# tensor([[0.9377]])

```

This snippet is an example of how you could use it. But in reality you should just use it wherever you encode a query using your teacher model. There's no need to keep the teacher in memory. This makes NIFE extremely flexible, because you can decouple the inference model from the indexing model. Because the models load extremely quickly, they can be used in edge environments and one-off things like lambda functions.

## Installation

On [PyPi](https://pypi.org/project/pynife/):

```
pip install pynife
```

## Usage

A NIFE model is just a [sentence transformer](https://github.com/huggingface/sentence-transformers) router model, so you don't need to install `pynife` to use NIFE models. Nevertheless, NIFE contains some helper functions for loading a model trained with NIFE.

Note that with all NIFE models the teacher model is unchanged; so if you have a large set of documents indexed with the teacher model, you can use the NIFE model as a drop-in replacement.

### Standalone

Use just like any other sentence transformer:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("stephantulkens/NIFE-mxbai-embed-large-v1", device="cpu")
X = model.encode(["What is the capital of France?"])
```

### As a router

You can also use the small model and big model together as a single [router](https://sbert.net/docs/package_reference/sentence_transformer/models.html#sentence_transformers.models.Router) using a helper function from `pynife`. This is useful for benchmarking; in production you should probably use the query model by itself.

```python
from pynife import load_as_router

model = load_as_router("stephantulkens/NIFE-mxbai-embed-large-v1")
# Use the fast model
query = model.encode_query("What is the capital of France?")
# Use the slow model
docs = model.encode_document("What is the capital of France?")

print(model.similarity(query, docs))
# Same result as above in the quickstart.
# tensor([[0.9377]])

```

## Pretrained models

I have two pretrained models:

* [`stephantulkens/NIFE-mxbai-embed-large-v1`](https://huggingface.co/stephantulkens/NIFE-mxbai-embed-large-v1): aligned with [`mxbai-embed-large-v1`](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1).
* [`stephantulkens/NIFE-gte-modernbert-base`](https://huggingface.co/stephantulkens/NIFE-gte-modernbert-base): aligned with [`gte-modernbert-base`](https://huggingface.co/Alibaba-NLP/gte-modernbert-base).

## Rationale

For retrieval using dense models, the normal mode of operation is to embed your documents, and put them in some index. Then, using that same model, also embed your queries. In general, larger embedding models are better than smaller models, so you're often better off by making your embedder as large as possible. This however, makes inference more difficult; you need to host a larger model, and embedding queries might take longer.

For sparse models, like [SPLADE](https://arxiv.org/pdf/2107.05720), there is an interesting alternative, which they call doc-SPLADE, and which sentence transformers calls _inference free_. In doc-SPLADE, you only embed using the full model for documents in your index. When querying, you just index the sparse index using the tokenizer.

NIFE is the answer to the question: what would inference free dense retrieval be? It is called _Nearly_ Inference Free, because you still need to have some mapping from tokens to embeddings.

See this table:

|                | Sparse     | Dense                |
|----------------|------------|----------------------|
| Full           | SPLADE     | Sentence transformer |
| Inference free | doc-SPLADE | NIFE                 |

As in doc-SPLADE, you lose performance. No real way about it, but as with other fast models, the gap is smaller than you might think.

## Benchmarks

I benchmark our models on [NanoBEIR](https://huggingface.co/collections/zeta-alpha-ai/nanobeir). I use two trained models:

* [`stephantulkens/NIFE-mxbai-embed-large-v1`](https://huggingface.co/stephantulkens/NIFE-mxbai-embed-large-v1)
* [`stephantulkens/NIFE-gte-modernbert-base`](https://huggingface.co/stephantulkens/NIFE-gte-modernbert-base)

For all models, I report NDC@10 and queries per second. I do this for the student model and teacher model, to show how much performance you lose when switching between them. Detailed benchmark performance can be found in [the benchmarks folder.](./benchmarks/). The query timings were performed on the first 1000 queries of the msmarco dataset, and averaged over 7 runs. The benchmarks were run on an Apple M3 pro.

### `gte-modernbert-base`

|         | Queries per second (CPU) | NDCG@10 |
|---------|--------------------------|---------|
| NIFE    | 71400 (14ms/1k queries)  | 59.2    |
| Teacher | 237 (4210ms/1k queries)  | 66.34   |

### `mxbai-embed-large-v1`

|         | Queries per second (CPU) | NDCG@10 |
|---------|--------------------------|---------|
| NIFE    | 65789 (15ms/1k queries)  | 59.2    |
| Teacher | 108 (9190ms/1k queries)  | 65.6    |

It is interesting that both NIFE models get the same performance, even with different teacher models. This could point towards a ceiling effect, where a certain percentage of queries can be answered correctly by static models, while others require contextualization.

## How does it work?

We use knowledge distillation from an initialized static model to the teacher we want to emulate. Some special things:

1) The static model is initialized directly from the teacher by inferring all tokens in the tokenizer through the whole model. This is similar to how this was done in [model2vec](https://github.com/MinishLab/model2vec), except we skip the PCA and weighting steps.
2) The knowledge distillation is done in cosine space. We don't guarantee any alignment in euclidean space. Using, e.g., MSE or KLDiv between the student and teacher did not work as well.
3) We train a custom tokenizer on our pre-training corpus, which is MsMARCO. This custom tokenizer is based on `bert-base-uncased`, but with a lot of added vocabulary. The models used in NIFE all have around 100k vocabulary size.
4) We perform two stages of training; following [LEAF](https://arxiv.org/pdf/2509.12539), we also train on _queries_. This raises performance considerably, but training on interleaved queries and documents does not work very well. So we first train on a corpus of documents (MsMarco), and then finetune using a lower learning rate on a large selection of queries from a variety of sources.
5) Unlike LEAF, we leave out all instructions from the knowledge distillation process. Static models can't deal with instructions, because there is no interaction between the instruction and other tokens in the document. Instructions can therefore at best be a constant _offset_ of your embedding space. This can be really useful, but not for this specific task.

## Caveats/weaknesses

NIFE can't do the following things:

1) Ignore words based on context: the query "What is the capital of France?" the word "France" will cause documents containing the term "France" to be retrieved. There is no way for the model to attenuate this vector and morph it into the answer ("Paris").
2) Deal with negation: for the same reason as above; there is no interaction between tokens, so the similarity between "Cars that aren't red" and "Cars that are red" will be really high.

## Inquiries

If you think NIFE could be interesting for your business let me know, I am open to consulting jobs regarding training models and fast inference. Just reach out to me [via e-mail](mailto:stephantul@gmail.com).

## License

MIT

## Author

Stéphan Tulkens

## Citation

If you use `pynife` or NIFE models in general, please cite this work as follows:

```bibtex
@software{Tulkens2025pyNIFE,
  author       = {St\'{e}phan Tulkens},
  title        = {pyNIFE: nearly inference free embeddings in python},
  year         = {2025},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.17512919},
  url          = {https://github.com/stephantul/pynife},
  license      = {MIT},
}
```
