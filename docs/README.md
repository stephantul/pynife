# Documentation

This section currently just includes documentation on how to create your own NIFE model.

## Creating a NIFE model

To create a NIFE model, you can run the scripts in `scripts`, or directly use the code from the repository. First, you should create a corpus of embeddings for your embedder. You can also use pre-computed collections of embeddings I created:

* [mixedbread-ai/mxbai-embed-large-v1](https://huggingface.co/collections/stephantulkens/mxbai-large-v1-embedpress)
* [Alibaba-NLP/gte-modernbert-base](https://huggingface.co/collections/stephantulkens/gte-modernbert-embedpress)

Broadly construed, training a NIFE model has 5 separate steps.

### 1. Create a set of embeddings using the teacher

Let's assume we want to create embeddings on [trivia QA](https://huggingface.co/mandarjoshi/trivia_qa), using `mxbai-embed-large-v1` as a teacher.

```python
from datasets import load_dataset
from pynife.distillation.infer import generate_and_save_embeddings
from sentence_transformers import SentenceTransformer

model_name = "mixedbread-ai/mxbai-embed-large-v1"
model = SentenceTransformer(model_name)

dataset_name = "mandarjoshi/trivia_qa"
dataset = load_dataset(dataset_name, "rc", split="train")
dataset_iterator = ({"text": x['question']} for x in dataset)

output_directory = "my-trivia-qa"

generate_and_save_embeddings(
    model=model,
    records=dataset_iterator,
    output_folder=output_directory,
    limit_batches=None,
    batch_size=8,
    save_every=512,
    max_length=512,
    model_name=model_name,
    dataset_name=dataset_name,
    lowercase=False,
    make_greedy=False,
    )

```

This piece of code loads the model, the dataset and then starts inference. Inference takes a while, and will stream snippets to disk as .txt files and torch tensor files. After the whole dataset has been inferenced, the .txt and tensor files are converted into parquet files, and the .txt and torch tensor files are deleted.

Your dataset will be ready and saved as parquet files in `output_directory`. If you want to upload these, please use the `HfAPI`, not `dataset.push_to_hub`, because we rely on some metadata embedded in the README to infer the base model later on. Note that the dataset iterator can be anything, and does not need to be a Hugging Face dataset. For example, it could also work with a stream from your database.

For a simple inference script with a lot of pre-made datasets, see [the infer_datasets script](../scripts/infer_datasets.py).

### 2. (optional) Expanding a tokenizer

NIFE models work really well if you create a custom tokenizer for your domain. Empirically, it also works really well if you just expand the tokenizer of your teacher model with additional words. We call this _tokenizer expansion_. We have a pre-defined corpus to work on:

```python
from transformers import AutoTokenizer

from datasets import load_dataset
from pynife.tokenizer.expand_tokenizer import expand_tokenizer


dataset = load_dataset("stephantulkens/msmarco-vocab", split="train")
print(dataset.tolist()[:5])
# [{'token': '.', 'frequency': 36174594, 'document_frequency': 8701009},
# {'token': 'the', 'frequency': 28806701, 'document_frequency': 7712172},
# {'token': ',', 'frequency': 25825435, 'document_frequency': 7411743},
# {'token': 'of', 'frequency': 15196930, 'document_frequency': 6562023},
# {'token': 'a', 'frequency': 13702107, 'document_frequency': 6064770},

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# Function expects an iterator over dictionaries with "token" and "frequency" as keys.
new_tokenizer = expand_tokenizer(tokenizer, data, new_vocabulary_size=30000)
new_tokenizer.save_pretrained("my_tokenizer")

```

This will do a couple of things:
1) It will remove all tokens from the original tokenizer that aren't present in your data.
2) It will then add the most frequent tokens until the size of the tokenizer == `new_vocabulary_size`.

This works a lot better than training a tokenizer from scratch on equivalent data. For a runnable version, see [the expand_tokenizer script](../scripts/expand_tokenizer.py).

To get frequency counts, you can use `count_tokens_in_dataset`, as follows:

```python
from datasets import load_dataset, Dataset

from pynife.tokenizer.count_vocabulary import count_tokens_in_dataset

dataset = load_dataset("sentence-transformers/msmarco", "corpus", split="train", streaming=True)
dataset_iterator = (item["passage"] for item in dataset)
counts = count_tokens_in_dataset(dataset_iterator)

# Save the counts as a dataset if you want.
dataset = Dataset.from_list(counts, split="train")
dataset.push_to_hub("my_hub")

```

This dataset can be used directly to expand your tokenizer, above. For a runnable version, see [the create_vocabulary script](../scripts/create_vocabulary.py)

### 3. Train

Given a dataset and optionally a tokenizer, there's two steps to complete for a successful training.

#### 3a Initialize a static model using your teacher

Using *your teacher model*, initialize a static model. For example, when using [`mixedbread-ai/mxbai-embed-large-v1`](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1):

```python
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from pynife.initialization import initialize_from_model

teacher = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
# The tokenizer you trained in step 2. or an off-the-shelf tokenizer.
tokenizer = AutoTokenizer.from_pretrained("my_tokenizer")
model = initialize_from_model(teacher, tokenizer)

```

#### 3b Actually train

Now you can train, just like a regular sentence transformer. In my experiments, I found that using the cosine distance as a loss function was superior to using MSE, so I recommend using that, find it in `pynife.losses`. In addition, I also recommend using [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147). There's a bunch of helper functions in `pynife` to make training easier. In general, I recommend using hyperparameters like the following:

* `batch_size`: 128
* `learning rate`: 0.01
* `scheduler`: "cosine_warmup_with_min_lr"
* `warmup_ratio`: 0.1
* `weight_decay`: 0.01
* `epochs`: 5

It can be tempting to move to very high batch sizes, but this has a very large detrimental effect on performance, even with higher learning rates. As a consequence, GPU usage during training is actually pretty low, because there's very little actual computation happening. For a complete runnable training loop, including model initialization, see [the training script](../scripts/experiment_distillation.py).

```python
from pynife.losses import CosineLoss
from pynife.data import get_datasets

# Fill with datasets you trained yourself.
datasets_you_made = [""]
train_dataset = get_datasets(datasets_you_made)

# Model is initialized in step 3a.
loss = CosineLoss(model=model)

# Train as usual.

```

This will train a model and report the result to wandb. The `experiment_distillation` script is otherwise completely the same as a regular sentence transformers training loop, so there's very little actual code involved.
