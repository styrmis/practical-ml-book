# Using Transformers

In order to do something practical with transformers we will need:

- A tokenizer, to take our input sentences and turn them into tokens and then into numerically-encoded inputs (token ids)
- The transformer model, which may be
  - **Encoder only:** for classification, Named Entity Recognition tasks,
  - **Decoder only:** for text generation tasks,
  - **Encoder-decoder (sequence-to-sequence):** for translation and summarisation tasks.
- A postprocessing step which will use a tokenizer to take the numerical outputs and map them back to tokens and then to human-readable words and sentences.

In practice, the tokenizer at each end is the same implementation.

## The HuggingFace pipeline

If a model and tokenizer are implemented within the HuggingFace transformers ecosystem then they can be used in a very straightforward manner:

```python
from transformers import pipeline

sentiment_classifier = pipeline("sentiment-analysis")

sentiment_classifier(
    [
        "This book is really helpful",
        "I was disappointed with this book"
    ]
)
```

As we didn't specify a model, it defaults (at time of writing) to [`distilbert/distilbert-base-uncased-finetuned-sst-2-english`](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) and gives us the following output:

```
[{'label': 'POSITIVE', 'score': 0.9996920824050903},
 {'label': 'NEGATIVE', 'score': 0.9997894167900085}]
```

### Structure of the pipeline

As mentioned earlier, the pipeline can be divided into three main steps: tokenizing, model inference, and then a postprocessing step:

![](2024-07-14-transformer-pipeline-overview.png)

Looking at these steps in detail:

We start with some **raw text*, e.g. "This book is amazing!". The tokenizer maps words (and/or sub-words) to ids which represent tokens in our model's vocabulary. Common words like 'this' will likely be intact—in other cases a word may be represented by more than one token.

We give these numerical ids to the model as input, and get back values called [logits](#what-are-logits). As these logits can be considered to be log-odds values, we reverse the application of log by exponentiation via the sigmoid function. In doing so we have converted the logit values into _predictions_, probabilities in the range `[0, 1]`$.

![](2024-07-14-transformers-tokenizer.png)

In practice, we can apply this process to some raw text using the HuggingFace `AutoTokenizer` class:

```python
from transformers import AutoTokenizer

# The default, as of June 2024
model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer("This book is great!")
```

Which returns our token ids:

```python
{'input_ids': [101, 2023, 2338, 2003, 2307, 999, 102], 'attention_mask': [1, 1, 1, 1, 1, 1, 1]}
```

{aside}
### HuggingFace Auto models

When it comes to using these models in practice, Auto models help to cut out a wide range of possible implementation errors that could otherwise creep in. By using an `AutoModel` (along with `AutoConfig`, `AutoTokenizer`) you will ensure that everything is aligned, e.g. the tokenizer will be using the precise version that was used in the pretraining of the model, ensuring that the token ids match. It also ensures that the model architecture is configured correctly, amongst many other details which are handled for you automatically.
{/aside}

In order to pass this to a transformer model we will further need to convert the input ids into a [tensor](#about-tensors). We will also apply **padding** and **truncation**, which we will explore shortly.

For now, let's transform the input such that it is ready to pass to the model. Note that we are returning PyTorch tensors (`torch.Tensor`)[^return_tensors]:

```python
tokenizer("This book is great!", padding=True, truncation=True, return_tensors="pt")
```

[^return_tensors]: Setting this to `pt` will return PyTorch `torch.Tensor` objects, while `tf` will cause TensorFlow `tf.constant` objects to be returned. By default you will get a Python `list` of numbers.

The output is almost identical, though we can see that our token ids are now returned as a `tensor`:

```python
{'input_ids': tensor([[ 101, 2023, 2338, 2003, 2307,  999,  102]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1]])}
```

Now that we have encoded our input numerically in a form that the model can use, we can now submit these inputs to the model and generate the embedding for our sentence.

{aside}
### What is the attention mask?

### Why is it all `1`s for the input? 
{/aside}

#### Generating an embedding

We start by loading just the model (rather than the complete pipeline):

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
```

We then generate our inputs, as before:

```python
inputs = tokenizer("This book is great!", padding=True, truncation=True, return_tensors="pt")
```

and pass them to the model:

```python
output = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
output
```

{blurb, class: tip}
Note that we could also just write `model(**input_ids)` for convenience (to splat the keys of the output as keyword arguments) but the inputs were written out in full in the example for clarity.
{/blurb}

which gives us the output:

```
BaseModelOutput(last_hidden_state=tensor([[[ 0.7444,  0.1403,  0.1392,  ...,  0.4110,  0.8821, -0.5004],
         [ 0.8811,  0.1455,  0.0829,  ...,  0.3491,  0.9634, -0.3885],
         [ 0.8345,  0.1600,  0.2252,  ...,  0.3854,  0.8153, -0.4369],
         ...,
         [ 1.0104,  0.1794,  0.1503,  ...,  0.4251,  0.8810, -0.4535],
         [ 0.8471,  0.1320,  0.1086,  ...,  0.3997,  0.8830, -0.4399],
         [ 1.2363,  0.1287,  0.7568,  ...,  0.5593,  0.6615, -0.8747]]],
       grad_fn=<NativeLayerNormBackward0>), hidden_states=None, attentions=None)
```

We are interested in the `last_hidden_state`—these are the embedding values for our inputs (in our case, just one input).

The shape of `output.last_hidden_state.shape` is `torch.Size([1, 7, 768])`. The dimensions of this tensor represent:

1. **Batch size:** the number of input sequences, in our case `1`
2. **Input sequence length:** the length of the numerical representation of our input (in our case 7)
3. **Embedding dimensions:** the length of the hidden state (the embedding), in this case 768

![](2024-07-14-transformer-model-breakdown.png)

#### Different transformer heads

The head of the model is the part that performs the problem-specific task of interest. When we instantiate an `AutoModel` there is no head—we get back as output an instance of `BaseModelOutput` that just gives us `last_hidden_state`.

To perform other tasks we will swap in other heads, and in general we are more likely to be instantiating the following kinds of auto model than we are the base (i.e. just `AutoModel`):

- `AutoModelForSequenceClassification` for sentiment or other classification (e.g. spam detection)
- `AutoModelForSeq2Seq` for translation, summarisation
- `AutoModelForQuestionAnswering` for extractive question answering, i.e. answering questions about the submitted input
- `AutoModelForMultipleChoice` which is similar to question answering, but we supply possible answers to choose from
- `AutoModelForTokenClassification` for tasks where we want to label each token, e.g. Named Entity Recognition. We could also do something like label the sentiment of each token to find the positive, neutral and negative parts of something like "the staff were lovely but the food was awful".

{blurb, class:information}
This list of auto model (or head) types isn't the complete list, and they are all specifically for Natural Language Processing (NLP) tasks—in the transformers library documentation you'll find the [full list of NLP auto model classes](https://huggingface.co/docs/transformers/en/model_doc/auto#natural-language-processing) as well as [model classes for computer vision](https://huggingface.co/docs/transformers/en/model_doc/auto#computer-vision), [audio](https://huggingface.co/docs/transformers/en/model_doc/auto#audio) and [multimodal tasks](https://huggingface.co/docs/transformers/en/model_doc/auto#multimodal) (where multimodal means multiple modalities, e.g. working with both images and text).
{/blurb}

Taking `AutoModelForSequenceClassification` as an example, we classify the sentiment of a sentence:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

inputs = tokenizer("This book is quite reasonable", return_tensors="pt")

output = model(**inputs)
```

This output differs from that of the `AutoModel` (which was of type `BaseModelOutput` and gave us `last_hidden_state`)—it is of type `SequenceClassifierOutput` and gives us a tensor of two [logits](#what-are-logits):

```
output.logits

=> tensor([[-3.6401,  3.8207]], grad_fn=<AddmmBackward0>)
```

From this we can see that we are getting a high value for the second logit, which is for the positive label. To test this hypothesis we classify `"this book is awful"` and get a high value for the first logit, which is for the negative label:

```
tensor([[ 4.7072, -3.7667]], grad_fn=<AddmmBackward0>)
```

Rather than guessing however we would generally check `model.config.id2label`, which returns `{0: 'NEGATIVE', 1: 'POSITIVE'}`.

To convert these logit values into probabilities we would apply a [softmax](#softmax-glossary) to get probability values for each case (negative or positive, in this example):

```python
import torch.nn. functional as F

output_probabilities = F.softmax(output.logits, dim=-1).tolist()

print([ round(prob, 4) for prob in output_probabilities[0] ])
```

which outputs:

```
[0.9998, 0.0002]
```

{blurb, class:information}
### Why is `dim=-1`, and what is `tolist()`?

The `dim` argument to `F.softmax` (and to many tensor-related functions) is the dimension across which to apply the function. In this case we have a 2d tensor, an array of logit pairs, or an array of arrays. In this case `dim=0` would be applying the function across the rows, but we would want it to be applied to the columns. We therefore want the second dimension `dim=1`—this also happens to be the last dimension, and you will commonly see this, as in the above example, written as `dim=-1`.

The call to `tolist()` on the instance of `torch.Tensor` simply returns the tensor as a regular python `list` (in this case an array of arrays, where we have one pair of logit values).
{/blurb}

## Handling inputs

As mentioned before, transformer models can only operate on numbers, and so we need to convert our input sentences (which we call sequences) into tensors—we call this process _encoding_. Then when we get the numerical output of the model we will need to _decode_ this back into textual form.

This process is straightforward when we are submitting a single sequence to the model, but it becomes a bit less convenient when we are submitting multiple sequences to the model, particularly if those sequences are different lengths, which is highly likely.

As to why we are submitting multiple sequences at once (in what we call a _batch_), this is primarily for the sake of efficiency—transfering data between CPU and GPU is slow relative to the actual computation that happens. For each such copy we pay a fixed time cost plus some additional time based on the size of the data—if we copy each sequence in turn then we will pay that fixed cost many times over, leading to reduced utilisation of our hardware. This is somewhat analogous to how it is generally faster to copy a 1GB file to object storage (e.g. S3, GCS) than it is to copy 1,000 smaller files which total 1GB.

A more direct constraint is that our inputs to the model are tensors, and tensors can't be ragged in their dimensions. As such when we have multiple sequences which are different lengths then we employ _padding_ to make them all the same size. Separately if one or more sequences are too long then we will _truncate_ them.

We will now look at how padding and truncation work in more detail—the transformers pipeline handles this for us so on a practical level we don't necessarily need to know this in great detail, but it's useful for understanding how the model works, and there may be occasions where this helps you in debugging unexpected behaviour.

#### Padding

If we have two or more inputs that are not the same length, then when we submit them for tokenization they will be padded so that they are the same length. If the inputs are not too long (see [truncation](#truncation) next) then the sequences will be padded to be the length of the longest sequence in the given batch.

Padding may seem straightforward, and it generally is especially when you use the right tokenizer via the transformers library, but it's still quite possible to get undesirable results. We'll look at an example of how it typically works (and works well), and then two cases that we should watch out for.

Returning to our familiar example:

```python
from transformers import AutoTokenizer

# The default, as of June 2024
model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer(["This book is great!"])
```

we get the expected output:

```python
{'input_ids': [[101, 2023, 2338, 2003, 2307, 999, 102]], 'attention_mask': [[1, 1, 1, 1, 1, 1, 1]]}
```

Note that we submitted our single input as a list of length 1. We'll now submit two inputs of different length:

```python
from rich import print

print(tokenizer([
    "This book is great!",
    "This book is quite useful"
]))
```

and from this we get:

```python
{
    'input_ids': [
        [101, 2023, 2338, 2003, 2307, 999, 102],
        [101, 2023, 2338, 2003, 3243, 6179, 102]
    ],
    'attention_mask': [[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1]]
}
```

Curiously these result in the same number of input ids. This is because all of our words and symbols are common enough to have their own token.

Let's try again, but with a word which will be sure to be broken up into multiple tokens:

```python
print(tokenizer([
    "This book is great!",
    "This book is positively pulchritudinous"
]))
```

which gives us:

```python
{
    'input_ids': [
        [101, 2023, 2338, 2003, 2307, 999, 102],
        [101, 2023, 2338, 2003, 13567, 16405, 29358, 14778, 21041, 18674, 102]
    ],
    'attention_mask': [[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
}
```

The sequences are not definitely different lengths, and our input ids are now ragged in their dimensions.

The second sequence tokenizes as:

```python
tokenizer.tokenize("This book is positively pulchritudinous")

=> ['this', 'book', 'is', 'positively', 'pu', '##lch', '##rit', '##udi', '##nous']
```

If we try to use our ragged input ids:

```python
inputs = tokenizer([
    "This book is great!",
    "This book is positively pulchritudinous"
], return_tensors="pt")

model = AutoModel.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

model(input_ids=inputs['input_ids'], attention_masks=inputs['attention_masks'])
```

Then we get an error stating the problem (the second sequence was expected to match the length of the first, but instead was longer), and a helpful hint regarding what the solution is likely to be:

```
...
ValueError: expected sequence of length 7 at dim 1 (got 11)
...
ValueError: Unable to create tensor, you should probably activate truncation and/or padding with 'padding=True' 'truncation=True' to have batched tensors with the same length. Perhaps your features (`input_ids` in this case) have excessive nesting (inputs type `list` where type `int` is expected).
```

Following this guidance, we will get exactly what we need in the majority of cases:

```python
print(tokenizer([
    "This book is great!",
    "This book is positively pulchritudinous"
], padding=True, truncation=True, return_tensors="pt"))
```

I> Note that we are applying padding, applying truncation and requesting pytorch tensors in the return value—while we are only considering padding in this section, this is the formulation that you are likely to use 99% of the time so it is included in full.

This gives us:

```python
{
    'input_ids': tensor([
        [ 101, 2023, 2338, 2003, 2307, 999, 102, 0, 0, 0, 0],
        [ 101, 2023, 2338, 2003, 13567, 16405, 29358, 14778, 21041, 18674,              102]
    ]),
    'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
}
```

Where the part of interest is that our shorter first sequence has been padded with `0`.

We can confirm the ids of special tokens:

```python
f"The tokenizer padding token is {tokenizer.pad_token} which has input id {tokenizer.pad_token_id}"
=> 'The tokenizer padding token is [PAD] which has input id 0'
```

{blurb, class: information}
We can look up the various special tokens of a particular tokenizer by looking at the methods that it defines:

```python
import re

# Methods not starting with an underscore, which end in '_token'
token_method_pattern = re.compile(r'^[^_].+_token$')

[ method for method in dir(tokenizer) if token_method_pattern.match(method) ]
```

which gives:

```
['bos_token',
 'cls_token',
 'eos_token',
 'mask_token',
 'pad_token',
 'sep_token',
 'unk_token']
 ```
{/blurb}

##### Padding gone wrong: outliers in your input data

#### Truncation {#truncation}

