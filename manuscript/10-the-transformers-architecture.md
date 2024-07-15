# The Transformer Architecture

In this chapter we will step through the workings of the transformer architecture at a high level, with a focus on how they are typically trained and used.

There are broadly three kinds of transformer models:

- GPT-like (auto-regressive Transformer models)
- BERT-like (auto-encoding Transformer models)
- BART/T5-like (sequence-to-sequence models)

There is a lot of jargon here already, but it turns out that they are simpler than they sound:

- For GPT, auto-regressive simply means that in order to produce the next word/token it looks at what it has already output (auto meaning self, regressive meaning to return to a previous state).
- For BERT, you can think of this as a foundational encoder model, the purpose of which is to encode the rich meaning of the texts that it is given.
- For BART/T5, sequence to sequence refers simply to the fact that it is given a sequence of words/tokens (e.g. and article) and outputs a sequence of words/tokens (e.g. a summary of the article).

All of the above are trained in a self-supervised manner, which means that the objective is determined from the training data (e.g. by masking different words and training the model to reconstruct the input).

A model trained in this manner will develop a statistical understanding of the input but won't be directly useful. The next step is Transfer Learning, where the model is fine-tuned on a particular task musing human-annotated (or otherwise reliable) labels. The earlier task is sometimes called Pretraining, which gives you a pretrained language model. In the fine-tuning process we are likely to replace the last layer (head) of the model with one that has the correct number of outputs for our task, and may include some additional layers before this if the problem warrants it. Ideally the problem domain and the domain of the pretrained/base model will be close, e.g. if the problem is in the German language then a German base model (or one trained on many languages) would appropriate.

In general, pretraining will require a large amount of compute, making it infeasible for the average person or smaller companies. Fine-tuning however can be done on a single GPU in a reasonable amount of time.

Such tasks might include causal language modelling, where we try to predict the next token/word given the previous n tokens/words.

Or masked language modelling where we ask the model to complete the missing word, which can be in the middle of a sentence.

## Encoder / Decoder

The transformer architecture is split broadly between an encoder and a decoder.

The encoder is bi-directional and uses self-attention. It converts words (or sequences of words?) into embeddings (which can also be called features), a numerical representation.

The decoder is uni-directional, uses masked self-attention and is traditionally used in an auto-regressive manner.

The connection of the encoder and decoder leads to this being a sequence-to-sequence model.

The decoder takes the output of the encoder and, along with other inputs, produces token output probabilities. The fact that the outputs will be used in later processing is what makes this model auto-regressive.

Each part can be used separately:

The encoder can be used on its own for sentence classification (distance between embeddings?) and named entity recognition (how?)
The decoder can be used on its own for text generation
Encoder-decoder (or sequence-to-sequence) models can be used for generative tasks that require an input, e.g. translation, summarisation.

### The original transformer architecture

The original transformer implementation was designed with translation in mind. The attention layers in the encoder portion can make use of all words in the input, which is necessary to know (for example) how to conjugate a given word based on the context (i.e. the subject/object). The decoder works sequentially—it is able to take as input the previous tokens that it has generated, plus the input from the encoder.

In training the decoder is fed the whole output sentence for the sake of efficiency, but is not allowed to peek at later words in the sequence.

### Encoder models

Encoder-only models are useful for when an understanding of the whole sentence is required, e.g. for sentence classification, named entity recognition or extractive quesiton answering. They are often characterised as having bi-directional attention and are often called auto-encoding models.

Given words (tokens?) the encoder outputs one embedding per word, although maybe it's better to call them feature vectors unless we are thinking of using them to calculate similarity.

An important point, given the string "Welcome to Tokyo", the feature vector for "to" will be the representation for that word but in the context, i.e. in relation to "Welcome" and "Tokyo". The aim therefore of this feature vector is to represent the meaning of the given word, in context.

This is bi-directional because the embedding for the word is based on context from the left and context from the right.

Encoder models are good at extracting meaning from sentences, lending to their use in:

- Sequence classification
- Question answering
- Masked language modelling (MLM is a pretraining technique, but it prepares the model to perform very well on a wide range of NLP tasks)

Typical models: BERT, RoBERTa, ALBERT

BERT is in fact an encoder-only model, which presumably makes it good as a base for training the classification head on top of.

It seems that when it comes to masked word prediction, encoder models can predict a word at any point in a sentence, rather than just at the end, e.g. `"My <MASK> is Stefan"`.

```python
import torch
from transformers import BertTokenizer, BertForMaskedLM

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Input sentence
sentence = "Hello, my [MASK] is Stefan"

# Tokenize input sentence
input_ids = tokenizer.encode(sentence, return_tensors='pt')

# Predict masked word
with torch.no_grad():
    outputs = model(input_ids)
    predictions = outputs.logits

# Get the predicted token id for the masked position
masked_index = (input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1].item()
predicted_token_id = predictions[0, masked_index].argmax(dim=-1).item()

# Decode the predicted token id to the word
predicted_word = tokenizer.decode([predicted_token_id])

print(f"The predicted word for the masked position is: {predicted_word}")
```

Running this, we get:

`The predicted word for the masked position is: name`

### Decoder models

It is surprising to me that there can even be such a thing as a decoder-only model, but an example of one is GPT 2. Curiously, you can use a decoder for many of the same tasks as encoders, but with seemingly slightly worse performance.

Some key points about decoder models:

- They're _uni-directional_, in that they only look at tokens to the left.
- They're _auto-regressive_, in that they produce outputs based on past outputs (fed in as inputs)
- They use _masked self-attention_ and _cross-attention_

Interestingly, a decoder model also produces feature vectors on a per-word basis, given an input of words (tokens).

{aside}
Encoders and decoders appear to have some similarities, in that they take a sequence of words and produce feature vectors (one per word). The encoder model excels at understanding and representing input sequences, while the decoder excels at generating output sequences based on context. Combined encoder/decoder architectures allow for the development of powerful sequence-to-sequence models.
{/aside}

Some examples of decoder-only models are CTRL, GPT, GPT-2 and Transformer XL.

A key difference between the encoder and decoder models is the way they use self-attention. With the decoder model the mode of self-attention is _masked self-attention_, where all tokens to the right of the token to generate are masked and therefore unavailable to the model.

### Encoder-Decoder Models

When we combine an encoder model with a decoder model we can build a _sequence-to-sequence_ model. By sequence we simply mean a sequence of words—we give the model some text (a sequence of words) and the model similarly outputs a sequence of words.

When generating text, the encoder portion of the model can access the full context of the given input text. This is fed to the decoder model, which is additionally able to look at the tokens which it has generated previously.

When generating each token, the decoder model will convert the previously-generated tokens into embeddings and will additionally incorporate a positional encoding so that the model can keep track of the position of each token in the sequence.

Sequence-to-sequence models are good at tasks like translation, summarisation and generative question answering.

Some examples of Encoder-Decoder models are BART, mBART, Marian and T5.

When we start generating, we have no output tokens to feed into the decoder. For this purpose we have a special 'start of sequence' token that we can use to signify that we want to start a sequence.

Once we have encoded the input sequence, passed it to the decoder and started the generation process with the 'start of sequence' token then we no longer need the encoder—from this point we can use the decoder in an auto-regressive manner to generate the first token, and then the next by feeding in again the output of the encoder (which has not changed), along with the special 'start of sequence' token plus the token that the decoder generated previously. We can repeat this process over and over to until we are done generating tokens.

The size of the context window of the model typically defines how many tokens can be taken into consideration **in both the input and the output combined**. For example with GPT-3, which has a context window of 2048 tokens, then if our prompt is 1024 tokens then we will only be able to generate 1024 tokens until we have exhausted the context window.

Q> Once we reach the limit of the context window of a model, how can we generate more tokens? Do we simply slide the window to the right, or are there more advanced techniques to avoid losing the context of the beginning of the input?

## Decoder models vs Sequence-to-sequence models

### Decoder-Focused Models (e.g., GPT-4)

- **Architecture**: Primarily use the decoder part of the Transformer architecture.
- **Operation**: Generate text auto-regressively (one token at a time) based on previous tokens.
- **Context Handling**: Use self-attention to consider the context of all previous tokens.
- **Applications**:
  - Text completion and generation
  - Conversational agents and chatbots
  - Creative writing and content creation
  - Code generation and completion

### Sequence-to-Sequence Models

- **Architecture**: Utilize both an encoder and a decoder.
- **Operation**: The encoder processes the input sequence to generate a context representation, which the decoder then uses to generate the output sequence.
- **Context Handling**: Encoder captures the entire input context, and the decoder generates output based on this context.
- **Applications**:
  - Machine translation (e.g., translating text from one language to another)
  - Text summarization (e.g., condensing long articles)
  - Paraphrasing and rephrasing text
  - Speech recognition and synthesis

### Summary

- **GPT-4**: Excels in tasks requiring fluent and coherent text generation based on a prompt, without needing a separate encoding step.
- **Seq2Seq Models**: Best suited for tasks that transform an input sequence into a distinct output sequence, leveraging both encoding and decoding processes.