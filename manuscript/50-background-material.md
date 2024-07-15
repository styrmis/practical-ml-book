# Background Material

The aim of this section is to provide a place to dive deeper into some topics, with a view to keeping the main flow of the book focussed on the practical use of transformer models without delving frequently into theory. While the theory can be critical to understand well if you want to get into model development or research, it is arguably not necessary (and may even be counterproductive[^theory-counterproductive] early on) if your main motivation is to put these models to practical use.

[^theory-counterproductive]: It may seem strange to almost discourage the learning of theory—the idea though is more to just delay it until you're comfortable applying the models to practice problems. The main trap I'd want to avoid here is where a domain term (like [logit](#what-are-logits), [tensor]{#about-tensors}, or any other term) might stop you in your tracks, thus keeping you from making practical progress. I say this as someone who is greatly motivated by solving problems but less motivated by the learning of theory without application.

## What are logits? {#what-are-logits}

W> This section is incomplete.

The term `logit` was coined in 1944 by [Joseph Berkson](https://en.wikipedia.org/wiki/Joseph_Berkson), a physicist, physician and statistician. Its name comes in part from the existence of the term `probit`, which had been introduced 10 years earlier by the biologist [Chester Ittner Bliss](https://en.wikipedia.org/wiki/Chester_Ittner_Bliss).

It is worth investing some time in getting comfortable with the concept of logits, as they appear frequently in Machine Learning, particularly with [logistic regression](#logistic-regression) and neural networks (with includes transformer models).

To motivate all of this with a practical example, let's say that we are trying to determine the probability that a given sentence has a positive sentiment. We define `p`$ as the probability (in the range `[0, 1]`$) that the sentence is positive, and `1 - p`$ as the probability that the sentence has negative sentiment.

For example, for the sentence "this book is alright" we might say that there's a 90% chance (`p = 0.9`) that the sentiment is positive, which leaves us a 10% (`(1 - p) = 0.1`$) chance that it is negative.

Given a probability, we can calculate the _odds_ for the event as follows:

```$
\textrm{odds} = \frac{p}{1 - p} = \frac{0.9}{0.1} = 9
```

The sigmoid:

```$
\frac{1}{1 + e^{-z}}
```

## About tensors {#about-tensors}

## Logistic regression {#logistic-regression}

## The Softmax function {#softmax-glossary}

