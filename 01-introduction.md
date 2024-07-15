# About this book

It has been possible for many decades now to achieve amazing feats with Machine Learning—there are almost too many to list that have contributed to making our lives better (or at the very least easier).

With the introduction of the transformer architecture and in particular a technique called _attention_ it has become possible though to develop models that can do a number of these tasks all at once, and discuss the results with you. You wouldn't have gotten very far trying to ask a spam filter what it was thinking (to use the term loosely), but now models like ChatGPT can even help you write your prompts that you use to get these results.

Machine Learning has always had quite a high bar for entry. Taking just Artificial Neural Networks as one sub-field, historically you would generally have needed to have good knowledge of calculus, linear algebra and regression analysis, in addition to programming. Then, to solve your problem, you would be quite likely to be building your training data from scratch. You would amass enough of it, learn how to train a model and repeat this process many times. Hopefully at the end of this process you would have something that you could put to good practical use.

In terms of not starting from scratch, there were some options for transfer learning, but this still required considerable effort, knowledge and accumulated experience to get right. Today this is still true, but I would argue that the bar has been very usefully lowered in terms of allowing anyone who persists at it to develop their own practical, useful models by standing on top of two recently-formed giants: the transformer architecture and its attention mechanism, and the myriad base models that are currently available as published by everyone from large companies down to individuals.

Now you can take a base model which may have cost some millions of dollars to train, and fine-tune it for some task, usefully leveraging the incredible depth and breadth of context and encoded knowledge that some of these models hold. In a number of cases the licensing is also sufficiently permissive to use the results commercially, meaning that in addition to research and hobby use you might actually build a business around your work in this field, if you wish to do so.

The purpose of this book is to lay out how we can solve practical problems using transformers and leveraging the incredible base models that we can access via the HuggingFace hub. A key aim here is to start with the practical and delve into the theoretical as an optional step—you can successfully train models and solve problems without knowing the underlying maths well. Not knowing the maths well can certainly hinder your progression in certain directions but solving practical problems using the transformers and pytorch libraries will always be open to you.

Being a Leanpub publication, this book is a work in progress and you are invited to ask questions and make requests. If some part is unclear then please let me know and I'll be happy to expand that section. If you find what looks to be an error then I will endeavour to fix it.

We'll start by focussing on natural language problems like classification (is the sentiment of this text positive, negative, or neutral?), question answering, translation and autocorrect. As transformers can also produce incredible results on vision tasks then we will look to explore that area also.

## How to study the material

As you go through the chapters, resist the urge to copy/paste anything—type out any code, ideally without looking at it. This will exercise your recall of the material right from the start, which is necessary if you want to be able to write this kind of code mostly from memory.

### Changelog

- **13th July 2024** `v0.1`