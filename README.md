# SolidGoldMagikarp

SolidGoldMagikarp is a collection of interesting posts, papers, and blog posts about various Artificial Intelligence models. The name stems from the intriguing fact that GPT-2 and GPT-3 models can be broken by specific prompts called anomalous tokens. This repository aims to be a starting point for those intersted in ML / AI research.

For more information see these blogposts:

- [SolidGoldMagikarp plus, prompt generation | Jessica Rumbelow, 2023-02-06](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation)
- [SolidGoldMagikarp II: technical details and more recent findings](https://www.lesswrong.com/posts/Ya9LzwEbfaAMY8ABo/solidgoldmagikarp-ii-technical-details-and-more-recent)

## Contents (summaries via GPT-3)

- [Interesting Papers](#interesting-papers)

  - [Large Language Models ](#large-language-models)

  <!-- - [Video](#video) -->

## Interesting Papers

### Large Language Models

Papers that are mandatory reading for anyone interested in LLM research. These papers are the foundation of the field and are the basis for many of the more recent papers.

<details>
<summary>Attention is all you need, Ashish Vaswani et al (2017)
</summary>

&nbsp;

[Link to Paper](https://arxiv.org/abs/1706.03762)

"Attention is All You Need" is a paper published in 2017 that proposed a new architecture for neural machine translation. The architecture, called the Transformer, introduced the concept of self-attention mechanisms, which allows the model to weigh the importance of each input element in a sequence when making predictions.

Before the Transformer, most neural machine translation models relied on recurrent neural networks (RNNs), which processed sequences by updating a hidden state over time. The problem with RNNs is that they have difficulty handling long sequences, as the hidden state becomes diluted with each step.

The Transformer solves this problem by using self-attention mechanisms, which allow the model to focus on the most important elements of the input sequence at each step. This allows the Transformer to process input sequences in parallel, making it much more efficient than traditional RNNs.

In addition to its use in machine translation, the Transformer architecture has since been applied to a wide range of natural language processing tasks, including text classification, text generation, and named entity recognition.

</details>

<details>
<summary>Bert: Pre-training of Deep Bidirectional Transformers for Language Understanding", Jacob Devlin et al (2019)
</summary>

&nbsp;

[Link to Paper](https://arxiv.org/abs/1810.04805)

BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained deep learning model developed by Google that is designed for natural language processing tasks. BERT was introduced in a paper called "Pre-training of Deep Bidirectional Transformers for Language Understanding" in 2018.

The main idea behind BERT is to pre-train a deep neural network on large amounts of text data, allowing it to learn the patterns and relationships between words in a language. Once pre-trained, the model can then be fine-tuned on specific NLP tasks, such as sentiment analysis, question answering, and named entity recognition.

BERT stands out from other NLP models in several ways. Firstly, it is a bidirectional model, meaning that it considers the context of a word from both its left and right context, rather than just its left context like traditional language models. Secondly, BERT uses a Transformer architecture, which is based on self-attention mechanisms, allowing it to weigh the importance of each input element in a sequence.

The pre-training step of BERT allows it to learn general knowledge about the language, such as the meaning of words and how words are related to each other. This makes BERT well-suited for a wide range of NLP tasks and has led to significant improvements in performance on many benchmarks.

</details>

<details>
<summary> RoBERTa: A Robustly Optimized BERT Pretraining Approach, Yinhan Liu et al (2019) </summary>

&nbsp;

[Link to Paper](https://arxiv.org/abs/1907.11692)

"RoBERTa: A Robustly Optimized BERT Pretraining Approach" is a research paper published in 2019 by researchers at Facebook AI. The paper introduces a new language model called RoBERTa, which is based on the popular BERT (Bidirectional Encoder Representations from Transformers) language model.

BERT was a breakthrough in the field of NLP (natural language processing) as it was able to achieve state-of-the-art results on many NLP tasks. However, the authors of the RoBERTa paper noticed that BERT was not optimized to its full potential and found several areas where it could be improved.

RoBERTa addresses these issues by making several changes to the training process of BERT. These changes include using a much larger training corpus, removing certain training data augmentation methods, and training for a longer period of time.

The results of the RoBERTa model were extremely promising, showing significant improvements over the original BERT model on a number of NLP tasks. This made RoBERTa one of the most popular language models used by NLP researchers and practitioners.

</details>

<details>
<summary>ELMo: Deep contextualized word representations (2018)
</summary>

&nbsp;

[Link to Paper](https://arxiv.org/abs/1802.05365)

ELMo, or Embeddings from Language Models, is a deep learning model for generating contextualized word representations, introduced in a paper called "Deep contextualized word representations" in 2018. ELMo is unique in its ability to generate word representations that take into account the context in which a word is used.

In traditional word embedding models, such as word2vec or GloVe, each word is represented by a fixed-length vector, regardless of the context in which it is used. This can be a limitation, as words can have multiple meanings depending on the context in which they are used.

ELMo solves this problem by representing words as a combination of character-based embeddings and context-sensitive representations learned from a deep bidirectional language model. The language model is trained on a large corpus of text and is able to generate representations that take into account the surrounding words and sentence structure.

These context-sensitive representations are then concatenated with the character-based embeddings to form a final representation for each word. The resulting ELMo representations have been shown to significantly improve performance on a wide range of NLP tasks, including named entity recognition, sentiment analysis, and text classification.

</details>


<details>
  <summary>The GPT Language Model Papers</summary>
The following are three research papers from OpenAI that describe the development and results of the Generative Pre-Training Transformer (GPT) language models for natural language processing (NLP) tasks.

## GPT: Generative Pre-Training from Language Models, Alec Radford et al (2018)

[Link to Paper](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)

This paper introduces GPT, a deep learning model that specializes in performing NLP tasks, such as text generation, translation, and question answering. It is known for its ability to perform these tasks with high accuracy after seeing only a few examples, a property known as "few-shot learning".

## GPT-2: Language Models are Unsupervised Multitask Learners, Alec Radford et al (2019)

[Link to Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)

This paper describes the GPT-2 language model, a state-of-the-art unsupervised deep learning model for NLP tasks. The model is capable of performing a variety of NLP tasks, such as text generation, translation, and question-answering, with high accuracy and only a few examples. This is attributed to its massive scale, diverse training data, and its ability to learn from patterns in the data and generalize to new tasks.

## GPT-3: Language Models are Few-Shot Learners, Alec Radford et al (2020)

[Link to Paper](https://arxiv.org/abs/2005.14165)

This paper describes GPT-3, a deep learning model that uses unsupervised learning to perform a wide range of NLP tasks, such as text generation, language translation, and question-answering. GPT-3 can perform these tasks with high accuracy after seeing only a few examples, a property known as few-shot learning. This is attributed to its massive scale, diverse training data, and its ability to learn from patterns in the data and generalize to new tasks.

</details>

<!-- 
- [Learning representations by back-propagating errors, Rumelhart et al (1986)](https://www.nature.com/articles/323533a0)

  "Learning representations by back-propagating errors" is a seminal paper in the field of machine learning published in 1986 by Rumelhart et al. The paper introduces the backpropagation algorithm, which is a method for training neural networks that is still widely used today.

  The backpropagation algorithm is a supervised learning method that uses gradient descent to train neural networks. It is based on the idea that the error of a neural network can be decomposed into the errors of its constituent layers, which can be calculated using the chain rule of differentiation. The algorithm then uses these errors to update the weights of the network in a way that minimizes the error.

  The backpropagation algorithm has been widely adopted in machine learning and has been the foundation of many state-of-the-art models, including OpenAI's GPT series of models. The paper has been highly influential and has received numerous citations, making it one of the most impactful papers in machine learning in recent years.

- [Sequence to Sequence Learning with Neural Networks, Sutskever et al (2014)](https://arxiv.org/abs/1409.3215)

  Sequence to Sequence Learning with Neural Networks" is a 2014 paper by Ilya Sutskever et al. that introduces a neural network architecture for performing sequence-to-sequence (Seq2Seq) tasks, such as machine translation and text summarization.

  The Seq2Seq architecture consists of two main components: an encoder network and a decoder network. The encoder network processes the input sequence and generates a fixed-length vector representation, known as the context vector, that summarizes the information in the input. The decoder network then uses the context vector to generate the output sequence.

  The authors show that the Seq2Seq architecture can be trained end-to-end using supervised learning, making it a powerful tool for NLP tasks. They also demonstrate that the architecture can be improved by incorporating attention mechanisms, which allow the decoder network to focus on different parts of the input sequence at different times.

  Overall, this paper introduced the Seq2Seq architecture and established it as a cornerstone of NLP research. The Seq2Seq architecture has been widely adopted and remains a popular choice for NLP tasks due to its simplicity, effectiveness, and versatility.

- [Long Short-term Memory, Hochreiter & Schmidhuber (1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)

  The unreasonable effectiveness of Recurrent Neural Network" is a seminal paper in the field of machine learning published in 1997 by Hochreiter & Schmidhuber. The paper introduces the Long Short-Term Memory (LSTM) architecture, which is a type of recurrent neural network (RNN) designed for sequence modeling tasks such as machine translation and text generation.

  The key idea behind the LSTM architecture is the use of a memory cell, which is a vector that stores information about the input sequence. The LSTM architecture is trained to update the memory cell in a way that allows it to store information about the input sequence and use this information to generate the output sequence.

- [Reinforcement Learning: An Introduction, Sutton & Barto (2018)](http://incompleteideas.net/book/the-book-2nd.html)

  "Reinforcement Learning: An Introduction" is a book by Richard S. Sutton and Andrew G. Barto, first published in 2018, that provides an introduction to the field of reinforcement learning (RL). RL is a subfield of machine learning that deals with the problem of decision-making in an environment where an agent must learn from its own experiences to maximize a reward signal.

  The book covers the fundamental concepts and algorithms of RL, including Markov decision processes, value functions, and policy gradient methods. The authors also discuss advanced topics, such as deep reinforcement learning, multi-agent RL, and inverse reinforcement learning.

  Throughout the book, the authors use concrete examples and intuitive explanations to make the concepts accessible to a wide audience. They also provide mathematical foundations and derivations to support the algorithms, making the book suitable for researchers and practitioners alike.

  Overall, "Reinforcement Learning: An Introduction" is a comprehensive and accessible resource for anyone interested in learning about RL. The book has become a seminal reference in the field and is widely regarded as one of the best introductory texts on RL. -->
