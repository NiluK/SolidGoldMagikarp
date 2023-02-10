# SolidGoldMagikarp

Solid AI is a collection of interesting posts, papers, and blog posts about various Artificial Intelligence models. The name stems from the intriguing fact that GPT-2 and GPT-3 models can be broken by specific prompts called anomalous tokens. This repository aims to be a starting point for those intersted in ML / AI research.

For more information see these blogposts:

- [SolidGoldMagikarp plus, prompt generation | Jessica Rumbelow, 2023-02-006](https://www.lesswrong.com/posts/aPeJE8bSo6rAFoLqg/solidgoldmagikarp-plus-prompt-generation)
- [SolidGoldMagikarp II: technical details and more recent findings](https://www.lesswrong.com/posts/Ya9LzwEbfaAMY8ABo/solidgoldmagikarp-ii-technical-details-and-more-recent)

## Contents (summaries via GPT-3)

- [Mandatory Reading](#mandatory-reading)
  - [Text](#text)

  <!-- - [Video](#video) -->

## Mandatory Reading

  ### Text 

  Papers that are mandatory reading for anyone interested in LLM research. These papers are the foundation of the field and are the basis for many of the more recent papers.

  1.  [Attention is all you need, Ashish Vaswani et al (2017)](https://arxiv.org/abs/1706.03762)

      "Attention is All You Need" is a seminal paper in the field of Natural Language Processing (NLP) published in 2017 by Ashish Vaswani et al. The paper introduces the Transformer architecture, which is a deep neural network designed for NLP tasks such as machine translation and text generation.

      The key idea behind the Transformer is the use of self-attention mechanisms, which allow the model to attend to different parts of the input sequence in a flexible and efficient manner. This contrasts with traditional NLP models, which use recurrent neural networks (RNNs) and can be difficult to parallelize and optimize. The Transformer architecture can be trained end-to-end, bypassing the need for traditional feature engineering methods.

      The Transformer has been widely adopted in NLP and has been the foundation of several state-of-the-art models, including OpenAI's GPT series of models. The paper has been highly influential and has received numerous citations, making it one of the most impactful papers in NLP in recent years.

  2.  [GPT-2: Language Models are Unsupervised Multitask Learners, Alec Radford et al (2019)](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)

      The paper demonstrates that GPT-2 can be trained on a massive amount of text data from the internet and can perform a variety of NLP tasks with high accuracy, even without task-specific fine-tuning. This is a departure from traditional NLP models that typically require task-specific training data and fine-tuning.

      The authors argue that the ability of GPT-2 to perform multiple tasks without task-specific fine-tuning is a result of its massive scale and the diversity of its training data, which allows it to learn a wide range of linguistic and contextual knowledge. They also highlight the potential limitations and ethical concerns of language models like GPT-2, such as their ability to generate biased or misleading text, and the need for caution when using these models in applications.

      Overall, the paper highlights the impressive capabilities of GPT-2 and its potential to revolutionize NLP, while also raising important questions about the ethical implications of large language models.

  3.  [GPT-3: Language Models are Few-Shot Learners, Alec Radford et al (2020)](https://arxiv.org/abs/2005.14165)

      GPT-3: Language Models are Few-Shot Learners" is a 2020 paper by Alec Radford et al. that describes the architecture and results of the GPT-3 language model developed by OpenAI. GPT-3 is a Transformer-based language model that uses unsupervised learning to perform multiple NLP tasks, such as text generation, translation, and question-answering.

      The paper demonstrates that GPT-3 can perform these tasks with high accuracy after seeing only a few examples, a property known as few-shot learning. This is a significant departure from previous NLP models, which typically require large amounts of task-specific training data and fine-tuning.

      The authors attribute the few-shot learning capabilities of GPT-3 to its massive scale, diverse training data, and its ability to learn from patterns in the data and generalize to new tasks. They also show that GPT-3 can perform various tasks, including language translation and answering questions, without any task-specific fine-tuning, making it a versatile tool for NLP applications.

  4. [Learning representations by back-propagating errors, Rumelhart et al (1986)](https://www.nature.com/articles/323533a0)

      "Learning representations by back-propagating errors" is a seminal paper in the field of machine learning published in 1986 by Rumelhart et al. The paper introduces the backpropagation algorithm, which is a method for training neural networks that is still widely used today.

      The backpropagation algorithm is a supervised learning method that uses gradient descent to train neural networks. It is based on the idea that the error of a neural network can be decomposed into the errors of its constituent layers, which can be calculated using the chain rule of differentiation. The algorithm then uses these errors to update the weights of the network in a way that minimizes the error.

      The backpropagation algorithm has been widely adopted in machine learning and has been the foundation of many state-of-the-art models, including OpenAI's GPT series of models. The paper has been highly influential and has received numerous citations, making it one of the most impactful papers in machine learning in recent years.

   5. [Sequence to Sequence Learning with Neural Networks, Sutskever et al (2014)](https://arxiv.org/abs/1409.3215)

      Sequence to Sequence Learning with Neural Networks" is a 2014 paper by Ilya Sutskever et al. that introduces a neural network architecture for performing sequence-to-sequence (Seq2Seq) tasks, such as machine translation and text summarization.

      The Seq2Seq architecture consists of two main components: an encoder network and a decoder network. The encoder network processes the input sequence and generates a fixed-length vector representation, known as the context vector, that summarizes the information in the input. The decoder network then uses the context vector to generate the output sequence.

      The authors show that the Seq2Seq architecture can be trained end-to-end using supervised learning, making it a powerful tool for NLP tasks. They also demonstrate that the architecture can be improved by incorporating attention mechanisms, which allow the decoder network to focus on different parts of the input sequence at different times.

      Overall, this paper introduced the Seq2Seq architecture and established it as a cornerstone of NLP research. The Seq2Seq architecture has been widely adopted and remains a popular choice for NLP tasks due to its simplicity, effectiveness, and versatility.


  6. [Long Short-term Memory, Hochreiter & Schmidhuber (1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)

      The unreasonable effectiveness of Recurrent Neural Network" is a seminal paper in the field of machine learning published in 1997 by Hochreiter & Schmidhuber. The paper introduces the Long Short-Term Memory (LSTM) architecture, which is a type of recurrent neural network (RNN) designed for sequence modeling tasks such as machine translation and text generation.

      The key idea behind the LSTM architecture is the use of a memory cell, which is a vector that stores information about the input sequence. The LSTM architecture is trained to update the memory cell in a way that allows it to store information about the input sequence and use this information to generate the output sequence.

  7. [Reinforcement Learning: An Introduction, Sutton & Barto (2018)](http://incompleteideas.net/book/the-book-2nd.html)

      "Reinforcement Learning: An Introduction" is a book by Richard S. Sutton and Andrew G. Barto, first published in 2018, that provides an introduction to the field of reinforcement learning (RL). RL is a subfield of machine learning that deals with the problem of decision-making in an environment where an agent must learn from its own experiences to maximize a reward signal.

      The book covers the fundamental concepts and algorithms of RL, including Markov decision processes, value functions, and policy gradient methods. The authors also discuss advanced topics, such as deep reinforcement learning, multi-agent RL, and inverse reinforcement learning.

      Throughout the book, the authors use concrete examples and intuitive explanations to make the concepts accessible to a wide audience. They also provide mathematical foundations and derivations to support the algorithms, making the book suitable for researchers and practitioners alike.

      Overall, "Reinforcement Learning: An Introduction" is a comprehensive and accessible resource for anyone interested in learning about RL. The book has become a seminal reference in the field and is widely regarded as one of the best introductory texts on RL.





