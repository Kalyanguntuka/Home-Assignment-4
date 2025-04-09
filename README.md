# Homeassignment4
# README - Home Assignment 4 (Neural Networks and Deep Learning)

## Student Information
**Name:** Kalyan Guntuka
**Course:** CS5720 Neural Networks and Deep Learning  
**Semester:** Spring 2025  
**University:** University of Central Missouri  
**Student ID :** 700757036

---

## Q1 NLP Preprocessing Pipeline

This Python script demonstrates a basic Natural Language Processing (NLP) preprocessing pipeline using **NLTK (Natural Language Toolkit)**. It includes the following steps:

1. **Tokenization** – Splitting a sentence into individual words.
2. **Stopword Removal** – Removing common English words that add little meaning (e.g., "the", "is", "in").
3. **Stemming** – Reducing words to their root form using the Porter Stemmer.

---

## 1. What is the difference between stemming and lemmatization?
Stemming: A simplified method for finding a word's basic form by removing suffixes. This approach is quick but can produce non-words. 
Lemmatization: A more advanced technique that identifies a word's dictionary form (lemma) by considering its grammatical role. This method yields actual words.

## 2. Why might removing stop words be useful in some NLP tasks, and when might it be harmful?
•	Useful in tasks like:
o	Text classification
o	Topic modeling
o	Information retrieval These tasks often benefit from removing stopwords to focus on more meaningful content.
•	Harmful in tasks like:
o	Sentiment analysis (e.g., removing "not" can change sentiment)
o	Machine translation
o	Question answering In these cases, stop words can carry important syntactic or semantic meaning.

---

## Q2: Named Entity Recognition with SpaCy 

This Python script uses **spaCy**, a powerful NLP library, to perform **Named Entity Recognition (NER)** on a sentence. NER automatically detects and classifies entities such as names, organizations, locations, dates, etc.

## Features

- Detects named entities in text
- Displays:
  - Entity text (e.g., "Barack Obama")
  - Entity label (e.g., PERSON, GPE, DATE)
  - Start and end character positions in the original string

---
## 1. How does NER differ from POS tagging in NLP?
Named Entity Recognition (NER): This process pinpoints and labels specific entities within text, such as people, organizations, and locations, classifying them into predefined categories. For instance, it recognizes "Apple" as an organization. 
Part-of-Speech (POS) Tagging: This process focuses on the grammatical role of each word in a sentence, assigning labels like noun, verb, or adjective. So, it identifies "Apple" as a proper noun. 
In essence: NER is about identifying the meaning of a word as a real-world entity, while POS tagging is about understanding the word's grammatical function within the sentence structure.

## 2. Two Real-World Applications of NER:
1.	Financial News Analysis:
o	Extracts companies, dates, events, and figures from articles.
o	Example: Identifying "Tesla", "Elon Musk", or "$300 million" in news for stock predictions.
2.	Search Engines:
o	Helps understand queries better by recognizing entities.
o	Example: In "weather in Paris next week", it detects "Paris" (GPE) and "next week" (DATE).

---

## Q3: Scaled Dot-Product Attention 

This script implements the **Scaled Dot-Product Attention** mechanism using **NumPy**. It is a core component of the Transformer architecture used in modern NLP models like BERT, GPT, and T5.

## What It Does

The script performs the following steps:

1. **Dot Product** of the Query (Q) and Transposed Key (Kᵀ)
2. **Scaling** the result by √dₖ (to reduce variance and improve stability)
3. **Softmax** to generate attention weights
4. **Weighted Sum** of values (V) using the attention weights

---
  
## 1.	Why do we divide the attention score by √d in the scaled dot-product attention formula?
•	Without scaling, the dot product values can become very large when the dimensionality d is high.
•	This leads to extremely small gradients after the softmax due to saturation, making learning unstable.
•	Dividing by √d normalizes the scores, reducing variance and helping the softmax function stay in a sensitive, non-saturated range

## 2.	How does self-attention help the model understand relationships between words in a sentence?

Self-attention: Enables words to directly consider the relevance of other words within the same sentence, building contextual understanding. 
•	For instance, "sat" in a sentence can focus on "cat" to understand who is performing the action.
Advantages: 
•	Effectively captures relationships between words, even those far apart, overcoming limitations of sequential models.
•	Generates rich, context-sensitive word representations, significantly improving performance in tasks like translation and summarization.

---

## Q4: Sentiment Analysis using HuggingFace Transformers

## Sentiment Analysis using HuggingFace Transformers

This script uses the  HuggingFace "transformers" library to perform **sentiment analysis** with a pre-trained model. It identifies whether a sentence expresses **positive** or **negative** sentiment, and gives a confidence score.


##  What It Does

- Loads a **pre-trained transformer model** for sentiment classification
- Analyzes the sentiment of a given input sentence
- Outputs:
  - **Label** (e.g., POSITIVE, NEGATIVE)
  - **Confidence score** (between 0 and 1)
 
---

## 1.	What is the main architectural difference between BERT and GPT? Which uses an encoder and which uses a decoder?

Model Comparison:
•	BERT: 
o	Architecture: Uses only the encoder part of the Transformer model.
o	Direction: Processes text in both directions simultaneously (bidirectional).
o	Best for: Tasks that require understanding the full context of a sentence, such as classification and question answering.
•	GPT: 
o	Architecture: Uses only the decoder part of the Transformer model.
o	Direction: Processes text sequentially, from left to right (unidirectional).
o	Best for: Tasks that involve generating text, like writing stories or engaging in conversations.
Key Difference:
•	BERT excels at understanding context from all sides, while GPT is designed to generate text by predicting what comes next in a sequence.


## 2.	Explain why using pre-trained models (like BERT or GPT) is beneficial for NLP applications instead of training from scratch.

Efficiency: They drastically reduce the time and computational power needed for training, as they've already been trained on vast amounts of data. 
Adaptability: They enable effective transfer learning, allowing you to achieve good results on your specific tasks with smaller datasets through fine-tuning. 
Performance: They provide strong baseline performance, as they have learned complex language patterns from extensive training.

---


