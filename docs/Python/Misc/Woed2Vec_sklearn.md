---
title: Integrating Word2Vec with Scikit-Learn Pipelines
date: 2025-02-03
author: Shanaka DeSoysa
description: Integrating Word2Vec with Scikit-Learn Pipelines.
---

# Integrating Word2Vec with Scikit-Learn Pipelines

In this post, we'll explore how to integrate a custom `Word2Vec` transformer with `scikit-learn` pipelines. This allows us to leverage the power of Word2Vec embeddings in a machine learning workflow.

## Introduction

Word2Vec is a popular technique for natural language processing (NLP) that transforms words into continuous vector representations. These vectors capture semantic relationships between words, making them useful for various NLP tasks. However, integrating Word2Vec with `scikit-learn` pipelines requires a custom transformer. Let's walk through the process step-by-step.

### Sample Data

We'll start with a small sample corpus and corresponding labels:

```python
corpus = [
    "This is the first document.",
    "This document is the second document.",
    "And this is the third one.",
    "Is this the first document?",
]
labels = [1, 0, 1, 1]
```

### Custom Word2Vec Transformer

We'll create a custom transformer class that fits a Word2Vec model and transforms documents into vector representations:

```python
from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class Word2VecTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=100, window=5, min_count=1, workers=4):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers

    def fit(self, X, y=None):
        tokenized_sentences = [sentence.split() for sentence in X]
        self.model = Word2Vec(sentences=tokenized_sentences, vector_size=self.vector_size, 
                              window=self.window, min_count=self.min_count, workers=self.workers)
        return self

    def transform(self, X):
        tokenized_sentences = [sentence.split() for sentence in X]
        return np.array([
            np.mean([self.model.wv[word] for word in sentence if word in self.model.wv] 
                    or [np.zeros(self.vector_size)], axis=0)
            for sentence in tokenized_sentences
        ])
```

### Defining the Pipeline

Next, we'll define a pipeline that includes `TfidfVectorizer`, our custom `Word2VecTransformer`, and a `LogisticRegression` classifier:

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('word2vec', Word2VecTransformer(vector_size=100, window=5, min_count=1, workers=4)),
    ('clf', LogisticRegression())
])
```

### Training and Prediction

We'll fit the pipeline on our sample data and make predictions on a new document:

```python
# Fit the pipeline
pipeline.fit(corpus, labels)

# Make predictions
predictions = pipeline.predict(["This is a new document."])
print(predictions)
```

### Explanation

1. **Import Libraries**: We import necessary libraries for data manipulation (`pandas`), Word2Vec model (`gensim`), and creating a pipeline (`sklearn`).
2. **Sample Data**: We create a sample corpus and corresponding labels.
3. **Custom Transformer**: We define a custom `Word2VecTransformer` class that fits a Word2Vec model and transforms documents into vector representations.
4. **Pipeline Definition**: We define a pipeline with `TfidfVectorizer`, our custom `Word2VecTransformer`, and `LogisticRegression`.
5. **Training and Prediction**: We fit the pipeline on the sample data and make predictions on a new document.

### Conclusion

By creating a custom `Word2Vec` transformer, we can seamlessly integrate Word2Vec embeddings into `scikit-learn` pipelines. This approach allows us to leverage the power of Word2Vec in a structured machine learning workflow.
