---
title: Draft ML
date: 2024-12-09
author: Shanaka DeSoysa
description: Draft ML WIP.
---

To classify survey comments into predefined topics, especially when allowing up to three classes per comment, you can follow these steps:

1. **Data Preparation**:
   - **Collect and Clean Data**: Gather all survey comments and clean the text by removing any irrelevant information (e.g., special characters, stop words).
   - **Label Data**: If you have a labeled dataset, use it to train your model. If not, you might need to manually label a subset of comments to create a training set.

2. **Feature Extraction**:
   - **Text Vectorization**: Convert text data into numerical format using techniques like TF-IDF (Term Frequency-Inverse Document Frequency) or word embeddings (e.g., Word2Vec, GloVe, BERT).

3. **Model Selection**:
   - **Choose a Classification Algorithm**: Use machine learning models such as Logistic Regression, Support Vector Machines (SVM), or more advanced models like BERT for text classification.
   - **Multi-label Classification**: Since each comment can belong to multiple topics, use multi-label classification techniques. Algorithms like Binary Relevance, Classifier Chains, or neural networks designed for multi-label tasks can be effective.

4. **Training the Model**:
   - **Split Data**: Divide your data into training and testing sets.
   - **Train the Model**: Use your training data to train the chosen model. Ensure you use appropriate evaluation metrics like F1-score, Precision, and Recall for multi-label classification.

5. **Classification**:
   - **Predict Topics**: For each new survey comment, use the trained model to predict the relevant topics. Ensure the model can assign up to three topics per comment.

6. **Post-processing**:
   - **Thresholding**: Apply a threshold to the model's output probabilities to decide which topics to assign to each comment.
   - **Evaluation**: Continuously evaluate the model's performance on new data and adjust as necessary.

7. **Automation and Integration**:
   - **Automate the Process**: Integrate the model into your survey analysis pipeline to automatically classify new comments.
   - **Feedback Loop**: Use feedback from the classified comments to retrain and improve the model over time.

By following these steps, you can effectively classify survey comments into predefined topics, allowing for a more nuanced analysis of employee experience. If you need more detailed guidance on any specific step, feel free to ask!

Here's an example of how you can use Python with the `scikit-learn` library to classify survey comments into predefined topics, allowing up to three topics per comment. This example uses a multi-label classification approach with a `OneVsRestClassifier` and `LogisticRegression`.

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Sample data
data = {
    'comment': [
        "I love the flexible work hours.",
        "The management is supportive.",
        "Great work-life balance.",
        "Need better communication from the top.",
        "More opportunities for growth would be nice."
    ],
    'topics': [
        ["Work Hours", "Work-Life Balance"],
        ["Management"],
        ["Work-Life Balance"],
        ["Communication"],
        ["Career Growth"]
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Binarize the labels
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['topics'])

# Vectorize the comments
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['comment'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = OneVsRestClassifier(LogisticRegression())
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred, target_names=mlb.classes_))

# Example prediction
new_comments = ["I appreciate the career development programs."]
X_new = vectorizer.transform(new_comments)
predictions = model.predict(X_new)
predicted_topics = mlb.inverse_transform(predictions)
print(predicted_topics)
```

This code does the following:
1. **Data Preparation**: Creates a sample dataset with comments and their associated topics.
2. **Label Binarization**: Converts the list of topics into a binary format suitable for multi-label classification.
3. **Text Vectorization**: Uses TF-IDF to convert text comments into numerical features.
4. **Model Training**: Trains a `OneVsRestClassifier` with `LogisticRegression` on the training data.
5. **Evaluation**: Evaluates the model's performance on the test set and prints a classification report.
6. **Prediction**: Demonstrates how to predict topics for new comments.

You can expand this code to include your full dataset and adjust the model parameters as needed. If you have any specific requirements or need further customization, let me know!

To test if the same topic structure works on another kind of survey and to include an "Other" topic for themes not covered, you can follow these steps:

1. **Data Collection and Preparation**:
   - **Collect New Survey Data**: Gather responses from the new survey.
   - **Clean and Preprocess**: Clean the text data similarly to how you did for the original survey.

2. **Feature Extraction**:
   - **Vectorize New Survey Comments**: Use the same vectorization method (e.g., TF-IDF) and the same vectorizer instance to transform the new survey comments into numerical features.

3. **Model Application**:
   - **Load Pre-trained Model**: Use the model trained on the original survey data.
   - **Predict Topics**: Apply the model to the new survey data to predict topics.

4. **Evaluation**:
   - **Manual Review**: Manually review a sample of the predictions to see if the topics make sense in the context of the new survey.
   - **Quantitative Metrics**: If you have labeled data for the new survey, calculate metrics like Precision, Recall, and F1-score to evaluate the model's performance.

5. **Incorporate "Other" Topic**:
   - **Thresholding**: Set a probability threshold for topic assignment. If no topic exceeds the threshold, classify the comment as "Other".
   - **Manual Labeling**: Manually review comments classified as "Other" to identify any new themes that might need to be added to the predefined topics.

Here's an example of how you can modify the code to include an "Other" topic:

```python
import numpy as np

# Function to predict topics with an "Other" category
def predict_with_other(model, vectorizer, mlb, comments, threshold=0.2):
    X_new = vectorizer.transform(comments)
    probabilities = model.predict_proba(X_new)
    predictions = (probabilities >= threshold).astype(int)
    
    # If no topic exceeds the threshold, classify as "Other"
    for i in range(predictions.shape[0]):
        if not np.any(predictions[i]):
            predictions[i] = np.zeros(predictions.shape[1])
            predictions[i, -1] = 1  # Assuming the last column is "Other"
    
    return mlb.inverse_transform(predictions)

# Example usage
new_comments = ["I appreciate the career development programs.", "The cafeteria food needs improvement."]
predicted_topics = predict_with_other(model, vectorizer, mlb, new_comments)
print(predicted_topics)
```

In this code:
1. **Thresholding**: The `predict_with_other` function applies a threshold to the predicted probabilities. If no topic exceeds the threshold, the comment is classified as "Other".
2. **Manual Review**: You can manually review comments classified as "Other" to identify any new themes.

By following these steps, you can test the applicability of your topic structure on a new survey and ensure that any new themes are captured effectively. If you need further customization or have specific requirements, feel free to ask!

----------------------------------

Here's a Python code snippet that classifies survey comments into predefined topics, including "No Topic" and "Other Topic," and assigns up to 3 classes per survey comment:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Predefined topics
topics = [
    "Work Environment",
    "Compensation",
    "Management",
    "Work-Life Balance",
    "Career Growth",
    "Company Culture",
    "Training and Development",
    "Job Security",
    "Benefits",
    "Communication",
    "No Topic",
    "Other Topic"
]

# Sample survey comments
survey_comments = [
    "I am happy with the work environment and the management is supportive.",
    "The compensation is not competitive and there are no benefits.",
    "I feel there is no work-life balance and the company culture is toxic.",
    "There are no opportunities for career growth and job security is a concern.",
    "The training and development programs are excellent.",
]

# Vectorize the topics and survey comments
vectorizer = TfidfVectorizer()
topic_vectors = vectorizer.fit_transform(topics)
comment_vectors = vectorizer.transform(survey_comments)

# Function to classify survey comments to topics
def classify_comments(comments, topic_vectors, n_classes=3):
    classifications = []
    
    for comment_vector in comments:
        # Calculate cosine similarity between comment and topics
        similarities = cosine_similarity(comment_vector, topic_vectors).flatten()
        
        # Get indices of top n_classes topics
        top_indices = similarities.argsort()[-n_classes:][::-1]
        
        # Get the corresponding topics
        top_topics = [topics[i] for i in top_indices]
        
        classifications.append(top_topics)
    
    return classifications

# Classify the survey comments
classified_comments = classify_comments(comment_vectors, topic_vectors)

# Print the results
for i, comment in enumerate(survey_comments):
    print(f"Survey Comment: {comment}")
    print(f"Classified Topics: {classified_comments[i]}")
    print()
```

This code uses TF-IDF vectorization to convert the survey comments and predefined topics into vectors. It then calculates the cosine similarity between each comment and the topics, selecting the top 3 most similar topics for each comment. You can adjust the `n_classes` parameter to classify a different number of topics per comment.

The code classifies a survey comment to "Other Topic" based on the cosine similarity between the comment and the predefined topics. If a comment doesn't closely match any of the specific topics, it might have a higher similarity score with "Other Topic."

Here's a more detailed explanation:

1. **Vectorization**: The `TfidfVectorizer` converts both the predefined topics and the survey comments into numerical vectors based on the importance of words.

2. **Cosine Similarity**: For each survey comment, the code calculates the cosine similarity between the comment vector and each topic vector. This measures how similar the comment is to each topic.

3. **Top Matches**: The code then identifies the top 3 topics with the highest similarity scores for each comment.

If a comment doesn't strongly match any specific topic, the similarity score for "Other Topic" might be among the highest, leading to its classification under "Other Topic."

For example, if a comment is about something not covered by the predefined topics, it might not have high similarity scores with any of them, but it could still have a relatively higher score with "Other Topic" due to the general nature of this category.

Would you like to see an example with a specific comment?

Got it! To validate if the existing topics will cover a new survey, you can follow these steps:

1. **Analyze Topic Coverage**: Check how well the predefined topics match the new survey comments. This involves calculating the similarity between the new comments and the predefined topics.

2. **Identify Uncovered Areas**: Determine if there are significant portions of the new survey comments that don't fit well into any of the predefined topics. This can be done by looking at the similarity scores and identifying comments with low scores across all topics.

3. **Adjust Topics if Necessary**: If you find that many comments don't fit well into the existing topics, you may need to adjust your predefined topics or add new ones.

Here's a step-by-step approach with code:

### Step 1: Vectorize New Survey Comments

```python
# New survey comments for testing
new_survey_comments = [
    "The office environment is very comfortable and the management is approachable.",
    "I am not satisfied with the salary and the benefits provided.",
    "There is a lack of communication and the work-life balance is poor.",
    "I appreciate the career growth opportunities and the training programs.",
    "The company culture is inclusive and supportive."
]

# Preprocess and vectorize the new survey comments
new_comment_vectors = vectorizer.transform(new_survey_comments)
```

### Step 2: Calculate Similarity Scores

```python
# Calculate similarity scores for new survey comments
def calculate_similarity_scores(comments, topic_vectors):
    similarity_scores = []
    
    for comment_vector in comments:
        # Calculate cosine similarity between comment and topics
        similarities = cosine_similarity(comment_vector, topic_vectors).flatten()
        similarity_scores.append(similarities)
    
    return similarity_scores

# Get similarity scores for new survey comments
similarity_scores = calculate_similarity_scores(new_comment_vectors, topic_vectors)
```

### Step 3: Analyze Coverage

```python
# Analyze coverage of predefined topics
def analyze_coverage(similarity_scores, threshold=0.2):
    uncovered_comments = []
    
    for i, scores in enumerate(similarity_scores):
        # Check if all similarity scores are below the threshold
        if all(score < threshold for score in scores):
            uncovered_comments.append(i)
    
    return uncovered_comments

# Identify comments that are not well covered by predefined topics
uncovered_comments = analyze_coverage(similarity_scores)

# Print results
for i in uncovered_comments:
    print(f"Uncovered Comment: {new_survey_comments[i]}")
```

### Step 4: Adjust Topics if Necessary

If you find that several comments are not well covered by the existing topics, you might need to:

- **Add New Topics**: Introduce new topics that better capture the themes in the new survey comments.
- **Refine Existing Topics**: Adjust the definitions or scope of existing topics to better match the new data.

By following these steps, you can validate whether your existing topics are sufficient for the new survey and make necessary adjustments to improve coverage.
