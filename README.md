# ✉ Spam-Ham Classification using Word2Vec and Avg Word2Vec

This project focuses on classifying emails or messages as **Spam** or **Ham (Not Spam)** using **Word Embedding Techniques** — specifically **Word2Vec** and **Average Word2Vec**.


## Dataset

- The dataset used in this project is a labeled email/message dataset sourced from [Kaggle].
- It contains two columns:
  - `label`: spam or ham
  - `message`: the actual text message


## 🔍 Objective

To build a machine learning model that:
- Converts text messages into vector form using Word2Vec & Avg Word2Vec.
- Classifies them into spam or ham using a suitable ML algorithm.
- Evaluates the model using metrics like precision score.


## 🛠️ Technologies Used

- Python
- TQDM
- Jupyter Notebook / Google Colab
- NLP with **gensim** for Word2Vec
- **Scikit-learn** for ML modeling and evaluation
- **Pandas**, **NumPy** for data manipulation


## 📈 Workflow

1. **Data Preprocessing**
   - Cleaning text (lowercasing, removing stopwords, punctuation, etc.)
   - Tokenization

2. **Feature Engineering**
   - Word Embedding with:
     - Word2Vec
     - Average Word2Vec (Mean of word vectors)

3. **Model Training**
   - Splitting dataset into training and test sets
   - Training with algorithms like **Random Fores Classifier**

4. **Evaluation**
   - Accuracy score
   - Classification Report : Precision, Recall, F1-Score



## 🧠 Word2Vec Techniques Explained

### 🔹 Word2Vec
- Maps words to vectors based on context using the **Skip-gram** or **CBOW** model.

### 🔹 Avg Word2Vec
- Takes the average of all word vectors in a sentence/message to get a single fixed-size vector.



## 📊 Results

                precision    recall  f1-score   support

       False       0.98      0.98      0.98       960
        True       0.89      0.85      0.87       154
     accuracy       _         _        0.96      1114


## ✅ Conclusion

- Word embeddings provide better feature representation for text classification compared to Bag of Words or TF-IDF.
- Avg Word2Vec performs efficiently for sentence-level classification.
- The classifier achieved good precision and recall in detecting spam.


## 📌 Future Work

- Try other embeddings like **FastText**, **BERT**.
- Use deep learning models like **LSTM** or **CNN** for better accuracy.
- Deploy the model using **Flask**, **Streamlit**, or other web frameworks.

