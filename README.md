# Sentiment Analysis on IMDB Movie Reviews



## About the Project
This project focuses on classifying IMDB movie reviews as positive or negative using advanced Natural Language Processing (NLP) techniques. It involves preprocessing text data with NLTK or spaCy to clean and tokenize reviews, followed by converting text into numerical embeddings using TF-IDF and Word2Vec. Multiple models, including Logistic Regression, Long Short-Term Memory (LSTM) networks, and BERT, are trained for sentiment classification. The project evaluates model performance through accuracy comparisons and other metrics, providing insights into effective NLP approaches for sentiment analysis. This work is applicable to review-based platforms for automated sentiment detection and analysis.

---

## Folder Structure
```
/README.md                # Project overview and instructions
/IMDB.ipynb               # Jupyter Notebook with text preprocessing, model training, and accuracy comparisons
/netflix_stock.csv               # Dataset containing IMDB movie reviews used in the analysis
```
---

## Tech Stack Used
- **Programming Language**: Python 3
- **Libraries**:
  - **NLTK/spaCy**: For text preprocessing
  - **Scikit-learn**: For TF-IDF vectorization and Logistic Regression
  - **Gensim**: For Word2Vec embeddings
  - **TensorFlow/Keras**: For LSTM model implementation
  - **Transformers (Hugging Face)**: For BERT model implementation
  - **Pandas**: Data manipulation and analysis
  - **NumPy**: Numerical computations
  - **Matplotlib/Seaborn**: For visualizations

---



## How to Access the Project
1. Ensure Jupyter Notebook or JupyterLab is installed (via Anaconda or `pip install notebook`).
2. Install required libraries: `pip install nltk spacy scikit-learn gensim tensorflow transformers pandas numpy matplotlib seaborn`.
3. Download spaCy model if used: `python -m spacy download en_core_web_sm`.
4. Clone or download the project folder from GitHub.
5. Navigate to the folder in your terminal or command prompt.
6. Run `jupyter notebook` to launch the Jupyter server.
7. Open `IMDB.ipynb` in the browser interface.
Alternatively, upload the `.ipynb` file to Google Colab (colab.research.google.com) for online execution. Ensure `netflix_stock.csv` is in the same directory or uploaded to Colab to load the data correctly.



---

## License
This project is open source and available under the [MIT License](LICENSE).

---

### **Author**  
 **Priya Sah**  
 Email: priyasah3005@gmail.com  
 GitHub: [github.com/PRIYA300705](https://github.com/PRIYA300705)  