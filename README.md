# Quant Document Classifer using RNNs

Manually sorting and labeling financial records—like Balance Sheets, Cash Flow statements, Income Statements, Notes, and miscellaneous documents—can be slow and error-filled. This initiative overcomes those bottlenecks by harnessing deep learning for automated classification. A bidirectional LSTM RNN built with TensorFlow accurately assigns each document to its correct category. The model powers an intuitive Streamlit interface and is hosted on Hugging Face, delivering fast, reliable document management for finance teams.

This project was a collaboration with my colleague Frank B., who happily insisted on helping with the front-end on `Streamlit`!

Uses the following libraries and models:
- Libraries: Tensorflow, Keras, spaCy, NLTK, Word2Vec, Pandas, MPL, Streamlit, HuggingFace, gensim, imblearn
- RNNs, Bi-LSTM

Clone the repo and run the app using:
```bash
streamlit run app.py
```

**Data Collection**
The dataset consists of HTML documents sorted into five folders—Balance Sheets, Cash Flow, Income Statement, Notes, and Others—each corresponding to a financial document category. You can download the full dataset here:
 [Dataset Link](https://drive.google.com/file/d/1yj_ucy-VuX7fjKAQsR23ViTc-odb-eD-/view)

---

**Data Preprocessing**

* **Text Extraction and Cleaning**

  * Parsed each HTML file with BeautifulSoup.
  * Used NLTK to split text into word tokens.
  * Leveraged spaCy for lemmatization and to strip out stop words, special characters, and duplicate tokens from every sentence.

* **Word Embedding**

  * Trained a Word2Vec model on the tokenized sentences with 300-dimensional vectors.
  * Converted all text into these 300-D embeddings and encoded the document labels.
  * Saved the trained Word2Vec model for later inference.

* **Balancing the Classes**

  * Applied the SMOTETomek technique (combining SMOTE oversampling with Tomek-link cleaning) to generate synthetic examples for underrepresented classes and remove borderline overlaps, resulting in a balanced dataset.

* **Tensor Conversion & Batching**

  * Transformed features and targets into TensorFlow tensors.
  * Built a `tf.data.Dataset` with a batch size of 32 to streamline model training.

* **Train/Validation/Test Split**

  * Split the data into 80% training, 10% validation, and 10% testing using a custom function, ensuring each set is representative for reliable training and evaluation.

---

**Model Building and Training**

* **Optimized Data Pipeline**

  * Incorporated `.cache()`, `.shuffle()`, and `.prefetch()` into the TensorFlow data pipeline to minimize I/O bottlenecks and speed up training.

* **Network Architecture**

  * Designed a bidirectional LSTM–based RNN with multiple LSTM layers and dropout layers to reduce overfitting by randomly disabling neurons during training.

* **Activation Functions**

  * Used `tanh` activations throughout the LSTM layers and `sigmoid` for their forget gates.
  * Employed a `softmax` output layer for multiclass probability predictions.

* **Training Configuration**

  * Optimized the model using the Adam optimizer.
  * Minimized the `SparseCategoricalCrossentropy` loss to guide parameter updates and ensure efficient convergence.

* **Performance**

  * The final model achieved 96.2% accuracy on the test set, demonstrating strong robustness and generalization for financial document classification.
 
**Model Deployment and Inference**

* **Saving the Trained Model**
  After training, the final model and its learned weights were serialized and stored, ensuring that it can be reloaded later to process new HTML documents without retraining.

* **Streamlit Application**
  An easy-to-use Streamlit web app was created to let users upload HTML files for classification. The interface displays both the predicted document category (with confidence scores) and the rendered HTML content, so users can immediately verify and interpret the results.

* **Hugging Face Deployment**
  The Streamlit app was launched as a Hugging Face Space, giving users one-click access to drop in new HTML documents, receive classification outputs, and view the documents—all directly through their browser.

## Conclusion
This project was able to classify financial documents with relatively strong accuracy using DL. 
