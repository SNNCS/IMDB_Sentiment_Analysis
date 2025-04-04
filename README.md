# IMDB Sentiment Analysis

This project implements a sentiment analysis model to classify IMDB movie reviews as either positive or negative using TensorFlow and Keras.

## Dataset

The dataset used is the [IMDB dataset](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz), which contains 50,000 movie reviews labeled as positive or negative. The dataset is divided into a training set, validation set, and test set.

- **Training set**: 25,000 reviews
- **Test set**: 25,000 reviews

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/SNNCS/IMDB_Sentiment_Analysis.git
cd IMDB_Sentiment_Analysis
```

### 2. Install Dependencies
Install the required dependencies using `pip`. If you're using a GPU, you can install the GPU-specific TensorFlow version.

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset
The dataset will be automatically downloaded when running the code. If you want to manually download it, visit this link and place it in the `data/` folder.

### 4. Train the Model
The model is defined and trained in `src/model.py`. To train the model, run the following command:

```bash
python src/model.py
```

### 5. Evaluate the Model
After training, evaluate the model's performance on the test dataset by running:

```bash
python src/evaluate.py
```

### 6. Results
After training for 10 epochs, the model's loss and accuracy on the test dataset will be printed. Additionally, a plot showing the training and validation loss over epochs will be displayed.

## Model Architecture
The model consists of the following layers:

- **Embedding layer**: Converts word indices into dense vectors of fixed size.
- **Dropout layers**: Helps prevent overfitting.
- **GlobalAveragePooling1D**: Averages the word embeddings to form a fixed-size output.
- **Dense layer**: A fully connected layer for binary classification.

## Dependencies
- TensorFlow >= 2.0
- NumPy
- Matplotlib

To install all dependencies:

```bash
pip install -r requirements.txt
```
