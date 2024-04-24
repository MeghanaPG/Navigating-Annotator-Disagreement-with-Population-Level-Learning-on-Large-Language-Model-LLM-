import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import torch
from transformers import BertTokenizer, BertModel
from cnn_model import create_cnn_model
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from transformers import BertTokenizer, BertModel
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import matplotlib.pyplot as plt
from cohens_kappa import calculate_cohens_kappa
import seaborn as sns
from keras.wrappers.scikit_learn import KerasClassifier


def generate_bert_embeddings(text_data):
    print("Generating BERT embeddings...")
    # Loading pre-trained BERT tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    concatenated_text = text_data['query'] + ' ' + text_data['safety_annotations'] + ' ' + \
                        text_data['safety_annotation_reasons'] + ' ' + text_data['source']

    # Tokenize the text and convert to input IDs
    inputs = tokenizer(concatenated_text.tolist(), return_tensors='pt', padding=True, truncation=True, max_length=128)

    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state

    # Convert PyTorch tensor to NumPy array
    embeddings_np = embeddings.numpy()

    bert_embeddings_file = "bert_embeddings_file.npy"
    np.save(bert_embeddings_file, embeddings_np)

    return embeddings_np

def split_dataset():
    print("Splitting the data")
    # splitting the dataset into features (X) and labels (y)
    # raw_input_file = "safest_labels_dataset.csv"
    # raw_input_file = "majority_labels_dataset.csv"
    raw_input_file = "least_safest_labels_dataset.csv"

    # reading the csv file
    dataset = pd.read_csv(raw_input_file)

    print("------------PRINTING DATASET SHAPE-------------")
    print(dataset.shape)
    dataset = dataset.sample(15000, random_state=42)
    # dataset = dataset.sample(100, random_state=42)

    ####### To remove rare classes ############
    # Count the occurrences of each class in 'y'
    class_counts = dataset['safety_label'].value_counts()

    # Identify classes with only one member
    rare_classes = class_counts[class_counts == 1].index

    # Filter out samples associated with rare classes
    dataset_filtered = dataset[~dataset['safety_label'].isin(rare_classes)]

    # dataset = dataset.head(1000)
    # print("------------PRINTING DATASET SHAPE-------------")
    # print(dataset_filtered.shape)
    # dataset = dataset.head()
    x = dataset_filtered.drop(columns=['safety_label'])
    print(x)
    y = dataset_filtered['safety_label']

    #generating BERT embeddings for text data
    embeddings = generate_bert_embeddings(x)

    # Initialize the LabelEncoder
    label_encoder = LabelEncoder()

    # Encode the categorical labels
    y_encoded = label_encoder.fit_transform(y)

    # Convert y_encoded to integer type
    y_encoded = y_encoded.astype(int)

    test_size = 0.5
    random_state = 35


    X_train, X_temp, y_train, y_temp, embeddings_train, embeddings_temp = train_test_split(
        x, y_encoded, embeddings, test_size=test_size, random_state=random_state, stratify=y_encoded)
    X_val, X_test, y_val, y_test, embeddings_val, embeddings_test = train_test_split(
        X_temp, y_temp, embeddings_temp, test_size=test_size, random_state=random_state, stratify=y_temp)


    return X_train, X_val, X_test, y_train, y_val, y_test, embeddings_train, embeddings_val, embeddings_test

def main():
    # splitting the dataset and obtain BERT embeddings
    print("_______________________________________________________________________")
    X_train, X_val, X_test, y_train, y_val, y_test, embeddings_train, embeddings_val, embeddings_test = split_dataset()

    input_shape = (768,)
    num_classes = 5
    bert_embeddings_file = np.load('bert_embeddings_file.npy')

    # loading the CNN model here
    model = create_cnn_model(num_classes, bert_embeddings_file)

    y_train_categorical = to_categorical(y_train, num_classes=num_classes)
    y_val_categorical = to_categorical(y_val, num_classes=num_classes)

    y_test_categorical = to_categorical(y_test, num_classes=num_classes)

    # Evaluate the model using the one-hot encoded y_test
    test_loss, test_accuracy = model.evaluate(embeddings_test, y_test_categorical)
    #
    history = model.fit(embeddings_train, y_train_categorical,
                        epochs=20,  # Specify the number of epochs
                        batch_size=32,  # Specify the batch size
                        validation_data=(embeddings_val, y_val_categorical))

    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)
    print("Below is history")
    # print(history.history)


    ############Evaluating the model##################
    # Make predictions on the test data
    y_pred = model.predict(embeddings_test)

    # Convert predicted probabilities to class labels (for binary classification)
    y_pred_binary = (y_pred > 0.5).astype(int)

    y_pred_classes = np.argmax(y_pred, axis=1)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred_classes)
    print("Accuracy:", accuracy)

    # Calculate precision
    precision = precision_score(y_test, y_pred_classes, average='weighted')
    print("Precision:", precision)

    # Calculate recall
    recall = recall_score(y_test, y_pred_classes, average='weighted')
    print("Recall:", recall)

    # Calculate F1-score
    f1 = f1_score(y_test, y_pred_classes, average='weighted')
    print("F1-score:", f1)

    # Generate a classification report
    report = classification_report(y_test, y_pred_classes, zero_division=1)
    print("Classification Report:\n", report)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    # plt.title('Majority labels train and test')
    # Save the plot as an image file
    # plt.savefig('confusion_matrix-majority.png')
    # plt.title('Confusion Matrix - Trained on Majority label - Tested on Safest label')
    # # Save the plot as an image file
    # plt.savefig('confusion_matrix_maj-safest.png')
    plt.title('Confusion Matrix - Trained on Majority label - Tested on Least safest label')
    # # Save the plot as an image file
    plt.savefig('confusion_matrix_maj-least-safest.png')
    plt.close()

    # Calculate Cohen's Kappa
    y_pred_classes = np.argmax(y_pred, axis=1)
    kappa = calculate_cohens_kappa(y_test, y_pred_classes)
    print("Cohen's kappa:", kappa)

    # Suppress UndefinedMetricWarning
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

if __name__ == "__main__":
    main()
