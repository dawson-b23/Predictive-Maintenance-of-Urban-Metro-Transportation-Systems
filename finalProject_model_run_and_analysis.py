## ## ## ## ## ## ## ## ## ## ## ##
##  finalProject_model_run_and_analysis.py
##
##  Dawson Burgess
##  burg1648
##  CS504 Final Project
##
##
## ## ## ## ## ## ## ## ## ## ## ##
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.utils import compute_class_weight, resample
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

## ## ## ## #
## GLOBALS ##
## ## ## ## #

key_features = [
    "Motor_current",
    "Oil_temperature",
    "DV_eletric",
    "TP2",
    "DV_pressure",
]

## ## ## ## ##
## METHODS ##
## ## ## ## ##


# resample data to deal with massive data imbalances
def resample_to_ratio(X, y, target_ratio=2):
    # Separate majority and minority classes
    X_majority = X[y == 0]
    y_majority = y[y == 0]
    X_minority = X[y == 1]
    y_minority = y[y == 1]

    # target majority size
    target_majority_size = min(len(X_majority), target_ratio * len(X_minority))

    # Downsample majority class
    X_majority_downsampled, y_majority_downsampled = resample(
        X_majority,
        y_majority,
        replace=False,
        n_samples=target_majority_size,
        random_state=42,
    )

    # Combine with minority class
    X_resampled = np.vstack((X_majority_downsampled, X_minority))
    y_resampled = np.hstack((y_majority_downsampled, y_minority))

    # Shuffle the dataset
    indices = np.arange(X_resampled.shape[0])
    np.random.shuffle(indices)

    return X_resampled[indices], y_resampled[indices]


# visualizing model training
def plot_training_history(history, title="Model Training"):
    plt.figure(figsize=(12, 6))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title(f"{title} - Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title(f"{title} - Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(title + ".png")
    # plt.show()
    plt.clf()


# confusion_matrix to see how model performs
def plot_confusion_matrix(true_labels, predictions, title="Confusion Matrix"):
    cm = confusion_matrix(true_labels, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap="Blues", values_format="d")
    plt.title(title)
    plt.savefig(title + ".png")
    # plt.show()
    plt.clf()


# roc curve for model
def plot_roc_curve(true_labels, predicted_probs, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")  # Random baseline
    plt.title(title)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig(title + ".png")
    # plt.show()
    plt.clf()


# for failure prediction / feature importance
def random_forest_model(X_train, X_test, y_train, y_test):

    # Initialize and train the Random Forest model
    rf = RandomForestClassifier(
        n_estimators=500,  # seems to work well
        max_depth=20,  # help prevent over-fitting
        random_state=42,
        n_jobs=-1,  # Utilize all available cores, REALLY speeds things up
        class_weight="balanced",
    )
    rf.fit(X_train, y_train)
    print("RF training complete")

    # Make predictions
    print("Making predicitions with RF")
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)[:, 1]

    # Evaluate the model
    print("Random Forest Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_pred_proba))

    feature_importance = rf.feature_importances_
    feature_importance_df = pd.DataFrame(
        {"Feature": key_features, "Importance": feature_importance}
    ).sort_values(by="Importance", ascending=False)

    # Display feature importances
    print("Feature Importance:")
    print(feature_importance_df)

    # Visualize feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(
        feature_importance_df["Feature"],
        feature_importance_df["Importance"],
        color="skyblue",
    )
    plt.gca().invert_yaxis()  # Most important at the top
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.title("Feature Importance in Random Forest")
    # plt.show()
    plt.savefig("Feature_importance_RF.png")
    plt.clf()


def svm_model(X_train, X_test, y_train, y_test):
    # Initialize and train the SVM
    # svm = LinearSVC(
    #     max_iter=1000, random_state=42, dual=False, verbose=1, class_weight="balanced"
    # )
    svm = SVC(kernel="rbf", random_state=42)
    svm.fit(X_train, y_train)
    print("SVM training complete")

    # Make predictions
    print("Making predicitons with SVM")
    y_pred = svm.predict(X_test)
    y_scores = svm.decision_function(X_test)

    # Evaluate the SVM
    print("SVM Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("ROC AUC score:", roc_auc_score(y_test, y_scores))

    # confusion_matrix
    plot_confusion_matrix(y_test, y_pred, title="svm confusion_matrix")

    # ROC curve
    plot_roc_curve(y_test, y_scores, title="SVM ROC Curve")


def neural_network_model(X_train, X_test, y_train, y_test):
    # Define the model
    model = Sequential(
        [
            Input(shape=(X_train.shape[1],)),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),  # Binary classification output
        ]
    )

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    # Early stopping
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=10,
        batch_size=64,
        callbacks=[early_stopping],
        verbose=1,
    )
    print("NN training complete")

    # Visualizations
    plot_training_history(history, title="Neural Network Model Training")

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train).astype(np.float32)
    y_test = np.array(y_test).astype(np.float32)

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print("Neural network results")
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")

    # Make predictions
    print("Making predictions with NN")
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(
        int
    )  # Threshold at 0.5 for binary classification

    # Evaluate the model
    print("Neural Network Classification Report:")
    print(classification_report(y_test, y_pred_binary))
    print("Neural Network Accuracy:", accuracy_score(y_test, y_pred_binary))
    print("Neural Network ROC AUC Score:", roc_auc_score(y_test, y_pred_binary))

    plot_confusion_matrix(
        y_test, y_pred_binary, title="Neural network confusion_matrix"
    )

    # Plot ROC Curve
    plot_roc_curve(y_test, y_pred_binary, title="Neural Network ROC Curve")


# for reshaping data into sequences for use in LSTM ---- SCRAPPED
def create_sequences(data, labels, time_steps):
    sequences, seq_labels = [], []
    for i in range(len(data) - time_steps):
        sequences.append(data[i : i + time_steps])
        seq_labels.append(
            labels[i + time_steps]
        )  # Label corresponds to the end of the sequence
    return np.array(sequences), np.array(seq_labels)


def lstm_model(X_train, X_test, y_train, y_test):
    time_steps = 20

    # Create sequences
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, time_steps)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, time_steps)

    # check seq
    print("Training Classes:", np.bincount(y_train_seq))
    print("Testing Classes:", np.bincount(y_test_seq))

    # Define LSTM model
    # i tried a LOT of different architectures here.... no luck
    model = Sequential(
        [
            Input(shape=(X_train_seq.shape[1], X_train_seq.shape[2])),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(64),
            Dropout(0.3),
            # Dense(64, activation="relu"),
            # Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ]
    )
    # model compile
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # class weights
    class_weights = {0: 1.0, 1: len(y_train_seq) / (2 * np.sum(y_train_seq))}

    # Train the model
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    history = model.fit(
        X_train_seq,
        y_train_seq,
        validation_split=0.2,
        epochs=20,
        batch_size=64,
        class_weight=class_weights,
        callbacks=[early_stopping],
        verbose=1,
    )
    print("LSTM training complete")

    # Visualizations
    plot_training_history(history, title="LSTM Model Training")

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test_seq, y_test_seq)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")

    # Predictions
    print("Making predicitions with LSTM")
    predictions = (model.predict(X_test_seq) > 0.5).astype("int32")

    # Metrics
    print("LSTM Classification Report")
    print(classification_report(y_test_seq, predictions))
    print("LSTM ROC AUC Score:", roc_auc_score(y_test_seq, predictions))

    plot_confusion_matrix(y_test_seq, predictions, title="LSTM confusion_matrix")


def run_lstm(df):

    # Separate features and labels
    X = df.drop("failure_anomaly", axis=1).values
    y = df["failure_anomaly"].values

    # Stratified split before sequence generation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("LSTM MODEL")
    lstm_model(X_train, X_test, y_train, y_test)


def run_other_models(df):
    # Separate features and labels
    X = df.drop("failure_anomaly", axis=1)
    y = df["failure_anomaly"]

    # Stratified train-test split to ensure test set contains both classes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Resample only the training set
    X_train_resampled, y_train_resampled = resample_to_ratio(
        X_train, y_train, target_ratio=1
    )

    # Convert resampled data back to DataFrame
    X_train_resampled = pd.DataFrame(X_train_resampled, columns=key_features)
    X_test = pd.DataFrame(X_test, columns=key_features)

    ## ## ## ## ## ## ## ## ## ##
    ## DEBUGGING INFO IF NEEDED ##
    ## ## ## ## ## ## ## ## ## ##
    print("Original Training Class Distribution:")
    print(f"Class 0: {np.sum(y_train == 0)}")
    print(f"Class 1: {np.sum(y_train == 1)}")

    print("Resampled Training Class Distribution:")
    print(f"Class 0: {np.sum(y_train_resampled == 0)}")
    print(f"Class 1: {np.sum(y_train_resampled == 1)}")

    print("Test Class Distribution:")
    print(f"Class 0: {np.sum(y_test == 0)}")
    print(f"Class 1: {np.sum(y_test == 1)}")

    # Train and evaluate Random Forest
    print("running rf")
    random_forest_model(X_train_resampled, X_test, y_train_resampled, y_test)

    print("running svm")
    # Train and evaluate SVM
    svm_model(X_train_resampled, X_test, y_train_resampled, y_test)

    print("running nn")
    # Train and evaluate Neural Network
    neural_network_model(X_train_resampled, X_test, y_train_resampled, y_test)


## ## ## ##
##  MAIN ##
## ## ## ##
if __name__ == "__main__":

    # open a file named model_output for saving output to
    # remove comment to write to an output file
    # sys.stdout = open("model_output_new", "w")

    # this assumes you have run the other file for data preperation and cleansing

    # loads cleaned csv
    df = pd.read_csv("data_with_failures_included.csv")
    df = df.drop(columns="timestamp")

    # filter out everything but key features
    df = df[key_features + ["failure_anomaly"]]

    # run models - rf, svm, nn
    run_other_models(df)

    # run lstm seperately
    # did not ultimately work, scrapping it entirely
    # run_lstm(df)

    # close and save output to file
    # remove comment to write to an output file
    # sys.stdout.close()
