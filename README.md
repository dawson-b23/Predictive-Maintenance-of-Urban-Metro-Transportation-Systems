# Predictive Maintenance of Urban Metro Transportation Systems

**Author:** Dawson Burgess, Computer Science Department, University of Idaho, Moscow, ID, United States  
**Email:** [burg1648@vandals.uidaho.edu](mailto:burg1648@vandals.uidaho.edu)

## Overview

This repository hosts the code, analysis, and research paper for the project "Predictive Maintenance of Urban Metro Transportation Systems." The project leverages machine learning to predict equipment failures in urban metro systems using the MetroPT-3 dataset. Models such as Random Forest, Support Vector Machines (SVM), and Neural Networks are employed to enable proactive maintenance, reducing downtime, costs, and safety risks. An exploratory LSTM model was tested but ultimately abandoned due to poor performance.

### Key Features

- **Data Preprocessing**: Timestamp conversion, noise reduction, class imbalance handling, and feature selection.
- **Models**: Random Forest, SVM, Neural Networks, and an exploratory LSTM (scrapped).
- **Visualization**: Time-series plots, correlation matrices, feature importance charts, and confusion matrices.
- **Results**: High failure prediction accuracy, with Random Forest leading at 99.5%.

## Project Structure

- **`data_cleaning_and_prep.py`**: Cleans the MetroPT-3 dataset, labels failures, removes noise, and generates visualizations.
- **`finalProject_model_run_and_analysis.py`**: Trains and evaluates machine learning models, including performance metrics and plots.
- **`Predictive_Maintenance_Urban_Metro_Systems.pdf`**: Research paper detailing the study’s methodology, results, and conclusions.
- **`figures/`**: Directory for visualization outputs (e.g., time-series plots, confusion matrices).
- **`data/`**: (Optional) Placeholder for the MetroPT-3 dataset, if included.

## Dataset

The **MetroPT-3 dataset** is a multivariate time-series dataset collected from metro compressor units in Porto, Portugal, in 2022. It includes over 15 million instances sampled roughly every 10 seconds, featuring:

- **Analog Sensors**: TP2, TP3, HI, DV Pressure, Reservoirs, Oil Temperature, Motor Current.
- **Digital Sensors**: COMP, DV Electric, TOWERS, MPG, LPS, Pressure Switch, Oil Level, Caudal Impulses.
- **Failure Information**: Labeled failure events from known anomaly periods (e.g., April 18–July 15, 2020).

**Preprocessing Steps**:

- Converted timestamps to datetime format and labeled failure events.
- Removed noise by filtering records with time differences >10 seconds (51,147 points dropped).
- Downsampled non-failure instances to balance the dataset (1:1 ratio with failures).

The **MetroPT-3 dataset** used in this project is publicly available and can be accessed [here](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset). It contains over 15 million instances of sensor data from urban metro compressor units in Porto, Portugal, sampled approximately every 10 seconds. The dataset includes both analog and digital sensor signals, along with labeled failure events, making it ideal for predictive maintenance applications.

For more details on the dataset, please refer to the original publication by Veloso et al. (2022).

## Methodology

### Data Preparation

- **Cleaning**: Adjusted timestamps and removed irregularities exceeding 10-second intervals.
- **Class Imbalance**: Reduced non-failure instances from 861,566 to 17,914 to match failure instances.
- **Feature Selection**: Focused on five key features: Motor Current, Oil Temperature, DV Electric, TP2, DV Pressure.
- **Normalization**: Applied `StandardScaler` for SVM and Neural Network models.

### Models

1. **Random Forest Classifier**:
   - Parameters: 500 estimators, max depth 20, balanced class weights.
   - Performance: 99.5% accuracy, 100% recall for failures.
2. **Support Vector Machine (SVM)**:
   - Kernel: Radial Basis Function (RBF).
   - Performance: 98.9% accuracy, 100% recall for failures.
3. **Neural Network**:
   - Architecture: 64-32-1 neurons with ReLU and sigmoid activations.
   - Performance: 98.7% accuracy, 100% recall for failures.
4. **LSTM (Exploratory)**:
   - Architecture: Two LSTM layers with dropout, abandoned due to 0% failure recall.

### Visualization Techniques

- **Time-Series Plots**: Displayed sensor trends with failure event overlays (e.g., `Motor_current_failure.png`).
- **Correlation Matrix**: Analyzed feature relationships (`correlation_matrix.png`).
- **Feature Importance**: Highlighted DV Pressure and Oil Temperature (`Feature_importance_RF.png`).
- **Confusion Matrices**: Evaluated model performance (e.g., `svm confusion_matrix.png`).

## Results

- **Random Forest**: Top performer with 99.5% accuracy, 100% recall, and 0.89 F1-score for failures.
- **SVM**: Achieved 98.9% accuracy, 100% recall, but lower precision (0.64) for failures.
- **Neural Network**: Recorded 98.7% accuracy, 100% recall, and strong non-linear pattern detection.
- **LSTM**: Failed with 0% recall for failures due to abrupt events and weak temporal patterns.

### Key Insights

- **Feature Importance**: DV Pressure (44.3%) and Oil Temperature (27.2%) were the most critical predictors.
- **Class Imbalance**: Resampling was essential for improving minority class detection.

## Challenges and Limitations

- **Class Imbalance**: Rare failure events skewed initial model predictions, mitigated by resampling.
- **Data Noise**: Significant timestamp variations required extensive cleaning.
- **LSTM Failure**: Abrupt failures and lack of temporal trends hindered sequential modeling.

## Future Work

- Test anomaly detection models like Autoencoders or Isolation Forests.
- Incorporate additional sensors (e.g., humidity, operational load) for richer data.
- Validate models on other metro datasets for generalizability.

## Installation and Usage

1. **Clone the repository**:

   ```bash
   git clone https://github.com/dawson-b23/Predictive-Maintenance-of-Urban-Metro-Transportation-Systems
   ```

2. Install Requirements

   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**:
   - Access the MetroPT-3 dataset [here](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset).
   - Place the dataset in the `data/` directory or update the file path in `data_cleaning_and_prep.py`.

4. Run data preparation

   ```bash
   python data_cleaning_and_prep.py
   ```

5. Run model training and analysis

  ```bash
  python finalProject_model_run_and_analysis.py
  ```

6. View outputs: Check figures/ for visualizations and the console for metrics.
