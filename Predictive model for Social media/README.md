# Predictive Models for Social Media Usage and Well-Being with EDA

## Overview

This project explores the relationship between social media usage patterns and users' well-being by developing predictive models. Utilizing both Random Forest and Neural Network algorithms, we aim to predict the dominant emotion experienced by a user based on their social media activity. An extensive Exploratory Data Analysis (EDA) was conducted on a Kaggle dataset to identify key features influencing well-being, such as social media frequency and interaction patterns. The Neural Network model demonstrated superior accuracy, offering insights that could guide healthier social media habits.

## Table of Contents

- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Models](#models)
  - [Feedforward Neural Network](#feedforward-neural-network)
  - [Random Forest Classifier](#random-forest-classifier)
- [Results](#results)
- [Conclusions](#conclusions)
- [Future Work](#future-work)
- [Dependencies](#dependencies)
- [Credits](#credits)
- [License](#license)

## Features

- **Exploratory Data Analysis (EDA)**: Comprehensive analysis to understand data distributions, patterns, and correlations.
- **Predictive Modeling**: Implementation of Random Forest and Feedforward Neural Network models for multiclass emotion prediction.
- **Data Preprocessing**: Handling missing values, encoding categorical variables, and feature scaling for optimal model performance.
- **Visualization**: Detailed plots and graphs for data interpretation and model evaluation.

## Dataset

The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/emirhanai/social-media-usage-and-emotional-well-being) and includes:

- **User Demographics**: Age, Gender.
- **Social Media Usage**: Platform used, Daily Usage Time, Posts Per Day, Likes Received, Comments Received, Messages Sent.
- **Well-Being Indicator**: Dominant Emotion experienced by the user.

## Installation

1. **Clone the repository**:

   ```bash
   git clone git@github.com:nesarks/Machine-Learning.git  // go to predictive model directory to access
   ```

2. **Navigate to the project directory**:

   ```bash
   cd Predictive-model-for-social-media-wellbeing
   ```

3. **Install the required packages**:

   ```bash
   pip install -r requirements.txt
   ```

   *Alternatively, ensure the following Python libraries are installed*:

   - numpy
   - pandas
   - matplotlib
   - seaborn
   - scikit-learn
   - tensorflow

## Usage

1. **Prepare the Dataset**:

   Ensure that `train.csv`, `test.csv`, and `val.csv` are placed in the `data/` directory.

2. **Run the Analysis**:

   - **Jupyter Notebook**:

     ```bash
     jupyter notebook analysis.ipynb
     ```

   - **Python Script**:

     ```bash
     python analysis.py
     ```

3. **Follow the Steps**:

   Execute the cells in the notebook or run the script to perform data preprocessing, EDA, model training, and evaluation.

## Data Preprocessing

- **Handling Missing Values**: Removed rows with missing values to maintain data integrity.
- **Encoding Categorical Variables**:
  - **One-Hot Encoding**: Applied to 'Gender' and 'Platform' columns using `pd.get_dummies()`.
  - **Label Encoding**: Applied to the target variable 'Dominant_Emotion' using `LabelEncoder`.
- **Feature Alignment**: Ensured all datasets have consistent features after encoding.
- **Feature Scaling**: Standardized numerical features using `StandardScaler`.

## Exploratory Data Analysis

- **Age Distribution**: Analyzed user age spread.
- **Gender Distribution**: Examined the proportion of different genders.
- **Platform Usage**: Compared average daily usage time across platforms.
- **Emotion Distribution**: Investigated the frequency of dominant emotions.
- **Engagement Metrics**: Explored relationships between posts, likes, comments, and messages.
- **Visualizations**: Generated plots using Seaborn and Matplotlib for insights.

## Models

### Feedforward Neural Network

- **Architecture**:
  - **Input Layer**: Matches the number of features after encoding and scaling.
  - **Hidden Layers**: Two layers with 64 and 32 neurons using ReLU activation.
  - **Dropout Layers**: Added to prevent overfitting (Dropout rate of 0.5).
  - **Output Layer**: Softmax activation for multiclass classification.
- **Compilation**:
  - **Optimizer**: Adam.
  - **Loss Function**: Sparse Categorical Crossentropy.
  - **Metrics**: Accuracy.
- **Training**:
  - **Epochs**: 500 (with considerations for early stopping to avoid overfitting).
  - **Batch Size**: 32.
  - **Validation**: Used validation set to monitor performance.
- **Evaluation**:
  - Achieved a test accuracy of **90.29%**.
  - Observed overfitting due to a gap between training and validation accuracy.
- **Recommendations**:
  - Implement early stopping.
  - Add regularization techniques.
  - Use data augmentation.
  - Apply cross-validation.

### Random Forest Classifier

- **Preprocessing**:
  - **Label Encoding**: Converted categorical features into numerical format.
  - **Feature Scaling**: Standardized numerical features.
- **Model Parameters**:
  - **Estimators**: 100 trees.
  - **Random State**: 42 for reproducibility.
- **Training**:
  - Trained on the preprocessed training set.
- **Evaluation**:
  - Achieved an accuracy of **99%** on the validation set.
  - Generated a classification report and confusion matrix.
- **Feature Importance**:
  - Analyzed to understand the impact of each feature on the prediction.

## Results

- **Neural Network**:
  - Test Accuracy: **90.29%**.
  - Issues with overfitting indicated by divergence between training and validation accuracy.
- **Random Forest**:
  - Validation Accuracy: **99%**.
  - High precision, recall, and F1-scores across all classes.
- **Insights**:
  - Random Forest outperformed the Neural Network on this dataset.
  - Important features included user engagement metrics and platform used.

## Conclusions

- Predictive models can effectively determine users' dominant emotions based on social media usage.
- Random Forest proved to be a robust model for this classification task.
- Overfitting in Neural Networks can be mitigated with proper techniques.
- EDA is crucial for understanding data and improving model performance.

## Future Work

- **Cross-Validation**: Implement to better assess model generalization.
- **Hyperparameter Tuning**: Optimize model parameters for improved accuracy.
- **Regularization Techniques**: Apply L1/L2 regularization in Neural Networks.
- **Additional Models**: Explore algorithms like Gradient Boosting or Support Vector Machines.
- **Feature Engineering**: Incorporate new features or transform existing ones.
- **Data Augmentation**: Increase dataset size synthetically to enhance model learning.

## Dependencies

- Python 3.x
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- tensorflow
- keras

## Credits

- **Dataset**: Provided by [Kaggle](https://www.kaggle.com/datasets/emirhanai/social-media-usage-and-emotional-well-being).
- **Inspiration**: Research on the impact of social media on mental health and well-being.
