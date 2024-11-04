# Customer Churn Prediction Using Artificial Neural Network (ANN)

This project predicts customer churn for a telecom company by analyzing customer demographics, services, and billing data. Using data processing, feature engineering, and an Artificial Neural Network (ANN) model, this project aims to classify customers likely to churn. This README outlines the methodology, libraries used, and the approach taken to achieve an accuracy of 77% and high precision and recall scores.

## Project Overview

- **Dataset**: The project utilizes a dataset of 7,000+ customer records, containing attributes such as tenure, monthly charges, contract type, and services used by each customer.
- **Objective**: To predict customer churn by building an ANN model that identifies patterns in the data, enabling proactive customer retention efforts.

## Key Steps

1. **Data Preprocessing**
   - **Data Cleaning**: Removed irrelevant fields, such as customer IDs, and handled missing values in `TotalCharges`.
   - **Encoding Categorical Features**: Converted categorical features to numerical values. Applied label encoding to binary columns and one-hot encoding to multi-category columns (e.g., Internet Service, Contract, and Payment Method).
   - **Scaling**: Used MinMaxScaler to normalize numerical features (tenure, MonthlyCharges, TotalCharges) for optimal model performance.

2. **Data Visualization**
   - Visualized customer distribution based on tenure and monthly charges, segregating churned and non-churned customers.
   - Histograms were used to understand the customer trends and patterns associated with churn.

3. **Model Building and Training**
   - Constructed an ANN model using **TensorFlow** and **Keras** libraries.
   - Architecture:
     - **Input Layer**: 26 features as input
     - **Hidden Layers**: Two dense layers with 26 and 15 neurons, ReLU activation function
     - **Output Layer**: Single neuron with sigmoid activation to classify churn probability.
   - **Optimizer**: Adam optimizer with a learning rate of 0.01.
   - **Loss Function**: Binary Crossentropy.
   - Trained the model for 100 epochs, achieving a training accuracy of approximately 77%.

4. **Model Evaluation**
   - Evaluated model performance on the test set, achieving an accuracy of 77%.
   - Generated a confusion matrix and classification report to assess precision and recall.
   - **Precision**: 81%
   - **Recall**: 88%

## Libraries Used

- **Data Processing**: `pandas`, `numpy`
- **Data Visualization**: `matplotlib`, `seaborn`
- **Model Building**: `tensorflow`, `keras`
- **Evaluation**: `sklearn.metrics` (confusion matrix and classification report)

## Usage

1. **Setup**:
   - Clone this repository.
   - Install required libraries:
     ```bash
     pip install numpy pandas matplotlib seaborn tensorflow scikit-learn
     ```

2. **Data Preparation**:
   - Ensure `customer_churn.csv` is in the same directory.
   - Run the data preprocessing steps provided in the code to clean and transform the data.

3. **Model Training**:
   - Use the provided ANN architecture to train the model. Modify the `epochs` and `learning rate` as needed.

4. **Evaluation**:
   - Use the confusion matrix and classification report to evaluate the model performance on unseen data.

## Results

- The model achieved **77% accuracy** in predicting customer churn.
- The precision and recall scores indicate that the model is effective in identifying customers likely to churn, aiding potential retention strategies.

## Future Enhancements

- Experiment with additional layers or different neural network architectures to improve accuracy.
- Incorporate hyperparameter tuning to find optimal values for model parameters.
- Deploy the model in a real-time environment using platforms like **Firebase** or **AWS Lambda** for practical applications.
