
# Stroke Disease Prediction

This project is a machine learning-based web application that predicts the likelihood of stroke disease using a Logistic Regression model. The application is built with a Python backend for data preprocessing and model training (`stroke_predict.py`) and a Streamlit-based web interface (`app.py`) for user interaction. The dataset is downsampled to address class imbalance, and missing values are imputed using a Decision Tree Regressor. The model is deployed as an interactive web app where users can input health-related features and receive a stroke risk prediction with probability.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Details](#model-details)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The Stroke Disease Prediction project aims to provide an accessible tool for predicting stroke risk based on user-provided health data. The backend script processes a healthcare dataset, handles missing values, balances the dataset, and trains a Logistic Regression model. The front-end is a Streamlit web app that collects user inputs, encodes them, and displays the prediction (stroke or no stroke) along with the probability.

## Features
- **Data Preprocessing**: Handles missing BMI values using a Decision Tree Regressor and balances the dataset via downsampling.
- **Model Training**: Uses a Logistic Regression model with feature scaling for robust predictions.
- **Web Interface**: Interactive Streamlit app for input collection and result visualization.
- **Error Handling**: Robust error handling for model loading and input validation.
- **Probability Output**: Displays the prediction probability for stroke risk.

## Technologies Used
- **Python 3.8+**
- **Pandas** and **NumPy** for data manipulation
- **Scikit-learn** for machine learning (Logistic Regression, Decision Tree Regressor, Pipeline)
- **Streamlit** for the web interface
- **Matplotlib** and **Seaborn** for data visualization
- **Joblib** for model serialization
- **Imbalanced-learn** (optional for alternative sampling methods)

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/KaanSezen1923/stroke-disease-prediction.git
   cd stroke-disease-prediction
   ```

2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the Dataset**:
   - Place the `healthcare-dataset-stroke-data.csv` file in the `Stroke Disease Predict` directory. You can obtain it from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) or a similar source.

5. **Directory Structure**:
   Ensure the `Stroke Disease Predict` directory exists and contains:
   - `healthcare-dataset-stroke-data.csv`
   - `stroke_model.pkl` (generated after running `stroke_predict.py`)

## Usage
1. **Train the Model**:
   Run the preprocessing and training script:
   ```bash
   python stroke_predict.py
   ```
   This will:
   - Load and preprocess the dataset
   - Downsample the majority class to 249 samples
   - Impute missing BMI values
   - Train a Logistic Regression model
   - Save the model as `stroke_model.pkl`

2. **Run the Web App**:
   Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```
   - Open your browser and navigate to `http://localhost:8501`.
   - Enter health-related inputs (e.g., gender, age, BMI, etc.).
   - Click the "Predict" button to view the stroke risk prediction and probability.

3. **Example Inputs**:
   - Gender: Male
   - Age: 45
   - Hypertension: No
   - Heart Disease: No
   - Work Type: Private
   - Average Glucose Level: 120.5
   - BMI: 28.2
   - Smoking Status: Never smoked
   - Ever Married: Yes

## Dataset
The dataset (`healthcare-dataset-stroke-data.csv`) contains 5,110 records with the following features:
- `id`: Unique identifier (dropped during preprocessing)
- `gender`: Male, Female, Other
- `age`: Patient age
- `hypertension`: 0 (No), 1 (Yes)
- `heart_disease`: 0 (No), 1 (Yes)
- `ever_married`: Yes, No
- `work_type`: Private, Self-employed, Govt_job, children, Never_worked
- `Residence_type`: Urban, Rural
- `avg_glucose_level`: Average glucose level
- `bmi`: Body Mass Index
- `smoking_status`: Never smoked, formerly smoked, smokes, Unknown
- `stroke`: Target variable (0: No stroke, 1: Stroke)

The dataset is imbalanced (4,861 no-stroke vs. 249 stroke cases), addressed via downsampling.

## Model Details
- **Preprocessing**:
  - Missing BMI values are imputed using a Decision Tree Regressor based on `gender` and `age`.
  - Categorical features are encoded (e.g., Male=0, Female=1).
  - The majority class is downsampled to 249 samples to match the minority class.
- **Model**: Logistic Regression with StandardScaler in a Scikit-learn Pipeline.
- **Features Used**: `gender`, `age`, `hypertension`, `heart_disease`, `work_type`, `avg_glucose_level`, `bmi`, `smoking_status`, `ever_married`
- **Evaluation**: Accuracy, confusion matrix, and classification report are printed during training.
- **Output**: Binary prediction (0 or 1) and probability of stroke.

## File Structure
```

├── Stroke Disease Predict/
├── healthcare-dataset-stroke-data.csv  # Input dataset
├── stroke_model.pkl                   # Trained model
├── app.py                                 # Streamlit web app
├── stroke_predict.py                      # Data preprocessing and model training
├── requirements.txt                       # Dependencies
├── README.md                              # This file
```

# Screenshots 

![image](https://github.com/user-attachments/assets/d86dd94e-4ea0-4a96-8c27-75196594083d)

![image](https://github.com/user-attachments/assets/b55e4594-8c27-402d-b3a4-36f879765bf8)

![image](https://github.com/user-attachments/assets/45861520-b3eb-4f02-a124-7c5b4ae621c3)

![image](https://github.com/user-attachments/assets/8d77e9cd-9cda-4f1c-bec6-8c63dc8b536e)





## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m 'Add your feature'`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

Please ensure your code follows PEP 8 style guidelines and includes appropriate documentation.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

</xaiArtifact>
