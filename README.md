# 🍽️ Restaurant Rating Prediction - Jupyter Notebook Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📋 Project Overview

This project analyzes restaurant data and builds machine learning models to predict aggregate ratings. The entire analysis is performed in a single Jupyter notebook, making it easy to follow and reproduce.

## 🎯 Objective

Build regression models to predict restaurant aggregate ratings based on available features, compare different algorithms, and identify the most important factors influencing restaurant ratings.

## 📊 Dataset

The dataset contains information about restaurants including:
- **Restaurant ID & Name**: Basic identifiers
- **Location**: Country Code, City, Address, Coordinates
- **Cuisines**: Types of food served
- **Cost**: Average cost for two, Currency
- **Services**: Table booking, Online delivery, Delivery status
- **Price range**: 1-4 scale
- **Rating**: Aggregate rating (0-5), Rating text
- **Votes**: Number of user votes

## 📓 Notebook Contents

The Jupyter notebook (`restaurant_rating_analysis.ipynb`) contains the following sections:

### 1. **Data Loading and Exploration** 📂
- Loading the dataset
- Understanding data structure and types
- Initial statistical summary

### 2. **Exploratory Data Analysis (EDA)** 🔍
- Distribution of ratings
- Rating by price range analysis
- Relationship between votes and ratings
- Top cuisines by average rating
- Visualizations with matplotlib and seaborn

### 3. **Data Preprocessing** 🧹
- Removing restaurants with 0 ratings (not rated)
- Feature engineering (extracting main cuisine, binary conversions)
- Handling categorical variables with one-hot encoding

### 4. **Model Building** 🏗️
Three regression models were implemented and compared:

| Model | Description |
|-------|-------------|
| **Linear Regression** | Baseline model for comparison |
| **Decision Tree** | Non-linear model with interpretability |
| **Random Forest** | Ensemble method for better performance |

### 5. **Model Evaluation** 📊
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- R² Score
- Train vs Test performance comparison

### 6. **Hyperparameter Tuning** ⚙️
- GridSearchCV for Random Forest optimization
- Finding best parameters:
  ```python
  Best parameters: {
      'max_depth': 10,
      'min_samples_split': 5,
      'n_estimators': 100
  }
