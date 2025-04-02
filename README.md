# 🧠 Decision Tree Classifier on Banknote Authentication Dataset

This Jupyter Notebook demonstrates the use of a Decision Tree Classifier on the **Banknote Authentication Dataset** using a from-scratch and Scikit-learn approach. It includes data shuffling, visual exploration, hyperparameter tuning, training, and performance evaluation.

---

## 📁 Project Structure

- `decision_tree.ipynb` – Main notebook with analysis and visualizations
- `data_banknote_authentication.txt` – Dataset file (from UCI ML Repository)

---

## 📊 Dataset Overview

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/banknote+authentication)
- **Features**:
  - Variance of Wavelet Transformed image
  - Skewness of Wavelet Transformed image
  - Kurtosis of Wavelet Transformed image
  - Entropy of image
- **Target**:
  - 0 = Fake
  - 1 = Authentic

---

## 🧪 Notebook Breakdown

### ✅ Q2.1 – Data Shuffling
Randomizes the dataset rows to simulate varied train-test splits.

### 📊 Q2.2 – Pairplot Visualization
Shows feature relationships using `seaborn.pairplot`, color-coded by class.

### 🛠️ Q2.3 – Hyperparameter Evaluation
Evaluates model performance across various combinations of:
- `criterion`: `gini`, `entropy`
- `max_depth`: `2`, `4`, `6`, `8`
- `min_samples_split`: `2`, `5`, `10`

Metrics evaluated:
- Accuracy
- Precision
- Recall
- F1 Score

### 🌳 Q2.4 – Decision Tree Visualization
Trains a final model and visualizes the full tree structure using `plot_tree`.

### 📈 Q2.5 – Feature Importance and Evaluation
Displays:
- Classification report
- Confusion matrix
- Bar chart of feature importances

---

## 🧰 Requirements

Install dependencies with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
