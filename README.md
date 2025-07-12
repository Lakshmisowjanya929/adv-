💼 Insurance Pricing Forecast using Linear Regression

This project builds a **Linear Regression model** to predict **insurance charges** based on demographic and health-related factors such as age, BMI, number of children, smoking habits, region, and gender. The model uses the **insurance dataset** from the book *"Machine Learning with R"*.

---
 🧾 Project Summary

Insurance companies collect small amounts of money called **premiums** to cover potential future losses. Setting the right premium price is critical, and data analysis can help forecast these values. This project applies machine learning to estimate insurance charges based on historical customer data.

---

📊 Features

- 📥 Loads real-world data from a CSV URL
- 🔄 Encodes categorical variables using one-hot encoding
- ✂️ Splits the dataset into training and test sets
- 🧠 Trains a Linear Regression model
- 📈 Evaluates the model using Mean Squared Error and R² Score
- 🖼 Visualizes Actual vs Predicted Charges

---

🛠️ Technologies Used

- Python
- pandas
- NumPy
- matplotlib
- seaborn
- scikit-learn


📁 Dataset

- **Source**: [GitHub – stedy/Machine-Learning-with-R-datasets](https://github.com/stedy/Machine-Learning-with-R-datasets)
- **File**: `insurance.csv`
- **Attributes**:
  - `age`
  - `sex`
  - `bmi`
  - `children`
  - `smoker`
  - `region`
  - `charges` (target variable)

---

## 📦 Installation

Make sure to install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
