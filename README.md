# 🏠 House Price Prediction using Regression Models with Streamlit Deployment

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b.svg)](https://streamlit.io/)
[![joblib](https://img.shields.io/badge/joblib-serialization-lightgrey.svg)](https://joblib.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

> An end-to-end machine learning pipeline that predicts residential property prices in **Melbourne, Australia**, benchmarks four regression algorithms, and deploys the winning model as an interactive **Streamlit** web app for real-time price prediction.

---
<img width="1342" height="795" alt="image" src="https://github.com/user-attachments/assets/823388fa-789e-4f57-8c14-b247df4d5eec" />

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Proposed Solution](#-proposed-solution)
- [Key Highlights](#-key-highlights)
- [Architecture](#-architecture)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Models Evaluated](#-models-evaluated)
- [Results & Feature Importance](#-results--feature-importance)
- [Streamlit Deployment](#-streamlit-deployment)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Tech Stack](#-tech-stack)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)
- [Contact](#-contact)

---

## 🔍 Problem Statement

The real estate market is one of the most dynamic sectors in any economy, where accurate property valuation plays a crucial role for **buyers, sellers, investors, and lenders**. Traditional valuation methods rely on manual appraisal and subjective judgment, which can introduce inconsistency, bias, and delays.

With the rise of data-driven decision-making, machine learning offers a more **reliable, scalable, and reproducible** approach to property valuation — capturing complex non-linear relationships between property features and market prices.

---

## 💡 Proposed Solution

This project builds a supervised regression pipeline that:

1. Ingests the **Melbourne Housing Snapshot** dataset from Kaggle.
2. Cleans, preprocesses, and engineers features from raw property data.
3. Trains and compares **four regression algorithms** — Linear Regression, Ridge Regression, Random Forest, and Gradient Boosting.
4. Selects the best-performing model using R², MAE, and RMSE.
5. Serializes the winning model with **joblib** and deploys it via an **interactive Streamlit app** — enabling non-technical users to get instant price predictions from property attributes.

The **Gradient Boosting Regressor** emerged as the top performer thanks to its ability to capture non-linear feature interactions without extensive feature engineering.

---

## ✨ Key Highlights

| Feature | Detail |
|---|---|
| 🎯 **Task** | Supervised regression — predict property selling price |
| 🏙 **Dataset** | Melbourne Housing Snapshot (Kaggle) |
| 🧠 **Best Model** | Gradient Boosting Regressor ⭐ |
| 🧪 **Models Compared** | Linear Regression · Ridge Regression · Random Forest · Gradient Boosting |
| 📊 **Metrics** | R² · MAE · RMSE |
| 🌐 **Deployment** | Interactive Streamlit web app |
| 🧰 **Persistence** | `joblib` model serialization (`.pkl`) |

---

## 🏛 Architecture

The diagram below illustrates the end-to-end pipeline — from raw data ingestion and preprocessing, through model benchmarking and selection, to Streamlit deployment.

<p align="center">
  <img src="architecture.svg" alt="House Price Prediction — Architecture Diagram" width="100%"/>
</p>

---

## 📊 Dataset

The project uses the **[Melbourne Housing Snapshot](https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot)** dataset from Kaggle, containing detailed records of residential property transactions in Melbourne.

### Key Features

| Feature | Description |
|---|---|
| `Rooms` | Number of bedrooms in the property |
| `Distance` | Distance from the Central Business District (km) |
| `Bathroom` | Number of bathrooms |
| `Car` | Number of car parking spaces |
| `Landsize` | Size of the property land (m²) |
| `BuildingArea` | Total built-up area of the property (m²) |
| `YearBuilt` | Year the property was constructed |
| `Suburb`, `Type`, `SellerG` | Categorical attributes (one-hot encoded) |

### Target Variable
- **`Price`** — selling price in Australian dollars (continuous, regression target).

---

## 🔬 Methodology

The project follows a standard, reproducible machine learning workflow:

1. **Data Preprocessing** — handle missing values, remove outliers, and select relevant features.
2. **Feature Engineering** — scale numerical features and encode categoricals.
3. **Train–Test Split** — 80 / 20 split with a fixed random seed for reproducibility.
4. **Model Development** — train four regression models with hyperparameter tuning.
5. **Evaluation** — compare models using R², MAE, and RMSE on the held-out test set.
6. **Deployment** — serialize the winning model and wrap it in a Streamlit UI.

### Preprocessing Details
- **Missing values:** numeric features imputed with the **median**; irrelevant or redundant columns dropped.
- **Outliers:** removed using the **Z-score method** (`|Z| > 3`) for Price, Landsize, and BuildingArea.
- **Categorical encoding:** **one-hot encoding** for nominal variables (Suburb, Type, SellerG, etc.).
- **Scaling:** **StandardScaler** applied to numerical features so all contribute on a comparable scale.

---

## 🧠 Models Evaluated

| Model | Rationale |
|---|---|
| **Linear Regression** | Interpretable baseline; quantifies linear signal. |
| **Ridge Regression** | L2-regularised linear model; controls overfitting on high-dimensional one-hot features. |
| **Random Forest Regressor** | Tree ensemble capturing non-linear interactions with minimal tuning. |
| **Gradient Boosting Regressor** ⭐ | Sequential boosting — **selected as the best performer** (highest R², lowest RMSE). |

---

## 📈 Results & Feature Importance

### Evaluation Metrics
<img width="857" height="561" alt="image" src="https://github.com/user-attachments/assets/9ab51bc6-00d4-494d-929b-01e2f578c13f" />

Each model was evaluated on the test set using:
- **R² Score** — proportion of variance explained (higher is better).
- **MAE** — mean absolute error in AUD (lower is better).
- **RMSE** — root mean squared error, penalising larger errors (lower is better).

The **Gradient Boosting Regressor** achieved the **highest R² and lowest RMSE** among all candidates.

### Feature Importance Insights

**Ensemble models (Random Forest, Gradient Boosting):**
- Heavily prioritise **BuildingArea** and **Distance**.
- Treat **Seller/Agency** categoricals as moderately to highly important.
- Rank **Rooms, Bathroom, and Landsize** as lower importance.

**Linear models (Linear Regression, Ridge):**
- Overwhelmingly prioritise **Location (Suburb)** — large coefficients on one-hot encoded suburb features dominate the prediction.

### Dominant Predictors
- 🏗 **BuildingArea** — largest positive influence on price.
- 🛏 **Rooms** — strong positive correlation.
- 🚗 **Distance** — negative correlation (closer to CBD → higher price).
- 🚿 **Bathroom / Car** — moderate positive influence.

---

## 🌐 Streamlit Deployment

The trained Gradient Boosting model is deployed as an interactive **Streamlit web app** for real-time predictions.

### Deployment Workflow
- 📦 **Model loading** — the `.pkl` model is loaded at app start-up via `joblib`.
- 🎛 **User input** — a sidebar form collects property attributes (rooms, bathrooms, land size, distance, etc.).
- ⚡ **Real-time prediction** — price is returned instantly on form submission.
- 📊 **Visualisation** — feature importance charts and model metrics rendered in-browser.

This pattern lets real-estate agents, buyers, and investors query the model without writing any code.

---

## 📁 Project Structure

```
House-Price-Prediction/
├── notebook.ipynb                        # Main Jupyter notebook (EDA, training, evaluation)
├── app.py                                # Streamlit application
├── model.pkl                             # Serialized trained model (joblib)
├── data/
│   └── melb_data.csv                     # Melbourne housing dataset
├── architecture.svg                      # Architecture diagram
├── House_Price_Prediction_Report.docx    # Industry-standard technical report
├── requirements.txt                      # Python dependencies
├── LICENSE                               # License file
└── README.md                             # Project documentation
```

> Adjust filenames to match your actual repository layout.

---

## ⚙️ Installation

### Prerequisites
- Python **3.8+**
- pip or conda

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/house-price-prediction.git
cd house-price-prediction
```

### 2. Create a Virtual Environment (recommended)
```bash
python -m venv venv
source venv/bin/activate       # macOS/Linux
venv\Scripts\activate          # Windows
```

### 3. Install Dependencies
```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib streamlit jupyter
```

Or, if a `requirements.txt` is provided:
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Option 1 — Run the Notebook
```bash
jupyter notebook notebook.ipynb
```
Step through the cells to reproduce the EDA, preprocessing, model training, and evaluation.

### Option 2 — Launch the Streamlit App
```bash
streamlit run app.py
```
The app opens in your browser at `http://localhost:8501`. Enter property attributes in the sidebar and get an instant price prediction.

---

## 🛠 Tech Stack

| Category | Tools |
|---|---|
| **Language** | Python 3 |
| **ML / Data** | scikit-learn, NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **Persistence** | joblib |
| **Deployment** | Streamlit |
| **Environment** | Jupyter Notebook |

---

## 🗺 Roadmap

- [ ] Benchmark against **XGBoost, LightGBM, and CatBoost** for further accuracy gains.
- [ ] Add **SHAP** for per-prediction explainability.
- [ ] Enrich features with **geospatial data** (schools, transport, amenities).
- [ ] Expose the model via a **FastAPI** endpoint for third-party integrations.
- [ ] Add **drift monitoring** and a scheduled retraining pipeline.
- [ ] Dockerize the Streamlit app for one-command deployment.
- [ ] Add unit tests and CI/CD.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/amazing-feature`).
3. Commit your changes (`git commit -m 'Add amazing feature'`).
4. Push to the branch (`git push origin feature/amazing-feature`).
5. Open a Pull Request.

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- **Kaggle** and *Dan Becker* for the Melbourne Housing Snapshot dataset.
- **Friedman, J. H.** — *"Greedy Function Approximation: A Gradient Boosting Machine"*, Annals of Statistics, 2001.
- **Pedregosa, F. et al.** — *"Scikit-learn: Machine Learning in Python"*, JMLR, 2011.
- The **Streamlit** team for the open-source data app framework.

---

## 📬 Contact

For questions, suggestions, or collaborations, please open an issue on this repository.

---

> ⚠️ **Disclaimer:** This project is for **educational and research purposes only**. Predictions should **not** be treated as a substitute for professional property valuation. Always consult a licensed real-estate professional for financial decisions.
