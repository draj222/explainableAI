# Explainable AI for Dynamic Ensemble Models in High-Stakes Decision-Making
**Overview**
This repository contains the code and resources for the research on applying Explainable Artificial Intelligence (XAI) to dynamic ensemble models, focusing on high-stakes decision-making in domains such as finance, healthcare, and criminal justice. The study demonstrates the use of ensemble machine learning techniques to improve prediction accuracy while addressing challenges like class imbalance and lack of interpretability.

The project leverages multiple classifiers, including RandomForest, GradientBoosting, and DecisionTree, to predict credit default risks based on credit and application record datasets. Synthetic Minority Oversampling Technique (SMOTE) is used to handle class imbalance, improving the ability of the model to predict high-risk cases.

**Key Features**
<ul>
  <li>Ensemble Learning: Combining RandomForest, GradientBoosting, and DecisionTree classifiers for robust predictions using a soft voting strategy.</li>
  <li>Class Imbalance Handling: Application of SMOTE to create synthetic samples of underrepresented classes, improving the model's performance on high-risk predictions.</li>
  <li>Explainability: Discussion on the integration of XAI techniques to provide real-time explanations for model predictions, enhancing transparency and stakeholder trust.</li>
  <li>High-Stakes Domains: Focus on financial credit risk prediction, but applicable to any domain where decision-making accuracy and transparency are crucial.</li>
</ul>

**Dataset**
<a href="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?resource=download"></a>

**Results**
ROC-AUC Score: 0.88 for Credit Record Dataset
Class imbalance was handled using SMOTE, with improvements in recall for high-risk cases.
Precision-Recall trade-off was analyzed in-depth, and results are posted

**Future Directions**
Integration of XAI techniques to provide real-time model explanations for better interpretability.
Exploration of hybrid models, combining deep learning techniques such as LSTM with ensemble methods for time-series financial data.







