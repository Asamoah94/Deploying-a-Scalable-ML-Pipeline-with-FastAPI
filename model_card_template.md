# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Model Name: Logistic Regression Classifier

Model Type: Supervised Learning

Model Version: 1.0

Model Developer: [Albert Asamoah]

Model Creation Date: [05/27/2024]

Model Last Updated: [05/27/2024]

Model Architecture: Logistic Regression

Programming Language: Python

Libraries Used: scikit-learn, pandas

Model Source Code: [https://github.com/Asamoah94/Deploying-a-Scalable-ML-Pipeline-with-FastAPI]


## Intended Use
The intended use of this model is to predict whether an individual's income exceeds $50,000 per year based on various demographic features such as workclass, education, occupation, etc
## Training Data
The model was trained on the UCI Adult Census Income dataset (census.csv), which contains demographic information such as age, education, marital status, occupation, etc., along with corresponding income labels.
## Evaluation Data
The model was evaluated on a separate test dataset split from the same UCI Adult Census Income dataset. Approximately 20% of the total dataset was used for evaluation.

## Metrics
_Please include the metrics used and your model's performance on those metrics._

Precision: 0.7285
Recall: 0.2699
F1 Score: 0.3939

## Ethical Considerations
Fairness: The model's predictions should not discriminate against any particular group based on protected attributes such as race, gender, or ethnicity. Fairness considerations should be addressed during feature selection, model training, and evaluation.
Privacy: The dataset used for training and evaluation should be handled with care to ensure the privacy of individuals' sensitive information.
Transparency: The model's decision-making process should be transparent and understandable to users and stakeholders.

## Caveats and Recommendations
Data Imbalance: The dataset may suffer from class imbalance, with fewer individuals having income greater than $50,000. Techniques such as oversampling or adjusting class weights may be necessary to mitigate this issue.
Feature Engineering: Additional feature engineering techniques such as feature scaling, encoding categorical variables, or creating new features may improve model performance.
Model Interpretability: While logistic regression is a simple and interpretable model, more complex models may offer better predictive performance at the cost of interpretability. Consider the trade-offs between model complexity and interpretability based on the specific use case.
