Code used for Melbourne Datathon 2016 Kaggle Competition (Predicting if a job posting was related to 'Hotel and Tourism'). Achieved accuracy metric (Gini Score) of 0.987 vs winner's metric 0.991

Two-level stacked model used. Uses an ensemble of Random Forests, Extra Trees, XGBoost and Logistic Regression to create meta features. The meta features are used by an ensemble of Logistic Regression and XGBoost, where the predictions of these two models are combined using bagging.
