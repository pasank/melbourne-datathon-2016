# Code used for Melbourne Datathon 2016 Kaggle Competition (Predicting if a job posting was related to 'Hotel and Tourism')
# Achieved accuracy metric (Gini Score) of 0.987 vs winner's metric 0.991
#
# Two-level stacked model used. Uses an ensemble of Random Forests, Extra Trees, XGBoost and Logistic Regression to
# create meta features. The meta features are used by an ensemble of Logistic Regression and XGBoost, where the predictions
# of these two models are combined using bagging.

from __future__ import division
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

import main as me

OUTPUT_FILE = "/home/ubuntu/melbourne_datathon/predictions/prediction_stackedlogregression.csv"
OUTPUT_FILE_BAGGING = "/home/ubuntu/melbourne_datathon/predictions/prediction_stackedlogregression_bagging.csv"


def create_blend_datasets(X, y, X_submission, label):
    skf = list(StratifiedKFold(y, n_folds))

    clfs = [RandomForestClassifier(n_jobs=8, max_depth=50, n_estimators=300),
            ExtraTreesClassifier(n_jobs=8),
            xgb.XGBClassifier(max_depth=6, n_estimators=400, objective='binary:logistic', learning_rate=0.3),
            LogisticRegression(n_jobs=8)
            ]

    print("Creating train and test sets for blending.")

    dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
    dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))

    for j, clf in enumerate(clfs):
        print(j, label, clf)
        dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
        for i, (train, test) in enumerate(skf):
            print("Fold", i)
            X_train = X[train]
            y_train = y[train]
            X_test = X[test]
            y_test = y[test]
            clf.fit(X_train, y_train)
            y_submission = clf.predict_proba(X_test)[:, 1]
            dataset_blend_train[test, j] = y_submission
            dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:, 1]
            print('Gini Score ' + str(me.getGiniScoreFromAUCScore(roc_auc_score(y_test, y_submission))))

        dataset_blend_test[:, j] = dataset_blend_test_j.mean(1)

        # fit on all data and generate further metafeature by predicting on all test data
        clf.fit(X, y)
        dataset_blend_train = np.hstack((dataset_blend_train, clf.predict_proba(X)[:, 1]))
        dataset_blend_test = np.hstack((dataset_blend_test, clf.predict_proba(X_submission)[:, 1]))

    return (dataset_blend_train, dataset_blend_test)


def blend(dataset_blend_train, dataset_blend_test):
    print("\nBlending.")

    print("Predicting on meta features using Logistic Regression")
    clf1 = LogisticRegression()
    clf1.fit(dataset_blend_train, y)
    y_submission1 = clf1.predict_proba(dataset_blend_test)[:, 1]

    print("Linear stretch of predictions to [0,1]")
    y_submission1 = (y_submission1 - y_submission1.min()) / (y_submission1.max() - y_submission1.min())

    print("Predicting on meta features using XGBoost")
    clf2 = xgb.XGBClassifier(n_estimators=200)
    clf2.fit(dataset_blend_train, y)
    y_submission2 = clf1.predict_proba(dataset_blend_test)[:, 1]

    print("Linear stretch of predictions to [0,1]")
    y_submission2 = (y_submission2 - y_submission2.min()) / (y_submission2.max() - y_submission2.min())

    print("Averaging (bagging) predictions based on meta features")
    y_submission = [sum(x) / len(x) for x in zip(y_submission1, y_submission2)]

    return y_submission


def blend_on_training_data_calc_metric(dataset_blend_train, trainingDataPredictionColValues):
    n_folds = 4
    skf = list(StratifiedKFold(trainingDataPredictionColValues, n_folds))
    clf1 = LogisticRegression()
    clf2 = xgb.XGBClassifier()

    for i, (train, test) in enumerate(skf):
        print('Fold ' + str(i))
        X_train = dataset_blend_train[train]
        y_train = trainingDataPredictionColValues[train]
        X_test = dataset_blend_train[test]
        y_test = trainingDataPredictionColValues[test]

        clf1.fit(X_train, y_train)
        predictions_X_test_blending1 = clf1.predict_proba(X_test)[:, 1]

        clf2.fit(X_train, y_train)
        predictions_X_test_blending2 = clf2.predict_proba(X_test)[:, 1]

        predictions_X_test_blending = [sum(x) / len(x) for x in
                                       zip(predictions_X_test_blending1, predictions_X_test_blending2)]
        predictions_X_test_averaging = [sum(x) / len(x) for x in X_test]

        print('Gini score after averaging' + str(
            me.getGiniScoreFromAUCScore(roc_auc_score(y_test, predictions_X_test_averaging))))
        print('Gini score after blending' + str(
            me.getGiniScoreFromAUCScore(roc_auc_score(y_test, predictions_X_test_blending))))


(allJobIdsFromTestingData,
 trainingVectorTitleRepresentation, trainingVectorAbstractRepresentation, trainingDataPredictionColValues,
 testingVectorTitleRepresentation,
 testingVectorAbstractRepresentation) = me.getJobIDsVectorizedRepresentationsTrainingLabels()


n_folds = 2
verbose = True
shuffle = False

X_title, y, X_submission = trainingVectorTitleRepresentation, np.array(
    trainingDataPredictionColValues), testingVectorTitleRepresentation

dataset_blend_train_title, dataset_blend_test_title = create_blend_datasets(trainingVectorTitleRepresentation, y,
                                                                            testingVectorTitleRepresentation,
                                                                          'TitleRepresentation')
dataset_blend_train_abstract, dataset_blend_test_abstract = create_blend_datasets(trainingVectorAbstractRepresentation, y,
                                                                                  testingVectorAbstractRepresentation,
                                                                                'AbstractRepresentation')

dataset_blend_train = np.hstack((dataset_blend_train_title, dataset_blend_train_abstract))
dataset_blend_test = np.hstack((dataset_blend_test_title, dataset_blend_test_abstract))

blend_on_training_data_calc_metric(dataset_blend_train, np.array(trainingDataPredictionColValues))

ensembledPredictedValuesForTestingData = blend(dataset_blend_train, dataset_blend_test)
averagedPredictedValuesForTestingData = [sum(x) / len(x) for x in dataset_blend_test]

# write blended predictions to console and file
f = open(OUTPUT_FILE, 'w')
jobIDsAndPredictions = zip(allJobIdsFromTestingData, ensembledPredictedValuesForTestingData)
f.write('job_id,hat' + '\n')
for jobID, titlePredictionValue in jobIDsAndPredictions:
    writeString = str(str(jobID) + ',' + str(titlePredictionValue))
    f.write(writeString + '\n')

# write averaged predictions to console and file
f = open(OUTPUT_FILE_BAGGING, 'w')
jobIDsAndPredictions = zip(allJobIdsFromTestingData, averagedPredictedValuesForTestingData)
f.write('job_id,hat' + '\n')
for jobID, titlePredictionValue in jobIDsAndPredictions:
    writeString = str(str(jobID) + ',' + str(titlePredictionValue))
    f.write(writeString + '\n')
