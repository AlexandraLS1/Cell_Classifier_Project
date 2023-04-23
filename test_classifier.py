

import pytest
import pandas as pd 
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
@pytest.fixture

#  unit test, generat synthetic data and testing that the function/ part of code  get the expected values
# run this code when we have any update of the function(any function can ve evaluates -  we need onlly make the correct tests)

def data():
    # Define a fixture to generate synthetic data for testing

    X, y = make_classification(n_samples=100, n_features=1500, n_classes=2, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Define a test function to check the performance of the logistic regression model on the test data
def test_pipline(data):
    X_train, X_test, y_train, y_test = data
def model_pipline_classification(X_train,X_test,y_train,y_test,data_type):

    df = pd.DataFrame(columns=['Data_type', 'Classifier', 'Accuracy score'])
    # Define a list of classifiers to use
    classifiers = [LogisticRegression(), RandomForestClassifier() ,SVC()] #KNeighborsClassifier()
    # make dict of classifiers 
    model_dict={i:(str(model)).strip('()') for i, model in enumerate(classifiers)}

    # Define a list of hyperparameters for each classifier
    hyperparameters = [{'logisticregression__penalty': ['l1', 'l2'], 'logisticregression__C': [0.1, 0.5 ,1, 10]},
                    {'randomforestclassifier__n_estimators': [10, 100, 1000], 
                        'randomforestclassifier__max_depth': [5,8,15,25,30,None],
                        'randomforestclassifier__min_samples_leaf':[1,2,5,10,100],
                    'randomforestclassifier__max_leaf_nodes': [2, 5,10]},
                    # {'kneighborsclassifier__n_neighbors': [1,3,5,7,],
                    # 'kneighborsclassifier__weights': ['uniform', 'distance'],
                    # 'kneighborsclassifier__metric': ['euclidean', 'manhattan']},
                    {'svc__C': [0.001,0.1, 1, 10], 'svc__kernel': ['linear', 'rbf']}]

    # Create a list of pipelines for each classifier with the same preprocessing steps
    pipelines = []
    for i,classifier in enumerate(classifiers):
        pipeline = make_pipeline(
            SimpleImputer(strategy='mean'),
            StandardScaler(),
            SelectKBest(f_classif, k=5),
           classifier
        )
        pipelines.append(pipeline)

    # Create a list of pipelines for each classifier with the same preprocessing steps
    pipelines = []
    for i ,classifier in enumerate(classifiers):
        pipeline = make_pipeline(
            SimpleImputer(strategy='mean'),
            StandardScaler(),
            SelectKBest(f_classif, k=5),
            classifier
        )
        pipelines.append(pipeline)

    # Perform grid search to find the best hyperparameters for each pipeline
    best_models = []
    best_models_param=[]
    xx=[]
    for i, pipline in enumerate(pipelines):
        grid_search = GridSearchCV(pipline, hyperparameters[i], cv=5)
        grid_search.fit(X_train, y_train)
        best_models.append(grid_search)

    for i, model_n in enumerate(best_models):
        model_bm=model_n.best_estimator_
        y_pred = model_bm.predict(X_test)
        #report = classification_report(y_test, y_pred, output_dict=True)
        # Evaluate the performance of the model using appropriate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print("Classifier ", i+1)
        print(classification_report(y_test, y_pred))
        print('{} Test Accuracy: {}'.format(model_dict[i],round(model_bm.score(X_test,y_test),4)))
        print('{} Best Params: {}'.format(model_dict[i], model_n.best_params_))
        df.loc[i,'Data_type']=data_type
        df.loc[i,'Classifier']=model_dict[i]
        df.loc[i,'Accuracy score']=round(model_bm.score(X_test,y_test),4)


        # Use the assert statement to check if the performance of the model meets the expected values
        assert accuracy >= 0.8, "Accuracy is lower than expected"
        assert precision >= 0.8, "Precision is lower than expected"
        assert recall >= 0.8, "Recall is lower than expected"
        assert f1 >= 0.8, "F1-score is lower than expected"

# Run the test function using pytest
pytest.main([__file__, '-v'])