import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from dm_tools import data_prep
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import time

# preprocessing step
def model_train(df,c):
# df = data_prep()

# set the random seed - consistent
    rs = 10

    # train test split  9000, train=6300, test = 2700
    y = df['TargetB']
    X = df.drop(['TargetB'], axis=1)
    X_mat = X.values
    X_train, X_test, y_train, y_test = train_test_split(X_mat, y,test_size=0.3,stratify=y,random_state=rs)

    # initialise a standard scaler object
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train, y_train)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(random_state=rs,C=c)

    # fit it to training data
    model.fit(X_train, y_train)

    # print("Train accuracy:", model.score(X_train, y_train))
    # print("Test accuracy:", model.score(X_test, y_test))

    # classification report on test data
    y_pred = model.predict(X_test)
    cr = classification_report(y_test, y_pred)
    trains=model.score(X_train, y_train)
    tests=model.score(X_test, y_test)
    return cr,trains,tests

def model_compare(df):
    rs = 10

    # train test split  9000, train=6300, test = 2700
    y = df['TargetB']
    X = df.drop(['TargetB'], axis=1)
    X_mat = X.values
    X_train, X_test, y_train, y_test = train_test_split(X_mat, y, test_size=0.3, stratify=y, random_state=rs)

    # initialise a standard scaler object
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train, y_train)
    X_test = scaler.transform(X_test)

    models = [LogisticRegression(),GaussianNB(),DecisionTreeClassifier(),GradientBoostingClassifier()]
    df_r = pd.DataFrame()
    i = 0
    for m in models:
        begin = time.time()
        m.fit(X_train, y_train)
        # y_pred=m.predict(X_test)
        trains = m.score(X_train, y_train)
        tests= m.score(X_test, y_test)
        end = time.time()
        gap = end-begin
        df_r.loc[i,"Model"] = m.__str__()
        df_r.loc[i,"Train Score"] = trains
        df_r.loc[i,"Test Score"] = tests
        df_r.loc[i,"Cost time"] = "%.2f s"%gap
        i+=1
    return df_r

if __name__ == '__main__':
    lr = LogisticRegression()
    print(lr.__str__())

