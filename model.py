from imports import *

def gaussianNaiveBayes(X, Y):

    # Split in train and test
    xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.25, random_state=0)
    classifier = GaussianNB()
    classifier.fit(xTrain, yTrain)

    # Make predictions
    yPred = classifier.predict(xTest)

    # Confusion Matrix and accuracy
    matrix = confusion_matrix(y_true=yTest, y_pred=yPred)
    accuracy = accuracy_score(yPred, yTest)

    # Precision, Recall and F-Score
    fScore = f1_score(yTest, yPred, average="macro")
    precision = precision_score(yTest, yPred, average="macro")
    recall = recall_score(yTest, yPred, average="macro")
    report = classification_report(yTest, yPred)


    return [matrix, accuracy, fScore, precision, recall, report]



def MultiLayerPerceptron(X, Y):

    # Split in train and test
    xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.25, random_state=0)

    # Fit the classifier
    classifier = MLPClassifier(alpha=10.0 ** -1, hidden_layer_sizes=(100,150), max_iter=100)
    classifier.fit(xTrain, yTrain)

    # Make predictions
    yPred = classifier.predict(xTest)

    # Confusion Matrix and accuracy
    matrix = confusion_matrix(y_true=yTest, y_pred=yPred)
    accuracy = accuracy_score(yPred, yTest)

    # Precision, Recall and F-Score
    fScore = f1_score(yTest, yPred, average="macro")
    precision = precision_score(yTest, yPred, average="macro")
    recall = recall_score(yTest, yPred, average="macro")
    report = classification_report(yTest, yPred)
    return [matrix, accuracy, fScore, precision, recall, report]


def SVM(X, Y):

    # Split in train and test
    xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=0.25, random_state=0)

    # Cross Validation
    # print("SVM Cross Validation!")
    # svm = SVC()
    # parameters = {'kernel': ['linear'], 'C': (1, 10)
    #     , 'gamma': ('auto', 'scale')
    #     # , 'gamma': (0.001, 0.01, 0.1, 1, 2, 3, 'auto')
    #     ,'decision_function_shape': ('ovo', 'ovr')
    #     # , 'shrinking': (True, False)
    #               }
    # clf = GridSearchCV(svm, parameters,scoring="accuracy",
    #                           cv=10,
    #                           n_jobs=8, verbose=10)
    # clf.fit(xTrain, yTrain)
    # bestAccuracy = clf.best_score_
    # bestParameters = clf.best_params_
    # print("The best parameters for MLP model are :\n{}\n".format(bestParameters))
    # print(bestAccuracy)


    # {'C': 1, 'decision_function_shape': 'ovo', 'gamma': 'auto', 'kernel': 'linear'}
    # # Fit the classifier
    classifier =SVC(C=1, kernel='linear', decision_function_shape='ovo', gamma='auto')
    classifier.fit(xTrain, yTrain)

    # Make predictions
    yPred = classifier.predict(xTest)

    # Confusion Matrix and accuracy
    matrix = confusion_matrix(y_true=yTest, y_pred=yPred)
    accuracy = accuracy_score(yPred, yTest)

    # Precision, Recall and F-Score
    fScore = f1_score(yTest, yPred, average="macro")
    precision = precision_score(yTest, yPred, average="macro")
    recall = recall_score(yTest, yPred, average="macro")
    report = classification_report(yTest, yPred)
    return [matrix, accuracy, fScore, precision, recall, report]

def trainBestClassifier(X, Y):


    # {'C': 1, 'decision_function_shape': 'ovo', 'gamma': 'auto', 'kernel': 'linear'}
    # Fit the classifier
    classifier =SVC(C=1, kernel='linear', decision_function_shape='ovo', gamma='auto')
    classifier.fit(X, Y)

    return classifier
