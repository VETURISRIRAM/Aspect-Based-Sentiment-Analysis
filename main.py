from imports import *
from preprocessing import *
from aspect import *
from model import *


def printOutput(df, Y, outFile):
    results = []
    for index, id in enumerate(df['example_id']):
        result = str(id) + ';;' + str(Y[index])
        results.append(result)
        print(result)
    f = open(outFile, "w")
    f.writelines(results)
    f.close()


if __name__ == "__main__":
    # Read two train datasets
    df_comp_in = pd.read_csv('data/data-2_train.csv', sep='\s*,\s*')
    df_comp_out = pd.read_csv('data/Data-2_test.csv', sep='\s*,\s*')
    
    # Your output file name
    outFile = "output.txt"

    df_comp_out['class'] = np.ones(len(df_comp_out))

    df = pd.concat([df_comp_in, df_comp_out])
    df = preprocessData(df)
    
    df, X, Y = aspectAnalysis(df)
    X_train = X[0:len(df_comp_in)]
    Y_train = Y[0:len(df_comp_in)]
    X_test = X[len(df_comp_in):]
    
    # Classifier
    classfier_comp = trainBestClassifier(X_train, Y_train)
    Y_test = classfier_comp.predict(X_test)
    printOutput(df_comp_out, Y_test, outFile)
