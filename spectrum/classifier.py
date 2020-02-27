import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# number of runs
num = 5
SVM = 0
LR = 0

for z in range(num):
    # read python dict back from the file
    pkl_file = open("features.pkl", "rb")

    data = pickle.load(pkl_file)
    pkl_file.close()
    X = data["data"]
    y = data["label"]
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        svclassifier = SVC(C=2.76, kernel="rbf", gamma=0.80)
        svclassifier.fit(X_train, y_train)
        # print('Accuracy on test set: {:.3f}'.format(svclassifier.score(X_test, y_test)))

        logreg = LogisticRegression(solver="liblinear", max_iter=1000)
        logreg.fit(X_train, y_train)
        # print('Accuracy on test set: {:.3f}'.format(logreg.score(X_test, y_test)))

        SVM += svclassifier.score(X_train, y_train)
        LR += logreg.score(X_test, y_test)
    except Exception as e:
        print(e)
        num -= 1
        print(num)


print("Average SVM: " + str(SVM / num))
print("Average LR: " + str(LR / num))
