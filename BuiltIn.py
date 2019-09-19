from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.neural_network import MLPClassifier
from Data import *

# Training Classifiers Using SkLearn
# ----------------------------------

# clf = sklearn.linear_model.LogisticRegressionCV(max_iter=10000)
# clf = BernoulliNB()
# clf = ComplementNB()
# clf = MultinomialNB()

clf = RandomForestClassifier(n_estimators=300, random_state=0, bootstrap=False)
# clf = MLPClassifier(solver='lbfgs', alpha=30, random_state=None, activation='tanh', hidden_layer_sizes=(100, 100),
#                     learning_rate='constant', learning_rate_init=1)

clf.fit(X_train.T, np.squeeze(Y_train.T))

# Prediction
predict_train = clf.predict(X_train.T)
predict_test = clf.predict(X_test.T)
predict_tot = clf.predict(X_tot.T)
predict_final = clf.predict(X_final.T)

# Training Set Accuracy
m1 = compute_metrics(Y_train, predict_train)
print('Training Set : ', m1)

# Test Set Accuracy
m2 = compute_metrics(Y_test, predict_test)
print('Test Set : ', m2)
print('Confusion : ', confusion_matrix(np.squeeze(Y_test), np.squeeze(predict_test)))

# Total Set Accuracy
m3 = compute_metrics(Y_tot, predict_tot)
print('Total Set : ', m3)

# Test Cases Prediction
print(np.count_nonzero(predict_final))

# Upload to File
df = pd.DataFrame(predict_final.T, dtype=int)
df.index += 1
df.to_csv('Data/Predict_skl.csv', sep=',', encoding='utf-8', header=['Revenue'], index_label='ID')
