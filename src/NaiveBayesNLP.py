import numpy as np
import pandas as pd
import sklearn
import sklearn.model_selection as ms
import sklearn.feature_extraction.text as text
import sklearn.naive_bayes as nb
import matplotlib.pyplot as plt

df = pd.read_csv('https://github.com/ipython-books/'
                 'cookbook-2nd-data/blob/master/'
                 'troll.csv?raw=true')

df[['Insult', 'Comment']].tail()

y = df['Insult']

tf = text.TfidfVectorizer()
X = tf.fit_transform(df['Comment'])
print(X.shape)

p = 100 * X.nnz / float(X.shape[0] * X.shape[1])
print(f"Each sample has ~{p:.2f}% non-zero features.")

(X_train, X_test, y_train, y_test) = \
    ms.train_test_split(X, y, test_size=.2)

bnb = ms.GridSearchCV(
    nb.BernoulliNB(),
    param_grid={'alpha': np.logspace(-2., 2., 50)})
bnb.fit(X_train, y_train)

bnb.score(X_test, y_test)

# We first get the words corresponding to each feature
names = np.asarray(tf.get_feature_names())
# Next, we display the 50 words with the largest
# coefficients.
print(','.join(names[np.argsort(
    bnb.best_estimator_.coef_[0, :])[::-1][:50]]))

print(bnb.predict(tf.transform([
    "I totally agree with you.",
    "You are so stupid."
])))
