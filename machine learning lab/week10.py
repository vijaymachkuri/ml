import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
columns = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
df = pd.read_csv(url, names=columns)

# Summarize dataset
print(df.shape, "\n", df.describe(), "\n", df.groupby('class').size())

# Data visualization
sns.scatterplot(x='sepal-length', y='petal-length', hue='class', data=df)
plt.show()
df.hist()
plt.show()
df.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
plt.show()

# Prepare data for modeling
X, y = df.iloc[:, :-1], df['class']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=7)

# Define and evaluate models
models = [
    ('LR', LogisticRegression()), 
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()), 
    ('CART', DecisionTreeClassifier()),
    ('NB', GaussianNB()), 
    ('SVM', SVC())
]

# Cross-validate and compare models
results, names = [], []
for name, model in models:
    scores = cross_val_score(model, X_train, y_train, cv=KFold(n_splits=10, shuffle=True, random_state=7), scoring='accuracy')
    results.append(scores)
    names.append(name)
    print(f"{name}: {scores.mean():.3f} ({scores.std():.3f})")

# Plot algorithm comparison
plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
plt.show()
