---
title: 'Evaluating the Model'
description: 'Evaluation Metrics'
---

## Accuracy and Confusion Matrix

```yaml
type: NormalExercise
key: f7302f2790
xp: 100
```



`@instructions`


`@hint`


`@pre_exercise_code`
```{python}
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve # ROC Curves
from sklearn.metrics import roc_auc_score # Calculating AUC for ROC's!

#read Wisconsin breast cancer data set
bc = pd.read_csv('http://assets.datacamp.com/production/repositories/3810/datasets/7c19b7d9c1db98790fcf3efc234807a478e6a53e/data.csv')

# Convert diagnosis to binary : M=1, B=0
bc['diagnosis'] = bc['diagnosis'].map({'M':1, 'B':0})

# Instantiate StandardScaler
sc = StandardScaler()

#Split the dataframe into an array 'X' with the input variables and an array 'y' with the outcome variable
X = bc[['fractal_dimension_mean', 'smoothness_se']].values
y = bc['diagnosis'].values

#Scale features
X = sc.fit_transform(X)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#instantiation, random_state is set to a constant so that we obtain the same result when re-executing
classifier = RandomForestClassifier(random_state=43)

#perform the fit, using the training subset
classifier = classifier.fit(X_train,y_train)

```

`@sample_code`
```{python}
#apply the classifier against the testset
y_predict= classifier.predict(X_test)


```

`@solution`
```{python}

```

`@sct`
```{python}

```

---

## Calculating Area Under the Curve

```yaml
type: NormalExercise
key: 9cfed5239c
xp: 100
```

Besides accuracy and the confusion matrix, there are other metrics and evaluations methods possible for the evaluation of the model. In this exercise, you will use the Area Under the Curve (AUC).

`@instructions`
Just press the 'submit answer' button. The points are yours to keep !

`@hint`


`@pre_exercise_code`
```{python}
from sklearn.metrics import roc_curve # ROC Curves
from sklearn.metrics import roc_auc_score # Calculating AUC for ROC's!
```

`@sample_code`
```{python}
#ROC curve analysis
#y_pred_proba = classifier.predict_proba(X=X_test)
#scores=y_pred_proba[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_predict)
roc_auc = roc_auc_score(y_test, y_predict)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating curve')
plt.legend(loc="lower right")
plt.show()
```

`@solution`
```{python}
#ROC curve analysis
y_pred_proba = classifier.predict_proba(X=X_test)
scores=y_pred_proba[:,1]

fpr, tpr, thresholds = roc_curve(y_test, scores)
roc_auc = roc_auc_score(y_test, scores)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating curve')
plt.legend(loc="lower right")
plt.show()
```

`@sct`
```{python}

```
