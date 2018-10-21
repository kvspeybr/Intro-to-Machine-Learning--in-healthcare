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

In this step, you will perform an evaluation of the model by calculating accuracy and showing the confusion matrix. Accuracy is only one metric that can be used to evaluate a model and it's the ratio of correctly classified 

The confusion matrix is a grid with the actual outcomes on the X-axis and the predicted outcomes on the Y-axis.

`@instructions`
- To calculate accuracy (accuracy = number of correct predictions / total number of predictions):

`ac = accuracy_score(parm1,parm2)` whereby parm1 will be the array of actual outcome values (y_test) and parm2 will be the predicted outcomes (y_predict)

- To calculate the confusion matrix:

`cm=confusion_matrix(parm1,parm2)` whereby parm1 will be the array of actual outcome values (y_test) and parm2 will be the predicted outcomes (y_predict)

- To visualize the confusion matrix:

`sns.heatmap(cm,annot=True,fmt="d")`
`plt.show()`

Use this visualization to select the correct answer(s) from the questions below.

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
X = bc[['radius_mean','texture_mean','smoothness_mean','concavity_mean','symmetry_mean','fractal_dimension_mean']].values
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
y_predict= classifier.predict(__)

#complete command to calculate accuracy on testset
ac=accuracy_score(__,__)
print("accuracy score: ",ac)

#complete command to calculate confusion matrix on training set
cm=confusion_matrix(__,__)

sns.heatmap(cm,annot=True,fmt="d")
plt.show()
```

`@solution`
```{python}
#apply the classifier against the testset
y_predict= classifier.predict(X_test)

#complete command to calculate accuracy on testset
ac=accuracy_score(y_test,y_predict)
print("accuracy score: ",ac)

#complete command to calculate confusion matrix on training set
cm=confusion_matrix(y_test,y_predict)

sns.heatmap(cm,annot=True,fmt="d")
plt.show()
```

`@sct`
```{python}
Ex().check_function("classifier.predict").check_args(0).has_equal_value()
Ex().check_function("sklearn.metrics.accuracy_score").multi(check_args(0).has_equal_value(),check_args(1).has_equal_value())
Ex().check_function("sklearn.metrics.confusion_matrix").multi(check_args(0).has_equal_value(),check_args(1).has_equal_value())
success_msg("Cool! Looks like the confusion matrix has no secrets for you !")
```

---

## Accuracy and Confusion Matrix - Contd

```yaml
type: MultipleChoiceExercise
key: deedb66802
xp: 50
```

Using the commands from the previous exercise, try to use those to select the correct statement below (there is only 1 statement true)

`@possible_answers`
- Accuracy is a good metric for evaluating a ML model for balanced and imbalanced datasets
- The accuracy would be higher if you set Benign tumors to 'True' and Malignant tumors to 'False' than in the opposite case
- Accuracy on the test set is higher than on the training set
- The number of False Positives is equal to the number of False Negatives

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

`@sct`
```{python}
msg1 = "Incorrect. Accuracy is not a good metric for unbalanced datasets"
msg2 = "Incorrect."
msg3 = "Incorrect. As the total number of correct predictions still would be the same, there would be no difference."
msg4 = "Correct!"
Ex().has_chosen(4,[msg1, msg2, msg3, msg4])
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

y_predict= classifier.predict(X_test)
```

`@sample_code`
```{python}
#ROC curve analysis
y_predict_a = classifier.predict_proba(X=X_test)
scores=y_predict_a[:,1]

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

`@solution`
```{python}
#ROC curve analysis
y_predict_a = classifier.predict_proba(X=X_test)
scores=y_predict_a[:,1]

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
