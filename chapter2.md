---
title: 'Applying the ML Model'
description: 'Define and run the ML model'
---

## Building the Model

```yaml
type: TabExercise
key: 318e9a3c2f
xp: 100
```

Prior to executing a machine learning model, a data normalisation step is executed. This will scale all variables so that the absolute value of a variable, does not influence the model build. Secondly, data is split between a training set and a test set. The training set is used to calculate the model parameters against, while the test set is used to validate the calculated model

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

#apply the classifier against the testset
y_predict= classifier.predict(X_test)


```

***

```yaml
type: NormalExercise
key: 90152f6a90
xp: 100
```

`@instructions`
In a first step the data will be normalized.

Split data in a training and test set. You will do this using the `train_test_split` method from the `sklearn.model_selection` library. Typically, the ratio training vs test data is in the range of 0.7 to 0.8. In this specific exercise, use a ratio of 0.25 for the fraction of test data.

`@hint`


`@sample_code`
```{python}
#Split the dataframe into an array 'X' with the input variables and an array 'y' with the outcome variable
X = bc[['fractal_dimension_mean', 'smoothness_se']].values
y = bc['diagnosis'].values

#Scale features
X = sc.fit_transform(X)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = __, random_state = 0)
```

`@solution`
```{python}
X = bc[['fractal_dimension_mean', 'smoothness_se']].values
y = bc['diagnosis'].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
```

`@sct`
```{python}
Ex().has_no_error()
Ex().check_function('sklearn.model_selection.train_test_split').multi(
    check_args(['options', 'test_size']).has_equal_value(),
    check_args(['options', 'random_state']).has_equal_value()
)
```

***

```yaml
type: NormalExercise
key: 5c790eff84
```

`@instructions`
In this step, you will actually run a Random Forest Classifier algorithm. After instantiating the RandomForestClassifier class, run the `.fit(X,y)` method. You should use the training subset to perform this `.fit(X,y)` method. The first parameter in the fit method will be the feature training set (`X_train`), the second parameter needs to be the outcome variable i.e. `y_train`. In a last step, you will use the classifier to predict the values of the test subset.

`@hint`


`@sample_code`
```{python}
#instantiation, random_state is set to a constant so that we obtain the same result when re-executing
classifier = RandomForestClassifier(random_state=43)

#perform the fit, using the training subset
classifier = classifier.fit(__,__)

#apply the classifier against the testset
y_predict= classifier.predict(__)

#print the first 10 predicted scores
print(__)
```

`@solution`
```{python}
#instantiation, random_state is set to a constant so that we obtain the same result when re-executing
classifier = RandomForestClassifier(random_state=43)

#perform the fit, using the training subset
classifier = classifier.fit(X_train,y_train)

#apply the classifier against the testset
y_predict= classifier.predict(X_test)

#print the first 10 predicted scores
print(y_predict.head(n=10))
```

`@sct`
```{python}
Ex().check_correct(check_function('classifier.fit').multi(
    check_args(0).has_equal_value(),
    check_args(1).has_equal_value()),
                   check_function('classifier.predict').check_args(0).has_equal_value())
success_msg("Congrats!")
```

***

```yaml
type: MultipleChoiceExercise
key: 06accfd534
```

`@question`
In this step, you will perform an evaluation of the model by calculating accuracy and showing the confusion matrix. Accuracy is only one metric that can be used to evaluate a model and it's the ratio of correctly classified 

The confusion matrix is a grid with the actual outcomes on the X-axis and the predicted outcomes on the Y-axis. 

- Run the following command to calculate accuracy (accuracy = number of correct predictions / total number of predictions)

`ac = accuracy_score(parm1,parm2)` whereby parm1 will be the array of actual outcome values (y_test) and parm2 will be the predicted outcomes (y_predict)

- To calculate the confusion matrix

`cm=confusion_matrix(parm1,parm2)` whereby parm1 will be the array of actual outcome values (y_test) and parm2 will be the predicted outcomes (y_predict)

- To visualize the confusion matrix:

`sns.heatmap(cm,annot=True,fmt="d")`

Use this visualization to select the correct answer(s) from the questions below.

`@possible_answers`
1. Accuracy is a good metric for model performance for balanced as well as unbalanced datasets
2. Accuracy for this model on test set is 0.91 while it's 0.98 on the **training** set
3. If we were to invert the definition of true / false for our outcomes i.e. true=benign and false=malignant we would get a higher accuracy with the same model
4. There are as many false negatives as false positives in the **test** set

`@hint`


`@sct`
```{python}
msg1 = "Incorrect. Accuracy is not a good metric for unbalanced datasets"
msg2 = "Incorrect."
msg3 = "Incorrect. As the total number of correct predictions still would be the same, there would be no difference."
msg4 = "Correct! Python is an extremely versatile language."
Ex().has_chosen(1, [msg4])
```

***

```yaml
type: NormalExercise
key: bc4eff4706
```

`@instructions`


`@hint`


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

`@sct`
```{python}
Ex().has_no_error()
```
