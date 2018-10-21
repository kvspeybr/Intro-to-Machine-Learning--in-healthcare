---
title: 'Bulding the ML Model'
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

Split data in a training and test set. You will do this using the `train_test_split` method from the `sklearn.model_selection` library. Typically, the ratio test vs training data is in the range of 0.2 to 0.3. In this specific exercise, use a ratio of 0.25 for the fraction of test data.

`@hint`


`@sample_code`
```{python}
#Split the dataframe into an array 'X' with the input variables and an array 'y' with the outcome variable
X = bc[['radius_mean','texture_mean','smoothness_mean','concavity_mean','symmetry_mean','fractal_dimension_mean']].values
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
Ex().check_object("y").has_equal_value
Ex().check_function('sklearn.model_selection.train_test_split').multi(
    check_args(['options', 'test_size']).has_equal_value(),
    check_args(['options', 'random_state']).has_equal_value()
)
success_msg("Great job! X is the array with the features, y is the vector with the outcome variable. You've correctly split the test/train set")
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
print(__[0:9])
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
print(y_predict[0:9])
```

`@sct`
```{python}
Ex().check_correct(check_function('classifier.fit').multi(
    check_args(0).has_equal_value(),
    check_args(1).has_equal_value()),
                   check_function('classifier.predict').check_args(0).has_equal_value())
success_msg("Congratulations! You've succesfully ran the rand forest classifier algorithm on the test set after having trained it on on the training dataset")
```
