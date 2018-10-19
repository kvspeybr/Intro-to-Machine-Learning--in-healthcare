---
title: 'Evaluating the Model'
description: 'Evaluation Metrics'
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
