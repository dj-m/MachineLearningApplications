# Machine Learning Applications #

**Process**

- Determine the proper **model** (evaluation) **metric** to use based on the type of response variable & business problem to-solve.
  - Additionally, how to communicate the model's performance to stakeholders.
- Determine the optimal tuning of your model's **hyperparameter**.
  - You set the value of a hyperparameter prior to training your model to determine its _learning process_.

## Machine Learning Model Evaluation Metrics ## 

Notes based on the AnacondaCon [video](https://www.youtube.com/watch?v=wpQiEHYkBys)

**Evaluation Metrics**

- Why're there so many?
- What's the point?
- What's the difference?
- What do they mean?

**Definition**

An evaluation metric is a way to quantify the performance of a machine learning model.

- It's a number that tells you if it's any good.
  - This number can be used to compare different models.
  
It's not the same as loss function: It can be, but it doesn't have to be.

- Loss function is used while training your model or optimizing it.
- Evaluation metric is used on an already trained machine learning model to see the result: if it's any good.

| Supervised Learning Metrics |
| :-: |
| ![Supervised Learning Metrics Slide](images/supv_learning_metrics.png) |

### Classification Metrics ###

**Binary Classification**

- Accuracy = Number of correct predictions / Total number of predictions
  - Ranges from 0 to 100% (or 0 to 1)
  
| Accuracy in scikit-learn is accessed via score() method |
| :-: |
| ![Accuracy in scikit-learn](images/scikit-learn_accuracy.png) |

- Even though you're getting close to 96% accuracy, it may not necessarily be a good thing.
  - The data can be class imbalanced, with 95% of examples being positive (label) and 5% negative.
    - Which means that if all the values are coded as positive (label) then you'd get 95% accuracy.

| If we don't know what our data looks like we can't say if 96% is good number |
| :-: |
| ![Is accuracy of 96% good?](images/accuracy.png) |

- From just the accuracy score we can't know what errors the model is doing nor how to improve it.

**Confusion Matrix**

A table (matrix) that gives you numbers of how many samples your model classified correctly for what they are and how it mistook for something else.

- Technically not a metric, more of a diagnostic tool.
- It helps to gain insight into the type of errors a model is making.
- It also helps to understand other metrics that're derived from it.

| Confusino Matrix in scikit-learn: frst, pass actual values then predictions |
| :-: |
| ![Confusion Matrix in scikit-learn](images/confmatrix.png) |
| Confusion Matrix table: rows = actual values, columns = predicted values |
| ![Confusion Matrix table](images/confmatrix_table.png) |
| True Negative [Upper Left] values were predicted negative and they were negative.<br><br> True Positive [Lower Right] were predicted positive and they actually were positive. |
| ![Confusion Matrix table highlighted](images/confmatrix_diag.png) | 

Another benefit of a confusion matrix is what other metrics can be derived from it:

- Precision = True Positives (TP) / TP + False Positives (FP)
  - If you wanted to improve this you'd get closer to 1 (or 100%).
- Recall = TP / TP + False Negatives (FN)
  - If you wanted to improve this, look to having as little false negatives as possible.
- F1 score = 2 * Precision * Recall / Precision + Recall = 2 * TP / 2 * TP + FP + FN
  - This is a harmonic mean of Precision & Recall

Depending on the business problem, what do you care about?

- Minimizing False Positives -> **Precision**
- Minimizing False Negatives -> **Recall**

| Precision, Recall & F1 in scikit-learn |
| :-: |
| ![Precision Recall F1 in scikit-learn](images/prec_recall_f1.png) |
| To optimize for recall, use GridSearchCV with scoring argument = recall |
| ![GridSearchCV for recall](images/gridsearch_recall.png) |

Another approach is to use all four matrix categories:

| Matthews Correlation Coefficient (MCC)|
| :-: |
| ![MCC](images/mcc.png) | 

This is important because so far, this would be the first time that scikit-learn would raise a red flag for unbalanced data:

| Dividing by Zero raises flags of unbalanced data |
| :-: |
| ![MCC raises flags](images/mcc_undefined.png) |

F1 score is very sensitive to what you call a positive class while MCC is not. 

In this example, the data is the same but flipped between the two confusion matrix tables.

| F1 Score changes while MCC does not |
| :-: |
| ![F1 Score sensitivity](images/f1_mcc.png) |

If you want to summarize a confusion matrix, in one number for a binary problem, MCC gives you a better feeling of what's going on.

The downside:

- It doesn't extend well into a multiclass problem.

