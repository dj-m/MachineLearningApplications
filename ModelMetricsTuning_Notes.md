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

### Classification Metrics (Binary) ###

**Accuracy**

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

- It doesn't extend well into a multiclass problems.

#### Metrics that incorporate probabilities ####

**Receiver Operating Characteristic (ROC) curve**

Looks at the probabily of one example belonging to one class or the other.

Typically it's reprepsented as as plot of FP Rate (FP / FP + TN) on the x-axis and TP Rate (TP / TP + FN) on the y-axis.

| ROC curve |
| :-: |
| ![ROC curve](images/roc.png) |

- Why is it a curve and not a dot?
  - Probability threshold come into play: by default, when you have a probability generated for an example belonging to one class or another, it's 50%
  - In the example above, moving the probability to 60% generates one true positive rate for the model and a false positive rate. These two values are then connected via a curve (line).
  
So, what's the best?

- A TP (y-axis) of 1 and a FP (x-axis) of 0.

Determining whether the ROC for a specific threshold is good is hard with a plot, so looking at the 'area under the curve' helps us with this:

| Area under ROC curve |
| :-: |
| ![AUC](images/auc.png) |

  - The calculation is the percentage of the plot that's under the curve.
  
There's also a Precision & Recall curve where you move the threshold, and plot the precision/recall values, connected by a line.

| Area under Precision & Recall curve |
| :-: |
| ![AUPRC](images/auprc.png) |

A lot depends on the data you're dealing wiht, for instance unbalanced data:

| Comparison of AUC between Rates of TP/FP & Precision/Recall |
| :-: |
| ![AUC comparison](images/auc_comp.png) |

**Log Loss Function**

It's a loss function that takes into account uncertainty of model predictions. There's a larger penalty for confident false predictions.

| Log Loss Function |
| :-: |
| ![log loss function](images/logloss.png) |
| Log Loss Function Plot |
| ![log loss plot](images/logloss_plot.png) |

  - The more wrong your predictions are the more confident your model is in wrong predictions and your log loss will skyrocket.

- You'd care about log loss when you not only care about the accuracy of your predictions but also how confident your model is in the predictions it makes.

### Classification Metrics (Multi-class) ###

**Confusion Matrix**

| Confusion matrix of [hand-writing recognition](https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html) |
| :-: |
| ![confusion matrix maulti-class](images/confmatrix_multi.png) |
| Here you can see that #8 (True label) was confused for a #1 (Predicted label) |

**Precision, Recall & F1 Score**

For multi-class problems the notions of true positives and true negatives, etc. don't really apply directly.

You can get the Precision, Recall & F1 Scores by calculateing them per label and then averging them using the more common micro, macro & weighted methods.

| Precision, Recall & F1 Score with Multi-class Problems<br> |
| :-: |
| ![x](images/prec_recall_f1_multi.png) |
| <br>Next create a confusion matrix for each class/label<br> |
| ![x](images/prec_recall_f1_to_conf.png) |
| <br>Then convert this to a Tp / FP table<br> |
| ![x](images/confmatrix_to_tpfp.png) |
| Here, rows = TP & columns = FP |
| <br>The number of samples are added to create the micro/macro/weighted averages<br> |
| ![x](images/tpfp_to_means.png) |

- Micro-averaged: all samples equally contribute to the average
- Macro-averaged: all classes equally contribute to the average
- Weighted-average: each class's contribution to the average is weighted by its size

If you have a class imbalanced data and you have one class that's underrepresented but you really want to get this one right you'd want to use **macro**-average to make sure this class's contribution is amplified and it's on the same level as other classes.

scikit-learn documentation recommends using micro-average for multi-labelled problem.

**Log Loss Function**

The sum of every sample's possible label and whether it's correct for that sample and then multiply by the log of probability of this label being right for that sample.

| Formula of Multi-class Log Loss |
| :-: |
| ![Log Loss for Multi-class Problems](images/logloss_multi.png) |

### Regression Metrics ###

Considered easier to conceptualize because you're not dealing with probabilities and you only have a continuous value and your prediction is a continuous value. You subtract one from the other to get residuals.

How do you evaluate a model based on all those residuals?

- scikit-learn offers a default evaluation metric for regressions: R<sup>2</sup> (or coefficient of determination)

