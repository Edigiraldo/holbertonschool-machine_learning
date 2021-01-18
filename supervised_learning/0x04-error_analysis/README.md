0x04. Error Analysis
====================

Learning Objectives
-------------------

### General

-   What is the confusion matrix?
-   What is type I error? type II?
-   What is sensitivity? specificity? precision? recall?
-   What is an F1 score?
-   What is bias? variance?
-   What is irreducible error?
-   What is Bayes error?
-   How can you approximate Bayes error?
-   How to calculate bias and variance
-   How to create a confusion matrix

Tasks
-----
#### 0\. Create Confusion

Write the function `def create_confusion_matrix(labels, logits):` that creates a confusion matrix:

-   `labels` is a one-hot `numpy.ndarray` of shape `(m, classes)` containing the correct labels for each data point
    -   `m` is the number of data points
    -   `classes` is the number of classes
-   `logits` is a one-hot `numpy.ndarray` of shape `(m, classes)` containing the predicted labels
-   Returns: a confusion `numpy.ndarray` of shape `(classes, classes)` with row indices representing the correct labels and column indices representing the predicted labels

#### 1\. Sensitivity

Write the function `def sensitivity(confusion):` that calculates the sensitivity for each class in a confusion matrix:

-   `confusion` is a confusion `numpy.ndarray` of shape `(classes, classes)` where row indices represent the correct labels and column indices represent the predicted labels
    -   `classes` is the number of classes
-   Returns: a `numpy.ndarray` of shape `(classes,)` containing the sensitivity of each class

#### 2\. Precision

Write the function `def precision(confusion):` that calculates the precision for each class in a confusion matrix:

-   `confusion` is a confusion `numpy.ndarray` of shape `(classes, classes)` where row indices represent the correct labels and column indices represent the predicted labels
    -   `classes` is the number of classes
-   Returns: a `numpy.ndarray` of shape `(classes,)` containing the precision of each class

#### 3\. Specificity

Write the function `def specificity(confusion):` that calculates the specificity for each class in a confusion matrix:

-   `confusion` is a confusion `numpy.ndarray` of shape `(classes, classes)` where row indices represent the correct labels and column indices represent the predicted labels
    -   `classes` is the number of classes
-   Returns: a `numpy.ndarray` of shape `(classes,)` containing the specificity of each class

#### 4\. F1 score

Write the function `def f1_score(confusion):` that calculates the F1 score of a confusion matrix:

-   `confusion` is a confusion `numpy.ndarray` of shape `(classes, classes)` where row indices represent the correct labels and column indices represent the predicted labels
    -   `classes` is the number of classes
-   Returns: a `numpy.ndarray` of shape `(classes,)` containing the F1 score of each class
-   You may use `sensitivity = __import__('1-sensitivity').sensitivity` and `precision = __import__('2-precision').precision`

#### 5\. Dealing with Error mandatory

In the text file `5-error_handling`, write the lettered answer to the question of how you should approach the following scenarios. Please write the answer to each scenario on a different line. If there is more than one way to approach a scenario, please use CSV formatting and place your answers in alphabetical order (ex. `A,B,C`):

Scenarios:

```
1\. High Bias, High Variance
2. High Bias, Low Variance
3. Low Bias, High Variance
4. Low Bias, Low Variance

```

Approaches:

```
A. Train more
B. Try a different architecture
C. Get more data
D. Build a deeper network
E. Use regularization
F. Nothing
```

#### 6\. Compare and Contrast mandatory

Given the following training and validation confusion matrices and the fact that human level performance has an error of ~14%, determine what the most important issue is and write the lettered answer in the file `6-compare_and_contrast`

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/11/03c511c109a790a30bbe.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210118%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210118T220237Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=036f18cbd4919d10d1beff13cb50a341d83cb8c0e909157f3980025f06f143bf)

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/11/8f5d5fdab6420a22471b.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20210118%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20210118T220237Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=56a1cfc845f7adca8b8b90b534e9dab4579ae574da86b87c924a9a07c83bf4fd)

Most important issue:

```
A. High Bias
B. High Variance
C. Nothing
```
