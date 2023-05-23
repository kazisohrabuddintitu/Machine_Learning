# Support Vector Machines(SVMs)

* Supervised Learning Methods
* Applicable => When the data points of a dataset are separable by a linear line.<br><br>
* <b>Hyperplane/ Decision boundary</b> => separates data points belonging to different classes in a given feature space.<br><br>
* <b>The margin</b> => It is defined as the perpendicular distance between the hyperplane and the closest data points from each class. <br><br>

<p align="center">
    <img width="300" src="https://github.com/sharminislamshroddha/machine_learning/assets/133664253/7ce2b769-3552-4176-94fe-e213c73b6f7f">
    <img width="300" src="https://github.com/sharminislamshroddha/machine_learning/assets/133664253/cc48d6a7-cfe1-44e9-89bc-7e021dba70d2">
</p>

* Because SVM can choose the best hyperplane that maximizes the margin. SVM is sometimes called <b>‘Maximum Margin Classifier’</b>.<br><br>
* <b>Goal of SVM</b> => find the optimal hyperplane that maximizes the margin between the two classes.<br><br>
* <b>Support vectors</b> => support vectors are the special data points that are closest to the hyperplane/ decision boundary in a support vector machine (SVM).
* During classification, the algorithm calculates the distances between the support vectors and the new data point. These distances, along with the associated weights, 
  determine the class assignment for the new point.<br><br>
* SVM aims to find a hyperplane that maximizes the margin between classes; <b>outliers</b> that are misclassified or lie close to the decision boundary
  can distort the optimal placement of the hyperplane.
* SVM can be transformed into a higher-dimensional space using the kernel trick to achieve separability.<br><br>
* Application: used for <b>classification, regression and outliers detection</b>.<br><br>
* <b>Confusion Matrix</b>: A confusion matrix is a tabular representation that summarizes the performance of a classification model by showing the counts of true positive (TP), 
  true negative (TN), false positive (FP), and false negative (FN) predictions.<br><br>
* <b>Classification Report</b>: A classification report is a textual summary that provides several performance metrics for each class in a classification problem. 
  It includes metrics such as precision, recall, F1-score, and support. 
  <br><br>
<ins>Precision:</ins> 
Also called positive predictive value, it represents the proportion of correctly predicted positive instances out of the total 
instances predicted as positive. It measures the model's ability to minimize false positives. <br><br>
<ins>Recall:</ins> Also known as sensitivity or true positive rate, it represents the proportion of correctly predicted positive instances 
out of the total actual positive instances. It measures the model's ability to minimize false negatives. <br><br>
<ins>F1-score:</ins> The harmonic mean of precision and recall, which provides a balanced measure of a model's performance. 
It combines precision and recall into a single metric. <br><br>
<ins>Support:</ins> The number of occurrences of each class in the true dataset.
<br><br>


#### Refference:
<a href="https://www.datacamp.com/tutorial/svm-classification-scikit-learn-python">Support Vector Machines with Scikit-learn Tutorial</a> <br>
<a href="https://youtu.be/ny1iZ5A8ilA">Support Vector Machines: All you need to know!</a>
