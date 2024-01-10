# 📖KickStarter Data Mining📖

We developed a classification model using the Kickstarter dataset to predict project success or failure and a clustering model to group similar project characteristics.

### **Classification Model** ###
- *Data Preprocessing:* Removed invalid features, dropped irrelevant variables, and dummified categorical variables.
- *Feature Selection:* Used Random Forest for feature selection, choosing predictors with the highest importance.
- *Model Choice:* Opted for Random Forest due to its low risk of overfitting and good efficiency with categorical predictors.
- *Model Evaluation:* Achieved an accuracy score of 76.44%, precision of 67.64%, recall of 55.65%, and f1 score of 60.87%.

### **Clustering Model** ###
- *Chosen Variables:* Selected numerical variables for clustering, focusing on project success, goal amount, name length, blurb length, and time between creation and launch.
- *Model Development:* Used K-Means with an optimal number of clusters determined by the Elbow Method.
- *Evaluation:* Achieved an average silhouette score of 0.6222, indicating good evidence of the reality of clusters.
- *Insights:* Identified two clusters, with one characterized by successful projects with higher goals and longer durations.

### **Conclusion** ###
In summary, our classification model accurately predicts project outcomes, while the clustering model effectively groups projects based on key characteristics, providing valuable insights for Kickstarter projects.
