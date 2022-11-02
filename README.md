<p align="center">
    <font size="+3"><b>Predicting Customer Reviews from Amazon</b></font>
</p>

# Introduction

Customer reviews are a very valuable tool for businesses as they provide feedback for the business to improve to increase customer satisfaction. Often, these reviews are on some scale (like from 1-10); obviously, a high score is great for the business while a low score is bad. However, to truly understand these reviews, one must go through them and determine exactly what is being praised or criticized. The goal of this project was to create a model that could predict whether online reviews for a product are positive or negative based on the text content alone. The ability to distinguish between good and bad reviews would allow for a more accurate classification of customer sentiment.

This project was originally done in the RStudio environment, but I moved the code into a JuPyterNotebook environment for easier viewing of the outputs. This part of the project focuses on machine learning prediction so I recommend first looking at the [first part](https://github.com/johnmtayag/Amazon-reviews-text-mining) where I explore and analyze the data.

## Business Task

Originally, the business task for this project was to build a model which could identify whether a review for headphones from Amazon is good or bad, depending on the text within the review. This data could be used to guide product development and quality control efforts to increase customer satisfaction with the business's products. However, I decided that the scope of that task was too large and decided to split the project into two sections - analyzing the review content and finding the most important words from either good or bad reviews, and building a model that could identify good or bad reviews.

# Dataset

The [dataset](https://www.kaggle.com/shitalkat/amazonearphonesreviews) used to create the model is called “Amazon Earphones Reviews” and was posted by user Shital Kat on Kaggle. It is a 2.5MB csv file that consists of 14,337 reviews and 4 columns:

* Title of review
* Body of review
    * As before, the title and bodies will be merged into a "Combined" column
* Rating (1-5)
    * As before, the data will be grouped into bins according to rating value:
    * Good Reviews: Rating >= 4
    * Bad Reviews: Rating < 4
* Product name

<p align="center">
    <img src="images\review_text_df.PNG" height="200"><br>
    <em>A couple of example rows from the dataset</em>
</p>
<br>

## Packages Used

>*tm*: Text mining package to build the document-term matrices (DTM)<br>
>*ggplot2*: For data visualization tools<br>
>*e1071*: For building prediction models

## Data Preprocessing

I built a simple SVM model to test if the review text dataset was enough to predict if a given review text was good or bad. A use case for this would be sorting unlabeled reviews into the respective categories. It would also be useful for identifying negative criticisms within otherwise positive reviews (especially considering the reviews are on a 1-5 scale rather than a binary scale).

As seen in the previous project section, measuring just the frequency of each token across the documents led to a lot of shared terms between the good and bad review sets. This isn't inherently bad and could still produce a working classifier, but I wondered if these shared terms would confuse the models.

To address this, I manipulated the document term matrices using 3 metrics to amplify the weight for terms that are more common in either good or bad reviews:

1. Total term frequency (TTF)
    * The total number of appearances in the DTM for each term
2. Average term frequency (ATF)
    * The average number of appearance for each term per document
3. Document frequency (DF)
    * The number of documents in which each term appears at least once

For a term to be weighted highly for a review set, it should have a high scores in all 3 metrics in one review set and lower scores in all 3 metrics in the other review set. I created another metric that combines these metrics that would weight the variables, which I called TFDF (Term Frequency - Document Frequency)

For each term in both the good and bad DTMs I calculated its weight with the following formulas:
<br><br>
$$TFDF\_Weight_{good} =  \frac{TTF_{good} * ATF_{good} * DF_{good}}{nrow(DTM_{good}) * TTF_{bad} * ATF_{bad} * DF_{bad}} $$
$$TFDF\_Weight_{bad} =  \frac{TTF_{bad} * ATF_{bad} * DF_{bad}}{nrow(DTM_{bad}) * TTF_{good} * ATF_{good} * DF_{good}} $$
<br>
To avoid division by zero , I initialized the default value for each variable as 1 (even if a given term is not present in one of the DTMs). I then scaled each weight set from 0 to 1 to reduce the skew toward the most frequent terms. The resulting weight set still had a massive skew, so I saved a log-transformed version which will be used for the model:

<br><br>
<p align="center">
    <b><font size = "+2">Original Weights</font></b><br>
        <img src="images\good_barh.PNG" height="500">
        <img src="images\bad_barh.PNG" height="500"><br><br>
    <b><font size = "+2">Adjusted Weights With Log Transformation</font></b><br>
        <img src="images\adjusted_scaled_good_barh.PNG" height="525">
        <img src="images\adjusted_scaled_bad_barh.PNG" height="525"><br><br>
    <b><font size = "+2">Adjusted Weights With Log Transformation<br></font><font size = "+1">(All Terms)</font></b><br>
        <img src="images\adjusted_scaled_good_all.PNG" height="500">
        <img src="images\adjusted_scaled_bad_all.PNG" height="500"><br>
    <em>The blue lines represent words more common in the other review set.<br>
    <b>Note</b>: The bad review set contains significantly more words that are more common in the good review set</em>
</p>
<br>

# Document Classification

I wanted to test if my TF-DF weighting could improve a machine learning algorithm's ability to classify documents into good or bad classes. To do this, I initalized different sets of training and testing DTMs with different preprocessing steps applied. Then, for each DTM set, I applied my TF-DF weighting scheme to get 3 more sets of DTMs.:

1. Base DTM
    * Implementation:
        * This DTM version has no alterations
    * Predicted performance:
        * As the base case, the other DTM types will be compared to this
    * Data splits:
        * Training: 10035 rows
        * Testing: 4302 rows
2. Resampled DTM
    * Implementation:
        * All rows from the bad review set were used (n_bad)
        * Then, n_bad rows from the good review set were randomly sampled (without replacement)
    * Predicted performance:
        * As the classes are balanced, the bad reviews should be categorized more accurately
        * However, as many good-class documents are dropped, the good review categorization would likely be less accurate
    * Data splits:
        * Training: 6909 rows
        * Testing: 2961 rows
3. Bagging DTM
    * Implementation:
        * All rows from the good review set were used (n_good)
        * Then, n_good rows from the bad review set were randomly sampled (with replacement)
    * Predicted performance:
        * Again, as the classes are balanced, the bad reviews should be categorized more accurately
        * In fact, as bagging has a tendency toward overfitting, both classes may be accurately categorized, but the model 
        may not be as generalized 
    * Data splits:
        * Training: 13162 rows
        * Testing: 5642 rows

I then created a version of the previous 6 sets of DTMs that are normalized by document length - essentially, this added a step where for each document, the term counts are divided by the number of terms in the document. This lessens the influence of terms from documents which are significantly longer than the others.

For every set of DTMs, I applied these preprocessing steps as well:

* Deleted stop words
    * Common words from the SnowballC English stopwords dictionary
    * Product names
* Deleted terms below certain sparse thresholds:
    * Terms must appear at least 20 times in total
    * Terms must be between 3 and 20 characters long (inclusive)
    * Terms must appear in at least 50 documents
* Divided the DTMs into training and test sets
    * 70% of documents were randomly selected for the training set
    * 30% of documents were randomly selected for the training set
    * This scheme was chosen for simple implementation, but for more accurate modelling, 10-fold cross-validation would be preferred
<br><br>
### Resulting Data Tables:

<p align="center">

|DTM type    |TF-DF|Normalized by<br>Doc Length|# Training<br>Samples|# Testing<br>Samples|
|:-----------|:---:|:-------------------------:|:-------------------:|:------------------:|
|Base        |NO   |NO                         |10035                |4302                |
|Base        |NO   |YES                        |-                    |-                   |
|Base        |YES  |NO                         |-                    |-                   |
|Base        |YES  |YES                        |-                    |-                   |
|            |     |                           |                     |                    |
|Resampled   |NO   |NO                         |6909                 |2961                |
|Resampled   |NO   |YES                        |-                    |-                   |
|Resampled   |YES  |NO                         |-                    |-                   |
|Resampled   |YES  |YES                        |-                    |-                   |
|            |     |                           |                     |                    |
|Bagging     |NO   |NO                         |13162                |5642                |
|Bagging     |NO   |YES                        |-                    |-                   |
|Bagging     |YES  |NO                         |-                    |-                   |
|Bagging     |YES  |YES                        |-                    |-                   |

</p>

<p align="center">
    <b>Note</b>: Neither applying the TF-DF weighting scheme nor normalization by doc length changed the number of test/train samples
</p>
<br><br>

## Creating the Model

I chose to model the data using an SVM (Support Vector Machine) which I implemented via the E1701 package. SVMs work by transforming the data representation using a kernel function in such a manner that the data classes can be separated. This process can enable very complex data to be categorized, but it can be very time-intensive with big datasets.

I used the Gaussian RBF (Radial Basis Function) kernel with the hyperparameters set to cost = 2.1 and gamma = 0.1. These hyperparameters were chosen by running the tune.svm() function on the unaltered base DTM sets. For more accurate results, the SVM model would preferably be tuned for each dataset (or at least, for each DTM type)

Again, the results for each model would be more accurate if the results were averaged across each run of a 10-fold cross-validation split. However, for this analysis, each model was ran once as building each SVM is very time intensive. I performed separate analyses later using the 10-fold cross validation method later.

I tabulated the main results below along with the individual results and confusion matrices. The best model overall for each DTM type is highlighted while the model for each DTM type that improved the performance from the base case the most is bolded.
<br><br>
## Results

<p align="center">

|<center>DTM type</center>      |Class Accuracy<br>"Bad"|% Difference|Class Accuracy<br>"Good"|% Difference|
|:------------------------------|:---------------------:|:----------:|:----------------------:|:----------:|
|Base                           |79.99%                 |-           |98.09%                  |-           |
|Base w/ Normalization          |78.70%                 |-1.61%      |98.94%                  |+0.87%      |
|<mark>**Base w/ TFDF**</mark>  |<mark>87.89%</mark>    |**+9.88%**  |<mark>99.07%</mark>     |**+0.99%**  |
|Base w/ TFDF and Normalization |87.75%                 |+9.70%      |98.97%                  |+0.90%      |
</p>
<br>
<p align="center">
    <img src="images\base_results.PNG" height="650">
    <br><b>Note</b>: The columns of the confusion matrix represent the actual classes while the rows represent the predictions
</p>
<br>
<p align="center">

|<center>DTM type</center>                           |Class Accuracy<br>"Bad"|% Difference|Class Accuracy<br>"Good"|% Difference|
|:---------------------------------------------------|:---------------------:|:----------:|:----------------------:|:----------:|
|Base                                                |79.99%                 |-           |98.09%                  |-           |
|                                                    |                       |            |                        |            |
|Resampled                                           |96.39%                 |+20.50%     |88.87%                  |-9.40%      |
|Resampled w/ Normalization                          |95.59%                 |+19.50%     |88.05%                  |-10.24%     |
|Resampled w/ TFDF                                   |99.80%                 |**+24.77%** |92.49%                  |-5.71%      |
|<mark>**Resampled w/ TFDF and Normalization**</mark>|<mark>99.67%</mark>    |+24.60%     |<mark>93.92%</mark>     |-4.25%      |
</p>
<br>

<p align="center">
    <img src="images\resampled_results.PNG" height="650">
    <br><b>Note</b>: The columns of the confusion matrix represent the actual classes while the rows represent the predictions
</p>
<br>
<p align="center">

|<center>DTM type</center>                          |Class Accuracy<br>"Bad"|% Difference|Class Accuracy<br>"Good"|% Difference|
|:--------------------------------------------------|:---------------------:|:----------:|:----------------------:|:----------:|
|Base                                               |79.99%                 |-           |98.09%                  |-           |
|                                                   |                       |            |                        |            |
|Bagging                                            |92.78%                 |+15.99%     |98.14%                  |+0.05%      |
|Bagging w/ Normalization                           |92.89%                 |+16.13%     |99.02%                  |**+0.95%**  |
|Bagging w/ TFDF                                    |95.43%                 |+19.30%     |98.63%                  |+0.55%      |
|<mark>**Bagging w/ TFDF and Normalization**</mark> |<mark>99.71%</mark>    |**+24.65%** |<mark>97.89%</mark>     |-0.20%      |
</p>
<br>
<p align="center">
    <img src="images\bagging_results.PNG" height="650">
    <br><b>Note</b>: The columns of the confusion matrix represent the actual classes while the rows represent the predictions
</p>
<br>

</p>

# Discussion

The models all performed decently well - as expected, the good reviews were generally predicted with much higher accuracy overall with some exceptions. The resampling method increased the bad review accuracy by 20.5% but decreased the good review accuracy by about 10%. The bagging method increased the bad review accuracy by about 16% while leaving the good review accuracy about the same. Normalizing the DTMs by document length had mixed results - for some DTMs, it improved accuracy slightly while for others, it decreased accuracy slightly.

Overall, the TF-DF weighting scheme improved the accuracy of predicting the bad reviews for all 3 models to varying degrees of success. The base DTMs had an accuracy increase of almost 10%, while the resampled DTMs had an increase of about 5%. The bagged DTMs were mixed, but still increased 5-10%. The good reviews had more mixed results, but were still largely positive. For the base case, the accuracy increased by about 1%, though the base DTMs already had a very high 98% accuracy. 

The best models for all 3 DTM types all used the TF-DF weighting scheme. The best resampled and bagged DTMs also used normalization, but these results were very similar to the DTMs that only used the TF-DF weighting scheme. The most accurate model was the bagged DTM with TF-DF and normalization. However, the SVMs for this configuration were extremely slow (likely due to the size of the respective DTMs) and likely overfit the data. On the other hand, the resampled DTMs had greatly increased bad review accuracy at the cost of good review accuracy (likely due to large amounts of good reviews being cut from the datasets).

# Modelling With 10-Fold Cross Validation

For this part of the analysis, I split the data and trained the models using a 10-fold cross-validation method. Instead of separating the data into explicit training and testing sets, this method involves splitting the data into 10 sets of equal size. For each run, 9 sets are used to train the model while the last hold-out set is used to test the model - this is performed 10 times and the results are averaged at the end. This reduces the chance that the model will overfit and gives a decent sense of how well the model would be able to generalize with unseen data.

While SVMs are generally highly accurate, they take a very long time to build, so I performed this part of the analysis using both Naive Bayes and K Nearest Neighbors classifiers. Naive Bayes algorithms are usually fairly fast, but not quite as accurate. It also assumes that all features are independent which, in this case, is most likely not true (given certain words or phrases are more likely to appear together than others). K Nearest Neighbors is a fairly robust algorithm that adapts its classification criteria according to each training instance that is added. The algorithm is simple, but can be fairly slow and the results are heavily dependent on the order that training instances are added.

My tests with the Naive Bayes algorithm were inconclusive - it seems that this algorithm cannot effectively classify the reviews. The best runs had lower accuracies than the SVM runs, and sometimes the model would overly bias toward one class. This was most evident on the runs using the TFDF weighting scheme.

On the other hand, the tests with the K Nearest Neighbors algorithm were much more successful. While the base DTMs for all 3 DTM types had lower accuracies than the SVM equivalents, once the the TFDF weighting scheme was applied, the accuracies skyrocketed up to nearly 100% across the board.

<font size = '+2'><b>Naive Bayes Classifier</b></font>
<p align="center">

|<center>DTM type</center>           |Class Accuracy<br>"Bad"|% Difference|Class Accuracy<br>"Good"|% Difference|
|:-----------------------------------|:---------------------:|:----------:|:----------------------:|:----------:|
|<mark>**Base**</mark>               |<mark>70.65%</mark>    |-           |<mark>84.68%</mark>     |-           |
|Base w/ Normalization               |73.68%                 |+4.29%      |76.49%                  |-9.67%      |
|Base w/ TFDF                        |97.97%                 |**+38.67%** |41.81%                  |-50.63%     |
|Base w/ TFDF and Normalization      |97.57%                 |+38.10%     |30.64%                  |-63.82%     |
|                                    |                       |            |                        |            |
|<mark>**Resampled**</mark>          |<mark>67.55%</mark>    |-           |<mark>87.45%</mark>     |-           |
|Resampled w/ Normalization          |69.84%                 |**+3.39%**  |81.78%                  |-6.48%      |
|Resampled w/ TFDF                   |23.12%                 |-65.77%     |98.99%                  |**+13.55%** |
|Resampled w/ TFDF and Normalization |20.08%                 |-70.27%     |97.77%                  |+13.30%     |
|                                    |                       |            |                        |            |
|<mark>**Bagging**</mark>            |<mark>64.26%</mark>    |-           |<mark>87.45%</mark>     |-           |
|Bagging w/ Normalization            |65.85%                 |**+2.47%**  |80.85%                  |-7.21%      |
|Bagging w/ TFDF                     |3.83%                  |-94.04%     |98.94%                  |**+13.55%** |
|Bagging w/ TFDF and Normalization   |8.82%                  |-86.27%     |98.72%                  |+13.30%     |
</p>

<font size = '+2'><b>K Nearest Neighbors Classifier</b></font>
<p align="center">

|<center>DTM type</center>           |Class Accuracy<br>"Bad"|% Difference|Class Accuracy<br>"Good"|% Difference |
|:-----------------------------------|:---------------------:|:----------:|:----------------------:|:-----------:|
|Base                                |71.20%                 |-           |89.47%                  |-            |
|Base w/ Normalization               |60.65%                 |-14.82%     |85.96%                  |-3.92%       |
|<mark>**Base w/ TFDF**</mark>       |<mark>97.97%</mark>    |**+37.60%** |<mark>99.57%</mark>     |**+11.29%**  |
|Base w/ TFDF and Normalization      |97.57%                 |+37.03%     |99.45%                  |+11.15%      |
|                                    |                       |            |                        |             |
|Resampled                           |79.31%                 |-           |83.20%                  |-            |
|Resampled w/ Normalization          |70.18%                 |-11.51%     |77.28%                  |-7.12%       |
|<mark>**Resampled w/ TFDF**</mark>  |<mark>97.77%</mark>    |**+23.28%** |<mark>98.99%</mark>     |**+18.98%**  |
|Resampled w/ TFDF and Normalization |97.57%                 |+23.02%     |97.98%                  |+17.76%      |
|                                    |                       |            |                        |             |
|Bagging                             |83.72%                 |-           |83.30%                  |-            |
|Bagging w/ Normalization            |75.96%                 |-9.27%      |78.19%                  |-6.13%       |
|<mark>**Bagging w/ TFDF**</mark>    |<mark>98.94%</mark>    |**+18.18%** |<mark>99.15%</mark>     |**+19.03%**  |
|Bagging w/ TFDF and Normalization   |98.51%                 |+17.67%     |98.40%                  |+18.13%      |
</p>

# Conclusion

Though a lot more tests are needed to verify its effectiveness, the TF-DF weighting scheme appears to be fairly effective in increasing the accuracy for classification tasks. With the exception of the Naive Bayes classifiers (which struggled to classify the data properly), the TF-DF weighting scheme greatly increased the classification accuracy for both classes, even with the original unbalanced data. In the future, it would be interesting to see how the weighting scheme could be adjusted 

This method could likely be combined with other data preprocessing methods to further increase classification accuracy. For example, I could use n-grams (terms consisting of up to n words) which help preserve a lot of context in the data. The models also could be adjusted to further increase accuracy. With more time, running the SVMs with the cross-validation method would give a better idea of how well it could generalize on unseen data. The models could also be improved by tuning the hyperparameters (for this quick analysis, the relevant hyperparameters were arbitrarily chosen). 




