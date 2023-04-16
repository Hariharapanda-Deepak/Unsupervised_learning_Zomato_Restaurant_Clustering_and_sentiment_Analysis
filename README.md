# Unsupervised Learning Zomato Restaurant Clustering and Sentiment Analysis

## Problem Statement

Zomato offers information about restaurants, menus, and reviews from customers. It also offers food delivery from affiliated restaurants in a few cities. India is well known for its wide variety of cuisine, which is offered in many restaurants and resort hotels and is suggestive of unity in diversity. In India, the restaurant industry is constantly changing. Indians are becoming more receptive to the concept of dining out or ordering meals to be delivered. In order to gain some insights, fascinating facts, and figures regarding the Indian food sector in each city, data were examined in light of the rising number of restaurants in each Indian state. Therefore, the analysis of Zomato restaurant data for each city in India is the main goal of this study.

In the project I have implemented K-Means Clustering Algorithm on the restaurant data. The objective of the clustering was to group similar restaurants together and discern patterns within the data. The features employed for the clustering process included the restaurant's ratings, cuisines, and average cost for two. The number of clusters was determined by utilizing the elbow method.

I then proceeded to conduct sentiment analysis on the user reviews to gain a comprehensive understanding of the overall sentiment towards the restaurants. Certain libraries were utilized to classify the reviews as positive, negative, or neutral.

## Datasets Information

There are two data sets Metadata dataset and reviews data set received from zomato. 

For Metadata Dataset:

I have found that the Metadata datset contains 105 rows and 6 columns. I have found there are no duplicate rows in the dataset. There are 54 rows with missing values present in columns Collection and 1 row with missing values in timings. I have not replaced or removed any missing values.

For Reviews dataset

I have found that the reviews dataset has 10000 rows and 7 columns. I have found that there are 36 duplicate rows. I have dropped all duplicate rows. I also found that there are 9 missing values of reviewer and 2 missing values in each column of review, reviewer, rating, Metadata and Time. I have observed that for the two rows, American Wild Wings and Arena Eleven has no reviews, ratings,metadata and we dont have the timings of the hotel. so I have dropperd these two rows. The final reviews dataset has 9962 rows and 7 columns.

## Variables Description

### Zomato Restaurant names and Metadata

**Name** : Name of Restaurants

**Links** : URL Links of Restaurants

**Cost** : Per person estimated Cost of dining

**Collection** : Tagging of Restaurants w.r.t. Zomato categories

**Cuisines** : Cuisines served by Restaurants

**Timings** : Restaurant Timings

## Zomato Restaurant Reviews

**Restaurant** : Name of the Restaurant

**Reviewer** : Name of the Reviewer

**Review** : Review Text

**Rating** : Rating Provided by Reviewer

**MetaData** : Reviewer Metadata - No. of Reviews and followers

**Time** : Date and Time of Review

**Pictures** : No. of pictures posted with review

## Data Vizualization, Storytelling & Experimenting with charts : Understand the relationships between variables

**Chart - 1 - Pie Chart on Cuisines of different hotels**

![image](https://user-images.githubusercontent.com/117559898/232266286-2f7261be-f450-4c63-8eba-2ac356c6e453.png)

I have found that North Indian cuisine and Chinese cuisine is highly offered cuisine by the hotels with a percentage of 19.5 and 13.7 respectively.

From the graph we can clearly understand the most important cuisines offered by the hotels. This indicates that the customers are highly choosing the north Indian cuisine, chinese cuisine. So, Zomato can target on customers intrested in these cuisine through marketing.

**Chart - 2 - Bar Chart on Number of different Cuisines offered by different hotels**

![image](https://user-images.githubusercontent.com/117559898/232266348-dc41fb5a-7b17-488c-bb02-858de3d40fe3.png)

From the graph I have identified the following observations:

* The hotel Beyond Flavours have the highest number of cuisines offered.

* The average number of cuisines provided by the restaurants is 3. This means mostly hotels are multicuisine.

**Chart - 3 - Bar Chart on Costs of different hotels**

![image](https://user-images.githubusercontent.com/117559898/232266492-43b2605f-2326-42bb-ade0-8e6c47a5b9b4.png)

From the above chart I have made the following conclusions:

* College-Hyatt Hyderabad Gachibowli has the highest price among the hotels.

* Amul has the lowest price among the hotels.

**Chart - 4 - Bar Chart on Ratings of different hotels**

![image](https://user-images.githubusercontent.com/117559898/232266611-268bc109-811b-4749-81d4-65b52940eea3.png)

From the above chart I have made the following conclusions:

* AB's Absolute Barbeque has highest ratings.

* KFC is the least rated restaurant.

**Chart - 5 - Bar Chart on Ratings of different hotels**

![image](https://user-images.githubusercontent.com/117559898/232266654-5e344fc8-0166-42ec-bb82-97c69695f0b4.png)

From the above charts I have observed the following:

* The hotel Pour House7 has the highest number of pictures uploaded.

* The hotel Hitech Bawarchi Food zone has least number of pictures uploaded.

**Chart - 6 - Scatter Plot on Cost and Ratings of different hotels**

![image](https://user-images.githubusercontent.com/117559898/232266691-c39bec48-59b4-494f-9c79-483362e9365b.png)

I have made a comparision between cost and ratings of the hotels and have found the following:

* Hotel zara Hi-Fi is the least rated hotel

* Hotel AB's Absolute Barbeque, B-Dubs, 3B's Buddies Bar and Barbeque are top rated restaurants and their cost vary between 1000 to 1600 rupees.

* Also there are many low rated hotels in price range below 1000 rupees.

**Chart - 7 - Scatter Plot on number of cuisines and Ratings of different hotels**

![image](https://user-images.githubusercontent.com/117559898/232266722-1380f04d-284a-4ab6-9814-de66081b952e.png)

From the above plots I have made the following observations:

* From the list of highest number of cuisines I have found that the hotel Flechazo has the highest rating and shanghai chef 2 has the lowest rating.

* From the lowest nuber of cuisines list I have found that B-Dubs has the highest rating and Hotel Zara-Hifi has the lowest number of ratings.

**Chart - 8 - Scatter Plot on number of Pictures and Ratings of different hotels**

![image](https://user-images.githubusercontent.com/117559898/232266788-b638b707-2288-425d-ad17-30b3ed8df15b.png)

From the observations made from the graphs. The hotels with high rating has less number of pictures uploaded. Uploading more pictures can be useful for the customers to review to take decisions about the hotel.

**Chart - 9 - Scatter Plot on number of cuisines and cost of different hotels**

![image](https://user-images.githubusercontent.com/117559898/232266819-c5690311-667c-4ae3-9a7a-cef545255851.png)

* B-Dubs and Arena Eleven are the two hotels whose cost per meal is above 1500 and offers only single cuisine.

* Collage Hyatt Hyderabad Gachibowli is the only hotel which offers high variety of cuisines which is priced above 2500

**Chart - 10- Correlation Heatmap**

![image](https://user-images.githubusercontent.com/117559898/232266857-9242e080-5e99-490d-9613-86a73b3ec99c.png)

From the above correlation map we can say that there is no correlation between any features in the dataset

# Feature Engineering & Data Pre-processing

## Handling Missing Values

I have Found the duplicates present in the dataset and found no duplicates because they are removed in earlier stages.

Also I have checked null values and found null values in reviews, rating and followers. I have replaced null values of Total Followers with 0. I have replaced rating values with mean and replaced review with an empty string.

## Handling Outliers

First I have identified numerical columns. Then I separated the skew symmetric and symmetric features and define the upper and lower boundry as defined below. Using the capping technique instead of removing the outliers, capped outliers with the highest and lowest limit using IQR method.

In a Gaussian distribution while it’s the symmetric curve and outlier are present. Then, we can set the boundary by taking standard deviation into action.

The describe function is used for describing the behavior of the data in the middle as well as at the ends of the distributions. The box plot uses the median and the lower and upper quartiles (defined as the 25th and 75th percentiles). If the lower quartile is Q1 and the upper quartile is Q3, then the difference (Q3 — Q1) is called the interquartile range or IQ. A box plot is constructed by drawing a box between the upper and lower quartiles with a solid line drawn across the box to locate the median. The following quantities (called fences) are needed for identifying extreme values in the tails of the distribution:

lower inner fence: Q1–1.5*IQ
upper inner fence: Q3 + 1.5*IQ
lower outer fence: Q1–3*IQ
upper outer fence: Q3 + 3*IQ

## Categorical Encoding

I have replaced the False values with 0 and True values with 1 in the datasets cuisine_bool and collections_bool.

## Textual Data Preprocessing

### Expand Contraction, Lower Casing, Removing Punctuations, Removing URLs & Removing words and digits contain digits, Removing Stopwords & Removing White spaces.

* I have imported the contractions library and used a lambda function on each rows of reviews to fix contractions.

* Using ' str.lower ' I have converted all the text into lower cases.

* I have created a function to remove punctuations and applied the function on review column to remove all punctuations.

* I have also removed digits and urls present in the reviews column using lambda function.

* From nltk library I have imported stop words and have removed stop words and removed white spaces using lambda function.

### Tokenization

From nltk library I have used word tokenization.

### Text Normalization

From nltk.stem I have imported WordNetLemmatizer to convert the word to its root form.

### Text Vectorization

In my work, I utilized the Tf-idf Vectorization technique, which is a process of assigning a weight to each word in a document. The weight is calculated by multiplying the term frequency (tf) with the inverse document frequency (idf).

Term frequency (tf) refers to the frequency of a word's appearance in a document, while inverse document frequency (idf) measures the rarity of a word across all documents in a collection. Tf-idf assigns higher weights to words that appear frequently in a document but are rare across the collection, indicating that they are more informative.

Tf-idf can be expressed mathematically as tf-idf(t, d, D) = tf(t, d) * idf(t, D), where t represents a term (word), d is a document, D is a collection of documents, tf(t, d) is the term frequency of t in d, and idf(t, D) is the inverse document frequency of t in D.

Tf-idf is commonly used in natural language processing for text classification and information retrieval tasks as it can down-weight the impact of common words and emphasize the importance of rare, informative words. Additionally, it can help to reduce data dimensionality and increase the weight of significant words, resulting in a more informative and robust feature set for machine learning models.

Text vectorization involves converting text data into numerical vectors that can be used as input for machine learning models. Tf-idf vectorization is one of the most common text vectorization methods alongside bag-of-words (BoW) which uses CountVectorizer, word2vec, and doc2vec.

## Feature Manipulation & Selection

 I have merged the metadata dataframe with cusine_bool and collections_bool.
 
 In the review dataframe I have created a column 'sentiment' with 1 corresponding to positive 0 corresponding to neutral and -1 corresponding to Negative.

I will be usign PCA for feature selection, which will be again beneficial for dimensional reduction, therefore will do the needfull in the precedding step.

The goal of PCA is to identify the most important variables or features that capture the most variation in the data, and then to project the data onto a lower-dimensional space while preserving as much of the variance as possible.

## Data Scaling

I have used Standard Scalar for scaling the data

## Dimesionality Reduction

###PCA

![image](https://user-images.githubusercontent.com/117559898/232272259-7f4a0717-ed37-40cb-81dc-7750d00219be.png)

I have used PCA as dimension reduction technique, PCA is a widely used technique for dimensionality reduction as it is able to identify patterns in the data that are responsible for the most variation, known as principal components. These components are uncorrelated linear combinations of the original features and can be used to effectively reduce the dimensionality of the data while retaining most of the important information. PCA is also a linear technique that is easy to interpret and can be used for data visualization.

When used before k-means, PCA transforms the original feature space into a new space of uncorrelated principal components. This helps to remove noise and correlated features from the data, making the clustering results more interpretable. However, the clusters may be harder to interpret in the original feature space.

When used after k-means, PCA is used to project the data into a lower-dimensional space for easier visualization and interpretation of the clusters. The advantage of this approach is that the clusters can be easily interpreted in the original feature space. However, it may not be as effective in removing noise and correlated features from the data.

## Data Splitting

I have used 80:20 split which is one the most used split ratio. Since there was only 9962 data, therefore I have used more in training set.

## Handling Imbalanced Dataset

![image](https://user-images.githubusercontent.com/117559898/232272325-722e7b5c-23b7-4880-bdee-cde6c209f50d.png)

There are three categories in which sentiment is classified. There is imbalance in dataset with 60: 40 ratio, where 60 is the majaority class(positive sentiment) and 40 is the minority class of Neutral sentiment. Even the CIR score suggest that majority class is 1.53 times greater than minority class.However it is considered as slight imbalance, therefore not performing any under or over sampling technique i.e., not required to treat class imabalance.

There is no imbalance in dataset for majority class (Positive sentiment) and minority class (negative sentiment).

# ML Model Implementation

## ML Model - 1 - Clustering

**KMeans Clustering**

K-Means Clustering is an Unsupervised Learning algorithm.The algorithm takes the unlabeled dataset as input, divides the dataset into k-number of clusters, and repeats the process until it does not find the best clusters. The value of k should be predetermined in this algorithm.

It is a centroid-based algorithm, where each cluster is associated with a centroid. The main aim of this algorithm is to minimize the sum of distances between the data point and their corresponding clusters.

The k-means clustering algorithm mainly performs two tasks:

Determines the best value for K center points or centroids by an iterative process.

Assigns each data point to its closest k-center. Those data points which are near to the particular k-center, create a cluster.

**ELBOW METHOD**

This method uses the concept of WCSS value. WCSS stands for Within Cluster Sum of Squares, which defines the total variations within a cluster.

**SILHOUETTE METHOD**

The silhouette coefficient or silhouette score kmeans is a measure of how similar a data point is within-cluster (cohesion) compared to other clusters (separation).

![image](https://user-images.githubusercontent.com/117559898/232272579-44739f8c-dc51-4fdb-aac2-2bee302b774d.png)

![image](https://user-images.githubusercontent.com/117559898/232272595-839c8dbc-25a6-400f-92ab-c1838f6e0e51.png)

![image](https://user-images.githubusercontent.com/117559898/232272630-b805d448-f149-49e8-8dc9-ec2cfb1fa6dd.png)

![image](https://user-images.githubusercontent.com/117559898/232272638-23ea2a58-c3f9-43cb-8847-37490bdd383b.png)

![image](https://user-images.githubusercontent.com/117559898/232272647-4a0746e9-834f-414c-a5d0-bbf607ebd8fa.png)

![image](https://user-images.githubusercontent.com/117559898/232272652-90a39a80-2353-4d43-a7a6-f4842eea92d6.png)

![image](https://user-images.githubusercontent.com/117559898/232272660-8c4c6af8-ef51-4bec-a5b5-b8d1eabca132.png)

![image](https://user-images.githubusercontent.com/117559898/232272674-b26e6287-18f6-4894-ac1f-9ae770848ef7.png)

![image](https://user-images.githubusercontent.com/117559898/232272689-6ab815d7-9ed9-45e2-b9b9-bd6da7e06967.png)

** For n=3

![image](https://user-images.githubusercontent.com/117559898/232272727-4b920922-7728-431d-8f4f-112109727f6d.png)

Cuisine List for Cluster : 0 

['Chinese' 'Continental' 'Kebab' 'European' 'South Indian' 'North Indian'
 'Biryani' 'Seafood' 'Beverages' 'Healthy Food' 'American' 'Italian'
 'Finger Food' 'Japanese' 'Salad' 'Sushi' 'Mexican' 'Andhra' 'Bakery'
 'Mughlai' 'Juices' 'Arabian' 'Hyderabadi' 'Spanish' 'Thai' 'Indonesian'
 'Asian' 'Momos' 'Fast Food' 'Burger' 'Desserts' 'Cafe'] 

========================================================================================================================
Cuisine List for Cluster : 2 

['Biryani' 'North Indian' 'Chinese' 'Asian' 'Mediterranean' 'Desserts'
 'Continental' 'Seafood' 'Goan' 'Kebab' 'BBQ' 'European' 'American'
 'Italian' 'South Indian' 'Modern Indian' 'Mughlai' 'Sushi'] 

========================================================================================================================
Cuisine List for Cluster : 1 

['Lebanese' 'Ice Cream' 'Desserts' 'Street Food' 'North Indian'
 'Fast Food' 'Burger' 'Chinese' 'Biryani' 'Continental' 'Mughlai' 'Cafe'
 'Bakery' 'American' 'Wraps' 'South Indian' 'Asian' 'Beverages'
 'Hyderabadi' 'Kebab' 'Momos' 'Pizza' 'Arabian' 'North Eastern' 'Seafood'] 

========================================================================================================================

**Agglomerative Hierarchical Clustering**

Hierarchial clustering algorithms group similar objects into groups called clusters. There are two types of hierarchical clustering algorithms:

Agglomerative — Bottom up approach. Start with many small clusters and merge them together to create bigger clusters. Divisive — Top down approach. Start with a single cluster than break it up into smaller clusters.

**Agglomerative hierarchical clustering**

The agglomerative hierarchical clustering algorithm is a popular example of HCA. To group the datasets into clusters, it follows the bottom-up approach. It means, this algorithm considers each dataset as a single cluster at the beginning, and then start combining the closest pair of clusters together. It does this until all the clusters are merged into a single cluster that contains all the datasets. This hierarchy of clusters is represented in the form of the dendrogram.

**Dendrogram in Hierarchical clustering**

The dendrogram is a tree-like structure that is mainly used to store each step as a memory that the HC algorithm performs. In the dendrogram plot, the Y-axis shows the Euclidean distances between the data points, and the x-axis shows all the data points of the given dataset.

![image](https://user-images.githubusercontent.com/117559898/232272901-02aef7b2-f670-4aa9-9063-d10b73adaeb3.png)

** For n Clusters=3

![image](https://user-images.githubusercontent.com/117559898/232272943-01c80606-2d81-43fb-b618-b7be5684b9c9.png)

K-means and hierarchical clustering are two different methods for grouping data points into clusters. K-means is a centroid-based method, where each cluster is defined by the mean of the data points assigned to it. Hierarchical clustering, on the other hand, is a linkage-based method, where clusters are defined by the similarity of data points. Because these methods use different criteria to define clusters, the labels they assign to data points can be different. Additionally, the number of clusters and initialization of the algorithm can also affect the outcome, which can cause the labels to differ.


**KMeans Clustering**

I applied K means Clustering to cluster the Restaurants based on the given features. I used both the Elbow and Silhuoette Methods to get an efficient number of K, and we discovered that n clusters = 3 was best for our model. The model was then fitted using K means, and each data point was labelled with the cluster to which it belonged using K means.labels. After labelling the clusters, we visualised them and counted the number of restaurants in each cluster, discovering that the majority of the restaurants belonged to the first cluster.

**Agglomerative Hierarchical Clustering**

I have used Hierarchial Clustering - Agglomerative Model to cluster the restaurants based on different features. This model uses a down-top approach to cluster the data. I have used Silhouette Coefficient Score and used clusters = 3 and then vizualized the clusters and the datapoints within it.

## ML Model - 2 - Sentiment Analysis

### Unsupervised Sentiment Analysis

### LDA

![image](https://user-images.githubusercontent.com/117559898/232273064-0706848c-de48-470a-8ac5-f1ea386129b1.png)

LDA is an unsupervised learning algorithm, it doesn't have any predefined labels. The labels are assigned based on the analysis done on the words, the weights of the words, and the context of the words in each topic. So, the predicted topic is not a definite answer, therfore experimenting with different techniques like using supervised algorithm and combining the results to make a more accurate sentiment labeling.

### Supervised Sentiment Analysis

**Logistic Regression**

![image](https://user-images.githubusercontent.com/117559898/232273208-15079643-0d0c-43a6-8d48-38022129312c.png)
![image](https://user-images.githubusercontent.com/117559898/232273217-5b0e1870-8ca3-41d5-b182-44ebd4d1ab19.png)

For train data logestic regression has correctly classified positive for 2666 values, neutral values for 1182 and negative values for 2723 values. I have evaluate the metrics of the train data and found that accuracy is 82.45%, Recall is 81.65%, Precession is 79.9%, f1 score is 80.25% and ROC_AUC is 93.96%.

For test data logestic regression has correctly classified positive for 610 values, neutral values for 201 and negative values for 620 values. I have evaluate the metrics of the train data and found that accuracy is 71.8%, Recall is 68.74%, Precession is 68.25%, f1 score is 68.12% and ROC_AUC is 87.09%.

 **Cross- Validation & Hyperparameter Tuning**

![image](https://user-images.githubusercontent.com/117559898/232273308-20024b20-9059-4082-9908-aed8ce7196b5.png)
![image](https://user-images.githubusercontent.com/117559898/232273311-fc5aa93d-762c-41db-ab13-849428dc760d.png)

For train data logestic regression with cross validation has correctly classified positive for 2866 values, neutral values for 1918 and negative values for 3022 values. I have evaluate the metrics of the train data and found that accuracy is 97.95%, Recall is 98.20%, Precession is 97.66%, f1 score is 97.90% and ROC_AUC is 99.87%.

For test data logestic regression with cross validation has correctly classified positive for 528 values, neutral values for 203 and negative values for 555 values. I have evaluate the metrics of the train data and found that accuracy is 64.52%, Recall is 62.09%, Precession is 61.87%, f1 score is 61.95% and ROC_AUC is 78.77%.

hyperparameter optimization technique used:

GridSearchCV which uses the Grid Search technique for finding the optimal hyperparameters to increase the model performance.

our goal should be to find the best hyperparameters values to get the perfect prediction results from our model. But the question arises, how to find these best sets of hyperparameters? One can try the Manual Search method, by using the hit and trial process and can find the best hyperparameters which would take huge time to build a single model.

For this reason, methods like Random Search, GridSearch were introduced. Grid Search uses a different combination of all the specified hyperparameters and their values and calculates the performance for each combination and selects the best value for the hyperparameters. This makes the processing time-consuming and expensive based on the number of hyperparameters involved.

In GridSearchCV, along with Grid Search, cross-validation is also performed. Cross-Validation is used while training the model.

That's why I have used GridsearCV method for hyperparameter optimization.

**Random Forest Classifier**

![image](https://user-images.githubusercontent.com/117559898/232273549-55693418-cdad-4ff2-b4ca-949ce3a043b6.png)
![image](https://user-images.githubusercontent.com/117559898/232273558-20ef7ef7-dda5-4a62-9646-0facfcbca1fa.png)

For train data Random Forest Classifier has correctly classified positive for 2875 values, neutral values for 1930 and negative values for 3023 values. I have evaluate the metrics of the train data and found that accuracy is 98.23%, Recall is 98.46%, Precession is 97.97%, f1 score is 98.19% and ROC_AUC is 99.59%.

For test data Random Forest has correctly classified positive for 612 values, neutral values for 120 and negative values for 660 values. I have evaluate the metrics of the train data and found that accuracy is 69.84%, Recall is 66.05%, Precession is 64.37%, f1 score is 63.11% and ROC_AUC is 85.15%.

![table2](https://user-images.githubusercontent.com/117559898/232273666-9aad2e9a-6a0f-4650-981a-e1d0e97cc76f.PNG)

From the above Table we can say that for Random Forest Classifier The ROC_AUC score for Train is high and ROC_AUC score of test is also higher. Clearly there is no bias and variance in the algorithm and have performed better than logistic regression and logistic regression with cross validation. So, I have chosed Random Forest Classifier. Further we will observe ROC curves for all the algorithms.

![image](https://user-images.githubusercontent.com/117559898/232273707-40fff3f9-dbe6-4482-8534-00c4e867c3a3.png)

![image](https://user-images.githubusercontent.com/117559898/232273743-ec14dd25-b6ad-49f8-9c3b-2fdeae66be6b.png)

![image](https://user-images.githubusercontent.com/117559898/232273762-db44ace6-2fb9-4082-a121-2db1db293ce7.png)

I have plotted ROC AUC Curves for the above algorithms. The ROC AUC curves for all classes are plotted and the AUC score in Random Forest is better and with AUC Score of 91 for negative class, 77 for neutral class and 88 for positive class

The Random Forest Classifier can be considered as an efficient model for the business, especially when it achieves high scores in all of these evaluation metrics, which would indicate that it can accurately predict outcomes, identify all positive instances, and correctly classify instances as positive or negative.

## ML Model - 3 - Recommendation System

Content-based filtering is a commonly used technique in recommendation systems. It recommends items to users based on their past preferences or interactions by analyzing the attributes of the items and the user's profile. To avoid plagiarism, it is important to use your own words when summarizing or paraphrasing information from a source. In this case, you could rephrase the text as follows:

Content-based filtering is a popular recommendation system technique that suggests items to users by examining the features of both the items and the user's history. By creating a user profile that outlines their preferences, such as the type of restaurants or books they enjoy, the system can suggest items that match those attributes. This technique can also be used for new users, by recommending items based on their attributes. Collaborative filtering can be used alongside content-based filtering to enhance recommendation accuracy. Various models, such as cosine similarity, nearest neighbors, and vector space models, can be employed depending on the type of data and attributes of the items.

Cosine Similarity:

Cosine similarity is a measure of similarity between two vectors that is commonly used in recommendation systems for content-based filtering. It determines the cosine of the angle between two vectors and ranges from -1 to 1. A value of 1 indicates that the two vectors are identical, while a value of -1 means they are completely dissimilar. Cosine similarity is used to compare the attributes of items, and the closer the value is to 1, the more similar the items are considered to be.

Nearest Neighbours:

Nearest neighbors is an algorithm used in recommendation systems for content-based filtering. It is used to find the k-nearest items to a given item based on their attributes. The algorithm calculates the distance between items and identifies the k closest items based on that distance metric. For example, in a movie recommendation system, if a user has watched and enjoyed a particular action movie, the system would find other action movies with similar attributes and recommend them to the user as the nearest neighbors. This algorithm is often used in conjunction with cosine similarity to improve the accuracy of recommendations.

Vector Space Model:

Vector Space Model is an algorithm used in recommendation systems for content-based filtering that represents items as vectors in a multi-dimensional space, where the dimensions represent the attributes of the items. Each attribute is assigned a weight, which determines its importance in the recommendation process. The model creates a profile for each user that captures their preferences by analyzing their interactions with items. To make recommendations, the algorithm calculates the similarity between items based on the distance between their vectors in the multi-dimensional space. The more similar the items are, the closer their vectors are in the space. This algorithm is commonly used in text-based recommendation systems, where the attributes represent the features of the text.

## Conclusion

Grouping restaurants according to their cuisine offerings can aid food delivery services like Zomato in helping clients discover the food they're looking for. In general, clustering based on cuisine offers can be helpful in revealing information about local residents' tastes in food and informing commercial decisions like the opening of restaurants, food delivery services, and advertising.

By using sentiment analysis and clustering on a dataset of customer reviews, the analysis's goal was to learn more about how customers felt about the food delivery service Zomato. Three classes indicating good, neutral and negative customer satisfaction levels were produced as a consequence of using the clustering approach to group consumers based on the text of their reviews. After classifying the review language as favourable or negative, sentiment analysis was used to identify particular areas where the service may be improved. Significant information from this research can be used to inform business choices and enhance the quality of the service. Integrating sentiment analysis and clustering algorithms enables a thorough comprehension of client comments.

Zomato's recommendations for restaurants are tailored to each customer based on their historical ordering patterns, culinary preferences, dietary restrictions, and reviews. As a result, finding suitable restaurants will take less time and effort, improving the user experience. Zomato can make recommendations for similar restaurants to a customer based on restaurant characteristics including cuisine,  pricing range, and reviews. Customers may find new eateries that are comparable to their favourite ones thanks to this.

Other important discoveries and recommendation during analysis are -

* There are many hotels which are poorly rated and have price range less than 1000. These hotels ought to be eliminated since they could damage Zomato's reputation.
* The hotel AB's - Absolute Barbecues, show maximum engagement and retention as it has maximum number of rating on average and Hotel Zara Hi-Fi show lowest engagement as has lowest average rating.
* The best food, as offered by the majority of restaurants, is north Indian, followed by Chinese.
restaurant Collage - Hyatt Hyderabad Gachibowli is most expensive restaurant in the locality which has a price of 2800 for order and has 3.5 average rating. Hotels like Amul and Mohammedia Shawarma are least expensive with price of 150 and has 3.9 average rating.
* Based on negative reviews like some focused on issues with delivery time or food quality, the company should prioritize addressing these issues to imporve customer satisfaction.
* Also use the clustering results to target specific customer segments and tailor marketing and promotional efforts accordingly.






















