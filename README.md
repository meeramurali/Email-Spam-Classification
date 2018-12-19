# Email-Spam-Classification

This project involved the implementation of Gaussian Naïve Bayes to classify the Spambase data from the UCI ML repository. 
The dataset contains 4601 examples which were split into training and test sets having around 2301 and 2300 instances respectively with roughly 40% spam and 60% non-spam data to reflect the statistics of the full dataset. 
Since each feature is assumed to be independent, no standardization of features is required.
A probabilistic model was created using the training set, by first computing the prior probability for each class – spam and non-spam. 
Then, given each class, for each of the 57 features, mean and standard deviation were computed. Standard deviation values of zero were replaced by 0.0001 to avoid divide-by-zero error.
Gaussian Naïve Bayes algorithm was then used to classify test set instances using:

