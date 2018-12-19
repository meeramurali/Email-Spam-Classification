"""
Author: Meera Murali
(Collaborated with Niveditha Venugopal)
Date: 6/1/2018
Course: CS545

Naive Bayes classifier on SpamBase dataset
"""

import numpy as np
import math
import collections


# Loads data from file into numpy array
# Arguments: file path to load from
# Returns numpy array
def loadDataFromFile(filePath):
    File = open(filePath,'r')
    dataArray = np.loadtxt(File,delimiter=',')
    return dataArray



# Splits data into training and test sets
# Arguments: numpy data array
# Returns training and test data and target arrays
def generate_train_test_data(dataArray):
    totalData = len(dataArray)

    # Counting the spam occurrences in the data set
    totalSpamDataSize = ((dataArray[:, -1]) != 0).sum(0)

    # Counting the non spam occurrences in the data set
    totalNonSpamDataSize = totalData - totalSpamDataSize
    testSpamSize = 0
    testNonSpamSize = 0

    # Initializing the test data array
    testDataArray = np.empty((int(totalData/2),58))
    j=0

    # Iterating over the source array and creating the test data array
    for i in range(len(dataArray)):

        # If the test data array reaches the size, break out of the loop
        if i == len(dataArray):
            break

        # Add spam data to test array till half of the total spam data is added to test array
        if dataArray[i][-1] == 1 and testSpamSize < int(totalSpamDataSize / 2):
            testSpamSize +=1
            testDataArray[j] = dataArray[i]
            dataArray = np.delete(dataArray, i, axis=0)
            j+=1

        # Add non spam data to test array till half of the total non spam data is added to the test array
        elif dataArray[i][-1] == 0 and testNonSpamSize < int(totalNonSpamDataSize / 2):
            testNonSpamSize +=1
            testDataArray[j] = dataArray[i]
            dataArray = np.delete(dataArray, (i), axis=0)
            j+=1

    # Creating the training data array and target array
    trainingDataArray = dataArray
    trainingTargetArray = trainingDataArray[:, -1]
    trainingTargetArray = np.reshape(trainingTargetArray,((len(trainingTargetArray),1)))
    trainingDataArray = trainingDataArray[:, 0:-1]

    # Creating the test data array and target array
    testTargetArray = testDataArray[:,-1]
    testTargetArray = np.reshape(testTargetArray,(len(testTargetArray),1))
    testDataArray = testDataArray[:,0:-1]

    return trainingDataArray,trainingTargetArray,testDataArray,testTargetArray



# Computes prior probabilities for each class
def computePrior(trainingTargetArray):
    spamDataSize = ((trainingTargetArray[:, 0]) != 0).sum(0)
    priorSpam = spamDataSize/len(trainingTargetArray)
    priorNonSpam = (len(trainingTargetArray)-spamDataSize)/len(trainingTargetArray)
    return priorSpam,priorNonSpam



# Computes mean and standard deviation for each class
def computeMeanAndSD(training_data, training_target):
    trainingArray = np.concatenate((training_data, training_target), axis=1)
    trainingSpamArray = trainingArray[np.where(trainingArray[:,-1]==1)]
    trainingNonSpamArray = trainingArray[np.where(trainingArray[:, -1] == 0)]
    spamMean = trainingSpamArray.mean(axis=0)
    spamMean = spamMean[:-1]
    spamSD = trainingSpamArray.std(axis=0)
    spamSD = spamSD[:-1]
    nonSpamMean = trainingNonSpamArray.mean(axis=0)
    nonSpamMean = nonSpamMean[:-1]
    nonSpamSD = trainingNonSpamArray.std(axis=0)
    nonSpamSD = nonSpamSD[:-1]
    spamSD[np.where(spamSD == 0)] = 0.0001
    nonSpamSD[np.where(nonSpamSD == 0)] = 0.0001
    return spamMean,nonSpamMean,spamSD,nonSpamSD



# Applies the Gaussian NB algorithm to classify instances in data set
# NB = (1/sqrt(2*pi) * standard deviation) e^(-(x-mean)^2 / 2*(standard deviation)^2)
def NBClassifier(mean,SD,testDataArray):
    test_data_length = len(testDataArray)
    mean_array = np.array([mean,]*test_data_length)
    SD_array = np.array([SD,]*test_data_length)
    NB = np.subtract(testDataArray,mean_array)
    NB = np.square(NB)
    NB = np.multiply(-1,NB)
    SD_array_square = np.square(SD_array)
    two_SD_array_square = np.multiply(2,SD_array_square)
    NB = np.divide(NB,two_SD_array_square)
    NB = np.exp(NB)
    constant_term = 1/math.sqrt(2*math.pi)
    NB = np.multiply(constant_term,NB)
    NB = np.divide(NB,SD_array)
    NB[np.where(NB == 0)] = math.pow(10,-55)
    return NB



# Computes log of product of probabilities for a class
def productProbability(NB,prior):
    NB = np.log(NB)
    cls = np.sum(NB,axis=1)
    cls = np.multiply(prior,cls)
    return cls



# Predicts the class for each example by determining the argmax between
# conditional probabilities for spam vs non-spam
def predictClass(spamClass,nonSpamClass):
    predictedClass = np.subtract(spamClass,nonSpamClass)
    predictedClass[np.where(predictedClass < 0)] = 0
    predictedClass[np.where(predictedClass > 0)] = 1
    return predictedClass



# Computes accuracy, precision, recall and confusion matrix on test set
def compute_accuracy(predicted_class, test_target):
    predicted_class = np.reshape(predicted_class, (len(predicted_class), 1))

    # Identify correct classfications as hits
    hits = np.equal(predicted_class, test_target)

    # TP: true positives, FP: false positives, TN: true negatives, FN: false negatives
    TP = TN = FP = FN = 0

    # Count TP, FP, TN, FN
    for i in range(len(test_target)):
        if(test_target[i][0] == 1):
            if(hits[i] == True):
                TP += 1
            else:
                FN += 1
        else:
            if(hits[i] == True):
                TN += 1
            else:
                FP += 1

    # Compute accuracy, precision, recall and confusion matrix
    precision = (TP * 100) / (TP + FP)
    recall = (TP * 100) / (TP + FN)
    accuracy = ((TP + TN) * 100) / (TP + TN + FP + FN)
    confusion_matrix = np.array([[TP,FN],[FP,TN]])

    return accuracy, precision, recall, confusion_matrix



if __name__ == "__main__":

    # Load data from file
    data = loadDataFromFile('spambase/spambase.data')

    # Split data into training and test sets
    training_data, training_target, test_data, test_target = generate_train_test_data(data)

    # Compute prior probabilities for each class
    prior_spam, prior_nonSpam = computePrior(training_target)

    # Compute mean and standard deviation for each class
    spam_mean, non_spam_mean, spam_SD, non_spam_SD = computeMeanAndSD(training_data, training_target)

    # Apply the Gaussian NB algorithm to classify instances in data set
    # NB = (1/sqrt(2*pi) * standard deviation) e^(-(x-mean)^2 / 2*(standard deviation)^2)
    spam_NB = NBClassifier(spam_mean, spam_SD, test_data)
    non_spam_NB = NBClassifier(non_spam_mean, non_spam_SD, test_data)

    # Compute log of product of probabilities for each class
    spam_Class = productProbability(spam_NB, prior_spam)
    non_spam_class = productProbability(non_spam_NB, prior_nonSpam)

    # Predicts the class for each example by determining the argmax between
    # conditional probabilities for spam vs non-spam
    predicted_class = predictClass(spam_Class, non_spam_class)

    # Compute accuracy, precision, recall and confusion matrix on test set
    accuracy, precision, recall, confusion_matrix = compute_accuracy(predicted_class, test_target)

    # Print results
    print("________________________________\n")
    print("Accuracy: %.3f" % accuracy)
    print("________________________________\n")
    print("Precision: %.3f" % precision)
    print("________________________________\n")
    print("Recall: %.3f" % recall)
    print("________________________________\n")
    print("Confusion Matrix:\n\n", confusion_matrix)
    print("________________________________\n")


