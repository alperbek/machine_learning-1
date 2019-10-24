import pandas as pd
import math
import numpy as np

images = pd.read_csv("hw01_images.csv", header=None)
training_images = images.loc[:199,:]
training_set = training_images.values.tolist()
test_set = images.loc[200:,:].values.tolist()
y = pd.read_csv("hw01_labels.csv", header=None, names=['class'])
training_y = y.loc[:199,:]
test_y = y.loc[200:,:]

means =  np.asarray([training_images[training_y['class'] == 1].mean().tolist() , training_images[training_y['class'] == 2].mean().tolist()])
deviations = [ np.sqrt(((training_images[training_y['class'] == 1] - means[0])**2).mean().tolist()) , np.sqrt(((training_images[training_y['class'] == 2] - means[1])**2).mean().tolist())]
variances = np.square(deviations)

num_of_class_one = training_images[training_y['class'] == 1].shape[0]
num_of_class_two = training_images[training_y['class'] == 2].shape[0]
total = num_of_class_one + num_of_class_two
priors = [ num_of_class_one / total , num_of_class_two / total ]

def get_scores(x_vector):
    scores = [0,0]
    for i in range(2):
        scores[i]  = np.sum((-0.5 * ( np.log(2 * math.pi * variances[i] ))) + (-0.5 * (means[i] - x_vector )**2 /  variances[i] )) + np.log(priors[i])
    return scores

def predict(x_vector):
    scores = get_scores(x_vector)
    return scores.index(max(scores)) + 1

def predict_set(image_set):
    predictions = []
    for image in image_set:
        prediction = predict(image)
        predictions.append(prediction)
    return predictions

training_predictions = predict_set(training_set)
test_predictions = predict_set(test_set)

def confusion_matrix(predictions, actual):
    pr_1_ac_1, pr_1_ac_2, pr_2_ac_1, pr_2_ac_2 = 0,0,0,0
    for i in range(len(training_predictions)):
        if predictions[i] == 1 and actual[i] == 1:
            pr_1_ac_1 += 1
        if predictions[i] == 1 and actual[i] == 2:
            pr_1_ac_2 += 1
        if predictions[i] == 2 and actual[i] == 1:
            pr_2_ac_1 += 1
        if predictions[i] == 2 and actual[i] == 2:
            pr_2_ac_2 += 1
    return pr_1_ac_1, pr_1_ac_2, pr_2_ac_1, pr_2_ac_2

print("means[0]: {}".format(means[0]))
print("means[1]: {}".format(means[1]))
print("deviations[0]: {}".format(deviations[0]))
print("deviations[1]: {}".format(deviations[1]))
print("priors: {}\n".format(priors))
pr_1_ac_1, pr_1_ac_2, pr_2_ac_1, pr_2_ac_2 = confusion_matrix(training_predictions, training_y['class'].tolist())
print("y_hat    1       2\ny_training  \n1        {}       {}\n2        {}       {}\n".format(pr_1_ac_1, pr_2_ac_1,pr_1_ac_2,pr_2_ac_2))
pr_1_ac_1, pr_1_ac_2, pr_2_ac_1, pr_2_ac_2 = confusion_matrix(test_predictions, test_y['class'].tolist())
print("y_hat    1       2\ny_test      \n1        {}       {}\n2        {}       {}".format(pr_1_ac_1, pr_2_ac_1,pr_1_ac_2,pr_2_ac_2))
