import numpy as np
import perceptron
import samples
import data_classification_utils as dcu
from util import raiseNotDefined

"""feel free to play with these values and see what happens"""
bias = False
num_times_to_train = 10
num_train_examples = 5000

def get_perceptron_training_data():
	training_data = samples.loadDataFile("digitdata/trainingimages", num_train_examples, 28, 28)
	training_labels = map(str, samples.loadLabelsFile("digitdata/traininglabels", num_train_examples))

	featurized_training_data = np.array(map(dcu.simple_image_featurization, training_data))
	return training_data, featurized_training_data, training_labels

def get_perceptron_test_data():
	test_data = samples.loadDataFile("digitdata/testimages", 1000, 28,28)
	test_labels = map(str, samples.loadLabelsFile("digitdata/testlabels", 1000))

	featurized_test_data = np.array(map(dcu.simple_image_featurization, test_data))
	return test_data, featurized_test_data, test_labels

"""
if you want a bias, then apply that bias to your data, then create a perceptron to identify digits

Next, train that perceptron on the entire set of training data num_times_to_train times on num_train_examples.

Finally, use the zero_one_loss defined in data_classification_utils to find the 
final accuracy on both the training set and the test set, assigning them to the 
variables training_accuracy and test_accuracy respectively"""



raw_training_data, featurized_training_data, training_labels = get_perceptron_training_data()
raw_test_data, featurized_test_data, test_labels = get_perceptron_test_data()

"""YOUR CODE HERE"""
if bias:
	featurized_training_data = dcu.apply_bias(raw_training_data)
	featurized_test_data = dcu.apply_bias(raw_test_data)
	p = perceptron.Perceptron(["0","1","2","3","4","5","6","7","8","9","10"], 785)
else:
	p = perceptron.Perceptron(["0","1","2","3","4","5","6","7","8","9"], 784)

# for i in range(num_train_examples):
for _ in range(num_times_to_train):
	p.train(featurized_training_data[:num_train_examples],training_labels[:num_train_examples])

training_accuracy = (1 - dcu.zero_one_loss(p, featurized_training_data, training_labels))*100
test_accuracy = (1 - dcu.zero_one_loss(p, featurized_test_data, test_labels)) * 100
print('Final training accuracy: ' + str(training_accuracy) + '% correct')

print("Test accuracy: " + str(test_accuracy) + '% correct')

dcu.display_digit_features(np.transpose(p.weightMatrix), bias)
