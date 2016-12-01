import numpy as np 
import data_classification_utils
from util import raiseNotDefined
import random

class Perceptron(object):
    def __init__(self, categories, numFeatures):
        """categories: list of strings 
           numFeatures: int"""
        self.categories = categories
        self.numFeatures = numFeatures

        """YOUR CODE HERE"""
        self.weightMatrix = np.zeros((len(self.categories), self.numFeatures))


    def classify(self, sample):
        """sample: np.array of shape (1, numFeatures)
           returns: category with maximum score, must be from self.categories"""

        """YOUR CODE HERE"""
        arr = []
        for i in range(len(self.categories)):
          arr.append(np.dot(self.weightMatrix[i], sample))
        return self.categories[np.argmax(arr)]


    def train(self, samples, labels):
        """samples: np.array of shape (numSamples, numFeatures)
           labels: list of numSamples strings, all of which must exist in self.categories 
           performs the weight updating process for perceptrons by iterating over each sample once."""

        """YOUR CODE HERE"""
        arr = []
        for i in range(len(labels)):
          labelprime = self.classify(samples[i])
          if labelprime != labels[i]:
            for j in range(len(self.categories)):
              if self.categories[j] == labelprime:
                self.weightMatrix[j] = self.weightMatrix[j] - samples[i]
              if self.categories[j] == labels[i]:
                self.weightMatrix[j] = self.weightMatrix[j] + samples[i]


