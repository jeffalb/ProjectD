# mira.py
# -------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# Mira implementation
import util

PRINT = True


class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """

    def __init__(self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter()  # this is the data-structure you should use

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        self.features = trainingData[0].keys()  # this could be useful for your code later...

        if (self.automaticTuning):
            Cgrid = [0.002, 0.004, 0.008]
        else:
            Cgrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
        """
        This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the validationData.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """
        "*** YOUR CODE HERE ***"
        self.features = trainingData[0].keys()
        bestResult = 0
        for C in Cgrid:
            weights = {}
            for label in self.legalLabels:
                weights[label] = util.Counter()
            for iteration in range(self.max_iterations):
                print "Starting iteration ", iteration, "..."
                for i in range(len(trainingData)):
                    # First we set up the training datum and default a scoreMax and y'
                    datum = trainingData[i].copy()
                    scoreMax = 0
                    yAccent = 0

                    # For every label in the perceptron's legalLabels we calculate the score
                    # If it is higher than the previous we set it as the max score and save the
                    # label (y') associated with it.
                    for y in self.legalLabels:
                        score = datum * weights[y]
                        if score > scoreMax:
                            scoreMax = score
                            yAccent = y

                    yTrue = trainingLabels[i]

                    # Next we compare the y and y'. If it differs the weights are adjusted
                    if yAccent != yTrue:
                        tau = 0.5 * ((weights[yAccent] - weights[yTrue]) * datum + 1.0) / (datum * datum)
                        if C < tau:
                            tau = C
                        if tau != 0:
                            datum.divideAll(1 / tau)  # util.counter lacks a multiplyAll method
                            weights[yTrue] = weights[yTrue] + datum
                            weights[yAccent] = weights[yAccent] - datum

            result = 0
            for i in range(len(validationData)):
                datum = validationData[i]
                scoreMax = 0
                yAccent = 0
                # For every label in the perceptron's legalLabels we calculate the score
                # If it is higher than the previous we set it as the max score and save the
                # label (y') associated with it.
                for y in self.legalLabels:
                    score = datum * weights[y]
                    if score > scoreMax:
                        scoreMax = score
                        yAccent = y
                yTrue = validationLabels[i]
                # check if the instance is classified correctly:
                if yTrue == yAccent:
                    result += 1
            print(str(bestResult) + " correct out of 100 with C value " + str(C))
            # check if the number of correct classifications with the current value of C is greater than
            # the number of correct classifications of previous values of C:
            if result > bestResult:
                bestResult = result
                self.weights = weights

    def classify(self, data):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = self.weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses

    def findHighWeightFeatures(self, label):
        """
        Returns a list of the 100 features with the greatest weight for some label
        """
        featuresWeights = []

        "*** YOUR CODE HERE ***"
        # we use util.counter to gives us a list of the keys sorted by highest weight
        # Next we set the featuresWeights as the first 100 of this list
        sortedlist = self.weights[label].sortedKeys()

        featuresWeights = sortedlist[:100]

        return featuresWeights
