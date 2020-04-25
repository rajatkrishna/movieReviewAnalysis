

import time
import sys
import numpy as np
from collections import Counter

# Encapsulate our neural network in a class
class SentimentNetwork:
    def __init__(self, reviews, labels, min_count, polarity_cutoff, hidden_nodes = 10, learning_rate = 0.1):
        """Create a SentimenNetwork with the given settings
        Args:
            reviews(list) - List of reviews used for training
            labels(list) - List of POSITIVE/NEGATIVE labels associated with the given reviews
            hidden_nodes(int) - Number of nodes to create in the hidden layer
            learning_rate(float) - Learning rate to use while training

        """

        np.random.seed(1)
        self.pre_process_data(reviews, labels, min_count, polarity_cutoff)

        self.init_network(len(self.review_vocab),hidden_nodes, 1, learning_rate)


    def pre_process_data(self, reviews, labels, min_count, polarity_cutoff):

        rev_count = Counter()
        pos_count = Counter()
        neg_count = Counter()
        review_vocab = set()
        for i in range(len(reviews)):
            wrds = reviews[i].split(' ')
            rev_count.update(wrds)
            if labels[i] == 'POSITIVE':
                pos_count.update(wrds)
            else:
                neg_count.update(wrds)
        for word in set(rev_count.elements()):
            if rev_count[word] > min_count and abs(np.log(pos_count[word] / (neg_count[word] + 1))) > polarity_cutoff:
                review_vocab.add(word)
        # Convert the vocabulary set to a list so we can access words via indices
        self.review_vocab = list(review_vocab)


        label_vocab = set(labels)
        self.label_vocab = list(label_vocab)

        # Store the sizes of the review and label vocabularies.
        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)

        # Create a dictionary of words in the vocabulary mapped to index positions
        self.word2index = {}
        for i, wrd in enumerate(self.review_vocab):
            self.word2index[wrd] = i
        # Create a dictionary of labels mapped to index positions
        self.label2index = {}
        for i, wrd in enumerate(self.label_vocab):
            self.label2index[wrd] = i


    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Store the number of nodes in input, hidden, and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Store the learning rate
        self.learning_rate = learning_rate

        # Initialize weights
        self.layer_1 = np.zeros([1, hidden_nodes])
        self.weights_0_1 = np.zeros([self.input_nodes, self.hidden_nodes])

        self.weights_1_2 = np.random.randn(self.hidden_nodes, self.output_nodes)


    def get_target_for_label(self,label):
        l = int(label == 'POSITIVE')
        return l

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_output_2_derivative(self,output):
        return output * (1 - output)

    def train(self, training_reviews_raw, training_labels):

        training_reviews = list()
        for review in training_reviews_raw:
            indices = set()
            for word in review.split(" "):
                if(word in self.word2index.keys()):
                    indices.add(self.word2index[word])
            training_reviews.append(list(indices))

        # Keep track of correct predictions to display accuracy during training
        correct_so_far = 0

        # Remember when we started for printing time statistics
        start = time.time()

        # loop through all the given reviews and run a forward and backward pass,
        # updating weights for every item
        for i in range(len(training_reviews)):

            rev = training_reviews[i]
            lab = training_labels[i]
            self.layer_1 *= 0
            for index in rev:
                self.layer_1 += self.weights_0_1[index]
            hid_to_out = self.layer_1.dot(self.weights_1_2)
            out = self.sigmoid(hid_to_out)

            error2 = out - self.get_target_for_label(lab)
            delta2 = error2 * self.sigmoid_output_2_derivative(out)

            error1 = delta2.dot(self.weights_1_2.T)
            delta1 = error1

            self.weights_1_2 -= self.layer_1.T.dot(delta2) * self.learning_rate

            for index in rev:
                self.weights_0_1[index] -= delta1[0] * self.learning_rate
            if(out >= 0.5 and lab == 'POSITIVE'):
                correct_so_far += 1
            elif(out < 0.5 and lab == 'NEGATIVE'):
                correct_so_far += 1

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0

            sys.stdout.write("\rProgress:" + str(100 * i/float(len(training_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct_so_far) + " #Trained:" + str(i+1) \
                             + " Training Accuracy:" + str(correct_so_far * 100 / float(i+1))[:4] + "%")
            if(i % 2500 == 0):
                print("")

    def test(self, testing_reviews, testing_labels):

        # keep track of how many correct predictions we make
        correct = 0
        start = time.time()

        # Loop through each of the given reviews and call run to predict
        # its label.
        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            if(pred == testing_labels[i]):
                correct += 1

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0

            sys.stdout.write("\rProgress:" + str(100 * i/float(len(testing_reviews)))[:4] \
                             + "% Speed(reviews/sec):" + str(reviews_per_second)[0:5] \
                             + " #Correct:" + str(correct) + " #Tested:" + str(i+1) \
                             + " Testing Accuracy:" + str(correct * 100 / float(i+1))[:4] + "%")

    def run(self, review):

        self.layer_1 *= 0
        unique_indices = set()
        for word in review.lower().split(" "):
            if word in self.word2index.keys():
                unique_indices.add(self.word2index[word])
        for index in unique_indices:
            self.layer_1 += self.weights_0_1[index]


        hid_to_out = self.layer_1.dot(self.weights_1_2)
        out = self.sigmoid(hid_to_out)

        if out[0] >= 0.5:
            return "POSITIVE"
        else:
            return "NEGATIVE"

mlp = SentimentNetwork(reviews[:-1000],labels[:-1000],min_count=20,polarity_cutoff=0.8,learning_rate=0.01)
mlp.train(reviews[:-1000],labels[:-1000])        
