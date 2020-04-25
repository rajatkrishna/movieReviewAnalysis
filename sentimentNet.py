import time
import sys
import numpy as np
from collections import Counter


class SentimentNetwork:
    def __init__(self, reviews, labels, hidden_nodes = 10, learning_rate = 0.1):


        np.random.seed(1)

        self.pre_process_data(reviews, labels)

        self.init_network(len(self.review_vocab),hidden_nodes, 1, learning_rate)

    def pre_process_data(self, reviews, labels):

        review_vocab = set()
        for rev in reviews:
            review_vocab.update(rev.split(' '))

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

        self.weights_0_1 = np.zeros([self.input_nodes, self.hidden_nodes])

        self.weights_1_2 = np.random.randn(self.hidden_nodes, self.output_nodes)

        self.layer_0 = np.zeros((1,input_nodes))


    def update_input_layer(self,review):

        self.layer_0 *= 0
        wrds = review.split(' ')

        for wrd in wrds:
            if(wrd in self.word2index.keys()):
                self.layer_0[0][self.word2index[wrd]] += 1

    def get_target_for_label(self,label):

        l = int(label == 'POSITIVE')
        return l

    def sigmoid(self,x):

        return 1 / (1 + np.exp(-x))

    def sigmoid_output_2_derivative(self,output):

        return output * (1 - output)

    def train(self, training_reviews, training_labels):

        assert(len(training_reviews) == len(training_labels))

        correct_so_far = 0

        start = time.time()

        for i in range(len(training_reviews)):

            rev = training_reviews[i]
            lab = training_labels[i]

            self.update_input_layer(rev)
            inp_to_hid = self.layer_0.dot(self.weights_0_1)
            hid_to_out = inp_to_hid.dot(self.weights_1_2)
            out = self.sigmoid(hid_to_out)

            error2 = out - self.get_target_for_label(lab)
            delta2 = error2 * self.sigmoid_output_2_derivative(out)

            error1 = delta2.dot(self.weights_1_2.T)
            delta1 = error1

            self.weights_1_2 -= inp_to_hid.T.dot(delta2) * self.learning_rate
            self.weights_0_1 -= self.layer_0.T.dot(delta1) * self.learning_rate

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

        correct = 0
        start = time.time()
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

        self.update_input_layer(review.lower())
        inp_to_hid = self.layer_0.dot(self.weights_0_1)
        hid_to_out = inp_to_hid.dot(self.weights_1_2)
        out = self.sigmoid(hid_to_out)
        
        if out[0] >= 0.5:
            return "POSITIVE"
        else:
            return "NEGATIVE"
