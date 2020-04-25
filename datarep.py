from collections import Counter
import numpy as np

positive_counts = Counter()
negative_counts = Counter()
total_counts = Counter()

g = open('reviews.txt','r')
reviews = list(map(lambda x:x[:-1],g.readlines()))
g.close()

g = open('labels.txt','r')
labels = list(map(lambda x:x[:-1].upper(),g.readlines()))
g.close()

#split words into positive and negative
for i in range(len(labels)):
    words = reviews[i].split(' ')
    if labels[i] == 'POSITIVE':
        positive_counts.update(words)
    else:
        negative_counts.update(words)
    total_counts.update(words)


pos_neg_ratios = Counter()

#log of positive to negative ratios of most common words
for term,cnt in list(total_counts.most_common()):
    if(cnt > 100):
        pos_neg_ratio = positive_counts[term] / float(negative_counts[term]+1)
        pos_neg_ratios[term] = np.log(pos_neg_ratio)


vocab = set(total_counts.elements())
vocab_size = len(vocab)

layer_0 = np.zeros([1, vocab_size])

word2index = {}
for i,word in enumerate(vocab):
    word2index[word] = i


def update_input_layer(review):

    global layer_0
    # clear out previous state by resetting the layer to be all 0s
    layer_0 *= 0
    i = 0

    wrds = review.split(' ')
    count = Counter(wrds)

    for wrd in vocab:
        layer_0[0, i] = count[wrd]
        i += 1


def get_target_for_label(label):

    l = int(label == 'POSITIVE')
    return l
