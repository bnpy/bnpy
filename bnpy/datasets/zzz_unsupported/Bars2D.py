'''
Bars2D.py

Generic functions for creating toy bars data
'''
import numpy as np


def Create2DBarsTopicWordParams(V, K, fracMassOnTopic=0.95, PRNG=np.random):
    ''' Create parameters of each topics distribution over words

        Args
        ---------
        V : int vocab size
        K : int number of topics
        fracMassOnTopic : fraction of probability mass for "on-topic" words
        PRNG : random number generator (for reproducibility)

        Returns
        ---------
        topics : K x V matrix, real positive numbers whose rows sum to one
    '''
    sqrtV = int(np.sqrt(V))
    BarWidth = sqrtV / (K / 2)  # number of consecutive words in each bar
    B = V / (K / 2)  # total number of "on topic" words in each bar

    topics = np.zeros((K, V))
    # Make horizontal bars
    for k in range(K / 2):
        wordIDs = list(range(B * k, B * (k + 1)))
        topics[k, wordIDs] = 1.0

    # Make vertical bars
    for k in range(K / 2):
        wordIDs = list()
        for b in range(sqrtV):
            start = b * sqrtV + k * BarWidth
            wordIDs.extend(list(range(start, start + BarWidth)))
        topics[K / 2 + k, wordIDs] = 1.0

    # Add smoothing mass to all entries in "topics"
    #  instead of picking this value out of thin air, instead,
    #  set so 95% of the mass of each topic is on the "on-topic" bar words
    #  if s is the smoothing mass added, and B is num "on topic" words, then
    # fracMassOnTopic = (1 + s) * B / ( (1+s)*B + s*(V-B) ), and we solve for
    # s
    smoothMass = (1 - fracMassOnTopic) / (fracMassOnTopic * V - B) * B
    topics += (2 * smoothMass) * PRNG.rand(K, V)

    # Ensure each row of topics is a probability vector
    for k in range(K):
        topics[k, :] /= np.sum(topics[k, :])

    assert np.sum(topics[0, :B]) > fracMassOnTopic - 0.05
    assert np.sum(topics[1, B:2 * B]) > fracMassOnTopic - 0.05
    assert np.sum(topics[-1, wordIDs]) > fracMassOnTopic - 0.05
    return topics


def Create2DBarsTopicWordParams2(V, K, r=0.5, fracMassOnTopic=0.95,
                                 PRNG=np.random):
    ''' Create parameters of each topics distribution over words

        Args
        ---------
        V : int vocab size
        K : int number of topics
        fracMassOnTopic : fraction of probability mass for "on-topic" words
        PRNG : random number generator (for reproducibility)

        Returns
        ---------
        topics : K x V matrix, real positive numbers whose rows sum to one
    '''
    topics = np.zeros((K, V))

    B = V // (K // 2 + 1)
    for k in range(K // 2):
        wordIDs = list(range(B * k, B * (k + 1)))
        topics[2 * k, wordIDs] = np.linspace(1.0, r, B)
        wordIDs = list(range(B // 2 + B * k, B // 2 + B * (k + 1)))
        topics[2 * k + 1, wordIDs] = np.linspace(1.0, r, B)

    topics = smoothAndNormalizeTopics(topics, fracMassOnTopic, PRNG)
    return topics


def smoothAndNormalizeTopics(topics, fracMassOnTopic=0.95, PRNG=np.random):
    ''' Produce topic-word parameters that are proper probabilities with no zeros

        Args
        --------
        topics : 2D array, size K x V
                 each row has two types of entries, "on-topic" and "off-topic"
                 on-topic entries have value > 0, off-topic have value = 0

        Returns
        --------
        topics : 2D array, size K x V
                 each row sums to one, has no non-zero entries
    '''
    for k in range(topics.shape[0]):
        onTopicMass = np.sum(topics[k])
        smoothMass = (1 - fracMassOnTopic) / fracMassOnTopic * onTopicMass
        offTopicWords = topics[k] == 0
        offProbs = PRNG.rand(np.sum(offTopicWords))
        offProbs /= offProbs.sum()
        topics[k, offTopicWords] = smoothMass * offProbs
    return topics / topics.sum(axis=1)[:, np.newaxis]
