'''
BinaryReviews.py

Movie review snippets classified as positive or negative.
'''
import numpy as np
from bnpy.data import BagOfWordsData
import string, os, sys, random
import io, glob
from scipy.sparse import csr_matrix

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


RANDOMSEED = 54321

datasetdir = os.path.sep.join(
    os.path.abspath(__file__).split(os.path.sep)[:-1])

if not os.path.isdir(datasetdir):
    raise ValueError('CANNOT FIND BINARY REVIEW DATASET DIRECTORY:\n' + datasetdir)

posfilepath = os.path.join(datasetdir, 'rawData', 'rt-polaritydata', 'rt-polarity.pos')
negfilepath = os.path.join(datasetdir, 'rawData', 'rt-polaritydata', 'rt-polarity.neg')

posfilepath_2 = os.path.join(datasetdir, 'rawData', 'rt-polaritydata_v2', 'pos')
negfilepath_2 = os.path.join(datasetdir, 'rawData', 'rt-polaritydata_v2', 'neg')

regressfilepath = os.path.join(datasetdir, 'rawData', 'scaledata')

stopfilepath = os.path.join(datasetdir, 'rawData', 'stop.txt')
if not os.path.isfile(posfilepath):
    raise ValueError('CANNOT FIND BINARY REVIEW DATASET MAT FILE:\n' + matfilepath)


def fprint(val):
    print val
    sys.stdout.flush()

def get_data_info():
    s = 'Binary movie review dataset.'
    return s

def get_data_arrays(version=1, numvocab=4310, use_stop=False, min_thresh=0, max_thresh=0, **kwargs):
    tf = dict()
    allwords = []
    labels = []

    if version == 1:
        processFile(posfilepath, 1, tf, allwords, labels)
        processFile(negfilepath, 0, tf, allwords, labels)
    elif version == 2:
        processFiles(posfilepath_2, 1, tf, allwords, labels)
        processFiles(negfilepath_2, 0, tf, allwords, labels)
    elif version == 'regress':
        processRegressionFiles(regressfilepath, 'rating', tf, allwords, labels)
    else:
        raise ValueError('Unknown dataset version!')

    fprint('Finished file processing')

    random.seed(RANDOMSEED)
    lst = zip(allwords, labels)
    random.shuffle(lst)
    allwords = [l[0] for l in lst]
    labels = [l[1] for l in lst]

    vocabList, invvocab = createVocab(tf, len(labels), numvocab, use_stop, min_thresh, max_thresh)
    vocab_size = len(vocabList)
    word_id = []
    word_count = []
    doc_range = [0]
    Y = []

    fprint('Finished vocab construction')
    
    for i, doc in enumerate(allwords):
        numwords = 0
        for word, count in doc.iteritems():
            if word in invvocab:
                word_id.append(invvocab[word])
                word_count.append(count)
                numwords += 1
        if numwords > 0:
            doc_range.append(doc_range[-1] + numwords)
            Y.append(labels[i])

    fprint('Finished data construction')
    return word_id, word_count, doc_range, vocab_size, vocabList, Y

def get_data_from_arrays(word_id, word_count, doc_range, vocab_size, vocabList, Y):
    Data = BagOfWordsData(word_id=np.array(word_id), word_count=np.array(word_count),
        doc_range=np.array(doc_range), vocab_size=vocab_size, vocabList=vocabList, Y=Y)

    Data.name = 'BinaryReviews'
    Data.summary = get_data_info()
    return Data


def get_data(version=1, numvocab=4310, use_stop=False, min_thresh=0, max_thresh=0, **kwargs):
    word_id, word_count, doc_range, vocab_size, vocabList, Y = get_data_arrays(version, numvocab, use_stop, min_thresh=min_thresh, max_thresh=max_thresh, **kwargs)
    Data = get_data_from_arrays(word_id, word_count, doc_range, vocab_size, vocabList, Y)
    return Data

def split_data_arrays(word_id, word_count, doc_range, vocab_size, vocabList, Y, split=0.7):
    ntrain = int(split * len(Y))
    splitind = doc_range[ntrain]

    train_doc_range = doc_range[0:ntrain+1]
    test_doc_range = [r - splitind for r in doc_range[ntrain:]]

    train_word_id, test_word_id = word_id[:splitind], word_id[splitind:]
    train_word_count, test_word_count = word_count[:splitind], word_count[splitind:]
    train_Y, test_Y = Y[:ntrain], Y[ntrain:]

    return (train_word_id, train_word_count, train_doc_range, vocab_size, vocabList, train_Y), \
        (test_word_id, test_word_count, test_doc_range, vocab_size, vocabList, test_Y)

def get_train_test_data(version=1, numvocab=4310, split=0.7, use_stop=False, min_thresh=0, max_thresh=0, **kwargs):
    word_id, word_count, doc_range, vocab_size, vocabList, Y = get_data_arrays(version, numvocab, use_stop=use_stop, min_thresh=min_thresh, max_thresh=max_thresh, **kwargs)
    
    train, test = split_data_arrays(word_id, word_count, doc_range, vocab_size, vocabList, Y, split=split)

    train = get_data_from_arrays(*train)
    test = get_data_from_arrays(*test)

    return train, test

def get_split_data(version=1, numvocab=4310, split=[0.8, 0.1, 0.1], use_stop=False, min_thresh=0, max_thresh=0, **kwargs):
    rem = get_data_arrays(version, numvocab, use_stop=use_stop, min_thresh=min_thresh, max_thresh=max_thresh, **kwargs)
    
    total = 1.0
    datasets = []
    for s in split:
        cursplit = s / total
        dataset, rem = split_data_arrays(*rem, split=cursplit)
        total -= s
        datasets.append(get_data_from_arrays(*dataset))
    return tuple(datasets)

def get_sklearn_data(version=1):
    if version != 1:
        raise
        
    corpus = []
    with open(posfilepath) as f:
        examples = [line.translate(None, string.punctuation) for line in f]
        corpus.extend(examples)
        poslabels = np.ones(len(examples))
    with open(negfilepath) as f:
        examples = [line.translate(None, string.punctuation) for line in f]
        corpus.extend(examples)
        neglabels = np.zeros(len(examples))
    Y = np.hstack([poslabels, neglabels])

    vectorizer = CountVectorizer(min_df=1)
    X = vectorizer.fit_transform(corpus)
    return X, Y

def processFiles(direc, label, tf, allwords, labels, byline=False):
    files = [os.path.join(direc, f) for f in os.listdir(direc) if os.path.isfile(os.path.join(direc, f))]
    for f in files:
        processFile(f, label, tf, allwords, labels, byline=byline)

def processRegressionFiles(direc, label, tf, allwords, labels, byline=True):
    subjfiles = glob.glob(os.path.join(direc, '*', 'subj*'))
    if type(label) is not str:
        label = 'rating'

    for sfile in subjfiles:
        lfilename = sfile.replace('subj.', label + '.')
        processFile(sfile, 0, tf, allwords, [], byline=byline)

        with open(lfilename) as lfile: 
            for line in lfile:
                try:
                    labels.append(float(line.strip()))
                except:
                    print 'Skipping label:', line
                    pass

def processFile(file, label, tf, allwords, labels, byline=True):
    with open(file) as f:
        docs = []
        if byline:
            for line in f:
                if len(line) > 0:
                    docs.append(line)
        else:
            doc = ''
            for line in f:
                doc += ' ' + line
            docs = [doc]

        for docnum, doc in enumerate(docs):
            words = doc.translate(None, string.punctuation).split()
            labels.append(label)

            counts = {}
            for word in words:
                if word not in counts:
                    counts[word] = 1
                else:
                    counts[word] += 1

            allwords.append(counts)
            for word, count in counts.iteritems():
                if word not in tf:
                    tf[word] = [count]
                else:
                    tf[word].append(count)

def createVocab(tf, D, numvocab=4310, use_stop=False, min_thresh=0, max_thresh=0):
    if min_thresh < 1.0 and min_thresh > 0.0:
        min_thresh = int(D * min_thresh)
    if max_thresh < 1.0 and max_thresh > 0.0:
        max_thresh = int(D * max_thresh)
    max_thresh = max_thresh if max_thresh >= 1 else 9999999999   

    stop = set()
    if use_stop:
        with open(stopfilepath) as f:
            for line in f:
                stop.add(line.strip())


    vocab = []
    for word, counts in tf.iteritems():
        tfidf = np.sum(np.array(counts)) * np.log(float(D) / len(counts))
        if word not in stop and len(counts) >= min_thresh and len(counts) <= max_thresh:
            vocab.append((word, tfidf))

    vocab.sort(key=lambda v: -v[1])
    vocab = vocab[:numvocab]
    vocab = [v[0] for v in vocab]

    invvocab = dict(zip(vocab, range(len(vocab))))
    return vocab, invvocab

def save_csr(filename,array):
    np.savez(filename, data=array.data ,indices=array.indices,
             indptr=array.indptr, shape=array.shape)

def save_sscape_data(direc, names=['train', 'test', 'valid'], split=[0.8, 0.1, 0.1], version=1, numvocab=4310, use_stop=True, min_thresh=0, max_thresh=0):
    try:
        os.makedirs(direc)
    except:
        pass

    datasets = get_split_data(version=version, split=split, numvocab=numvocab, use_stop=use_stop, min_thresh=min_thresh, max_thresh=max_thresh)
    with open(os.path.join(direc, 'X_colnames.txt'), 'w') as file:
        file.write('\n'.join(datasets[0].vocabList) + '\n')
    with open(os.path.join(direc, 'Y_colnames.txt'), 'w') as file:
        file.write('bow_pang_verision_' + str(version))

    P_start = 0
    for dataset, name in zip(datasets, names):
        Y = dataset.Y.reshape((-1,1))
        np.save(os.path.join(direc, 'Y_' + name + '.npy'), Y)
        
        P = np.arange(Y.shape[0]) + P_start
        np.save(os.path.join(direc, 'P_' + name + '.npy'), P)
        P_start += np.max(P)

        csrmat = dataset.getSparseDocTypeCountMatrix()
        save_csr(os.path.join(direc, 'X_csr_' + name + '.npz'), csrmat)


if __name__ == '__main__':
    train_v1, test_v1 = get_train_test_data(version=1, numvocab=4310, split=0.7, use_stop=True)
    train_v1.to_npz('train_v1.npz')
    test_v1.to_npz('test_v1.npz')

    train_v2, test_v2 = get_train_test_data(version=1, numvocab=10000, split=0.7, use_stop=True)
    train_v2.to_npz('train_v2.npz')
    test_v2.to_npz('test_v2.npz')

    train_regress, test_regress = get_train_test_data(version='regress', numvocab=4310, split=0.7, use_stop=True, min_thresh=5, max_thresh=0.75)
    train_regress.to_npz('train_regress.npz')
    test_regress.to_npz('test_regress.npz')

