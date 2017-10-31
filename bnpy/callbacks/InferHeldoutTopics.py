from builtins import *
import argparse
import time
import os
import numpy as np
import scipy.io
import sklearn.metrics
import bnpy
import glob
import warnings
import logging

from scipy.special import digamma
from scipy.misc import logsumexp
from bnpy.allocmodel.topics.LocalStepSingleDoc import calcLocalParams_SingleDoc
from bnpy.ioutil.ModelReader import \
    getPrefixForLapQuery, loadTopicModel, loadModelForLap
from bnpy.ioutil.DataReader import \
    loadDataFromSavedTask, loadLPKwargsFromDisk, loadDataKwargsFromDisk
from bnpy.ioutil.DataReader import str2numorstr

VERSION = 0.1

def evalTopicModelOnTestDataFromTaskpath(
        taskpath='',
        queryLap=0,
        nLap=0,
        elapsedTime=None,
        seed=42,
        dataSplitName='test',
        fracHeldout=0.2,
        printFunc=None,
        **kwargs):
    ''' Evaluate trained topic model saved in specified task on test data
    '''
    stime = time.time()

    LPkwargs = dict(
        nnzPerRowLP=0,
        nCoordAscentItersLP=100,
        convThrLP=0.01,
        restartLP=0,
        initDocTopicCountLP='setDocProbsToEGlobalProbs')
    for key in kwargs:
        if key in LPkwargs and kwargs[key] is not None:
            LPkwargs[key] = str2val(kwargs[key])
    # Force to be 0, which gives better performance
    # (due to mismatch in objectives)
    if 'restartLP' in LPkwargs:
        LPkwargs['restartLP'] = 0
    # Force to be 0, so we are fair at test time
    if 'nnzPerRowLP' in LPkwargs:
        LPkwargs['nnzPerRowLP'] = 0

    # Load test dataset
    Data = loadDataFromSavedTask(taskpath, dataSplitName=dataSplitName)

    # Check if info is stored in topic-model form
    topicFileList = glob.glob(os.path.join(taskpath, 'Lap*Topic*'))
    if len(topicFileList) > 0:
        topics, probs, alpha = loadTopicModel(
            taskpath, queryLap=queryLap,
            returnTPA=1, normalizeTopics=1, normalizeProbs=1)
        K = probs.size
    else:
        hmodel, foundLap = loadModelForLap(taskpath, queryLap)
        if hasattr(Data, 'word_count'):
            # Convert to topics 2D array (K x V)
            topics = hmodel.obsModel.getTopics()
            probs = hmodel.allocModel.get_active_comp_probs()
        else:
            hmodel.obsModel.setEstParamsFromPost(hmodel.obsModel.Post)
            hmodel.obsModel.inferType = "EM" # Point estimate!

        assert np.allclose(foundLap, queryLap)
        if hasattr(hmodel.allocModel, 'alpha'):
            alpha = hmodel.allocModel.alpha
        else:
            try:
                DataKwargs = loadDataKwargsFromDisk(taskpath)
                alpha = float(DataKwargs['alpha'])
            except Exception:
                alpha = 0.5
        K = hmodel.allocModel.K
    # Prepare debugging statements
    if printFunc:
        startmsg = "Heldout Metrics at lap %.3f" % (queryLap)
        filler = '=' * (80 - len(startmsg))
        printFunc(startmsg + ' ' + filler)
        if hasattr(Data, 'word_count'):
            nAtom = Data.word_count.sum()
        else:
            nAtom = Data.nObs
        msg = "%s heldout data. %d documents. %d total atoms." % (
            Data.name, Data.nDoc, nAtom)
        printFunc(msg)
        printFunc("Using trained model from lap %7.3f with %d topics" % (
            queryLap, K))
        printFunc("Using alpha=%.3f for heldout inference." % (alpha))
        printFunc("Local step params:")
        for key in ['nCoordAscentItersLP', 'convThrLP', 'restartLP']:
            printFunc("    %s: %s" % (key, str(LPkwargs[key])))
        msg = "Splitting each doc" + \
            " into %3.0f%% train and %3.0f%% test, with seed %d" % (
            100*(1-fracHeldout), 100*fracHeldout, seed)
        printFunc(msg)

    # Preallocate storage for metrics
    KactivePerDoc = np.zeros(Data.nDoc)
    logpTokensPerDoc = np.zeros(Data.nDoc)
    nTokensPerDoc = np.zeros(Data.nDoc, dtype=np.int32)
    if hasattr(Data, 'word_count'):
        aucPerDoc = np.zeros(Data.nDoc)
        RprecisionPerDoc = np.zeros(Data.nDoc)
    for d in range(Data.nDoc):
        Data_d = Data.select_subset_by_mask([d], doTrackFullSize=0)
        if hasattr(Data, 'word_count'):
            Info_d = calcPredLikForDoc(
                Data_d, topics, probs, alpha,
                fracHeldout=fracHeldout,
                seed=seed + d,
                LPkwargs=LPkwargs)
            logpTokensPerDoc[d] = Info_d['sumlogProbTokens']
            nTokensPerDoc[d] = Info_d['nHeldoutToken']
            aucPerDoc[d] = Info_d['auc']
            RprecisionPerDoc[d] = Info_d['R_precision']
            KactivePerDoc[d] = np.sum(Info_d['DocTopicCount'] >= 1.0)
            avgAUCscore = np.mean(aucPerDoc[:d+1])
            avgRscore = np.mean(RprecisionPerDoc[:d+1])
            scoreMsg = "avgLik %.4f avgAUC %.4f avgRPrec %.4f medianKact %d" % (
                np.sum(logpTokensPerDoc[:d+1]) / np.sum(nTokensPerDoc[:d+1]),
                avgAUCscore, avgRscore, np.median(KactivePerDoc[:d+1]))
            SVars = dict(
                avgRPrecScore=avgRscore,
                avgAUCScore=avgAUCscore,
                avgAUCScorePerDoc=aucPerDoc,
                avgRPrecScorePerDoc=RprecisionPerDoc)
        else:
            Info_d = calcPredLikForDocFromHModel(
                Data_d, hmodel,
                alpha=alpha,
                fracHeldout=fracHeldout,
                seed=seed + d,
                LPkwargs=LPkwargs)
            logpTokensPerDoc[d] = Info_d['sumlogProbTokens']
            nTokensPerDoc[d] = Info_d['nHeldoutToken']
            scoreMsg = "avgLik %.4f" % (
                np.sum(logpTokensPerDoc[:d+1]) / np.sum(nTokensPerDoc[:d+1]),
                )
            SVars = dict()

        if d == 0 or (d+1) % 25 == 0 or d == Data.nDoc - 1:
            if printFunc:
                etime = time.time() - stime
                msg = "%5d/%d after %8.1f sec " % (d+1, Data.nDoc, etime)
                printFunc(msg + scoreMsg)
    # Aggregate results
    meanlogpTokensPerDoc = np.sum(logpTokensPerDoc) / np.sum(nTokensPerDoc)
    '''
    # Compute heldout Lscore
    if not hasattr(Data, 'word_count'):
        if hasattr(hmodel.allocModel, 'gamma'):
            gamma = hmodel.allocModel.gamma
        else:
            gamma = hmodel.allocModel.gamma0
        aParams = dict(gamma=gamma, alpha=alpha)
        oParams = hmodel.obsModel.get_prior_dict()
        del oParams['inferType']

        # Create DP mixture model from current hmodel
        DPmodel = bnpy.HModel.CreateEntireModel('VB', 'DPMixtureModel',
            hmodel.getObsModelName(),
            aParams, oParams,
            Data)
        DPmodel.set_global_params(hmodel=hmodel)
        LP = DPmodel.calc_local_params(Data, **LPkwargs)
        SS = DPmodel.get_global_suff_stats(Data, LP, doPrecompEntropy=1)
        dpLscore = DPmodel.calc_evidence(SS=SS)

        # Create HDP topic model from current hmodel
        HDPmodel = bnpy.HModel.CreateEntireModel('VB', 'HDPTopicModel',
            hmodel.getObsModelName(),
            aParams, oParams,
            Data)
        HDPmodel.set_global_params(hmodel=hmodel)
        LP = HDPmodel.calc_local_params(Data, **LPkwargs)
        SS = HDPmodel.get_global_suff_stats(Data, LP, doPrecompEntropy=1)
        hdpLscore = HDPmodel.calc_evidence(SS=SS)

        SVars['dpLscore'] = dpLscore
        SVars['hdpLscore'] = hdpLscore
        printFunc("~~~ dpL=%.6e\n~~~hdpL=%.6e" % (dpLscore, hdpLscore))
    '''
    # Prepare to save results.
    if dataSplitName.count('test'):
        outfileprefix = 'predlik-'
    else:
        outfileprefix = dataSplitName + '-predlik-'
    prefix, lap = getPrefixForLapQuery(taskpath, queryLap)
    outmatfile = os.path.join(taskpath, prefix + "Heldout_%s.mat"
        % (dataSplitName))
    # Collect all quantities to save into giant dict.
    SaveVars = dict(
        version=VERSION,
        outmatfile=outmatfile,
        fracHeldout=fracHeldout,
        predLLPerDoc=logpTokensPerDoc,
        avgPredLL=np.sum(logpTokensPerDoc) / np.sum(nTokensPerDoc),
        K=K,
        KactivePerDoc=KactivePerDoc,
        nTokensPerDoc=nTokensPerDoc,
        **LPkwargs)
    SaveVars.update(SVars)
    scipy.io.savemat(outmatfile, SaveVars, oned_as='row')
    SVars['avgLikScore'] = SaveVars['avgPredLL']
    SVars['lapTrain'] = queryLap
    SVars['K'] = K
    for p in [10, 50, 90]:
        SVars['KactivePercentile%02d' % (p)] = np.percentile(KactivePerDoc, p)


    # Record total time spent doing current work
    timeSpent = time.time() - stime
    if elapsedTime is not None:
        SVars['timeTrainAndEval'] = elapsedTime + timeSpent
    # Load previous time spent non training from disk
    timeSpentFilepaths = glob.glob(os.path.join(taskpath, '*-timeEvalOnly.txt'))
    totalTimeSpent = timeSpent
    splitTimeSpent = timeSpent
    for timeSpentFilepath in timeSpentFilepaths:
        with open(timeSpentFilepath,'r') as f:
            for line in f.readlines():
                pass
            prevTime = float(line.strip())
        cond1 = dataSplitName.count('valid')
        cond2 = timeSpentFilepath.count('valid')
        if cond1 and cond2:
            splitTimeSpent += prevTime
        elif (not cond1) and (not cond2):
            splitTimeSpent += prevTime
        totalTimeSpent += prevTime
    SVars['timeEvalOnly'] = splitTimeSpent
    # Mark total time spent purely on training
    if elapsedTime is not None:
        SVars['timeTrain'] = SVars['timeTrainAndEval'] - totalTimeSpent
    for key in SVars:
        if key.endswith('PerDoc'):
            continue
        outtxtfile = os.path.join(taskpath, outfileprefix + '%s.txt' % (key))
        with open(outtxtfile, 'a') as f:
            f.write("%.6e\n" % (SVars[key]))
    if printFunc:
        printFunc("DONE with heldout inference at lap %.3f" % queryLap)
        printFunc("Wrote per-doc results in MAT file:" +
            outmatfile.split(os.path.sep)[-1])
        printFunc("      Aggregate results in txt files: %s__.txt"
            % (outfileprefix))

    # Write the summary message
    if printFunc:
        etime = time.time() - stime
        curLapStr = '%7.3f' % (queryLap)
        nLapStr = '%d' % (nLap)
        logmsg = '  %s/%s %s metrics   | K %4d | %s'
        logmsg = logmsg % (curLapStr, nLapStr, '%5s' % (dataSplitName[:5]), K, scoreMsg)
        printFunc(logmsg, 'info')

    return SaveVars


def createTrainTestSplitOfVocab(
        seen_wids,
        seen_wcts,
        vocab_size,
        fracHeldout=0.2,
        ratioSeenToUnseenInHoldout=1/9.,
        MINSIZE=10,
        seed=42):
    ''' Create train/test split of the vocab words

    Returns
    -------
    Info : dict with fields
    * tr_seen_wids
    * tr_unsn_wids
    * ho_seen_wids
    * ho_unsn_wids
    * ratio

    Example
    -------
    >>> seen_wids = np.arange(100)
    >>> swc = np.ones(100)
    >>> Info = createTrainTestSplitOfVocab(seen_wids, swc, 1000, 0.2, 1.0/10.0)
    >>> print Info['ratio']
    0.1
    >>> I213 = createTrainTestSplitOfVocab(seen_wids, 213, 0.2, 1.0/10.0)
    >>> print I213['ratio']
    0.1
    >>> print len(I213['tr_seen_wids'])
    80
    >>> print len(I213['ho_seen_wids'])
    11
    >>> print len(I213['ho_unsn_wids'])
    110
    >>> # Here's an example that fails
    >>> I111 = createTrainTestSplitOfVocab(seen_wids, 111, 0.2, 1.0/10.0)
    raises ValueError
    '''
    seen_wids = np.asarray(seen_wids, dtype=np.int32)
    # Split seen words into train and heldout
    # Enforcing the desired fraction as much as possible
    # while guaranteeing minimum size
    n_ho_seen = int(np.ceil(fracHeldout * len(seen_wids)))
    if len(seen_wids) < 2 * MINSIZE:
        raise ValueError(
            "Cannot create training and test set with " +
            "at least MINSIZE=%d seen (present) words" % (MINSIZE))
    elif n_ho_seen < MINSIZE:
        n_ho_seen = MINSIZE
    n_tr_seen = len(seen_wids) - n_ho_seen
    assert n_tr_seen >= MINSIZE
    assert n_ho_seen >= MINSIZE

    # Now, divide the un-seen words similarly
    n_ttl_unsn = vocab_size - len(seen_wids)
    n_ho_unsn = int(np.ceil(n_ho_seen / ratioSeenToUnseenInHoldout))
    while n_ho_unsn > n_ttl_unsn and n_ho_seen > MINSIZE:
        # Try to shrink heldout set
        n_ho_seen -= 1
        n_ho_unsn = int(np.ceil(n_ho_seen / ratioSeenToUnseenInHoldout))

    if n_ho_seen < MINSIZE:
        raise ValueError(
            "Cannot create test set with " +
            "at least MINSIZE=%d seen/present words" % (MINSIZE))
    if n_ho_unsn > n_ttl_unsn:
        raise ValueError(
            "Cannot create heldout set with desired ratio of unseen words")
    assert n_ho_unsn >= MINSIZE
    ratio = n_ho_seen / float(n_ho_unsn)

    # Now actually do the shuffling of vocab ids
    PRNG = np.random.RandomState(seed)
    shuffled_inds = PRNG.permutation(len(seen_wids))
    tr_seen_wids = seen_wids[shuffled_inds[:n_tr_seen]].copy()
    tr_seen_wcts = seen_wcts[shuffled_inds[:n_tr_seen]].copy()
    ho_seen_wids = seen_wids[
        shuffled_inds[n_tr_seen:n_tr_seen+n_ho_seen]].copy()
    ho_seen_wcts = seen_wcts[
        shuffled_inds[n_tr_seen:n_tr_seen+n_ho_seen]].copy()
    assert len(ho_seen_wids) == n_ho_seen
    assert len(tr_seen_wids) == n_tr_seen

    unsn_wids = np.setdiff1d(np.arange(vocab_size), seen_wids)
    PRNG.shuffle(unsn_wids)
    ho_unsn_wids = unsn_wids[:n_ho_unsn].copy()
    tr_unsn_wids = unsn_wids[n_ho_unsn:].copy()
    assert len(ho_unsn_wids) == n_ho_unsn

    Info = dict(
        ratio=ratio,
        tr_seen_wcts=tr_seen_wcts,
        ho_seen_wcts=ho_seen_wcts,
        tr_seen_wids=tr_seen_wids,
        ho_seen_wids=ho_seen_wids,
        tr_unsn_wids=tr_unsn_wids,
        ho_unsn_wids=ho_unsn_wids,)

    ho_all_wids = np.hstack([ho_seen_wids, ho_unsn_wids])
    tr_all_wids = np.hstack([tr_seen_wids, tr_unsn_wids])
    n_all = len(ho_all_wids) + len(tr_all_wids)
    if n_all < vocab_size:
        xtra_seen_wids = seen_wids[
            shuffled_inds[n_tr_seen+n_ho_seen:]].copy()
        xtra_seen_wcts = seen_wcts[
            shuffled_inds[n_tr_seen+n_ho_seen:]].copy()
        assert vocab_size - n_all == len(xtra_seen_wids)
        Info['xtra_seen_wids'] = xtra_seen_wids
        Info['xtra_seen_wcts'] = xtra_seen_wcts
    return Info

def calcPredLikForDoc(docData, topics, probs, alpha,
                      LPkwargs=dict(),
                      **kwargs):
    ''' Calculate predictive likelihood for single doc under given model.

    Returns
    -------
    '''
    assert docData.nDoc == 1
    Info = createTrainTestSplitOfVocab(
        docData.word_id, docData.word_count, docData.vocab_size,
        **kwargs)
    # # Run local step to get DocTopicCounts
    DocTopicCount_d, moreInfo_d = inferDocTopicCountForDoc(
        Info['tr_seen_wids'], Info['tr_seen_wcts'],
        topics, probs, alpha, **LPkwargs)
    Info.update(moreInfo_d)

    # # Compute point-estimate of topic probs in this doc
    theta_d = DocTopicCount_d + alpha * probs
    Epi_d = theta_d / np.sum(theta_d)

    # # Evaluate likelihood
    Info['DocTopicCount'] = DocTopicCount_d
    ho_wcts = Info['ho_seen_wcts'].copy()
    ho_wids = Info['ho_seen_wids'].copy()
    if 'xtra_seen_wids' in Info:
        ho_wids = np.hstack([ho_wids, Info['xtra_seen_wids']])
        ho_wcts = np.hstack([ho_wcts, Info['xtra_seen_wcts']])

    probPerToken_d = np.dot(topics[:, ho_wids].T, Epi_d)
    logProbPerToken_d = np.log(probPerToken_d)
    Info['sumlogProbTokens'] = np.sum(logProbPerToken_d * ho_wcts)
    Info['nHeldoutToken'] = np.sum(ho_wcts)

    # # Eval retrieval metrics
    ho_all_wids = np.hstack([Info['ho_seen_wids'], Info['ho_unsn_wids']])
    scoresOfHeldoutTypes_d = np.dot(topics[:, ho_all_wids].T, Epi_d)
    trueLabelsOfHeldoutTypes_d = np.zeros(ho_all_wids.size, dtype=np.int32)
    trueLabelsOfHeldoutTypes_d[:len(Info['ho_seen_wids'])] = 1
    assert np.sum(trueLabelsOfHeldoutTypes_d) == len(Info['ho_seen_wids'])
    # AUC metric
    fpr, tpr, thr = sklearn.metrics.roc_curve(
        trueLabelsOfHeldoutTypes_d, scoresOfHeldoutTypes_d)
    auc = sklearn.metrics.auc(fpr, tpr)
    # Top R precision, where R = total num positive instances
    topR = len(Info['ho_seen_wids'])
    topRHeldoutWordTypes = np.argsort(-1 * scoresOfHeldoutTypes_d)[:topR]
    R_precision = sklearn.metrics.precision_score(
        trueLabelsOfHeldoutTypes_d[topRHeldoutWordTypes],
        np.ones(topR))
    Info['auc'] = auc
    Info['R_precision'] = R_precision
    Info['scoresOfHeldoutTypes'] = scoresOfHeldoutTypes_d
    # # That's all folks
    return Info

    """
    # Split document into training and heldout
    # assigning each unique vocab type to one or the other
    nSeen_d = docData.word_id.size
    nUnseen_d = docData.vocab_size - nSeen_d

    # Randomly assign seen words to TRAIN or HELDOUT
    nHeldout = int(np.ceil(fracHeldout * nSeen_d))
    nHeldout = np.maximum(MINSIZE, nHeldout)
    PRNG = np.random.RandomState(int(seed))
    shuffleIDs = PRNG.permutation(nSeen_d)
    heldoutIDs = shuffleIDs[:nHeldout]
    trainIDs = shuffleIDs[nHeldout:]
    if len(heldoutIDs) < MINSIZE:
        raise ValueError('Not enough unique IDs to make good test split')
    if len(trainIDs) < MINSIZE:
        raise ValueError('Not enough unique IDs to make good train split')

    # Randomly assign unseen words to TRAIN or HELDOUT
    unseen_mask_d = np.ones(docData.vocab_size, dtype=np.bool8)
    unseen_mask_d[docData.word_id] = 0
    unseenWordTypes = np.flatnonzero(unseen_mask_d)
    PRNG.shuffle(unseenWordTypes)
    # Pick heldout set size,
    # so that among all heldout types (seen & unseen)
    # the ratio of seen / total equals the desired fracHeldoutSeen
    nUHeldout = int(np.floor(nHeldout / fracHeldoutSeen))
    if nUHeldout > unseenWordTypes.size:
        nUHeldout = unseenWordTypes.size - nHeldout

    fracHeldoutPresent = nHeldout / float(unseenWordTypes.size)
    if fracHeldoutPresent < 0.5 * fracHeldoutSeen:
        warnings.warn(
            "Frac of heldout types thare are PRESENT is %.3f" % (
                fracHeldoutPresent))
    elif fracHeldoutPresent > 1.5 * fracHeldoutSeen:
        warnings.warn(
            "Frac of heldout types thare are PRESENT is %.3f" % (
                fracHeldoutPresent))
    heldoutUWords = unseenWordTypes[:nUHeldout]
    trainUWords = unseenWordTypes[nUHeldout:]

    ho_word_id = docData.word_id[heldoutIDs]
    ho_word_ct = docData.word_count[heldoutIDs]
    tr_word_id = docData.word_id[trainIDs]
    tr_word_ct = docData.word_count[trainIDs]
    # Run local step to get DocTopicCounts
    DocTopicCount_d, Info = inferDocTopicCountForDoc(
        tr_word_id, tr_word_ct, topics, probs, alpha, **LPkwargs)
    # # Compute expected topic probs in this doc
    theta_d = DocTopicCount_d + alpha * probs
    Epi_d = theta_d / np.sum(theta_d)
    # # Evaluate log prob per token metric
    probPerToken_d = np.dot(topics[:, ho_word_id].T, Epi_d)
    logProbPerToken_d = np.log(probPerToken_d)
    sumlogProbTokens_d = np.sum(logProbPerToken_d * ho_word_ct)
    nHeldoutToken_d = np.sum(ho_word_ct)

    # # Evaluate retrieval metrics
    heldoutSWords = ho_word_id
    heldoutWords = np.hstack([heldoutSWords, heldoutUWords])
    scoresOfHeldoutTypes_d = np.dot(topics[:, heldoutWords].T, Epi_d)
    trueLabelsOfHeldoutTypes_d = np.zeros(heldoutWords.size, dtype=np.int32)
    trueLabelsOfHeldoutTypes_d[:heldoutSWords.size] = 1
    assert np.sum(trueLabelsOfHeldoutTypes_d) == ho_word_id.size
    # AUC metric
    fpr, tpr, thr = sklearn.metrics.roc_curve(
        trueLabelsOfHeldoutTypes_d, scoresOfHeldoutTypes_d)
    auc = sklearn.metrics.auc(fpr, tpr)
    # Top R precision, where R = total num positive instances
    topR = ho_word_id.size
    topRHeldoutWordTypes = np.argsort(-1 * scoresOfHeldoutTypes_d)[:topR]
    R_precision = sklearn.metrics.precision_score(
        trueLabelsOfHeldoutTypes_d[topRHeldoutWordTypes],
        np.ones(topR))
    '''
    # unseen_mask_d : 1D array, size vocab_size
    #   entry is 0 if word is seen in training half
    #   entry is 1 if word is unseen
    unseen_mask_d = np.ones(docData.vocab_size, dtype=np.bool8)
    unseen_mask_d[tr_word_id] = 0
    probOfUnseenTypes_d = np.dot(topics[:, unseen_mask_d].T, Epi_d)
    unseen_mask_d = np.asarray(unseen_mask_d, dtype=np.int32)
    unseen_mask_d[ho_word_id] = 2
    trueLabelsOfUnseenTypes_d = unseen_mask_d[unseen_mask_d > 0]
    trueLabelsOfUnseenTypes_d -= 1
    assert np.sum(trueLabelsOfUnseenTypes_d) == ho_word_id.size
    fpr, tpr, thr = sklearn.metrics.roc_curve(
        trueLabelsOfUnseenTypes_d, probOfUnseenTypes_d)
    auc = sklearn.metrics.auc(fpr, tpr)
    # top R precision, where R = total num positive instances
    topR = ho_word_id.size
    topRUnseenTypeIDs = np.argsort(-1 * probOfUnseenTypes_d)[:topR]
    R_precision = sklearn.metrics.precision_score(
        trueLabelsOfUnseenTypes_d[topRUnseenTypeIDs],
        np.ones(topR))
    # Useful debugging
    # >>> unseenTypeIDs = np.flatnonzero(unseen_mask_d)
    # >>> trainIm = np.zeros(900); trainIm[tr_word_id] = 1.0
    # >>> testIm = np.zeros(900); testIm[ho_word_id] = 1.0
    # >>> predictIm = np.zeros(900);
    # >>> predictIm[unseenTypeIDs[topRUnseenTypeIDs]] = 1;
    # >>> bnpy.viz.BarsViz.showTopicsAsSquareImages( np.vstack([trainIm, testIm, predictIm]) )
    '''
    Info['auc'] = auc
    Info['R_precision'] = R_precision
    Info['ho_word_ct'] = ho_word_ct
    Info['tr_word_ct'] = tr_word_ct
    Info['DocTopicCount'] = DocTopicCount_d
    Info['nHeldoutToken'] = nHeldoutToken_d
    Info['sumlogProbTokens'] = sumlogProbTokens_d
    return Info
    """


def calcPredLikForDocFromHModel(
        docData, hmodel,
        fracHeldout=0.2,
        seed=42,
        MINSIZE=10,
        LPkwargs=dict(),
        alpha=None,
        **kwargs):
    ''' Calculate predictive likelihood for single doc under given model.

    Returns
    -------
    '''
    Info = dict()
    assert docData.nDoc == 1

    # Split document into training and heldout
    # assigning each unique vocab type to one or the other
    if hasattr(docData, 'word_id'):
        N = docData.word_id.size
    else:
        N = docData.nObs
    nHeldout = int(np.ceil(fracHeldout * N))
    nHeldout = np.maximum(MINSIZE, nHeldout)
    PRNG = np.random.RandomState(int(seed))
    shuffleIDs = PRNG.permutation(N)
    heldoutIDs = shuffleIDs[:nHeldout]
    trainIDs = shuffleIDs[nHeldout:]
    if len(heldoutIDs) < MINSIZE:
        raise ValueError('Not enough unique IDs to make good test split')
    if len(trainIDs) < MINSIZE:
        raise ValueError('Not enough unique IDs to make good train split')

    hoData = docData.select_subset_by_mask(atomMask=heldoutIDs)
    trData = docData.select_subset_by_mask(atomMask=trainIDs)

    Epi_global = hmodel.allocModel.get_active_comp_probs()
    LP = hmodel.obsModel.calc_local_params(hoData)
    hoLik_d = LP['E_log_soft_ev']
    hoLik_d += np.log(Epi_global)[np.newaxis,:]
    logProbPerToken_d = logsumexp(hoLik_d, axis=1)
    Info['sumlogProbTokens'] = np.sum(logProbPerToken_d)
    Info['nHeldoutToken'] = len(heldoutIDs) * hoData.dim
    return Info
    '''
    # Run local step to get DocTopicCounts
    DocTopicCount_d, Info = inferDocTopicCountForDocFromHModel(
        trData, hmodel, **LPkwargs)
    probs = hmodel.allocModel.get_active_comp_probs()
    # Compute expected topic probs in this doc
    theta_d = DocTopicCount_d + alpha * probs
    E_log_pi_d = digamma(theta_d) - digamma(np.sum(theta_d))
    # Evaluate log prob per token metric
    LP = hmodel.obsModel.calc_local_params(hoData)
    logProbArr_d = LP['E_log_soft_ev']
    logProbArr_d += E_log_pi_d[np.newaxis, :]
    logProbPerToken_d = logsumexp(logProbArr_d, axis=1)
    # Pack up and ship
    Info['DocTopicCount'] = DocTopicCount_d
    Info['nHeldoutToken'] = len(heldoutIDs)
    Info['sumlogProbTokens'] = np.sum(logProbPerToken_d)
    return Info
    '''

def inferDocTopicCountForDoc(
        word_id, word_ct, topics, probs, alpha,
        **LPkwargs):
    K = probs.size
    K2, W = topics.shape
    assert K == K2
    # topics : 2D array, vocab_size x K
    # Each col is non-negative and sums to one.
    topics = topics.T.copy()
    assert np.allclose(np.sum(topics, axis=0), 1.0)
    # Lik_d : 2D array, size N x K
    # Each row is non-negative
    Lik_d = np.asarray(topics[word_id, :].copy(), dtype=np.float64)
    # alphaEbeta : 1D array, size K
    alphaEbeta = np.asarray(alpha * probs, dtype=np.float64)
    DocTopicCount_d, _, _, Info = calcLocalParams_SingleDoc(
        word_ct, Lik_d, alphaEbeta,
        alphaEbetaRem=None,
        **LPkwargs)
    assert np.allclose(DocTopicCount_d.sum(), word_ct.sum())
    return DocTopicCount_d, Info

def inferDocTopicCountForDocFromHModel(
        docData, hmodel, alpha=0.5, **LPkwargs):
    # Lik_d : 2D array, size N x K
    # Each row is non-negative
    LP = hmodel.obsModel.calc_local_params(docData)
    Lik_d = LP['E_log_soft_ev']
    Lik_d -= Lik_d.max(axis=1)[:,np.newaxis]
    np.exp(Lik_d, out=Lik_d)

    # alphaEbeta : 1D array, size K
    alphaEbeta = alpha * hmodel.allocModel.get_active_comp_probs()
    DocTopicCount_d, _, _, Info = calcLocalParams_SingleDoc(
        1.0, Lik_d, alphaEbeta,
        alphaEbetaRem=None,
        **LPkwargs)
    assert np.allclose(DocTopicCount_d.sum(), Lik_d.shape[0])
    return DocTopicCount_d, Info

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('taskpath', type=str)
    parser.add_argument('--queryLap', type=float, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--printStdOut', type=int, default=1)
    parser.add_argument('--printLevel', type=int, default=logging.INFO)
    parser.add_argument('--elapsedTime', type=float, default=None)
    parser.add_argument('--dataSplitName', type=str, default="test")
    #parser.add_argument('--restartLP', type=int, default=None)
    #parser.add_argument('--fracHeldout', type=float, default=0.2)
    args = parser.parse_args()

    if args.printStdOut:
        def printFunc(x, level='debug', **kwargs):
            if level == 'debug':
                level = logging.DEBUG
            elif level == 'info':
                level = logging.INFO
            if level >= args.printLevel:
                print(x)
        args.__dict__['printFunc'] = printFunc
    evalTopicModelOnTestDataFromTaskpath(**args.__dict__)
