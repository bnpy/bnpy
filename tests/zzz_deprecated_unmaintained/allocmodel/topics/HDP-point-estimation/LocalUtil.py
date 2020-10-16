import numpy as np
import copy
from scipy.special import digamma, gammaln

from bnpy.util import NumericUtil

nCoordAscentIters = 20
convThr = 0.001

def calcLocalParams(Data, LP, aModel, 
                          methodLP='scratch',
                          routineLP='simple',
                          **kwargs):
  ''' Calculate all local parameters for provided dataset under a topic model

      Returns
      -------
      LP : dict of local params, with fields
      * DocTopicCount
      * resp
      * model-specific fields for doc-topic probabilities
  ''' 
  kwargs['methodLP'] = methodLP

  ## Prepare the log soft ev matrix
  ## Make sure it is C-contiguous, so that matrix ops are very fast
  Lik = np.asarray(LP['E_log_soft_ev'], order='C') 
  Lik -= Lik.max(axis=1)[:,np.newaxis] 
  NumericUtil.inplaceExp(Lik)
  K = Lik.shape[1]
  hasDocTopicCount = 'DocTopicCount' in LP \
                     and LP['DocTopicCount'].shape == (Data.nDoc, K)
  if methodLP == 'memo' and hasDocTopicCount:
    initDocTopicCount = LP['DocTopicCount']
  else:
    initDocTopicCount = None

  if routineLP == 'simple':
    DocTopicCount, Prior, sumR, AI = calcDocTopicCountForData_Simple(Data, 
                                      aModel, Lik,
                                      initDocTopicCount=initDocTopicCount,
                                      **kwargs)
  elif routineLP == 'fast':
    DocTopicCount, Prior, sumR = calcDocTopicCountForData_Fast(Data, aModel,
                                      Lik,
                                      initDocTopicCount=initDocTopicCount,
                                      **kwargs)
  else:
    raise ValueError('Unrecognized routine ' + routineLP)

  LP['DocTopicCount'] = DocTopicCount
  LP = aModel.updateLPGivenDocTopicCount(LP, DocTopicCount)
  LP = updateLPWithResp(LP, Data, Lik, Prior, sumR)

  if kwargs['restartremovejunkLP'] == 1:
    LP, RInfo = removeJunkTopicsFromAllDocs(aModel, Data, LP, **kwargs)

  if 'lapFrac' in kwargs and 'batchID' in kwargs:
    if hasattr(Data, 'batchID') and Data.batchID == kwargs['batchID']:     
      perc = [0, 5, 10, 50, 90, 95, 100]
      siter = ' '.join(['%4d' % np.percentile(AI['iter'], p) for p in perc])
      sdiff = ['%6.4f' % np.percentile(AI['maxDiff'], p) for p in perc]
      sdiff = ' '.join(sdiff)
      nFail = np.sum(AI['maxDiff'] > kwargs['convThrLP'])
      msg = '%4.2f %3d %4d %s %s' % (kwargs['lapFrac'], Data.batchID,
                                     nFail, siter, sdiff)

      if kwargs['restartremovejunkLP'] == 1:
        msg += " %4d/%4d %4d/%4d" % (RInfo['nDocRestartsAccepted'],
                                     RInfo['nDocRestartsTried'],
                                     RInfo['nRestartsAccepted'],
                                     RInfo['nRestartsTried'])
      elif kwargs['restartremovejunkLP'] == 2:
        msg += " %4d/%4d" % (AI['nRestartsAccepted'],
                             AI['nRestartsTried'])

  LP['Info'] = AI
  return LP


def removeJunkTopicsFromAllDocs(aModel, Data, LP, 
                     maxTryPerDoc=5, minThrResp=0.05, maxThrResp=0.9,
                     maxThrDocTopicCount=25, **kwargs):
  ''' Remove junk topics from each doc, if they exist.
  '''
  origLP = copy.deepcopy(LP)
  kwargs = dict(**kwargs)
  kwargs['nCoordAscentItersLP'] = kwargs['restartNumItersLP']
  from bnpy.allocmodel.admix2.HDPDir import calcELBOSingleDoc
  Lik = LP['E_log_soft_ev'] # already applied exp to this matrix
  nDocRestartsTried = 0
  nDocRestartsAccepted = 0
  nRestartsTried = 0
  nRestartsAccepted = 0
  for d in range(Data.nDoc):
    start = Data.doc_range[d]
    stop = Data.doc_range[d+1]
    resp_d = LP['resp'][start:stop]

    maxResp = resp_d.max(axis=0)
    
    if kwargs['restartCriteriaLP'] == 'maxResp':
      eligibleIDs = np.flatnonzero(np.logical_and(maxResp > minThrResp,
                                                  maxResp < maxThrResp))
      ## Sort smallest to largest, keep the best few
      sortIDs = np.argsort(maxResp[eligibleIDs])
      eligibleIDs = eligibleIDs[sortIDs[:maxTryPerDoc]]
    else:
      Nd = LP['DocTopicCount'][d]
      eligibleIDs = np.flatnonzero(np.logical_and(Nd > minThrResp,
                                                  Nd < maxThrDocTopicCount))
      sortIDs = np.argsort(Nd[eligibleIDs])
      eligibleIDs = eligibleIDs[sortIDs[:maxTryPerDoc]]      

    Nactivetopics = np.sum(LP['DocTopicCount'][d] > minThrResp)
    if eligibleIDs.size < 1 or Nactivetopics < 2:
      ## Skip if we don't have eligible topics to "delete"
      ## or if we cannot delete any more without removing the final topic
      continue


    Lik_d = Lik[start:stop]
    logLik_d = np.log(np.maximum(Lik_d, 1e-100))
    ## Calculate "baseline" ELBO
    curELBO = calcELBOSingleDoc(Data, d, 
                                resp_d=resp_d, logLik_d=logLik_d,
                                theta_d=LP['theta'][d][np.newaxis,:],
                                thetaRem=LP['thetaRem'])
    assert np.isfinite(curELBO)

    if hasattr(Data, 'word_count'):
      wc_d = Data.word_count[start:stop]
    else:
      wc_d = 1.0
    
    nDocRestartsTried += 1
    didAcceptAny = 0
    ## Attempt each move
    for kID in eligibleIDs:
      DocTopicCount_d = LP['DocTopicCount'][d].copy()
      origSize = DocTopicCount_d[kID]
      DocTopicCount_d[kID] = 0
      if np.sum(DocTopicCount_d > minThrResp) < 1:
        ## Never delete the final topic
        break
      nRestartsTried += 1
      Prior_d = aModel.calcLogPrActiveComps_Fast(DocTopicCount_d[np.newaxis,:])
      Prior_d -= Prior_d.max(axis=1)[:, np.newaxis]
      np.exp(Prior_d, out=Prior_d)

      DocTopicCount_d, Prior_d, sumR_d, I = calcDocTopicCountForDoc(d, aModel,
                              DocTopicCount_d, Lik_d,
                              Prior_d[0], np.zeros(stop-start), 
                              wc_d=wc_d,
                              **kwargs)
      DocTopicCount_d = DocTopicCount_d[np.newaxis,:]
      propLP_d = dict()
      propLP_d['DocTopicCount'] = DocTopicCount_d
      propLP_d = aModel.updateLPGivenDocTopicCount(propLP_d, DocTopicCount_d)
      propLP_d = updateSingleDocLPWithResp(propLP_d, Lik_d, Prior_d, sumR_d)
      # Evaluate ELBO and accept if improved!
      propELBO = calcELBOSingleDoc(Data, d, 
                                   resp_d=propLP_d['resp'], logLik_d=logLik_d,
                                   theta_d=propLP_d['theta'],
                                   thetaRem=LP['thetaRem'])
      assert np.isfinite(propELBO)

      if propELBO > curELBO:
        msg = '**** accepted'
        didAccept = 1
        nRestartsAccepted += 1
      else:
        msg = '     rejected'
        didAccept = 0

      '''
      if d < 10:
        print '   %3d %.2f %6.2f %12.5f %12.5f %s %.5f' \
               % (kID, maxResp[kID], origSize, curELBO, propELBO, msg,
                  I['maxDiff'])
      '''
      didAcceptAny += didAccept
      ## If accepted, make necessary changes
      if didAccept:
        curELBO = propELBO
        LP['DocTopicCount'][d] = DocTopicCount_d[0]
        LP['theta'][d] = propLP_d['theta'][0]
        LP['ElogPi'][d] = propLP_d['ElogPi'][0]
        LP['resp'][start:stop] = propLP_d['resp']
    if didAcceptAny > 0:
      nDocRestartsAccepted += 1
  AI = dict(nDocRestartsAccepted=nDocRestartsAccepted,
            nDocRestartsTried=nDocRestartsTried,
            nRestartsTried=nRestartsTried,
            nRestartsAccepted=nRestartsAccepted)
  return LP, AI

def updateLPWithResp(LP, Data, Lik, Prior, sumRespTilde):
  LP['resp'] = Lik.copy()
  for d in range(Data.nDoc):
    start = Data.doc_range[d]
    stop  = Data.doc_range[d+1]
    LP['resp'][start:stop] *= Prior[d]
  LP['resp'] /= sumRespTilde[:, np.newaxis]
  np.maximum(LP['resp'], 1e-300, out=LP['resp'])
  return LP

def updateSingleDocLPWithResp(LP_d, Lik_d, Prior_d, sumR_d):
  resp_d = Lik_d.copy()
  resp_d *= Prior_d
  resp_d /= sumR_d[:, np.newaxis]
  np.maximum(resp_d, 1e-300, out=resp_d)
  LP_d['resp'] = resp_d
  return LP_d

def calcDocTopicCountForData_Simple(Data, aModel, Lik,
                   initDocTopicCount=None,
                   initPrior=None, 
                   **kwargs
                  ):
  ''' Calculate updated doc-topic counts for every document in provided set

      Will loop over all docs, and at each one will run coordinate ascent
      to alternatively update its doc-topic counts and the doc-topic prior.
      Ascent stops after convergence or a maximum number of iterations.
    
      Returns
      ---------
      DocTopicCount : 2D array, size nDoc x K
      DocTopicCount[d,k] is effective number of tokens in doc d assigned to k

      Prior : 2D array, size nDoc x K
      Prior[d,k] = exp( E[log pi_{dk}] )

      sumRespTilde : 1D array, size N = # observed tokens
                     sumRespTilde[n] = normalization for the responsibility          
                     parameters for token n
  '''
  sumRespTilde = np.zeros(Lik.shape[0])

  ## Initialize DocTopicCount and Prior
  if initDocTopicCount is not None:
    DocTopicCount = initDocTopicCount.copy()
    Prior = aModel.calcLogPrActiveComps_Fast(DocTopicCount)
    Prior -= Prior.max(axis=1)[:, np.newaxis]
    np.exp(Prior, out=Prior)
  else:
    DocTopicCount = np.zeros((Data.nDoc, aModel.K))
    if initPrior is None:
      if kwargs['methodLP'] == 'scratch':
        Prior = np.ones((Data.nDoc, aModel.K))
      elif kwargs['methodLP'] == 'prior':
        probs = aModel.get_active_comp_probs().copy()
        Prior = np.tile(probs, (Data.nDoc, 1))
      else:
        Prior = np.ones((Data.nDoc, aModel.K))
    else:
      Prior = initPrior.copy()
  AggInfo = dict()
  AggInfo['maxDiff'] = np.zeros(Data.nDoc)
  AggInfo['iter'] = np.zeros(Data.nDoc, dtype=np.int32)
  for d in range(Data.nDoc):
    start = Data.doc_range[d]
    stop  = Data.doc_range[d+1]
    Lik_d = Lik[start:stop].copy() # Local copy
    if hasattr(Data, 'word_count'):
      wc_d = Data.word_count[start:stop].copy()
    else:
      wc_d = 1.0
    sumR_d = np.zeros(stop-start)

    DocTopicCount[d], Prior[d], sumR_d, Info = calcDocTopicCountForDoc(
                                      d, aModel, 
                                      DocTopicCount[d], Lik_d,
                                      Prior[d], sumR_d, 
                                      wc_d,
                                      **kwargs)
    sumRespTilde[start:stop] = sumR_d

    AggInfo['maxDiff'][d] = Info['maxDiff']
    AggInfo['iter'][d] = Info['iter']
    if 'ELBOtrace' in Info:
      AggInfo['ELBOtrace'] = Info['ELBOtrace']
    if 'nAccept' in Info:
      if 'nRestartsAccepted' not in AggInfo:
        AggInfo['nRestartsAccepted'] = 0
        AggInfo['nRestartsTried'] = 0
      AggInfo['nRestartsAccepted'] += Info['nAccept']
      AggInfo['nRestartsTried'] += Info['nTrial']

  return DocTopicCount, Prior, sumRespTilde, AggInfo


def calcDocTopicCountForDoc(d, aModel,
                            DocTopicCount_d, Lik_d,
                            Prior_d, sumR_d, 
                            wc_d=None,
                            nCoordAscentItersLP=nCoordAscentIters,
                            convThrLP=convThr,
                            **kwargs
                            ):
  '''
     Returns
      ---------
      DocTopicCount : 1D array, size K
                      DocTopicCount[k] is effective number of tokens 
                      assigned to topic k in the current document d

      Prior_d      : 1D array, size K
                     Prior_d[k] : probability of topic k in current doc d

      sumRespTilde : 1D array, size Nd = # observed tokens in current doc d
                     sumRespTilde[n] = normalization for the responsibility          
                     parameters for token n
  '''
  prevDocTopicCount_d = DocTopicCount_d.copy()
  if hasattr(aModel, 'calcLogPrActiveCompsForDoc'):
    aFunc = aModel.calcLogPrActiveCompsForDoc
  else:
    aFunc = aModel
  
  doLogELBO = False
  if 'logELBOLP' in kwargs and kwargs['logELBOLP']:
    if hasattr(aModel, 'alphaEbeta'):
      alphaEbeta = aModel.alphaEbeta
      doLogELBO = True
      ELBOtrace = list()
      
  for iter in range(nCoordAscentItersLP):
    ## Update Prob of Active Topics
    if iter > 0:
      aFunc(DocTopicCount_d, Prior_d) # Prior_d = E[ log pi_dk ]
      Prior_d -= Prior_d.max()
      np.exp(Prior_d, out=Prior_d)    # Prior_d = exp E[ log pi_dk ]
      
    ## Update sumR_d for all tokens in document
    np.dot(Lik_d, Prior_d, out=sumR_d)

    ## Update DocTopicCounts
    np.dot(wc_d / sumR_d, Lik_d, out=DocTopicCount_d)
    DocTopicCount_d *= Prior_d

    if doLogELBO:
      ELBO = calcELBOSingleDoc_Fast(wc_d, DocTopicCount_d,
                                    Prior_d, sumR_d, alphaEbeta)
      ELBOtrace.append(ELBO)

    ## Check for convergence
    maxDiff = np.max(np.abs(DocTopicCount_d - prevDocTopicCount_d))
    if maxDiff < convThrLP:
      break
    prevDocTopicCount_d[:] = DocTopicCount_d

  Info = dict(maxDiff=maxDiff, iter=iter)
  if doLogELBO:
    Info['ELBOtrace'] = np.asarray(ELBOtrace)
  if kwargs['restartremovejunkLP'] == 2:
    DocTopicCount_d, Prior_d, sumR_d, RInfo = removeJunkTopicsFromDoc(
                     wc_d, 
                     DocTopicCount_d, Prior_d, sumR_d, 
                     Lik_d, aModel.alphaEbeta, aFunc, **kwargs)
    Info.update(RInfo)
    if 'ELBOtrace' in Info:
      Info['ELBOtrace'].append(RInfo['finalELBO'])
  return DocTopicCount_d, Prior_d, sumR_d, Info

def removeJunkTopicsFromDoc(wc_d, DocTopicCount_d, Prior_d, sumR_d, 
                     Lik_d, alphaEbeta, aFunc,
                     restartNumTrialsLP=5,
                     restartNumItersLP=2, 
                     MIN_USAGE_THR=0.01, **kwargs):
  ''' Create candidate models that remove junk topics, accept if improved.
  '''
  Info = dict(nTrial=0, nAccept=0)
  usedTopicMask = DocTopicCount_d > MIN_USAGE_THR
  nUsed = np.sum(usedTopicMask)
  if nUsed < 2:
    return DocTopicCount_d, Prior_d, sumR_d, Info
  usedTopics = np.flatnonzero(usedTopicMask)
  smallIDs = np.argsort(DocTopicCount_d[usedTopics])[:restartNumTrialsLP]
  smallTopics = usedTopics[smallIDs]
  smallTopics = smallTopics[:nUsed-1]
  curELBO = calcELBOSingleDoc_Fast(wc_d, DocTopicCount_d, 
                                   Prior_d, sumR_d, alphaEbeta)
  Info['startELBO'] = curELBO
  pDocTopicCount_d =  DocTopicCount_d.copy()
  pPrior_d = Prior_d.copy() 
  psumR_d = np.zeros_like(sumR_d)  
  for kID in smallTopics:
    pDocTopicCount_d[:] = DocTopicCount_d
    pDocTopicCount_d[kID] = 0
    pPrior_d[:] = Prior_d
    
    for iter in range(restartNumItersLP):
      ## Update Prob of Active Topics
      aFunc(pDocTopicCount_d, pPrior_d) # Prior_d = E[ log pi_dk ]
      pPrior_d -= pPrior_d.max()
      np.exp(pPrior_d, out=pPrior_d)    # Prior_d = exp E[ log pi_dk ]
      
      ## Update sumR_d for all tokens in document
      np.dot(Lik_d, pPrior_d, out=psumR_d)

      ## Update DocTopicCounts
      np.dot(wc_d / psumR_d, Lik_d, out=pDocTopicCount_d)
      pDocTopicCount_d *= pPrior_d
    Info['nTrial'] += 1
    ## Evaluate proposal and accept/reject
    propELBO = calcELBOSingleDoc_Fast(wc_d, pDocTopicCount_d, 
                                      pPrior_d, psumR_d, alphaEbeta)
    if not np.isfinite(propELBO):
      continue
    if propELBO > curELBO:
      nUsed -= 1
      Info['nAccept'] += 1
      curELBO = propELBO
      DocTopicCount_d[:] = pDocTopicCount_d
      Prior_d[:] = pPrior_d
      sumR_d[:] = psumR_d
  Info['finalELBO'] = curELBO
  return DocTopicCount_d, Prior_d, sumR_d, Info





def calcDocTopicCountForData_Fast(Data, *args, **kwargs):
  if hasattr(Data, 'word_count'):
    return calcDocTopicCountForData_Fast_wordcount(Data, *args, **kwargs)
  else:
    return calcDocTopicCountForData_Fast_nowordcount(Data, *args, **kwargs)

def calcDocTopicCountForData_Fast_wordcount(Data, aModel, Lik,
                   initDocTopicCount=None,
                   initPrior=None, 
                   nCoordAscentItersLP=nCoordAscentIters,
                   convThrLP=convThr,
                   **kwargs
                  ):
  ''' Calculate updated doc-topic counts for every document in provided set

      Will loop over all docs, and at each one will run coordinate ascent
      to alternatively update its doc-topic counts and the doc-topic prior.
      Ascent stops after convergence or a maximum number of iterations.
    
      Returns
      ---------
      DocTopicCount : 2D array, size nDoc x K
      DocTopicCount[d,k] is effective number of tokens in doc d assigned to k

      Prior : 2D array, size nDoc x K
      Prior[d,k] = exp( E[log pi_{dk}] )

      sumRespTilde : 1D array, size N = # observed tokens
                     sumRespTilde[n] = normalization for the responsibility          
                     parameters for token n
  '''
  ## Initialize 
  tmpLP = dict()
  sumRespTilde = np.zeros(Lik.shape[0])

  if initDocTopicCount is not None:
    DocTopicCount = initDocTopicCount.copy()
    Prior = aModel.calcLogPrActiveComps_Fast(DocTopicCount, None, tmpLP)
    np.exp(Prior, out=Prior)
  else:
    DocTopicCount = np.zeros((Data.nDoc, aModel.K))
    if initPrior is None:
      Prior = np.ones((Data.nDoc, aModel.K))
    else:
      Prior = initPrior.copy()

  activeDocs = np.arange(Data.nDoc, dtype=np.int32)
  prev_DocTopicCount = DocTopicCount.copy()

  for ii in range(nCoordAscentItersLP):
    ## Update Prior for active documents
    if ii > 0:
      aModel.calcLogPrActiveComps_Fast(DocTopicCount, activeDocs, tmpLP,
                                       out=Prior)
      # Unfortunately, cannot update only activeDocs inplace (fancy idxing)
      Prior[activeDocs] = np.exp(Prior[activeDocs])

    for d in activeDocs:
      start = Data.doc_range[d]
      stop = Data.doc_range[d+1]
      Lik_d = Lik[start:stop]

      ## Update sumRtilde for all tokens in document
      np.dot(Lik_d, Prior[d], out=sumRespTilde[start:stop])

      ## Update DocTopicCount with Likelihood
      wc_d = Data.word_count[start:stop]
      np.dot(wc_d / sumRespTilde[start:stop], Lik_d, out=DocTopicCount[d])
 
    ## Update DocTopicCount with Prior
    DocTopicCount[activeDocs] *= Prior[activeDocs]

    # Assess convergence
    docDiffs = np.max(np.abs(prev_DocTopicCount - DocTopicCount), axis=1)
    if np.max(docDiffs) < convThrLP:
      break
    activeDocs = np.asarray(np.flatnonzero(docDiffs >= convThrLP),
                            dtype=np.int32)

    # Store DocTopicCount for next round's convergence test
    # Here, the "[:]" syntax ensures we do NOT copy by reference
    prev_DocTopicCount[activeDocs] = DocTopicCount[activeDocs]
    ### end loop over alternating-ascent updates

  return DocTopicCount, Prior, sumRespTilde



def calcDocTopicCountForData_Fast_nowordcount(Data, aModel, Lik,
                   initDocTopicCount=None,
                   initPrior=None, 
                   nCoordAscentItersLP=nCoordAscentIters,
                   convThrLP=convThr,
                   **kwargs
                  ):
  ''' Calculate updated doc-topic counts for every document in provided set

      Will loop over all docs, and at each one will run coordinate ascent
      to alternatively update its doc-topic counts and the doc-topic prior.
      Ascent stops after convergence or a maximum number of iterations.
    
      Returns
      ---------
      DocTopicCount : 2D array, size nDoc x K
      DocTopicCount[d,k] is effective number of tokens in doc d assigned to k

      Prior : 2D array, size nDoc x K
      Prior[d,k] = exp( E[log pi_{dk}] )

      sumRespTilde : 1D array, size N = # observed tokens
                     sumRespTilde[n] = normalization for the responsibility          
                     parameters for token n
  '''
  ## Initialize 
  tmpLP = dict()
  sumRespTilde = np.zeros(Lik.shape[0])

  if initDocTopicCount is not None:
    DocTopicCount = initDocTopicCount.copy()
    Prior = aModel.calcLogPrActiveComps_Fast(DocTopicCount, None, tmpLP)
    np.exp(Prior, out=Prior)
  else:
    DocTopicCount = np.zeros((Data.nDoc, aModel.K))
    if initPrior is None:
      Prior = np.ones((Data.nDoc, aModel.K))
    else:
      Prior = initPrior.copy()

  activeDocs = np.arange(Data.nDoc, dtype=np.int32)
  prev_DocTopicCount = DocTopicCount.copy()

  for ii in range(nCoordAscentItersLP):
    ## Update Prior for active documents
    if ii > 0:
      aModel.calcLogPrActiveComps_Fast(DocTopicCount, activeDocs, tmpLP,
                                       out=Prior)
      # Unfortunately, cannot update only activeDocs inplace (fancy idxing)
      Prior[activeDocs] = np.exp(Prior[activeDocs])

    for d in activeDocs:
      start = Data.doc_range[d]
      stop = Data.doc_range[d+1]
      Lik_d = Lik[start:stop]

      ## Update sumRtilde for all tokens in document
      np.dot(Lik_d, Prior[d], out=sumRespTilde[start:stop])

      ## Update DocTopicCount with Likelihood
      np.dot(1.0 / sumRespTilde[start:stop], Lik_d, out=DocTopicCount[d])
    ## Update DocTopicCount with Prior
    DocTopicCount[activeDocs] *= Prior[activeDocs]

    # Assess convergence
    docDiffs = np.max(np.abs(prev_DocTopicCount - DocTopicCount), axis=1)
    if np.max(docDiffs) < convThrLP:
      break
    activeDocs = np.asarray(np.flatnonzero(docDiffs >= convThrLP),
                            dtype=np.int32)

    # Store DocTopicCount for next round's convergence test
    # Here, the "[:]" syntax ensures we do NOT copy by reference
    prev_DocTopicCount[activeDocs] = DocTopicCount[activeDocs]
    ### end loop over alternating-ascent updates

  return DocTopicCount, Prior, sumRespTilde


def printVectors(aname, a, fmt='%9.6f', Kmax=10):
  if len(a) > Kmax:
    print('FIRST %d' % (Kmax))
    printVectors(aname, a[:Kmax], fmt, Kmax)
    print('LAST %d' % (Kmax))
    printVectors(aname, a[-Kmax:], fmt, Kmax)

  else:
    print(' %10s %s' % (aname, np2flatstr(a, fmt, Kmax)))

def np2flatstr(xvec, fmt='%9.3f', Kmax=10):
  return ' '.join( [fmt % (x) for x in xvec[:Kmax]])


def calcELBOSingleDoc_Fast(wc_d, DocTopicCount_d, Prior_d, sumR_d, alphaEbeta):
  ''' Evaluate ELBO contributions for single doc, dropping terms constant wrt local step.

      Note: key to some progress was remembering that Prior_d is not just exp(ElogPi)
            but that it is usually altered by a multiplicative constant for safety
            we can find this constant offset (in logspace), and adjust sumR_d accordingly
  '''
  theta_d = DocTopicCount_d + alphaEbeta[:-1]
  thetaRem = alphaEbeta[-1]

  digammaSum = digamma(theta_d.sum() + thetaRem)
  ElogPi_d = digamma(theta_d) - digammaSum
  ElogPiRem = digamma(thetaRem) - digammaSum                  

  cDir = -1 * c_Dir(theta_d[np.newaxis,:], thetaRem)
  slackOn = np.inner(DocTopicCount_d + alphaEbeta[:-1] - theta_d,
                     ElogPi_d.flatten())
  slackOff = (alphaEbeta[-1] - thetaRem) * ElogPiRem
  rest = np.inner(wc_d, np.log(sumR_d)) - np.inner(DocTopicCount_d, np.log(Prior_d+1e-100))
  
  return cDir + slackOn + slackOff + rest  
