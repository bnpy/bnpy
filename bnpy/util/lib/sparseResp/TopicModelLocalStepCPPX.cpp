// TopicModelLocalStepCPPX.cpp
// Define this symbol to enable runtime tests for allocations
#define EIGEN_RUNTIME_NO_MALLOC 

#include <math.h>
#include "Eigen/Dense"
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>

using namespace Eigen;
using namespace std;

// ======================================================== Declare funcs
// ======================================================== visible externally

extern "C" {
    void sparseLocalStepSingleDoc(
        double* ElogLik_d_IN,
        double* alphaEbeta_IN,
        int nnzPerRow,
        int N,
        int K,
        int nCoordAscentIterLP,
        double convThrLP,
        int initProbsToEbeta,
        double* topicCount_d_OUT,
        double* spResp_data_OUT,
        int* spResp_colids_OUT,
        int d,
        int D,
        int* numIterVec_OUT,
        double* maxDiffVec_OUT
        );

    void sparseLocalStepSingleDoc_ActiveOnly(
        double* ElogLik_d_IN,
        double* wc_d_IN,
        double* alphaEbeta_IN,
        int nnzPerRow,
        int N,
        int K,
        int nCoordAscentIterLP,
        double convThrLP,
        int initProbsToEbeta,
        double* topicCount_d_OUT,
        double* spResp_data_OUT,
        int* spResp_colids_OUT,
        int d,
        int D,
        int* numIterVec_OUT,
        double* maxDiffVec_OUT,
        int doTrackELBO,
        double* elboVec_OUT,
        int numRestarts,
        int REVISE_FIRST,
        int REVISE_EVERY,
        int *rAcceptVec_IN, int* rTrialVec_IN,
        int verbose
        );


    void sparseLocalStepSingleDocWithWordCounts(
        double* wordcounts_d_IN,
        double* ElogLik_d_IN,
        double* alphaEbeta_IN,
        int nnzPerRow,
        int N,
        int K,
        int nCoordAscentIterLP,
        double convThrLP,
        int initProbsToEbeta,
        double* topicCount_d_OUT,
        double* spResp_data_OUT,
        int* spResp_colids_OUT
        );   
}

// ======================================================== Custom Type Defs
// ========================================================
// Simple names for array types
typedef Matrix<double, Dynamic, Dynamic, RowMajor> Mat2D_d;
typedef Matrix<double, 1, Dynamic, RowMajor> Mat1D_d;
typedef Array<double, Dynamic, Dynamic, RowMajor> Arr2D_d;
typedef Array<double, 1, Dynamic, RowMajor> Arr1D_d;
typedef Array<int, 1, Dynamic, RowMajor> Arr1D_i;

// Simple names for array types with externally allocated memory
typedef Map<Mat2D_d> ExtMat2D_d;
typedef Map<Mat1D_d> ExtMat1D_d;
typedef Map<Arr2D_d> ExtArr2D_d;
typedef Map<Arr1D_d> ExtArr1D_d;
typedef Map<Arr1D_i> ExtArr1D_i;

double calcELBOForSingleDoc_V2(
    Arr1D_d alphaEbeta,
    Arr1D_d topicCount_d,
    Arr1D_d spResp_data,
    Arr1D_i spResp_colids,
    Arr2D_d ElogLik_d,
    int K,
    int N,
    int nnzPerRow
    )
{
    double ELBO = 0.0;
    // Lalloc
    for (int k = 0; k < K; k++) {
        ELBO += boost::math::lgamma(topicCount_d(k) + alphaEbeta(k));
    }
    
    // Ldata and Lentropy
    for (int n = 0; n < N; n++) {
        for (int nzk = n * nnzPerRow; nzk < (n+1) * nnzPerRow; nzk++) {
            int k = spResp_colids(nzk);
            ELBO += spResp_data(nzk) * ElogLik_d(n,k);
            if (spResp_data(nzk) > 1e-9) {
                ELBO -= spResp_data(nzk) * log(spResp_data(nzk));
            }
        }
    }
    return ELBO;
}

double calcELBOForSingleDoc_V1(
    ExtArr1D_d alphaEbeta,
    ExtArr1D_d topicCount_d,
    Arr1D_d ElogProb_d,
    Arr1D_i activeTopics_d,
    double totalLogSumResp,
    double sum_gammalnalphaEbeta,
    int Kactive
    )
{
    // Compute Lalloc = \sum_k gammaln(\theta_dk)
    double ELBO = sum_gammalnalphaEbeta;
    for (int ka = 0; ka < Kactive; ka++) {
        int k = activeTopics_d(ka);
        ELBO += (
            boost::math::lgamma(topicCount_d(k) + alphaEbeta(k))
            - boost::math::lgamma(alphaEbeta(k))
            - topicCount_d(k) * ElogProb_d(k)
            );
    }
    ELBO += totalLogSumResp;
    return ELBO;
}

double updateAssignments_FixPerTokenActiveSet(
    ExtArr2D_d ElogLik_d,
    ExtArr1D_d wc_d,
    ExtArr1D_d alphaEbeta,
    Arr1D_i activeTopics_d,
    ExtArr1D_d& topicCount_d, // & makes sure we can edit passed array in place
    Arr1D_d& ElogProb_d,
    Arr1D_d& logScores_n,
    ExtArr1D_d& spResp_data,
    ExtArr1D_i& spResp_colids,
    int N,
    int K, 
    int Kactive,
    int nnzPerRow,
    int doTrackELBO
    )
{
    assert(nnzPerRow > 1);
    double totalLogSumResp = 0.0;
    // UPDATE ElogProb_d using input doc-topic counts
    for (int ka = 0; ka < Kactive; ka++) {
        int k = activeTopics_d(ka);
        ElogProb_d(k) = boost::math::digamma(
            topicCount_d(k) + alphaEbeta(k));
    }
    // RESET topicCounts to all zeros
    topicCount_d.fill(0);
    // UPDATE assignments, obeying sparsity constraint
    for (int n = 0; n < N; n++) {
        double sumResp_n = 0.0;
        int m = n * nnzPerRow;
        double maxScore_n;
        for (int ka = 0; ka < nnzPerRow; ka++) {
            int k = spResp_colids(m + ka);
            logScores_n(ka) = ElogProb_d(k) + ElogLik_d(n,k);
            if (ka == 0 || logScores_n(ka) > maxScore_n) {
                maxScore_n = logScores_n(ka);
            }
        }
        for (int nzk = 0; nzk < nnzPerRow; nzk++) {
            spResp_data(m + nzk) = \
                exp(logScores_n(nzk) - maxScore_n);
            sumResp_n += spResp_data(m + nzk);
        }

        for (int nzk = m; nzk < m + nnzPerRow; nzk++) {
            spResp_data(nzk) /= sumResp_n;
            topicCount_d(spResp_colids(nzk)) += \
                wc_d(n) * spResp_data(nzk);
        }
        if (doTrackELBO) {
            totalLogSumResp += wc_d(n) * (maxScore_n + log(sumResp_n));
        }
    } // end for loop over tokens n
    assert(abs(wc_d.sum() - topicCount_d.sum()) < .000001);
    return totalLogSumResp;
}

double updateAssignments_ActiveOnly(
    ExtArr2D_d ElogLik_d,
    ExtArr1D_d wc_d,
    ExtArr1D_d alphaEbeta,
    Arr1D_i activeTopics_d,
    ExtArr1D_d& topicCount_d, // & makes sure we can edit passed array in place
    Arr1D_d& ElogProb_d,
    Arr1D_d& logScores_n,
    Arr1D_d& tempScores_n,
    ExtArr1D_d& spResp_data,
    ExtArr1D_i& spResp_colids,
    int N,
    int K, 
    int Kactive,
    int nnzPerRow,
    int iter,
    int initProbsToEbeta,
    int doTrackELBO
    )
{
    double totalLogSumResp = 0.0;

    // UPDATE ElogProb_d using input doc-topic counts
    if (iter == 0 and initProbsToEbeta == 1) {
        for (int ka = 0; ka < Kactive; ka++) {
            int k = activeTopics_d(ka);
            ElogProb_d(k) = log(alphaEbeta(k));
        }
    } else {
        for (int ka = 0; ka < Kactive; ka++) {
            int k = activeTopics_d(ka);
            ElogProb_d(k) = boost::math::digamma(
                topicCount_d(k) + alphaEbeta(k));
        }
    }
    // RESET topicCounts to all zeros
    topicCount_d.fill(0);
    // UPDATE assignments, obeying sparsity constraint
    for (int n = 0; n < N; n++) {
        int m = n * nnzPerRow;
        int argmax_n = 0;
        double maxScore_n;
        for (int ka = 0; ka < Kactive; ka++) {
            int k = activeTopics_d(ka);
            logScores_n(ka) = ElogProb_d(k) + ElogLik_d(n,k);
            if (ka == 0 || logScores_n(ka) > maxScore_n) {
                maxScore_n = logScores_n(ka);
                if (nnzPerRow == 1) {
                    argmax_n = k;
                }
            }
        }
        if (nnzPerRow == 1) {
            spResp_data(m) = 1.0;
            spResp_colids(m) = argmax_n;
            topicCount_d(argmax_n) += wc_d(n);
            if (doTrackELBO) {
                totalLogSumResp += wc_d(n) * maxScore_n;
            }
        } else {
            // Find the top L entries in logScores_n
            // Copy current row over into a temp buffer
            std::copy(
                logScores_n.data(),
                logScores_n.data() + Kactive,
                tempScores_n.data());
            // Sort the data in the temp buffer (in place)
            std::nth_element(
                tempScores_n.data(),
                tempScores_n.data() + Kactive - nnzPerRow,
                tempScores_n.data() + Kactive);
            // Walk thru this row and find the top "nnzPerRow" positions
            double pivotScore = tempScores_n(Kactive - nnzPerRow);

            int nzk = m;
            double sumResp_n = 0.0;
            for (int ka = 0; ka < Kactive; ka++) {
                if (logScores_n(ka) >= pivotScore) {
                    spResp_data(nzk) = \
                        exp(logScores_n(ka) - maxScore_n);
                    spResp_colids(nzk) = activeTopics_d(ka);
                    sumResp_n += spResp_data(nzk);
                    nzk += 1;                        
                }
            }
            //assert(nzk - m == nnzPerRow);

            for (nzk = m; nzk < m + nnzPerRow; nzk++) {
                spResp_data(nzk) /= sumResp_n;
                topicCount_d(spResp_colids(nzk)) += \
                    wc_d(n) * spResp_data(nzk);
            }
            if (doTrackELBO) {
                totalLogSumResp += wc_d(n) * (maxScore_n + log(sumResp_n));
            }
        } // end if statement branch for nnz > 1
    } // end for loop over tokens n
    return totalLogSumResp;
}

void sparseLocalStepSingleDoc_ActiveOnly(
        double* ElogLik_d_IN,
        double* wc_d_IN,
        double* alphaEbeta_IN,
        int nnzPerRow,
        int N,
        int K,
        int nCoordAscentIterLP,
        double convThrLP,
        int initProbsToEbeta,
        double* topicCount_d_OUT,
        double* spResp_data_OUT,
        int* spResp_colids_OUT,
        int d,
        int D,
        int* numIterVec_OUT,
        double* maxDiffVec_OUT,
        int doTrackELBO,
        double* elboVec_OUT,
        int numRestarts,
        int REVISE_FIRST,
        int REVISE_EVERY,
        int* rAcceptVec_IN,
        int* rTrialVec_IN,
        int verbose
        )
{
    nCoordAscentIterLP = max(nCoordAscentIterLP + max(0, initProbsToEbeta), 1);

    // Unpack inputs, treated as fixed constants
    ExtArr2D_d ElogLik_d (ElogLik_d_IN, N, K);
    ExtArr1D_d wc_d (wc_d_IN, N);
    ExtArr1D_d alphaEbeta (alphaEbeta_IN, K);

    // Unpack outputs
    ExtArr1D_d spResp_data (spResp_data_OUT, N * nnzPerRow);
    ExtArr1D_i spResp_colids (spResp_colids_OUT, N * nnzPerRow);
    ExtArr1D_d topicCount_d (topicCount_d_OUT, K);

    ExtArr1D_i numIterVec (numIterVec_OUT, D);
    ExtArr1D_d maxDiffVec (maxDiffVec_OUT, D);
    ExtArr1D_d elboVec (elboVec_OUT, nCoordAscentIterLP);

    ExtArr1D_i rAcceptVec (rAcceptVec_IN, 1);
    ExtArr1D_i rTrialVec (rTrialVec_IN, 1);
    
    // Temporary storage
    Arr1D_d ElogProb_d (K);
    Arr1D_d prevTopicCount_d (K);
    Arr1D_d logScores_n (K);
    Arr1D_d tempScores_n (K);

    Arr1D_i activeTopics_d (K);
    Arr1D_i spareActiveTopics_d (nnzPerRow);

    prevTopicCount_d.fill(-1);
    double maxDiff = N;
    int iter = 0;
    int Kactive = K;
    double ACTIVE_THR = 1e-9;
    double totalLogSumResp = 0.0;
    double sum_gammalnalphaEbeta = 0.0;
    if (doTrackELBO || numRestarts > 0) {
        for (int k = 0; k < K; k++) {
            sum_gammalnalphaEbeta += boost::math::lgamma(alphaEbeta(k));
        }
    }
    // Before any updates... All topics are active!
    for (int k = 0; k < K; k++) {
        activeTopics_d(k) = k;
    }
    if (verbose > 1) {
        printf("Initial topic counts \n");
        for (int k = 0; k < K; k++) {
            printf("%d:%6.1f ", k, topicCount_d(k));
        }
        printf("\n");
    }

    for (iter = 0; iter < nCoordAscentIterLP; iter++) {
        int doReviseActiveSet;
        if (iter >= REVISE_FIRST && Kactive <= nnzPerRow) {
            // Set of active docs is already very small,
            // so nothing big to gain from revising
            doReviseActiveSet = 0;
        } else if (iter < REVISE_FIRST || (iter - 1) % REVISE_EVERY == 0) {
            doReviseActiveSet = 1;
        } else {
            doReviseActiveSet = 0;
        }

        if (iter > 0 || initProbsToEbeta < 0) {
            if (doReviseActiveSet) {
                int newKactive = 0;
                int ia = 0; // index for spare inactive topics
                for (int a = 0; a < Kactive; a++) {
                    int k = activeTopics_d(a);
                    if (topicCount_d(k) > ACTIVE_THR) {
                        activeTopics_d(newKactive) = k;
                        prevTopicCount_d(k) = topicCount_d(k);
                        newKactive += 1;
                    } else if (newKactive < nnzPerRow - ia) {
                        spareActiveTopics_d(ia) = k;
                        ia += 1;
                    }
                }
                // If num topics above threshold is less than nnzPerRow,
                // We need to fill in with some spare empty topics.
                Kactive = newKactive;
                while (Kactive < nnzPerRow) {
                    int k = spareActiveTopics_d(Kactive - newKactive);
                    activeTopics_d(Kactive) = k;
                    Kactive++;
                }
            } else {
                for (int a = 0; a < Kactive; a++) {
                    int k = activeTopics_d(a);
                    prevTopicCount_d(k) = topicCount_d(k);
                }
            }
        }
        assert(Kactive >= nnzPerRow);
        assert(Kactive <= K);
        if (nnzPerRow == 1 || doReviseActiveSet) {
            totalLogSumResp = updateAssignments_ActiveOnly(
                ElogLik_d, wc_d, alphaEbeta, activeTopics_d,
                topicCount_d, ElogProb_d,
                logScores_n, tempScores_n,
                spResp_data, spResp_colids,
                N, K, Kactive, nnzPerRow, 
                iter, initProbsToEbeta, doTrackELBO
                );
        } else {
            totalLogSumResp = updateAssignments_FixPerTokenActiveSet(
                ElogLik_d, wc_d, alphaEbeta, activeTopics_d,
                topicCount_d, ElogProb_d,
                logScores_n,
                spResp_data, spResp_colids,
                N, K, Kactive, nnzPerRow, 
                doTrackELBO
                );
        }

        if (doTrackELBO) {
            elboVec(iter) = calcELBOForSingleDoc_V1(
                alphaEbeta, topicCount_d, ElogProb_d, activeTopics_d,
                totalLogSumResp, sum_gammalnalphaEbeta, Kactive);
            /*
            double elboV2 = calcELBOForSingleDoc_V2(
                alphaEbeta, topicCount_d, spResp_data, spResp_colids,
                ElogLik_d, K, N, nnzPerRow);
            
            double elboV1 = calcELBOForSingleDoc_V1(
                alphaEbeta, topicCount_d, ElogProb_d, activeTopics_d,
                totalLogSumResp, sum_gammalnalphaEbeta, Kactive);
            printf(" V1: %.6f\n V2: %.6f\n", elboV2, elboV1);
            */
        }

        if (verbose > 1) {
            printf("end of iter %3d Kactive %3d maxDiff %10.6f\n", 
                iter, Kactive, maxDiff);
            for (int ka = 0; ka < Kactive; ka++) {
                int k = activeTopics_d(ka);
                printf("%d:%6.1f ", k, topicCount_d(k));
            }
            printf("\n");
        }

        // END ITERATION. Decide whether to quit early
        if (iter > 0 && iter % 5 == 0) {
            double absDiff_k = 0.0;
            maxDiff = 0.0;
            for (int ka = 0; ka < Kactive; ka++) {
                int k = activeTopics_d(ka);
                absDiff_k = abs(prevTopicCount_d(k) - topicCount_d(k));
                if (absDiff_k > maxDiff) {
                    maxDiff = absDiff_k;
                }
            }
            if (maxDiff <= convThrLP) {
                break;
            }
        }
    }

    // Figure out which topics are eligible for sparse restarts
    double smallThr = 1e-6;
    double curELBO = 0.0;
    for (int riter = 0; riter < numRestarts; riter++) {
        if (riter == 0) {
            totalLogSumResp = updateAssignments_ActiveOnly(
                ElogLik_d, wc_d, alphaEbeta, activeTopics_d,
                topicCount_d, ElogProb_d,
                logScores_n, tempScores_n,
                spResp_data, spResp_colids,
                N, K, Kactive, nnzPerRow, 
                0, 0, 1);
            curELBO = calcELBOForSingleDoc_V1(
                alphaEbeta, topicCount_d, ElogProb_d, activeTopics_d,
                totalLogSumResp, sum_gammalnalphaEbeta, Kactive);
            // Remember the best-known topic-count vector!
            for (int k = 0; k < K; k++) {
                prevTopicCount_d(k) = topicCount_d(k);
            }
        }
        // SEARCH FOR SMALLEST TOPIC HAVE NOT YET TRIED YET
        int numAboveThr = 0;
        double minVal = N + 1.0; // topicCount_d must never have this value
        int minLoc = 0;
        for (int ka = 0; ka < Kactive; ka++) {
            int k = activeTopics_d(ka);
            if (topicCount_d(k) > smallThr) {
                numAboveThr += 1;
                if (topicCount_d(k) < minVal) {
                    minVal = topicCount_d(k);
                    minLoc = k;
                }
            }
        }
        if (numAboveThr == 1) {
            break;
        }

        if (verbose) {
            printf("START: best known counts. ELBO=%.5e \n", curELBO);
            for (int ka = 0; ka < Kactive; ka++) {
                int k = activeTopics_d(ka);
                printf("%02d:%06.2f ", k, topicCount_d(k));
            }
            printf("\n");
        }
        smallThr = minVal;
        topicCount_d(minLoc) = 0.0;
        if (verbose) {        
            printf(
                "RESTART: Set index %d to zero (%d left)\n", 
                minLoc, numAboveThr - 1);
            for (int ka = 0; ka < Kactive; ka++) {
                int k = activeTopics_d(ka);
                printf("%02d:%06.2f ", k, topicCount_d(k));
            }
            printf("\n");
        }
        int NSTEP = 2;
        for (int step = 0; step < NSTEP; step++) {        
            totalLogSumResp = updateAssignments_ActiveOnly(
                ElogLik_d, wc_d, alphaEbeta, activeTopics_d,
                topicCount_d, ElogProb_d,
                logScores_n, tempScores_n,
                spResp_data, spResp_colids,
                N, K, Kactive, nnzPerRow, 
                step, 0, step == NSTEP-1);
        }
        // If the change is small, abandon this
        double propELBO;
        if (abs(prevTopicCount_d(minLoc) -topicCount_d(minLoc)) < 1e-6) {
            propELBO = curELBO;
        } else {
            propELBO = calcELBOForSingleDoc_V1(
                alphaEbeta, topicCount_d, ElogProb_d, activeTopics_d,
                totalLogSumResp, sum_gammalnalphaEbeta, Kactive);
        }

        if (verbose) {
            for (int ka = 0; ka < Kactive; ka++) {
                int k = activeTopics_d(ka);
                printf("%02d:%06.2f ", k, topicCount_d(k));
            }
            printf("\n");
            printf("propELBO % .6e\n", propELBO);
            printf(" curELBO % .6e\n", curELBO);
            if (propELBO > curELBO) {
                printf("beforeCount: %.6f\n", prevTopicCount_d(minLoc));
                printf(" afterCount: %.6f\n", topicCount_d(minLoc));
                printf("gainELBO % .6e  *** ACCEPTED \n", propELBO - curELBO);
            } else {
                printf("gainELBO % .6e      rejected \n", propELBO - curELBO);
            }
        }
        // If accepted, set current best doc-topic counts to latest proposal
        // Otherwise, reset the starting point for the next proposal.
        if (propELBO > curELBO) {
            curELBO = propELBO;
            for (int ka = 0; ka < Kactive; ka++) {
                int k = activeTopics_d(ka);
                prevTopicCount_d(k) = topicCount_d(k);
            }
            rAcceptVec(0) += 1;
        } else {
            for (int ka = 0; ka < Kactive; ka++) {
                int k = activeTopics_d(ka);
                topicCount_d(k) = prevTopicCount_d(k);
            }
        }
        rTrialVec(0) += 1;
    }
    if (numRestarts > 0) {
        for (int ka = 0; ka < Kactive; ka++) {
            int k = activeTopics_d(ka);
            topicCount_d(k) = prevTopicCount_d(k);
        }
        // Final update! Make sure spResp reflects best topicCounts found
        updateAssignments_ActiveOnly(
            ElogLik_d, wc_d, alphaEbeta, activeTopics_d,
            topicCount_d, ElogProb_d,
            logScores_n, tempScores_n,
            spResp_data, spResp_colids,
            N, K, Kactive, nnzPerRow, 
            0, 0, 0);
    }
    maxDiffVec(d) = maxDiff;
    numIterVec(d) = iter; // will already have +1 from last iter of for loop
}


void sparseLocalStepSingleDoc(
        double* ElogLik_d_IN,
        double* alphaEbeta_IN,
        int nnzPerRow,
        int N,
        int K,
        int nCoordAscentIterLP,
        double convThrLP,
        int initProbsToEbeta,
        double* topicCount_d_OUT,
        double* spResp_data_OUT,
        int* spResp_colids_OUT,
        int d,
        int D,
        int* numIterVec_OUT,
        double* maxDiffVec_OUT
        )
{
    // Unpack inputs, treated as fixed constants
    ExtArr2D_d ElogLik_d (ElogLik_d_IN, N, K);
    ExtArr1D_d alphaEbeta (alphaEbeta_IN, K);
    // Unpack outputs
    ExtArr1D_d spResp_data (spResp_data_OUT, N * nnzPerRow);
    ExtArr1D_i spResp_colids (spResp_colids_OUT, N * nnzPerRow);
    ExtArr1D_d topicCount_d (topicCount_d_OUT, K);

    ExtArr1D_i numIterVec (numIterVec_OUT, D);
    ExtArr1D_d maxDiffVec (maxDiffVec_OUT, D);

    // Temporary storage
    Arr1D_d ElogProb_d (K);
    Arr1D_d prevTopicCount_d (K);
    Arr1D_d logScores_n (K);
    Arr1D_d tempScores_n (K);

    prevTopicCount_d.fill(-1);
    double maxDiff = N;
    int iter = 0;

    for (iter = 0; iter < nCoordAscentIterLP + initProbsToEbeta; iter++) {

        if (iter == 0 and initProbsToEbeta == 1) {
            for (int k = 0; k < K; k++) {
                ElogProb_d(k) = log(alphaEbeta(k));
            }
        } else {
            for (int k = 0; k < K; k++) {
                ElogProb_d(k) = boost::math::digamma(
                    topicCount_d(k) + alphaEbeta(k));
            }
        }
    
        topicCount_d.fill(0);
        // Step over each data atom
        for (int n = 0; n < N; n++) {
            int m = n * nnzPerRow;
            int argmax_n = 0;
            logScores_n(0) = ElogProb_d(0) + ElogLik_d(n,0);
            double maxScore_n = logScores_n(0);
            for (int k = 1; k < K; k++) {
                logScores_n(k) = ElogProb_d(k) + ElogLik_d(n,k);
                if (logScores_n(k) > maxScore_n) {
                    maxScore_n = logScores_n(k);
                    argmax_n = k;
                }
            }
            if (nnzPerRow == 1) {
                spResp_data(m) = 1.0;
                spResp_colids(m) = argmax_n;
                // Update topicCount_d
                topicCount_d(argmax_n) += 1.0;
            } else {
                // Find the top L entries in logScores_n
                // Copy current row over into a temp buffer
                std::copy(
                    logScores_n.data(),
                    logScores_n.data() + K,
                    tempScores_n.data());
                // Sort the data in the temp buffer (in place)
                std::nth_element(
                    tempScores_n.data(),
                    tempScores_n.data() + K - nnzPerRow,
                    tempScores_n.data() + K);
                // Walk thru this row and find the top "nnzPerRow" positions
                double pivotScore = tempScores_n(K - nnzPerRow);
                int nzk = 0;
                double sumResp_n = 0.0;
                for (int k = 0; k < K; k++) {
                    if (logScores_n(k) >= pivotScore) {
                        spResp_data(m + nzk) = \
                            exp(logScores_n(k) - maxScore_n);
                        spResp_colids(m + nzk) = k;
                        sumResp_n += spResp_data(m + nzk);
                        nzk += 1;
                    }
                }
                // Normalize for doc-topic counts
                for (int nzk = 0; nzk < nnzPerRow; nzk++) {
                    spResp_data(m + nzk) /= sumResp_n;
                    topicCount_d(spResp_colids(m + nzk)) += \
                        spResp_data(m + nzk);
                }
            }
        }
        // END ITERATION. Decide whether to quit early
        if (iter > 0 && iter % 5 == 0) {
            double absDiff_k = 0.0;
            maxDiff = 0.0;
            for (int k = 0; k < K; k++) {
                absDiff_k = abs(prevTopicCount_d(k) - topicCount_d(k));
                if (absDiff_k > maxDiff) {
                    maxDiff = absDiff_k;
                }
                prevTopicCount_d(k) = topicCount_d(k); // copy over
            }
            if (maxDiff <= convThrLP) {
                break;
            }
        }
    }
    maxDiffVec(d) = maxDiff;
    numIterVec(d) = iter + 1;
}


void sparseLocalStepSingleDocWithWordCounts(
        double* wordcounts_d_IN,
        double* ElogLik_d_IN,
        double* alphaEbeta_IN,
        int nnzPerRow,
        int N,
        int K,
        int nCoordAscentIterLP,
        double convThrLP,
        int initProbsToEbeta,
        double* topicCount_d_OUT,
        double* spResp_data_OUT,
        int* spResp_colids_OUT
        )
{
    // Unpack inputs, treated as fixed constants
    ExtArr1D_d wc_d (wordcounts_d_IN, N);
    ExtArr2D_d ElogLik_d (ElogLik_d_IN, N, K);
    ExtArr1D_d alphaEbeta (alphaEbeta_IN, K);
    // Unpack outputs
    ExtArr1D_d spResp_data (spResp_data_OUT, N * nnzPerRow);
    ExtArr1D_i spResp_colids (spResp_colids_OUT, N * nnzPerRow);
    ExtArr1D_d topicCount_d (topicCount_d_OUT, K);
    // Temporary storage
    VectorXd ElogProb_d (K);
    VectorXd prevTopicCount_d (K);
    VectorXd logScores_n (K);
    VectorXd tempScores_n (K);
    prevTopicCount_d.fill(-1);
    int iter = 0;
    double maxDiff = 0.0;
    for (iter = 0; iter < nCoordAscentIterLP + initProbsToEbeta; iter++) {

        if (iter == 0 and initProbsToEbeta == 1) {
            for (int k = 0; k < K; k++) {
                ElogProb_d(k) = log(alphaEbeta(k));
            }
        } else {
            for (int k = 0; k < K; k++) {
                ElogProb_d(k) = boost::math::digamma(
                    topicCount_d(k) + alphaEbeta(k));
            }
        }
    
        topicCount_d.fill(0);
        // Step over each data atom
        for (int n = 0; n < N; n++) {
            int m = n * nnzPerRow;
            int argmax_n = 0;
            logScores_n(0) = ElogProb_d(0) + ElogLik_d(n,0);
            double maxScore_n = logScores_n(0);
            for (int k = 1; k < K; k++) {
                logScores_n(k) = ElogProb_d(k) + ElogLik_d(n,k);
                if (logScores_n(k) > maxScore_n) {
                    maxScore_n = logScores_n(k);
                    argmax_n = k;
                }
            }
            if (nnzPerRow == 1) {
                spResp_data(m) = 1.0;
                spResp_colids(m) = argmax_n;
                // Update topicCount_d
                topicCount_d(argmax_n) += wc_d(n);
            } else {
                // Find the top L entries in logScores_n
                // Copy current row over into a temp buffer
                std::copy(
                    logScores_n.data(),
                    logScores_n.data() + K,
                    tempScores_n.data());
                // Sort the data in the temp buffer (in place)
                std::nth_element(
                    tempScores_n.data(),
                    tempScores_n.data() + K - nnzPerRow,
                    tempScores_n.data() + K);
                // Walk thru this row and find the top "nnzPerRow" positions
                double pivotScore = tempScores_n(K - nnzPerRow);
                int nzk = 0;
                double sumResp_n = 0.0;
                for (int k = 0; k < K; k++) {
                    if (logScores_n(k) >= pivotScore) {
                        spResp_data(m + nzk) = \
                            exp(logScores_n(k) - maxScore_n);
                        spResp_colids(m + nzk) = k;
                        sumResp_n += spResp_data(m + nzk);
                        nzk += 1;
                    }
                }
                // Normalize for doc-topic counts
                for (int nzk = 0; nzk < nnzPerRow; nzk++) {
                    spResp_data(m + nzk) /= sumResp_n;
                    topicCount_d(spResp_colids(m + nzk)) += \
                        wc_d(n) * spResp_data(m + nzk);
                }
            }
        }
        // END ITERATION. Decide whether to quit early
        if (iter > 0 && iter % 5 == 0) {
            double absDiff_k = 0.0;
            maxDiff = 0.0;
            for (int k = 0; k < K; k++) {
                absDiff_k = abs(prevTopicCount_d(k) - topicCount_d(k));
                if (absDiff_k > maxDiff) {
                    maxDiff = absDiff_k;
                }
                prevTopicCount_d(k) = topicCount_d(k); // copy over
            }
            if (maxDiff <= convThrLP) {
                break;
            }
        }
    }
}



