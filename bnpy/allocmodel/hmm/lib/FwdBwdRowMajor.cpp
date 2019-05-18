#include <iostream>
#include "Eigen/Dense"

using namespace Eigen;
using namespace std;

// ======================================================== Func Declaration
// ========================================================

// Declare functions that need to be visible externally
extern "C" {
  void FwdAlg(
    double * initPiIN,
    double * transPiIN,
    double * SoftEvIN,
    double * fwdMsgOUT,
    double * margPrObsOUT,
    int K,
    int T);

  void FwdAlg_sparse(
    double * initPiIN,
    double * transPiIN,
    double * SoftEvIN,
    double * fwdMsgOUT,
    double * margPrObsOUT,
    int * topColIDsOUT,
    int K,
    int T,
    int L);

  void BwdAlg(
    double * initPiIN,
    double * transPiIN,
    double * SoftEvIN,
    double * margPrObsIN,
    double * bwdMsgOUT,
    int K,
    int T);

  void SummaryAlg(
    double * initPiIN,
    double * transPiIN,
    double * SoftEvIN,
    double * margPrObsIN,
    double * fwdMsgIN,
    double * bwdMsgIN,
    double * TransStateCountOUT,
    double * HtableOUT,
    double * mPairIDsIN,
    double * mergeHtableOUT,
    int K,
    int T,
    int M);
}



// ======================================================== Custom Type Defs
// ========================================================
// Simple names for array types
typedef Array<double, Dynamic, Dynamic, RowMajor> Arr2D;
typedef Array<double, 1, Dynamic, RowMajor> Arr1D;
typedef Array<int, Dynamic, Dynamic, RowMajor> Arr2D_i;

// Simple names for array types with externally allocated memory
typedef Map<Arr2D> ExtArr2D;
typedef Map<Arr1D> ExtArr1D;
typedef Map<Arr2D_i> ExtArr2D_i;

// Exactly the same as in util/lib/sparseResp/SparsifyRespCPPX.cpp - how to reuse?
struct LessThanFor1DArray {
    const double* xptr;

    LessThanFor1DArray(const double * xptrIN) {
        xptr = xptrIN;
    }

    bool operator()(int i, int j) {
        return xptr[i] < xptr[j];
    }

};

struct GreaterThanFor1DArray {
    const double* xptr;

    GreaterThanFor1DArray(const double * xptrIN) {
        xptr = xptrIN;
    }

    bool operator()(int i, int j) {
        return xptr[i] > xptr[j];
    }
};

struct Argsortable1DArray {
    double* xptr;
    int* iptr;
    int size;
    
    // Constructor
    Argsortable1DArray(double* xptrIN, int sizeIN) {
        xptr = xptrIN;
        size = sizeIN;
        iptr = new int[size];
        resetIndices(size);
    }

    // Helper method: reset iptr array to 0, 1, ... K-1 
    void resetIndices(int cursize) {
        assert(cursize <= size);
        for (int i = 0; i < cursize; i++) {
            iptr[i] = i;
        }
    }

    void pprint() {
        for (int i = 0; i < size; i++) {
            printf("%03d:% 05.2f ",
                this->iptr[i],
                this->xptr[this->iptr[i]]);
        }
        printf("\n");
    }

    void argsort() {
        this->argsort_AscendingOrder();
    }

    void argsort_AscendingOrder() {
        std::sort(
            this->iptr,
            this->iptr + this->size,
            LessThanFor1DArray(this->xptr)
            );
    }

    void argsort_DescendingOrder() {
        std::sort(
            this->iptr,
            this->iptr + this->size,
            GreaterThanFor1DArray(this->xptr)
            );
    }

    void findSmallestL(int L) {
        assert(L >= 0);
        assert(L < this->size);
        std::nth_element(
            this->iptr,
            this->iptr + L,
            this->iptr + this->size,
            LessThanFor1DArray(this->xptr)
            );
    }

    void findLargestL(int L, int Kactive) {
        assert(L >= 0);
        assert(L <= Kactive);
        std::nth_element(
            this->iptr,
            this->iptr + L,
            this->iptr + Kactive,
            GreaterThanFor1DArray(this->xptr)
            );
    }
};

// ======================================================== Forward Algorithm
// ======================================================== 
void FwdAlg(
    double * initPiIN,
    double * transPiIN,
    double * SoftEvIN,
    double * fwdMsgOUT,
    double * margPrObsOUT,
    int K,
    int T)
{
    // Prep input
    ExtArr1D initPi (initPiIN, K);
    ExtArr2D transPi (transPiIN, K, K);
    ExtArr2D SoftEv (SoftEvIN, T, K);

    // Prep output
    ExtArr2D fwdMsg (fwdMsgOUT, T, K);
    ExtArr1D margPrObs (margPrObsOUT, T);

    // Base case update for first time-step
    fwdMsg.row(0) = initPi * SoftEv.row(0);
    // find L largest 
    margPrObs(0) = fwdMsg.row(0).sum();
    fwdMsg.row(0) /= margPrObs(0);
    
    // Recursive update of timesteps 1, 2, ... T-1
    // Note: fwdMsg.row(t) is a *row vector* 
    //       so needs to be left-multiplied to square matrix transPi
    for (int t = 1; t < T; t++) {
        fwdMsg.row(t) = fwdMsg.row(t-1).matrix() * transPi.matrix();
        fwdMsg.row(t) *= SoftEv.row(t);
        margPrObs(t) = fwdMsg.row(t).sum();
        fwdMsg.row(t) /= margPrObs(t);
    }
}

void FwdAlg_sparse(
    double * initPiIN,
    double * transPiIN,
    double * SoftEvIN,
    double * fwdMsgOUT,
    double * margPrObsOUT,
    int * topColIDsOUT,
    int K,
    int T,
    int L)
{
    // Prep input
    ExtArr1D initPi (initPiIN, K);
    ExtArr2D transPi (transPiIN, K, K);
    ExtArr2D SoftEv (SoftEvIN, T, K);

    // Prep output
    //SparseMatrix<double, RowMajor> fwdMsg(T, L);
    //fwdMsg.reserve(VectorXi::Constant(T, L));
    ExtArr2D fwdMsg (fwdMsgOUT, T, L); // (T, L)
    ExtArr1D margPrObs (margPrObsOUT, T);
    ExtArr2D_i topColIDs (topColIDsOUT, T, L); // (T, L)

    // Prep tmp vars
    Arr1D fwdMsgTmp (K);

    // Base case update for first time-step
    fwdMsgTmp = initPi * SoftEv.row(0);

    // Pick and save top states
    Argsortable1DArray sorter = Argsortable1DArray(fwdMsgTmp.data(), K);
    sorter.findLargestL(L, K);
    for (int ell = 0; ell < L; ell++) {
        int k = sorter.iptr[ell];
        topColIDs(0, ell) = k;
        fwdMsg(0, ell) = sorter.xptr[k]; // This insertion takes L time
    }

    margPrObs(0) = fwdMsg.row(0).sum();
    fwdMsg.row(0) /= margPrObs(0);
    
    // Recursive update of timesteps 1, 2, ... T-1
    // Note: fwdMsg.row(t) is a *row vector* 
    //       so needs to be left-multiplied to square matrix transPi
    for (int t = 1; t < T; t++) {
        Arr2D transPi_t = transPi(topColIDs.row(t-1), all);
        fwdMsgTmp = fwdMsg.row(t-1).matrix() * transPi_t.matrix();
        fwdMsgTmp *= SoftEv.row(t);

        //sorter = Argsortable1DArray(fwdMsgTmp.data(), K);
        sorter.resetIndices(K);        
        sorter.findLargestL(L, K);
        for (int ell = 0; ell < L; ell++) {
            int k = sorter.iptr[ell];
            topColIDs(t, ell) = k;
            fwdMsg(t, ell) = sorter.xptr[k]; // This insertion takes L time
        }

        margPrObs(t) = fwdMsg.row(t).sum();
        fwdMsg.row(t) /= margPrObs(t);
    }
}

// ======================================================== Backward Algorithm
// ======================================================== 
void BwdAlg(
    double * initPiIN,
    double * transPiIN,
    double * SoftEvIN,
    double * margPrObsIN,
    double * bwdMsgOUT,
    int K,
    int T)
{
    // Prep input
    ExtArr1D initPi (initPiIN, K);
    ExtArr2D transPi (transPiIN, K, K);
    ExtArr2D SoftEv (SoftEvIN, T, K);
    ExtArr1D margPrObs (margPrObsIN, T);

    // Prep output
    ExtArr2D bMsg (bwdMsgOUT, T, K);

    // Base case update for last time-step
    bMsg.row(T-1).fill(1.0);
    
    // Recursive update of timesteps T-2, T-3, ... 3, 2, 1, 0
    // Note: bMsg.row(t) is a *row vector*
    //       so needs to be left-multiplied to square matrix transPi.T
    for (int t = T-2; t >= 0; t--) {
        bMsg.row(t) = (bMsg.row(t+1) * SoftEv.row(t+1)).matrix() \
                       * transPi.transpose().matrix();
        bMsg.row(t) /= margPrObs(t+1);
    }

}


// ======================================================== Summary Algorithm
// ======================================================== 
void SummaryAlg(
    double * initPiIN,
    double * transPiIN,
    double * SoftEvIN,
    double * margPrObsIN,
    double * fwdMsgIN,
    double * bwdMsgIN,
    double * TransStateCountOUT,
    double * HtableOUT,
    double * mPairIDsIN,
    double * mergeHtableOUT,
    int K,
    int T,
    int M)
{
    // Prep input
    ExtArr1D initPi (initPiIN, K);
    ExtArr2D transPi (transPiIN, K, K);
    ExtArr2D SoftEv (SoftEvIN, T, K);
    ExtArr1D margPrObs (margPrObsIN, T);
    ExtArr2D fwdMsg (fwdMsgIN, T, K);
    ExtArr2D bwdMsg (bwdMsgIN, T, K);

    // Prep output
    ExtArr2D TransStateCount (TransStateCountOUT, K, K);
    ExtArr2D Htable (HtableOUT, K, K);

    // Temporary KxK array for storing respPair at timestep t
    Arr2D respPair_t = ArrayXXd::Zero(K, K);
    Arr1D rowwiseSum = ArrayXd::Zero(K);
    Arr1D logrowwiseSum = ArrayXd::Zero(K);
    Arr2D epsArr = 1e-100 * ArrayXXd::Ones(K, K);

    ExtArr2D mPairIDs (mPairIDsIN, M, 2);
    ExtArr2D m_Htable (mergeHtableOUT, 2*M, K);
    // Temporary array for storing temp merge calculations at timestep t
    Arr2D m_respPair_t = ArrayXd::Zero(K);
    
    for (int t = 1; t < T; t++) {
        // In Python, we want:
        // >>> respPair[t] = np.outer(fmsg[t-1], bmsg[t] * SoftEv[t])
        // >>> respPair[t] *= PiMat / margPrObs[t]
        respPair_t = fwdMsg.row(t-1).transpose().matrix() \
                      * (bwdMsg.row(t) * SoftEv.row(t)).matrix();
        respPair_t *= transPi;
        respPair_t /= margPrObs(t);


        // Aggregate pairwise transition counts
        TransStateCount += respPair_t;


        // Aggregate entropy in a KxK matrix

        // Make numerically safe for logarithms
        // Each entry in respPair_t will be at least eps (1e-100)
        // Remember, cwiseMax only works with arrays, not scalars :(
        // https://forum.kde.org/viewtopic.php?f=74&t=98384
        respPair_t = respPair_t.cwiseMax(epsArr);

        rowwiseSum = respPair_t.rowwise().sum();
        logrowwiseSum = log(rowwiseSum);

        // ----------------------------------- Start merge logic
        for (int m = 0; m < M; m++) {
          int kA = (int) mPairIDs(m,0);
          int kB = (int) mPairIDs(m,1);

          // Construct new respPair_t[kA, :], respPair_t[kB, :]
          double m_logrowwiseSum = log(rowwiseSum(kA) + rowwiseSum(kB));
          for (int k = 0; k < K; k++) {
              if (k == kA) {
                double mrP = respPair_t(kA, kA) + respPair_t(kB, kB) \
                            +respPair_t(kA, kB) + respPair_t(kB, kA);
                m_Htable(2*m, kA) -= mrP * log(mrP);
                m_Htable(2*m, kA) += mrP * m_logrowwiseSum;

              } else if (k == kB) {
                continue;
              } else {
                double mrP = respPair_t(kA, k) + respPair_t(kB, k);
                m_Htable(2*m, k) -= mrP * log(mrP);
                m_Htable(2*m, k) += mrP * m_logrowwiseSum;
              }
          } 

          // Construct new respPair_t[:, kA], respPair_t[:, kB]
          for (int k = 0; k < K; k++) {
              if ((k == kA) || (k == kB)) {
                continue;
              } else {
                double mrP = respPair_t(k, kA) + respPair_t(k, kB);
                m_Htable(2*m+1, k) -= mrP * log(mrP);
                m_Htable(2*m+1, k) += mrP * logrowwiseSum(k);
              }
          }
          m_Htable(2*m+1, kA) = m_Htable(2*m, kA);
          m_Htable(2*m, kB) = 0;
          m_Htable(2*m+1, kB) = 0;
        }
        // ----------------------------------- End merge logic

        // Increment by rP log rP
        Htable += respPair_t * respPair_t.log();

        // Decrement by rP log rP.rowwise().sum()
        // Remember, broadcasting with *= doesnt work
        // https://forum.kde.org/viewtopic.php?f=74&t=95629 
        // so we use a forloop instead
        for (int k=0; k < K; k++) {
          respPair_t.col(k) *= logrowwiseSum;
        }
        Htable -= respPair_t; 

        /*
        printf("----------- t=%d\n", t);
        for (int j = 0; j < K; j++) {
          for (int k = 0; k < K; k++) {
            printf(" %.3f", respPair_t(j,k));
          }
          printf("\n");
        }
        */
    }

    Htable *= -1.0;
}
