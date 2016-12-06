/* RlogRCore.cpp
Author: Mike Hughes (www.michaelchughes.com)
Date:   24 Jan 2013
*/

#include <iostream>
#include "Eigen/Dense"

using namespace Eigen;
using namespace std;

// ======================================================== Declare funcs
// ========================================================  visible externally

extern "C" {
  void CalcRlogR_AllPairs(double *R, double *ElogqZ, int N, int K);
  void CalcRlogR_AllPairsDotV(double *R, double *V, double *ElogqZ, int N, int K);
  void CalcRlogR_SpecificPairsDotV(double *R, double *V, double *as, double *bs, 
                                double *ElogqZ, int N, int nPairs, int K);
}

// ======================================================== Custom Defs
// ======================================================== 
// ExtMat :  2-dim matrix/array externally pre-defined (via Matlab or Python)

typedef Map<ArrayXXd> ExtMat;
typedef Map<ArrayXd> ExtVec;
typedef ArrayXXd Mat;
typedef ArrayXd Vec;

// ======================================================== Main function
// ======================================================== 

void CalcRlogR_AllPairs(double *R_IN, double *ElogqZ_OUT, int N, int K) {
  ExtMat R  ( R_IN, N, K);
  ExtMat ElogqZ ( ElogqZ_OUT, K, K);

  Vec mergeR = Vec::Zero(N);
  for (int kk=0; kk<K; kk++) {
    for (int kB=kk+1; kB<K; kB++) {
      mergeR = R.col(kk) + R.col(kB);
      mergeR *= log(mergeR);
      ElogqZ(kk, kB) = mergeR.sum();
    }
  }
}


void CalcRlogR_AllPairsDotV(double *R_IN, double *V_IN, 
                            double *ElogqZ_OUT, int N, int K) {
  ExtMat R  ( R_IN, N, K);
  ExtVec V  ( V_IN, N);
  ExtMat ElogqZ ( ElogqZ_OUT, K, K);

  Vec mergeR = Vec::Zero(N);
  for (int kk=0; kk<K; kk++) {
    for (int kB=kk+1; kB<K; kB++) {
      mergeR = R.col(kk) + R.col(kB);
      mergeR *= log(mergeR);
      mergeR *= V;
      ElogqZ(kk, kB) = mergeR.sum();
    }
  }
}


void CalcRlogR_SpecificPairsDotV(double *R_IN, double *V_IN, 
                            double *a_IN, double *b_IN,
                            double *ElogqZ_OUT, int N, int nPairs, int K) {
  ExtMat R  ( R_IN, N, K);
  ExtVec V  ( V_IN, N);
  ExtVec avec  ( a_IN, nPairs);
  ExtVec bvec  ( b_IN, nPairs);
  ExtMat ElogqZ ( ElogqZ_OUT, K, K);

  Vec mergeR = Vec::Zero(N);
  for (int kk=0; kk<nPairs; kk++) {
      int kA = (int) avec(kk);
      int kB = (int) bvec(kk);
      mergeR = R.col(kA) + R.col(kB);
      mergeR *= log(mergeR);
      mergeR *= V;
      ElogqZ(kA, kB) = mergeR.sum();
  }
}


/*
void calcRlogR_vectorized(ExtMat &R, ExtMat &Z) {
    int N = R.rows();
    int K = R.cols();

    Mat R2 = Mat::Zero(N,K);
    for (int kk=0; kk<K-1; kk++) {
      R2.rightCols(K-kk) = R.rightCols(K-kk);
      //R2.rightCols(K-kk-1) += R2.col(kk);
      //R2.rightCols(K-kk-1) *= log(R2.rightCols(K-kk-1));
      Z.rightCols(K-kk) = R2.rightCols(K-kk).colwise().sum();
    }
}
*/

/*
void CalcRlogR_Vectorized(double *R_IN, double *Z_OUT, int N, int K) {
  ExtMat R (R_IN, N, K);
  ExtMat Z (Z_OUT, K, K);
  calcRlogR_vectorized(R, Z);
}
*/
