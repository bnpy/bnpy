/* LocalStepCore.cpp
Author: Mike Hughes (www.michaelchughes.com)
Date:   May 2014
*/

#include <iostream>
#include "Eigen/Dense"

using namespace Eigen;
using namespace std;

// ======================================================== Declare funcs
// ========================================================  visible externally

extern "C" {
  void CalcDocTopicCount(int N, int D, int K, int A, int *a, int *b, double *c,                         double *d, double *e, double *f, double *g);
}

// ======================================================== Custom Defs
// ======================================================== 
// ExtMat :  2-dim matrix/array externally pre-defined (via Matlab or Python)

typedef Map<MatrixXd> ExtMat;
typedef Map<VectorXd> ExtVec;

// ======================================================== Main function
// ======================================================== 

void CalcDocTopicCount(int Ntoken, int D, int K, int A,
                       int *activeDocsIN,
                       int *dptr,
                       double *wordcountIN,
                       double *PriorIN,
                       double *LikIN, 
                       double *sumROUT,  
                       double *DocTopicCountOUT) {
  ExtMat L (LikIN, Ntoken, K);
  ExtMat P (PriorIN, D, K);
  ExtMat DocTopicCount (DocTopicCountOUT, D, K);
  ExtVec sumR (sumROUT, Ntoken);
  ExtVec wordcount (wordcountIN, Ntoken);
  VectorXd u;
  VectorXd pd;

  for (int aa=0; aa < A; aa++) {
    int d = activeDocsIN[aa];
    int Nd = dptr[d+1] - dptr[d];
    MatrixXd Ld = L.block(dptr[d],0, Nd, K);

    pd = P.row(d);
    sumR.segment(dptr[d], Nd) = Ld * pd;
    u = wordcount.segment(dptr[d], Nd).cwiseQuotient( \
                                       sumR.segment(dptr[d], Nd) \
                                       );
    DocTopicCount.row(d) = Ld.transpose() * u;
    DocTopicCount.row(d).array() *= P.row(d).array();
  }

}
