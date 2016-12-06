/* ForwardBackward.cpp
Fast implementation of forward-backward algorithm
*/
#include <math.h>
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


// ======================================================== calcForwardAsgnProbs
// ========================================================
/* 
  calcForwardAsgnProbs
  
  Args
  --------
  T : int
  K : int
  piInit : vector, size K
  Pi  : matrix, size KxK
  Lik : matrix, size TxK  
  
  Returns
  --------
  alpha : T x K matrix,
          alpha[t,k] := p( x[t], z[1], z[2], ... z[t]=k )
*/
void calcForwardAsgnProbs(int T, int K, 
                          double* piInit_IN,
                          double* pi_IN,
                          double* Lik_IN,
                          double* alpha_OUT,
                          double* aScale_OUT,
                          ) {
  
  ExtVec piInit  (piInit_IN, K);
  ExtMat pi  (pi_IN, K, K);
  ExtMat Lik (Lik_IN, T, K);
  ExtMat alpha (alpha_OUT, T, K);
  ExtMat aScale (aScale_OUT, T, K);

  Mat piTmatrix = pi.matrix().transpose();

  alpha.row(0) = piInit * Lik.row(0);
  aScale(0) = alpha.row(0).sum();
  alpha.row(0) /= aScale(0);

  for (int t = 1; t < T; t++) {
    alpha.row(t) = (piTmatrix * alpha.row(t-1).matrix()).array();
    alpha.row(t) *= Lik.row(t);
    aScale(t) = alpha.row(t).sum();
    alpha.row(t) /= aScale(t);
  }  
}

/*
void 
FilterFwd(const Eigen::Map<MatrixType>& transmat, const Eigen::Map<MatrixType>& softev, 
          const Eigen::Map<VectorType>& init, double& loglik, MatrixType& alpha)
{
    int T = (int) softev.cols();
    int K = (int) softev.rows();

    if (alpha.cols() != T && alpha.rows() != K) {
        alpha.resize(K, T);
    }
    VectorType scale = VectorType::Zero(T);
    Eigen::MatrixXd at = transmat.matrix().transpose();

    alpha.col(0) = init * softev.col(0);
    scale(0) = alpha.col(0).sum();
    alpha.col(0) /= scale(0);

    for (int t = 1; t < T; ++t) {
        alpha.col(t) = (at.matrix() * alpha.col(t-1).matrix()).array();
        alpha.col(t) *= softev.col(t);
        scale(t) = alpha.col(t).sum();
        alpha.col(t) /= scale(t);
    }
    loglik = scale.log().sum();
}

//
// Returns alpha and loglik
// [alpha, loglik] = function FilterFwdC(transmat, softev, init)
//
// alpha is [K x T]
// loglik is [1x1]
//
// softev is [K x T]
// transmat is [K x K]
// init is [K x 1]  
//
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{
	if (nrhs != 3) {
		mexErrMsgTxt("Needs 3 arguments -- transmat, softev, init");
		return;
	}

	double* A = mxGetPr(prhs[0]);	
	double* D = mxGetPr(prhs[1]);
	double* Pi = mxGetPr(prhs[2]);

	const mwSize* A_dims = mxGetDimensions(prhs[0]);
	const mwSize* D_dims = mxGetDimensions(prhs[1]);

	int K = D_dims[0];	
	int T = D_dims[1];

	if (K != A_dims[0]) {
		mexErrMsgTxt("Softev must be K x T");
		return;
	}

	double loglik;

	Eigen::Map<MatrixType> softev(D, K, T);
	Eigen::Map<VectorType> init(Pi, K);
	Eigen::Map<MatrixType> transmat(A, K, K);
	MatrixType alpha = MatrixType::Zero(K,T);

	FilterFwd(transmat, softev, init, loglik, alpha);
	
	double* outputToolPtr;
	plhs[0] = mxCreateDoubleMatrix(K, T, mxREAL);
	outputToolPtr = mxGetPr(plhs[0]);
	memcpy(outputToolPtr, alpha.data(), K*T*sizeof(double));

	plhs[1] = mxCreateDoubleMatrix(1, 1, mxREAL);
	outputToolPtr = mxGetPr(plhs[1]);
	outputToolPtr[0] = loglik;
}
*/
