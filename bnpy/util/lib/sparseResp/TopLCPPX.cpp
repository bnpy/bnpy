#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <time.h>
#include "Eigen/Dense"

using namespace std;
using namespace Eigen;

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
        resetIndices();
    }

    // Helper method: reset iptr array to 0, 1, ... K-1 
    void resetIndices() {
        for (int i = 0; i < size; i++) {
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

    void findLargestL(int L) {
        assert(L >= 0);
        assert(L < this->size);
        std::nth_element(
            this->iptr,
            this->iptr + this->size - L,
            this->iptr + this->size,
            LessThanFor1DArray(this->xptr)
            );
    }
};


void findTopL_WITHSTRUCT(
    Argsortable1DArray & arrStruct,
    Arr1D_d & topLdata,
    Arr1D_i & topLinds,
    int topL,
    int K)
{
    arrStruct.resetIndices();
    arrStruct.findLargestL(topL);
    for (int j = 0; j < topL; j++) {
        int k = arrStruct.iptr[K - j - 1];
        topLinds(j) = k;
        topLdata(j) = arrStruct.xptr[k];
    }
}


void findTopL_WITHCOPY(
    Arr1D_d & scoreVec,
    Arr1D_d & tempScoreVec,
    Arr1D_d & topLdata,
    Arr1D_i & topLinds,
    int topL,
    int K)
{
    std::copy(
        scoreVec.data(),
        scoreVec.data() + K,
        tempScoreVec.data());

    // Sort the data in the temp buffer (in place)
    std::nth_element(
        tempScoreVec.data(),
        tempScoreVec.data() + K - topL,
        tempScoreVec.data() + K);
    
    // Walk thru this row and find the top "L" positions
    double pivotScore = tempScoreVec(K - topL);
    int i = 0;
    for (int k = 0; k < K; k++) {
        if (scoreVec(k) >= pivotScore) {
            topLdata(i) = scoreVec(k);
            topLinds(i) = k;
            i += 1;
        }
    }
    assert(i == topL);
}

double calcElapsedTime(timespec start_time, timespec end_time) {
    double diffSec = (double) end_time.tv_sec - start_time.tv_sec;
    double diffNano = (double) end_time.tv_nsec - start_time.tv_nsec;
    return diffSec + diffNano / 1.0e9;
}

void testSpeed_findTopL(
    int topL,
    int K,

    int nRep)
{
    if (topL >= K) {
        printf("K=%d topL=%d SKIPPED\n", K, topL);
        return;
    }
    Arr1D_d scoreVec = Arr1D_d::Random(K);
    Arr1D_d topLdata_WITHCOPY = Arr1D_d::Zero(topL);
    Arr1D_i topLinds_WITHCOPY = Arr1D_i::Zero(topL);

    Arr1D_d topLdata_WITHSTRUCT = Arr1D_d::Zero(topL);
    Arr1D_i topLinds_WITHSTRUCT = Arr1D_i::Zero(topL);

    timespec start_time, end_time;

    double elapsedSec_WITHSTRUCT = 0.0;
    double elapsedSec_WITHCOPY = 0.0;

    // Setup for struct version
    Argsortable1DArray arr = Argsortable1DArray(scoreVec.data(), K);    

    // Setup for copy version
    Arr1D_d tempScoreVec = Arr1D_d::Zero(K);

    for (int rep = 0; rep < nRep; rep++) {
        std::srand(rep); // Set random seed
        // Fill with random values
        for (int k = 0; k < K; k++) {
            int val = std::rand();
            scoreVec(k) = ((double) val / (double) RAND_MAX);
        }
        scoreVec(0) = 55.0;
        scoreVec(K-1) = 55.0; // Make a duplicate!!
        
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start_time);
        findTopL_WITHSTRUCT(arr,
            topLdata_WITHSTRUCT, topLinds_WITHSTRUCT, topL, K);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end_time);
        elapsedSec_WITHSTRUCT += calcElapsedTime(start_time, end_time);

        int found0 = 0;
        int foundKm1 = 0;
        for (int ka = 0; ka < topL; ka++) {
            //printf("%4d %6.2f\n", 
            //    topLinds_WITHSTRUCT(ka), topLdata_WITHSTRUCT(ka));
            if (topLinds_WITHSTRUCT(ka) == 0) {
                found0 = 1;
            } else if (topLinds_WITHSTRUCT(ka) == K - 1) {
                foundKm1 = 1;
            }
        }
        //printf("\n");
        assert(found0 > 0);
        assert(foundKm1 > 0);
        /*
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start_time);
        findTopL_WITHCOPY(scoreVec, tempScoreVec,
            topLdata_WITHCOPY, topLinds_WITHCOPY, topL, K);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end_time);
        elapsedSec_WITHCOPY += calcElapsedTime(start_time, end_time);

        // VERIFY
        if (rep < 10) {
            std::sort(topLinds_WITHCOPY.data(), 
                topLinds_WITHCOPY.data() + topL);
            std::sort(topLinds_WITHSTRUCT.data(), 
                topLinds_WITHSTRUCT.data() + topL);
            for (int j = 0; j < topL; j++) {
                if (topLinds_WITHCOPY(j) != topLinds_WITHSTRUCT(j)) {
                    printf("BAD!\n");
                }
            }
        }*/
    }

    printf("K=%d topL=%d\n", K, topL);
    printf("%11.6f sec for WITHSTRUCT\n", elapsedSec_WITHSTRUCT);
    printf("%11.6f sec for WITHCOPY\n", elapsedSec_WITHCOPY);
    printf("\n");

}

int main(int argc, char** argv) {
    int K = 12;
    int topL = 4;
    Arr1D_d scoreVec = Arr1D_d::Random(K);
    Arr1D_d tempScoreVec = Arr1D_d::Zero(K);
    Arr1D_d topLdata = Arr1D_d::Zero(topL);
    Arr1D_i topLinds = Arr1D_i::Zero(topL);

    Argsortable1DArray AV = Argsortable1DArray(scoreVec.data(), K);
    AV.pprint();

    printf("argsort_AscendingOrder()\n");
    AV.resetIndices();
    AV.argsort_AscendingOrder();
    AV.pprint();

    printf("argsort_DescendingOrder()\n");
    AV.resetIndices();
    AV.argsort_DescendingOrder();
    AV.pprint();
    AV.resetIndices();

    printf("findSmallestL(3)\n");
    AV.resetIndices();
    AV.findSmallestL(3);
    AV.pprint();

    printf("findLargestL(4)\n");
    AV.resetIndices();
    AV.findLargestL(4);
    AV.pprint();

    printf("");
    printf("findTopL_WITHCOPY\n");
    findTopL_WITHCOPY(scoreVec, tempScoreVec, topLdata, topLinds, topL, K);
    for (int j = 0; j < topL; j++) {
        printf(" %03d:% 5.2f ", topLinds(j), topLdata(j));
    }
    printf("\n");

    topLdata.fill(0);
    topLinds.fill(0);
    printf("");
    printf("findTopL_WITHSTRUCT\n");
    findTopL_WITHSTRUCT(AV, topLdata, topLinds, topL, K);
    for (int j = 0; j < topL; j++) {
        printf(" %03d:% 5.2f ", topLinds(j), topLdata(j));
    }
    printf("\n");

    
    printf("\nCan we modify our scores data in place?");
    printf("\n  CHANGE: Updated data with larger values at inds 7, 11\n\n");
    scoreVec(7) = 7.331;
    scoreVec(11) = 1.337;


    topLdata.fill(0);
    topLinds.fill(0);
    printf("");
    printf("findTopL_WITHCOPY\n");
    findTopL_WITHCOPY(scoreVec, tempScoreVec, topLdata, topLinds, topL, K);
    for (int j = 0; j < topL; j++) {
        printf(" %03d:% 5.2f ", topLinds(j), topLdata(j));
    }
    printf("\n");

    topLdata.fill(0);
    topLinds.fill(0);
    printf("");
    printf("findTopL_WITHSTRUCT\n");
    findTopL_WITHSTRUCT(AV, topLdata, topLinds, topL, K);
    for (int j = 0; j < topL; j++) {
        printf(" %03d:% 5.2f ", topLinds(j), topLdata(j));
    }
    printf("\n");

    int nRep = 5000;
    for (int KK = 20; KK < 2000; KK = 2 * KK) {
        testSpeed_findTopL(2,   KK, nRep);
        testSpeed_findTopL(4,   KK, nRep);
        testSpeed_findTopL(8,   KK, nRep);
        testSpeed_findTopL(16,  KK, nRep);
        testSpeed_findTopL(32,  KK, nRep);
        testSpeed_findTopL(64,  KK, nRep);
        testSpeed_findTopL(128, KK, nRep);
        testSpeed_findTopL(256, KK, nRep);
    }
    return 0;
}
