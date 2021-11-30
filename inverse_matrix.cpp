#include <iostream>
#include <math.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <omp.h>

#include <sys/time.h>




const uint64_t MY_MATRIX_N = 4048;
double myMatrix[MY_MATRIX_N * MY_MATRIX_N] = { 0 };



struct MatrixDataForMultiplication
{
    // startingI: starting I of the interested area
    // startingJ: starting J of the interested area
    // iSize: the number of rows for area of interest
    // jSize: the number of columns  for area of interest
    // matrixJSize: this is the main actual matrix N number of columns, 
    // since I am not using a 2-dimensional array this is needed to make sure I index correctly.
    uint64_t startingI, startingJ, iSize, jSize, matrixJSize;
};



void printMatrix(const double matrix[], const uint64_t n)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            //printf("  %lf", matrix[j + i * n]);
            printf("  %lf", matrix[j + i * n]);
        printf("\n");
    }
}



void multiplyMatrixSerial(const double matrixA[], const MatrixDataForMultiplication &matrixAData,
    const double matrixB[], const MatrixDataForMultiplication &matrixBData,
    double result[], const MatrixDataForMultiplication &resultData, bool isPositive)
{
    for (uint64_t i = 0; i < matrixAData.iSize; i++)
    {
        for (uint64_t j = 0; j < matrixBData.jSize; j++)
        {
            uint64_t matrixAIndex = matrixAData.startingJ + (matrixAData.startingI + i) * matrixAData.matrixJSize;
            uint64_t matrixBIndex = matrixBData.startingJ + j + matrixBData.startingI * matrixBData.matrixJSize;
            uint64_t resultIndex = resultData.startingJ + j + (resultData.startingI + i) * resultData.matrixJSize;
            result[resultIndex] = 0;
            // printf("[%lu, %lu] =>", i, j);
            for (uint64_t k = 0; k < matrixAData.jSize; k++)
            {
                // printf("%.2lf X %.2lf + ", matrixA[matrixAIndex], matrixB[matrixBIndex]);
                result[resultIndex] += matrixA[matrixAIndex] * matrixB[matrixBIndex];
                matrixAIndex++;
                matrixBIndex += matrixBData.matrixJSize;
            }

            if (!isPositive)
            {
                result[resultIndex] = -result[resultIndex];
            }
            // printf("= %.2lf \n", result[resultIndex]);
        }
    }
}



void multiplyMatrixParallel(const double matrixA[], const MatrixDataForMultiplication &matrixAData,
    const double matrixB[], const MatrixDataForMultiplication &matrixBData,
    double result[], const MatrixDataForMultiplication &resultData, bool isPositive)
{
    uint64_t i, j, k;
    uint64_t matrixAIndex, matrixBIndex, resultIndex;
    #pragma omp for nowait collapse(2) private(i, j, k, matrixAIndex, matrixBIndex, resultIndex)
        for (i = 0; i < matrixAData.iSize; i++)
        {
            for (j = 0; j < matrixBData.jSize; j++)
            {
                matrixAIndex = matrixAData.startingJ + (matrixAData.startingI + i) * matrixAData.matrixJSize;
                matrixBIndex = matrixBData.startingJ + j + matrixBData.startingI * matrixBData.matrixJSize;
                resultIndex = resultData.startingJ + j + (resultData.startingI + i) * resultData.matrixJSize;
                result[resultIndex] = 0;
                // printf("[%lu, %lu] =>", i, j);
                for (k = 0; k < matrixAData.jSize; k++)
                {
                    // printf("%.2lf X %.2lf + ", matrixA[matrixAIndex], matrixB[matrixBIndex]);
                    result[resultIndex] += matrixA[matrixAIndex] * matrixB[matrixBIndex];
                    matrixAIndex++;
                    matrixBIndex += matrixBData.matrixJSize;
                }

                if (!isPositive)
                {
                    result[resultIndex] = -result[resultIndex];
                }
                // printf("= %.2lf \n", result[resultIndex]);
            }
        }
}



// startingI: starting I of the interested area
// startingJ: starting J of the interested area
// n: the N for area of interest
// mainMatrixN: this is the main actual matrix N, since I am not using a 2-dimensional array this is needed to make sure I index correctly.
bool computeInverseSerial(const double matrix[], double inverseMatrix[],
    const uint64_t startingI, const uint64_t startingJ, const uint64_t n,
    const uint64_t mainMatrixN)
{
    if (n == 1)
    {
        uint64_t matrixIndex = startingJ + startingI * mainMatrixN;
        inverseMatrix[matrixIndex] = 1 / matrix[matrixIndex];
        return true;
    }

    uint64_t newN = static_cast<uint64_t>(round(static_cast<double>(n) / 2));

    computeInverseSerial(matrix, inverseMatrix, startingI, startingJ, newN, mainMatrixN);
    computeInverseSerial(matrix, inverseMatrix, startingI + newN, startingJ + newN, n - newN, mainMatrixN);

    double* temp = new double[(n - newN) * (newN)];

    // uint64_t startingI, startingJ, iSize, jSize, matrixJSize; // matrixJ size is used for C++ 2 dimension array behaviour on 1 dimension array
    struct MatrixDataForMultiplication inverseR11 = { startingI, startingJ, newN, newN, mainMatrixN };
    struct MatrixDataForMultiplication inverseR22 = { startingI + newN, startingJ + newN, n - newN, n - newN, mainMatrixN };
    struct MatrixDataForMultiplication r12 = { startingI, startingJ + newN, newN, n - newN, mainMatrixN };
    struct MatrixDataForMultiplication tempMatrixData = { 0, 0, newN, n - newN, n - newN };
    struct MatrixDataForMultiplication inverse12 = { startingI, startingJ + newN, newN, n - newN, mainMatrixN };

    multiplyMatrixSerial(inverseMatrix, inverseR11, matrix, r12, temp, tempMatrixData, false);
    multiplyMatrixSerial(temp, tempMatrixData, inverseMatrix, inverseR22, inverseMatrix, inverse12, true);

    delete[] temp;

    return true;
}



// startingI: starting I of the interested area
// startingJ: starting J of the interested area
// n: the N for area of interest
// mainMatrixN: this is the main actual matrix N, 
// since I am not using a 2-dimensional array this is needed to make sure I index correctly.
bool computeInverseParallel(const double matrix[], double inverseMatrix[],
    const uint64_t startingI, const uint64_t startingJ, const uint64_t n,
    const uint64_t mainMatrixN)
{
    if (n < 16)
    {
        computeInverseSerial(matrix, inverseMatrix, startingI, startingJ, n, mainMatrixN);
        return true;
    }

    uint64_t newN = static_cast<uint64_t>(round(static_cast<double>(n) / 2));


    #pragma omp task shared(inverseMatrix) firstprivate(matrix, startingI, startingJ, newN, mainMatrixN) default(none)  
    {
        //printf("1 T_ID: %d\n", omp_get_thread_num());
        computeInverseParallel(matrix, inverseMatrix, startingI, startingJ, newN, mainMatrixN);
    }
    #pragma omp task shared(inverseMatrix) firstprivate(matrix, startingI, startingJ, n, newN, mainMatrixN) default(none) 
    {
        // printf("2 T_ID: %d\n", omp_get_thread_num());
        computeInverseParallel(matrix, inverseMatrix, startingI + newN, startingJ + newN, n - newN, mainMatrixN);
    }

    double* temp = new double[(n - newN) * (newN)];

    // uint64_t startingI, startingJ, iSize, jSize, matrixJSize; // matrixJ size is used for C++ 2 dimension array behaviour on 1 dimension array
    struct MatrixDataForMultiplication inverseR11 = { startingI, startingJ, newN, newN, mainMatrixN };
    struct MatrixDataForMultiplication inverseR22 = { startingI + newN, startingJ + newN, n - newN, n - newN, mainMatrixN };
    struct MatrixDataForMultiplication r12 = { startingI, startingJ + newN, newN, n - newN, mainMatrixN };
    struct MatrixDataForMultiplication tempMatrixData = { 0, 0, newN, n - newN, n - newN };
    struct MatrixDataForMultiplication inverse12 = { startingI, startingJ + newN, newN, n - newN, mainMatrixN };

    #pragma omp taskwait
    multiplyMatrixParallel(inverseMatrix, inverseR11, matrix, r12, temp, tempMatrixData, false);
    multiplyMatrixParallel(temp, tempMatrixData, inverseMatrix, inverseR22, inverseMatrix, inverse12, true);

    delete[] temp;

    return true;
}



void generateMatrix(double matrix[], const uint64_t n)
{
    uint64_t index = 0;
    for (uint64_t i = 0; i < n; i++)
    {
        index = i * n + i;
        matrix[index] = n + (double)rand() / (double)RAND_MAX;
        for (uint64_t j = i + 1; j < n; j++)
        {
            index++;
            matrix[index] = (double)rand() / (double)RAND_MAX;
        }
    }
}




int main()
{
    struct timeval start, stop;
    double total_time;
    double inverseMatrix[MY_MATRIX_N * MY_MATRIX_N] = { 0 };
    uint64_t n;

    #pragma omp parallel
    {
        int np = omp_get_num_threads();
        int myid = omp_get_thread_num();
        printf("From process # %d out of %d!\n", myid, np);
    }
    printf("End of testing number of threads!\n\n");



    n = 15;
    printf("\ntesting for n = %lu\n", n);
    for (int i = 0; i < 5; i++)
    {
        generateMatrix(myMatrix, n);

        uint64_t startingI = 0;
        uint64_t startingJ = 0;

        gettimeofday(&start, NULL); 
        #pragma omp parallel shared(myMatrix, inverseMatrix, n, startingI, startingJ) num_threads(1)
        {
            #pragma omp single
            {
                computeInverseParallel(myMatrix, inverseMatrix, startingI, startingJ, n, n);
            }
        }
        gettimeofday(&stop, NULL); 

        total_time = (stop.tv_sec-start.tv_sec)+0.000001*(stop.tv_usec-start.tv_usec);


        printf("iteration %d : time it took %lf\n", i, total_time);
    }


    n = 15 * 2;
    printf("\ntesting for n = %lu\n", n);
    for (int i = 0; i < 5; i++)
    {
        generateMatrix(myMatrix, n);

        uint64_t startingI = 0;
        uint64_t startingJ = 0;

        gettimeofday(&start, NULL); 
        #pragma omp parallel shared(myMatrix, inverseMatrix, n, startingI, startingJ) num_threads(1)
        {
            #pragma omp single
            {
                computeInverseParallel(myMatrix, inverseMatrix, startingI, startingJ, n, n);
            }
        }
        gettimeofday(&stop, NULL); 

        total_time = (stop.tv_sec-start.tv_sec)+0.000001*(stop.tv_usec-start.tv_usec);


        printf("iteration %d : time it took %lf\n", i, total_time);
    }



    n = 15 * 4;
    printf("\ntesting for n = %lu\n", n);
    for (int i = 0; i < 5; i++)
    {
        generateMatrix(myMatrix, n);

        uint64_t startingI = 0;
        uint64_t startingJ = 0;

        gettimeofday(&start, NULL); 
        #pragma omp parallel shared(myMatrix, inverseMatrix, n, startingI, startingJ) num_threads(1)
        {
            #pragma omp single
            {
                computeInverseParallel(myMatrix, inverseMatrix, startingI, startingJ, n, n);
            }
        }
        gettimeofday(&stop, NULL); 

        total_time = (stop.tv_sec-start.tv_sec)+0.000001*(stop.tv_usec-start.tv_usec);


        printf("iteration %d : time it took %lf\n", i, total_time);
    }


    n = 100;
    printf("\ntesting for n = %lu\n", n);
    for (int i = 0; i < 5; i++)
    {
        generateMatrix(myMatrix, n);

        uint64_t startingI = 0;
        uint64_t startingJ = 0;

        gettimeofday(&start, NULL); 
        #pragma omp parallel shared(myMatrix, inverseMatrix, n, startingI, startingJ) num_threads(1)
        {
            #pragma omp single
            {
                computeInverseParallel(myMatrix, inverseMatrix, startingI, startingJ, n, n);
            }
        }
        gettimeofday(&stop, NULL); 

        total_time = (stop.tv_sec-start.tv_sec)+0.000001*(stop.tv_usec-start.tv_usec);


        printf("iteration %d : time it took %lf\n", i, total_time);
    }



    n = 128;
    printf("\ntesting for n = %lu\n", n);
    for (int i = 0; i < 5; i++)
    {
        generateMatrix(myMatrix, n);

        uint64_t startingI = 0;
        uint64_t startingJ = 0;

        gettimeofday(&start, NULL); 
        #pragma omp parallel shared(myMatrix, inverseMatrix, n, startingI, startingJ) num_threads(1)
        {
            #pragma omp single
            {
                computeInverseParallel(myMatrix, inverseMatrix, startingI, startingJ, n, n);
            }
        }
        gettimeofday(&stop, NULL); 

        total_time = (stop.tv_sec-start.tv_sec)+0.000001*(stop.tv_usec-start.tv_usec);


        printf("iteration %d : time it took %lf\n", i, total_time);
    }



    n = 15 * 8;
    printf("\ntesting for n = %lu\n", n);
    for (int i = 0; i < 5; i++)
    {
        generateMatrix(myMatrix, n);

        uint64_t startingI = 0;
        uint64_t startingJ = 0;

        gettimeofday(&start, NULL); 
        #pragma omp parallel shared(myMatrix, inverseMatrix, n, startingI, startingJ) num_threads(1)
        {
            #pragma omp single
            {
                computeInverseParallel(myMatrix, inverseMatrix, startingI, startingJ, n, n);
            }
        }
        gettimeofday(&stop, NULL); 

        total_time = (stop.tv_sec-start.tv_sec)+0.000001*(stop.tv_usec-start.tv_usec);


        printf("iteration %d : time it took %lf\n", i, total_time);
    }


    n = 15 * 16;
    printf("\ntesting for n = %lu\n", n);
    for (int i = 0; i < 5; i++)
    {
        generateMatrix(myMatrix, n);

        uint64_t startingI = 0;
        uint64_t startingJ = 0;

        gettimeofday(&start, NULL); 
        #pragma omp parallel shared(myMatrix, inverseMatrix, n, startingI, startingJ) num_threads(1)
        {
            #pragma omp single
            {
                computeInverseParallel(myMatrix, inverseMatrix, startingI, startingJ, n, n);
            }
        }
        gettimeofday(&stop, NULL); 

        total_time = (stop.tv_sec-start.tv_sec)+0.000001*(stop.tv_usec-start.tv_usec);


        printf("iteration %d : time it took %lf\n", i, total_time);
    }


    n = 15 * 32;
    printf("\ntesting for n = %lu\n", n);
    for (int i = 0; i < 5; i++)
    {
        generateMatrix(myMatrix, n);

        uint64_t startingI = 0;
        uint64_t startingJ = 0;

        gettimeofday(&start, NULL); 
        #pragma omp parallel shared(myMatrix, inverseMatrix, n, startingI, startingJ) num_threads(1)
        {
            #pragma omp single
            {
                computeInverseParallel(myMatrix, inverseMatrix, startingI, startingJ, n, n);
            }
        }
        gettimeofday(&stop, NULL); 

        total_time = (stop.tv_sec-start.tv_sec)+0.000001*(stop.tv_usec-start.tv_usec);


        printf("iteration %d : time it took %lf\n", i, total_time);
    }


    n = 15 * 64;
    printf("\ntesting for n = %lu\n", n);
    for (int i = 0; i < 5; i++)
    {
        generateMatrix(myMatrix, n);

        uint64_t startingI = 0;
        uint64_t startingJ = 0;

        gettimeofday(&start, NULL); 
        #pragma omp parallel shared(myMatrix, inverseMatrix, n, startingI, startingJ) num_threads(1)
        {
            #pragma omp single
            {
                computeInverseParallel(myMatrix, inverseMatrix, startingI, startingJ, n, n);
            }
        }
        gettimeofday(&stop, NULL); 

        total_time = (stop.tv_sec-start.tv_sec)+0.000001*(stop.tv_usec-start.tv_usec);


        printf("iteration %d : time it took %lf\n", i, total_time);
    }



    n = 1000;
    printf("\ntesting for n = %lu\n", n);
    for (int i = 0; i < 5; i++)
    {
        generateMatrix(myMatrix, n);

        uint64_t startingI = 0;
        uint64_t startingJ = 0;

        gettimeofday(&start, NULL); 
        #pragma omp parallel shared(myMatrix, inverseMatrix, n, startingI, startingJ) num_threads(1)
        {
            #pragma omp single
            {
                computeInverseParallel(myMatrix, inverseMatrix, startingI, startingJ, n, n);
            }
        }
        gettimeofday(&stop, NULL); 

        total_time = (stop.tv_sec-start.tv_sec)+0.000001*(stop.tv_usec-start.tv_usec);


        printf("iteration %d : time it took %lf\n", i, total_time);
    }



    n = 1024;
    printf("\ntesting for n = %lu\n", n);
    for (int i = 0; i < 5; i++)
    {
        generateMatrix(myMatrix, n);

        uint64_t startingI = 0;
        uint64_t startingJ = 0;

        gettimeofday(&start, NULL); 
        #pragma omp parallel shared(myMatrix, inverseMatrix, n, startingI, startingJ) num_threads(1)
        {
            #pragma omp single
            {
                computeInverseParallel(myMatrix, inverseMatrix, startingI, startingJ, n, n);
            }
        }
        gettimeofday(&stop, NULL); 

        total_time = (stop.tv_sec-start.tv_sec)+0.000001*(stop.tv_usec-start.tv_usec);


        printf("iteration %d : time it took %lf\n", i, total_time);
    }



    n = 15 * 128;
    printf("\ntesting for n = %lu\n", n);
    for (int i = 0; i < 5; i++)
    {
        generateMatrix(myMatrix, n);

        uint64_t startingI = 0;
        uint64_t startingJ = 0;

        gettimeofday(&start, NULL); 
        #pragma omp parallel shared(myMatrix, inverseMatrix, n, startingI, startingJ) num_threads(1)
        {
            #pragma omp single
            {
                computeInverseParallel(myMatrix, inverseMatrix, startingI, startingJ, n, n);
            }
        }
        gettimeofday(&stop, NULL); 

        total_time = (stop.tv_sec-start.tv_sec)+0.000001*(stop.tv_usec-start.tv_usec);


        printf("iteration %d : time it took %lf\n", i, total_time);
    }



    n = 2000;
    printf("\ntesting for n = %lu\n", n);
    for (int i = 0; i < 5; i++)
    {
        generateMatrix(myMatrix, n);

        uint64_t startingI = 0;
        uint64_t startingJ = 0;

        gettimeofday(&start, NULL); 
        #pragma omp parallel shared(myMatrix, inverseMatrix, n, startingI, startingJ) num_threads(1)
        {
            #pragma omp single
            {
                computeInverseParallel(myMatrix, inverseMatrix, startingI, startingJ, n, n);
            }
        }
        gettimeofday(&stop, NULL); 

        total_time = (stop.tv_sec-start.tv_sec)+0.000001*(stop.tv_usec-start.tv_usec);


        printf("iteration %d : time it took %lf\n", i, total_time);
    }



    n = 2048;
    printf("\ntesting for n = %lu\n", n);
    for (int i = 0; i < 5; i++)
    {
        generateMatrix(myMatrix, n);

        uint64_t startingI = 0;
        uint64_t startingJ = 0;

        gettimeofday(&start, NULL); 
        #pragma omp parallel shared(myMatrix, inverseMatrix, n, startingI, startingJ) num_threads(1)
        {
            #pragma omp single
            {
                computeInverseParallel(myMatrix, inverseMatrix, startingI, startingJ, n, n);
            }
        }
        gettimeofday(&stop, NULL); 

        total_time = (stop.tv_sec-start.tv_sec)+0.000001*(stop.tv_usec-start.tv_usec);


        printf("iteration %d : time it took %lf\n", i, total_time);
    }



    n = 15 * 256;
    printf("\ntesting for n = %lu\n", n);
    for (int i = 0; i < 5; i++)
    {
        generateMatrix(myMatrix, n);

        uint64_t startingI = 0;
        uint64_t startingJ = 0;

        gettimeofday(&start, NULL); 
        #pragma omp parallel shared(myMatrix, inverseMatrix, n, startingI, startingJ) num_threads(1)
        {
            #pragma omp single
            {
                computeInverseParallel(myMatrix, inverseMatrix, startingI, startingJ, n, n);
            }
        }
        gettimeofday(&stop, NULL); 

        total_time = (stop.tv_sec-start.tv_sec)+0.000001*(stop.tv_usec-start.tv_usec);


        printf("iteration %d : time it took %lf\n", i, total_time);
    }


    n = 4000;
    printf("\ntesting for n = %lu\n", n);
    for (int i = 0; i < 5; i++)
    {
        generateMatrix(myMatrix, n);

        uint64_t startingI = 0;
        uint64_t startingJ = 0;

        gettimeofday(&start, NULL); 
        #pragma omp parallel shared(myMatrix, inverseMatrix, n, startingI, startingJ) num_threads(1)
        {
            #pragma omp single
            {
                computeInverseParallel(myMatrix, inverseMatrix, startingI, startingJ, n, n);
            }
        }
        gettimeofday(&stop, NULL); 

        total_time = (stop.tv_sec-start.tv_sec)+0.000001*(stop.tv_usec-start.tv_usec);


        printf("iteration %d : time it took %lf\n", i, total_time);
    }



    n = 4048;
    printf("\ntesting for n = %lu\n", n);
    for (int i = 0; i < 5; i++)
    {
        generateMatrix(myMatrix, n);

        uint64_t startingI = 0;
        uint64_t startingJ = 0;

        gettimeofday(&start, NULL); 
        #pragma omp parallel shared(myMatrix, inverseMatrix, n, startingI, startingJ) num_threads(1)
        {
            #pragma omp single
            {
                computeInverseParallel(myMatrix, inverseMatrix, startingI, startingJ, n, n);
            }
        }
        gettimeofday(&stop, NULL); 

        total_time = (stop.tv_sec-start.tv_sec)+0.000001*(stop.tv_usec-start.tv_usec);


        printf("iteration %d : time it took %lf\n", i, total_time);
    }
}
