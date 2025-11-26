#include <stdio.h>
#include <iostream>
#include <fstream>
#include <time.h>
#include <omp.h>
#include <math.h>
#include <vector>
#include <malloc.h>
using namespace std;

//функция, вычисляющая определитиель матрицы (без параллельных вычислений, т.к. матрица не превосходит 4х4) методом разложения по нулевой строке
double det (vector<vector<double>> M, int n)
{
    double d=0;
    if (n > 1)
    {
        vector<vector<double>> M1(n - 1, vector<double>(n - 1));
        for (int i = 0; i < n; i++)
        {
            //построение матрицы-дополнения к элементу  M[0][i]
            for (int j = 0; j < n - 1; j++)
                for (int k = 0; k < n - 1; k++)
                {
                    if (j < i)
                        M1[k][j] = M[k + 1][j];
                    else M1[k][j] = M[k + 1][j + 1];
                };
            //разложение по нулевой строке
            d += M[0][i] * (1 - (i % 2) * 2) * det(M1, n - 1);
        };
    }
    else d = M[0][0];
    return d;
};
// функция, строящая обратную матрицу (без параллельных вычислений, т.к.матрица не превосходит 4х4)
vector<vector<double>> invM (vector<vector<double>> M, int n)
{
    vector<vector<double>> M1(n - 1, vector<double>(n - 1));//матрица-дополнение к элементу M[i][j]
    vector<vector<double>> C(n, vector<double>(n));//обратная матрица
    double D = det(M, n);
    for (int i = 0; i < n; i++)
        for (int j = i; j < n; j++)
        {
            for (int k = 0; k < n - 1; k++)
                for (int l = 0; l < n - 1; l++)
                    if ((k < i) && (l < j))
                        M1[k][l] = M[k][l];
                    else if (k < i)
                        M1[k][l] = M[k][l + 1];
                    else if (l < j)
                        M1[k][l] = M[k + 1][l];
                    else M1[k][l] = M[k + 1][l + 1];
        C[j][i] = (1 - ((i+j) % 2) * 2) * det(M1, n - 1)/ D;
        C[i][j] = C[j][i];
        }
    return C;

}

    void main()
    {
        int m;
        std::cout << "enter degree of a polynom m (0<m<=3): ";
        cin >> m;
        std::ifstream in("points.txt"); //название файла с тестом
        int i = 0;
        double* X;
        double* Y;
        double* Y1;
        X = (double*)malloc(sizeof(double));
        Y = (double*)malloc(sizeof(double));
        if (in.is_open())//чтение данных из файла
            while (in >> X[i] >> Y[i])
            {
                i++;
                X = (double*)realloc(X, (i + 1) * sizeof(double));
                Y = (double*)realloc(Y, (i + 1) * sizeof(double));
            };
        Y1 = (double*)malloc((i + 1) * sizeof(double));
        //будем искать решение A*b=B в виде b=A^{-1}*B, 
        // где:
        // B=(mean(X), mean(X*Y), mean(X^2*Y)...), вектор-результат
        // A=([1, mean(X), mean(X^2),...] [mean(X), mean(X^2), mean(X^3),...] [mean(X^2), mean(X^3), mean(X^4),...] ...) - матрица коэффициентов
        vector<vector<double>> A(m+1, vector<double>(m+1));
        vector<double> B(m+1);
        
        in.close();
        int n = i;
        vector<double> b(m + 1);//результирующий вектор коэффициентов полинома
        clock_t start = clock();
        A[0][0] = 1.0;
        double B0=0, B1=0, B2=0, B3=0, A01=0, A12=0, A23=0, A11=0, A22=0, A33=0;


#pragma omp parallel for reduction(+: B0, B1, B2, B3, A01, A12, A23, A11, A22, A33)
        for (int i = 0; i < n; i++)
        {   
            B0 += Y[i];
            B1 += X[i] * Y[i];
            A01 += pow(X[i], 1);
            A11 += pow(X[i], 2);
            if (m > 1)
            {
                B2 += pow(X[i], 2) * Y[i];
                A12 += pow(X[i], 3);
                A22 += pow(X[i], 4);
                if (m > 2)
                {
                    B3 += pow(X[i], 3) * Y[i];
                    A23 += pow(X[i], 5);
                    A33 += pow(X[i], 6);
                };
            };
        };

        B[0] = B0/n;
        B[1] = B1 / n;
        A[0][1] = A01 / n;
        A[1][0] = A[0][1];
        A[1][1] = A11 / n;
        if (m > 1)
        {
            B[2] = B2 / n;
            A[1][2] = A12 / n;
            A[2][1] = A[1][2];
            A[2][2] = A22 / n;
            A[0][2] = A[1][1];
            A[2][0] = A[1][1];
            if (m > 2)
            {
                B[3] = B3 / n;
                A[0][3] = A[1][2];
                A[3][0] = A[1][2];
                A[1][3] = A[2][2];
                A[3][1] = A[2][2];
                A[2][3] = A23 / n;
                A[3][2] = A[2][3];
                A[3][3] = A33 / n;
            };
        };


        //время, затраченное на вычисление элементов матрицы коэффициентов
        std::cout << "\n" << "time for matrix calculation, ms: " << clock() - start << "\n";


        //обращение матрицы коэффициентов
        A = invM(A, m + 1);
        //нахождение результирующего вектора коэффициентов
        for (int i = 0; i <= m; i++)
            {
            b[i] = 0;
            for (int j = 0; j <= m; j++)
                b[i] += A[i][j] * B[j];
            };
        std::cout << "\n" << "number of points: " <<n<< " \n";
        //вывод вектора коэффициентов на экран
        std::cout << "\n" << "approximating curve:  " << " \n";
        for (int i = 0; i <= m; i++)
        {
            std::cout << b[i] << "*x^"<< i<< "+";
        };

        std::cout << "\n" << "total time: " << clock() - start << " \n";
        std::cout << "\n" << "output is in a file points_out.txt of a format x, y(x)";
        std::ofstream out("points_out.txt");
        //вывод результата интерполяции в файл
        
    #pragma omp parallel for
            for (int i = 0; i < n; i++)
            {
                Y1[i] = 0.0;
                for (int j = 0; j <= m; j++)
                    Y1[i] += b[j] * pow(X[i], j);
                
            };
        if (out.is_open())
            for (int i = 0; i < n; i++)
                out << X[i] << "    " << Y1[i] << std::endl;

        out.close();
        free(X);
        free(Y);
    }

