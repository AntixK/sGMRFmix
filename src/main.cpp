#include <iostream>
#include <armadillo>
#include <chrono>
#include "sgmrfmix/sGMRFmix.h"


using namespace std;


//void blank_test(){
//    chrono::steady_clock sc;
//
//    int K = 2, M = 5, N = 20;
//
//    arma::Mat<double> X(N, M, arma::fill::ones);
//    double rho = 0.5;
//    arma::rowvec m0(M, arma::fill::ones);
//
//    auto start = sc.now();     // start timer
//    sGMRFmix(X,K, rho, m0, false, 0.01,1.0, 1500, 1e-1, true);
//    auto end = sc.now();
//    auto time_span = static_cast<chrono::duration<double>>(end - start);
//    cout << "Operation took: " << time_span.count() << " seconds"<<endl;
//}

void toy_test(){
    chrono::steady_clock sc;
    int K = 5, M = 5, N = 1000;

    arma::Mat<double> train(N, M),
                      test(N, M);
    train.load("../train.csv", csv_ascii);
    test.load("../test.csv", csv_ascii);

//    cout<<arma::approx_equal(train, test, "absdiff", 1e-1)<<endl;

    train.shed_col(0);
    test.shed_col(0);

    train.shed_row(0);
    test.shed_row(0);

    cout<<"Size(Train):"<<size(train)<<"\nSize(Test):"<<size(test)<<endl;

    double rho = 0.8,
           pi_threshold=(double)((1./K)/100.0);
    arma::rowvec m0(M, arma::fill::zeros);

    Cube<double> A;
    Mat<double> m;
    Mat<double> g_mat;
    auto start = sc.now();     // start timer
    sGMRFmix(train, K, rho, m0, A, m,  g_mat, false, pi_threshold,1.0, 500, 1e-1, true, 314);
    Mat<double> anomaly_score(N, M, fill::zeros);
    compute_anomaly_score(test, A, m, g_mat, anomaly_score, true);

//    cout<< anomaly_score.is_zero()<<endl;

    anomaly_score.save("anomaly.csv", csv_ascii);
    auto end = sc.now();
    auto time_span = static_cast<chrono::duration<double>>(end - start);
    cout << "Operation took: " << time_span.count() << " seconds"<<endl;
}

void real_test(){}


    int main() {
//   blank_test();
   toy_test();
//    vec q = {};
//    arma::mat M1 = {{  0.2821,   0.3611,  -0.1907,  -0.1455,  -0.2434},
//                    { 0.3611,   0.6788,  -0.1226,  -0.4604,  -0.1754},
//                    {-0.1907,  -0.1226,   0.3724,  -0.1257,  -0.0536},
//                    {-0.1455,  -0.4604,  -0.1257,   0.4503,   0.0712},
//                    {-0.2434,  -0.1754,  -0.0536,   0.0712,   0.7906}};
//    arma::SpMat<double> M2 = GLasso(M1, 0.1);
//    M2.print("M2=");
//
//    M1.print("M1=");
//
//    cout<<M1.is_symmetric()<<endl;
//
//    arma::mat M2 = arma::inv(M1);
//    M2.print("M2");
//
//    cout<<M2.is_symmetric()<<endl;
//        arma::Mat<double> train(20, 5);
//        train.load("../train.csv", csv_ascii);
//        train.shed_col(0);
//        train.shed_row(0);
//arma::mat L = {{0.38210195,  0.26107560, -0.09069755, -0.11389571, -0.14344540},
//                {0.26107560,  0.77879649, -0.03311371, -0.36043209, -0.09801073},
//                            {-0.09069755, -0.03311371,  0.47236249, -0.02568835,  0.03404889},
//                                        {-0.11389571, -0.36043209, -0.02568835,  0.55030498,  0.04275774},
//                                                    {-0.14344540, -0.09801073,  0.03404889,  0.04275774,  0.89063938}};
//    arma::rowvec m = {-1.0249093,  0.1480137,  0.1874360, -0.6168614, -0.2418323};
//    arma::vec y = dmvnorm(train, m, L, true);
//    y.print("y=");
//    train.print();
//    cout<<M1.is_symmetric()<<endl;
//    to_symmetric(M1);
//    cout<<M1.is_symmetric()<<endl;
    return 0;
}