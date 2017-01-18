//  @file main.cpp
//  @brief Quasi-Newton algorithm.
//
//  Created by MM on 12/20/16.
//  Copyright © 2016 MM. All rights reserved.
//

#include <iostream>
#include "qnClasses.h"
using namespace std;
#include <Eigen/Dense>
using namespace Eigen;

int main(int argc, const char * argv[]) {

    // Input list:
    int N(2);   // number of variables
    Vector_ x0(N);
    x0 << -1.2, 1.0; // initial guess
    Algorithm alg(100,1e-9); // max iter, stopping tolerance
    // End of input list
    
    Variable var(x0);
    QuasiNewton qn(N);
    
    qn.solve(N,var,alg);
    
    return 0;
}
