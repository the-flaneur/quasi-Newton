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
    VectorXd x0(N);
    x0 << -1.2, 1.0; // initial guess
    int MaxIter(100);  // max number of iterations
    double tol(1e-9);  // stopping tolerance
    // End of input list

    Algorithm alg(MaxIter,tol);
    Variable var(x0);
    ObjectiveGrad obj(N);
    QuasiNewton qn(N);
    
    VectorXd dir(N);        // search direction
    double alpha(0.0);      // line search steplength
    VectorXd deltaX(N);     // step from one iterate to the next
    double deltaF(0.0);     // objective value change
    VectorXd deltaGrad(N);  // gradient change
    double fOld(0.0);       // previous objective value
    VectorXd gradOld(N);    // previous gradient value
    unsigned int iter(0);   // MM
    
    // Evaluate objective and gradient at initial point.
    obj.evaluate(var.getVarValue());

    // Display initial point info.
    cout << iter << ") x = " << var.getVarValue().transpose() << " f = " << obj.getFval()
    << " g = " << obj.getGrad() << endl;

    do {
        iter = iter + 1; // MM
        // Compute search direction
        dir = qn.searchDirection(obj.getGrad());
        // Store current objective and grad values before
        // they get over-written by the line search.
        fOld = obj.getFval();
        gradOld = obj.getGrad();
        alpha = alg.lineSearch(var.getVarValue(),dir,obj);
        deltaX = alpha*dir;
        var.update(deltaX);
        // Evaluate gradient at new iterate (also re-computes fval)
        obj.evaluate(var.getVarValue());
        // Compute deltas
        deltaF = obj.getFval() - fOld;
        deltaGrad = obj.getGrad() - gradOld;
        // Update QN matri using BFGS formula.
        qn.update(deltaX,deltaGrad);
        // Display info.
        cout << iter << ") x = " << var.getVarValue().transpose() << " f = " << obj.getFval()
        << " g = " << obj.getGrad().transpose() << endl;
    } while (!alg.hasConverged(deltaX,deltaF,obj) && iter <= 100); // MM: the 100

    return 0;
}
