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
    ObjectiveGrad obj(N);
    QuasiNewton qn(N);
    
    Vector_ dir(N);        // search direction
    double alpha(0.0);      // line search steplength
    Vector_ deltaX(N);     // step from one iterate to the next
    Vector_ deltaGrad(N);  // gradient change
    double fOld(0.0);       // previous objective value
    Vector_ gradOld(N);    // previous gradient value
    
    // Evaluate objective and gradient at initial point.
    obj.evaluate(var.getVarValue());
    // Calculate the 2-norm of the initial gradient
    alg.setGradNorm(obj.getGrad());

    // Display initial point info.

    alg.displayIterInfo(obj);

    do {
        alg.iterCountPlusOne();
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
        // Calculate the 2-norm of the gradient
        alg.setGradNorm(obj.getGrad());
        // Compute deltas
        alg.setDeltaFval(obj.getFval() - fOld);
        deltaGrad = obj.getGrad() - gradOld;
        // Update QN matri using BFGS formula.
        qn.update(deltaX,deltaGrad);
        // Calculate the norm of deltaX
        alg.setDeltaXNorm(deltaX);
        // Display info.
        alg.displayIterInfo(obj,alpha);

    } while (!alg.hasConverged(obj) && !alg.reachedMaxIter());

    return 0;
}
