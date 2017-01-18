/// @file qnClasses.cpp
/// @brief Code for classes used in a quasi-Newton algorithm.
//
//  Created by MM on 12/21/16.
//  Copyright © 2016 MM. All rights reserved.
//

#include <iostream> // MM
#include <fstream>
#include <iomanip>
#include <stdio.h>
#include "qnClasses.h"
#include <cmath>
#include <cassert>
using namespace std;

void Variable::update(const Vector_& deltaX) {
    value = value + deltaX;
}

void ObjectiveFunc::evaluate(Vector_ x) {
    // Compute function value
    fval = 100.0*pow(x(1) - x(0)*x(0),2) + pow(1.0-x(0),2);
}

void ObjectiveGrad::evaluate(Vector_ x) {
    // Compute function and gradient values
    ObjectiveFunc::evaluate(x);
    grad << -400.0*(x(1)-x(0)*x(0))*x(0)-2.0*(1.0-x(0)), 200.0*(x(1)-x(0)*x(0));
}

void QuasiNewton::update(Vector_ deltaX, Vector_ deltaGrad) {
    // BFGS approximation to Hessian
//    matrix = matrix + deltaGrad*deltaGrad/(deltaGrad*deltaX)
//    - matrix*deltaX*matrix*deltaX/(deltaX*matrix*deltaX);
    Vector_ matrix_deltaX(matrix*deltaX);
    
    matrix = matrix + deltaGrad*deltaGrad.transpose()/(deltaGrad.dot(deltaX))
    - (matrix_deltaX)*matrix_deltaX.transpose()/(deltaX.dot(matrix_deltaX));

}

void QuasiNewton::solve(int N,Variable var,Algorithm alg) {
// Quasi-Newton algorithm
    
    ObjectiveGrad obj(N);
    
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
        dir = searchDirection(obj.getGrad());
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
        update(deltaX,deltaGrad);
        // Calculate the norm of deltaX
        alg.setDeltaXNorm(deltaX);
        // Display info.
        alg.displayIterInfo(obj,alpha);
        
    } while (!alg.hasConverged(obj) && !alg.reachedMaxIter());
}

Vector_ QuasiNewton::searchDirection(Vector_ g) {
    return matrix.llt().solve(-g);
}
double Algorithm::lineSearch(const Vector_& x,const Vector_& dir,ObjectiveFunc obj) {
    // Back-tracking line search
    double fval_current(obj.getFval());
    double alpha(2.0);
    unsigned int iter(0);
    do {
        iter = iter + 1;
        alpha = alpha/2.0;
        obj.evaluate(x + alpha*dir);

        // cout << "--- alpha = " << alpha << " f = " << obj.getFval() << endl; // MM

    } while (obj.getFval() >= fval_current && iter <= 100);
    // Update data member deltaX
    return alpha;
}

void Algorithm::displayIterInfo(const ObjectiveGrad& obj, double alpha) {
    std::ofstream writeOutput(outputFileName,std::ios::app); // open file for appending
    assert(writeOutput.is_open()); // verify file is open before attempting to write to it
    if (iterCount % 25 == 0) {
        // Display header periodically
        writeOutput << endl; // display a blank line
        writeOutput << setw(5) << "iter" << setw(15) << "fval" << setw(15) << "grad norm"
        << setw(15) << "step-length" << setw(15) << "step norm" << endl;
    }
    writeOutput << setw(5) << iterCount << setw(15) << obj.getFval() << setw(15);
    writeOutput << gradNorm;
    // If zero-th iteration, terminate line of output.
    if (iterCount == 0) writeOutput << endl;
    // If iteration count > 0, display additional quantities that are available.
    if (iterCount > 0) {
        writeOutput << setw(15) << alpha << setw(15) << deltaXNorm << endl;
    };
    writeOutput.close(); // close file before exiting
}

bool Algorithm::hasConverged(const ObjectiveGrad& obj) const {
    return gradNorm < tolerance || deltaXNorm < tolerance
             || fabs(deltaFval) < tolerance;
}

