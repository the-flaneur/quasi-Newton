/// @file qnClasses.cpp
/// @brief Code for classes used in a quasi-Newton algorithm.
//
//  Created by MM on 12/21/16.
//  Copyright © 2016 MM. All rights reserved.
//

#include <iostream> // MM
#include <stdio.h>
#include "qnClasses.h"
#include <cmath>
using namespace std;

void Variable::update(const VectorXd& deltaX) {
    value = value + deltaX;
}
void ObjectiveFunc::evaluate(VectorXd x) {
    // Compute function value
    fval = 100.0*pow(x(1) - x(0)*x(0),2) + pow(1.0-x(0),2);
}
void ObjectiveGrad::evaluate(VectorXd x) {
    // Compute function and gradient values
    ObjectiveFunc::evaluate(x);
    grad << -400.0*(x(1)-x(0)*x(0))*x(0)-2.0*(1.0-x(0)), 200.0*(x(1)-x(0)*x(0));
}
void QuasiNewton::update(VectorXd deltaX, VectorXd deltaGrad) {
    // BFGS approximation to Hessian
//    matrix = matrix + deltaGrad*deltaGrad/(deltaGrad*deltaX)
//    - matrix*deltaX*matrix*deltaX/(deltaX*matrix*deltaX);
    VectorXd matrix_deltaX(matrix*deltaX);
    
    matrix = matrix + deltaGrad*deltaGrad.transpose()/(deltaGrad.dot(deltaX))
    - (matrix_deltaX)*matrix_deltaX.transpose()/(deltaX.dot(matrix_deltaX));

}
VectorXd QuasiNewton::searchDirection(VectorXd g) {
    return matrix.llt().solve(-g);
}
double Algorithm::lineSearch(const VectorXd& x,const VectorXd& dir,ObjectiveFunc obj) {
    // Back-tracking line search
    double fval_current(obj.getFval());
    double alpha(2.0);
    unsigned int iter(0);
    do {
        iter = iter + 1;
        alpha = alpha/2.0;
        obj.evaluate(x + alpha*dir);
        cout << "--- alpha = " << alpha << " f = " << obj.getFval() << endl; // MM
    } while (obj.getFval() >= fval_current && iter <= 100);
    // Update data member deltaX
    return alpha;
}
bool Algorithm::hasConverged(const VectorXd& deltaX,const double& deltaF,const ObjectiveGrad& obj) const {
    return obj.getGrad().norm() < tolerance || deltaX.norm() < tolerance
             || fabs(deltaF) < tolerance;
}

