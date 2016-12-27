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

void Variable::update(const double& deltaX) {
    value = value + deltaX;
}
void ObjectiveFunc::evaluate(double x) {
    // Compute function value
    fval = x*x*x*x;
}
void ObjectiveGrad::evaluate(double x) {
    // Compute function and gradient values
    ObjectiveFunc::evaluate(x);
    grad = 4.0*x*x*x;
}
void QuasiNewton::update(double deltaX, double deltaGrad) {
    // BFGS approximation to Hessian
    matrix = matrix + deltaGrad*deltaGrad/(deltaGrad*deltaX)
    - matrix*deltaX*matrix*deltaX/(deltaX*matrix*deltaX);
}
double QuasiNewton::searchDirection(double g) {
    return -g/matrix;
}
double Algorithm::lineSearch(const double& x,const double& dir,ObjectiveFunc obj) {
    // Back-tracking line search
    double fval_current(obj.getFval());
    double alpha(2.0);
    unsigned int iter(0);
    do {
        iter = iter + 1;
        alpha = alpha/2.0;
        cout << "--- alpha = " << alpha << endl; // MM
        obj.evaluate(x + alpha*dir);
    } while (obj.getFval() >= fval_current && iter <= 10);
    // Update data member deltaX
    return alpha;
}
bool Algorithm::hasConverged(const double& deltaX,const double& deltaF,const ObjectiveGrad& obj) const {
    return fabs(obj.getGrad()) < tolerance || fabs(deltaX) < tolerance
             || fabs(deltaF) < tolerance;
}

