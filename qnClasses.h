/// @file qnClasses.h
/// @brief Declaration of classes used in a quasi-Newton algorithm.
//
//  Created by MM on 12/20/16.
//  Copyright © 2016 MM. All rights reserved.
//

#ifndef qnClasses_h
#define qnClasses_h
#include <Eigen/Dense>
using namespace Eigen;

// Convenience typedef's for debugging: replace X by the number
// of variables (e.g. VectorXd by Vector2d) in order to be able
// to inspect vector and matrix values in a debugging session.
typedef VectorXd Vector_;
typedef MatrixXd Matrix_;

/// Problem definition
class ObjectiveFunc {
protected:
    /// Current objective value.
    double fval;
public:
    /// Default constructor
    ObjectiveFunc() : fval(0.0) {}
    /// Evaluates objective.
    /// @param [in] x the current iterate.
    virtual void evaluate(Vector_ x);
    /// @return current objective value.
    /// @return current gradient value.
    double getFval() const {return fval;}
};

class ObjectiveGrad : public ObjectiveFunc {
private:
    /// Current gradient value.
    Vector_ grad;
public:
    /// Default constructor
    ObjectiveGrad(int N) : ObjectiveFunc(),grad(Vector_::Zero(N)) {}
    /// Evaluates objective and gradient.
    /// @param [in] x the current iterate.
    virtual void evaluate(Vector_ x) override;
    /// @return current gradient value.
    Vector_ getGrad() const {return grad;}
};

/// Class that represents the variable we are optimizing over.
class Variable {
private:
    /// Variable value of current iterate.
    Vector_ value;
public:
    /// Default and single-input constructor.
    /// @param [in] x0 Vector_ specifies the starting value of the variable.
    Variable(Vector_ x0) : value(x0) {};
    /// Updates the variable value.
    /// @param [in] deltaX Vector_: search step.
    void update(const Vector_& deltaX);
    /// @return value of variable at current iteration.
    Vector_ getVarValue() const {return value;}
};

/// Class that holds quasi-Newton algorithm quantities
class QuasiNewton {
private:
    /// QN matrix.
    Matrix_ matrix;
public:
    /// Default constructor.
    QuasiNewton(int N) : matrix(Matrix_::Identity(N,N)) {}
    /// Updates QN matrix via de BFGS formula.
    /// @param [in] deltaX search step.
    /// @param [in] deltaGrad gradient change.
    void update(Vector_ deltaX, Vector_ deltaGrad);
    /// Computes search direction.
    /// @param [in] g current gradient.
    /// @returns search direction \f$d_k = - H_kg_k\f$, where \f$H_k\f$ is the QN matrix.
    Vector_ searchDirection(Vector_ g);
};

/// Class that holds generic (not QN-specific) quantities
class Algorithm {
private:
    /// Iteration counter.
    unsigned int iterCount;
    /// Maximum number of iterations allowed.
    unsigned int maxIter;
    /// Stopping tolerance.
    double tolerance;
    /// Norm-2 of gradient
    double gradNorm;
    /// Change in objective value.
    double deltaFval;
    /// Norm of change in variables (norm of step)
    double deltaXNorm;
public:
    /// Constuctor.
    /// @param [in] mi maximum number of iterations allowed.
    /// @param [in] tol stopping tolerance.
    Algorithm(unsigned int mi,double tol) : iterCount(0),maxIter(mi),tolerance(tol),gradNorm(0.0),
    deltaFval(0.0),deltaXNorm(0.0) {};
    /// Backtracking line search. Halves step until simple decrease occurs
    /// or line-search's MaxIter reached.
    /// @param [in] x current iterate.
    /// @param [in] dir search direction.
    /// @param [in] obj objective object.
    /// @returns steplength \f$\alpha\f$ such that the objective decerases over line \f$x_k + \alpha d_k\f$.
    double lineSearch(const Vector_& x,const Vector_& dir,ObjectiveFunc obj);
    /// Iterative display.
    /// @param [in] obj is an ObjectiveGrad object.
    /// @param [in] alpha line search steplength. Default value = 0.0 provided because alpha is not available
    /// in zero-th iteration.
    void displayIterInfo(const ObjectiveGrad& obj, double alpha = 0.0);
    /// Indicates whether convergence has occurred.
    /// @param [in] obj objective object.
    /// @return true if convergence has occurred, false otherwise.
    bool hasConverged(const ObjectiveGrad& obj) const;
    /// Incraese iteration counter by one.
    void iterCountPlusOne() {iterCount = iterCount + 1;}
    /// Check whether maximum number of iterations allowed has been reached.
    /// @return true if maximum has been reached, false otherwise.
    bool reachedMaxIter() {return iterCount >= maxIter;}
    /// Set norm of gradient.
    void setGradNorm(Vector_ grad) {gradNorm = grad.norm();}
    /// Set change in objective value.
    void setDeltaFval(double df) {deltaFval = df;}
    /// Set norm of change in variables (norm of step).
    void setDeltaXNorm(const Vector_& deltaX) {deltaXNorm = deltaX.norm();}
};

#endif /* qnClasses_h */
