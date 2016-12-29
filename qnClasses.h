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
    virtual void evaluate(VectorXd x);
    /// @return current objective value.
    /// @return current gradient value.
    double getFval() const {return fval;}
};

class ObjectiveGrad : public ObjectiveFunc {
private:
    /// Current gradient value.
    VectorXd grad;
public:
    /// Default constructor
    ObjectiveGrad(int N) : ObjectiveFunc(),grad(VectorXd::Zero(N)) {} 
    /// Evaluates objective and gradient.
    /// @param [in] x the current iterate.
    virtual void evaluate(VectorXd x) override;
    /// @return current gradient value.
    VectorXd getGrad() const {return grad;}
};

/// Class that represents the variable we are optimizing over.
class Variable {
private:
    /// Variable value of current iterate.
    VectorXd value;
public:
    /// Default and single-input constructor.
    /// @param [in] x0 VectorXd specifies the starting value of the variable.
    Variable(VectorXd x0) : value(x0) {};
    /// Updates the variable value.
    /// @param [in] deltaX VectorXd: search step.
    void update(const VectorXd& deltaX);
    /// @return value of variable at current iteration.
    VectorXd getVarValue() const {return value;}
};

/// Class that holds quasi-Newton algorithm quantities
class QuasiNewton {
private:
    /// QN matrix.
    MatrixXd matrix;
public:
    /// Default constructor.
    QuasiNewton(int N) : matrix(MatrixXd::Identity(N,N)) {}
    /// Updates QN matrix via de BFGS formula.
    /// @param [in] deltaX search step.
    /// @param [in] deltaGrad gradient change.
    void update(VectorXd deltaX, VectorXd deltaGrad);
    /// Computes search direction.
    /// @param [in] g current gradient.
    /// @returns search direction \f$d_k = - H_kg_k\f$, where \f$H_k\f$ is the QN matrix.
    VectorXd searchDirection(VectorXd g);
};

/// Class that holds generic (not QN-specific) quantities
class Algorithm {
private:
    // MM unsigned int iterCount;
    /// Maximum number of iterations allowed.
    unsigned int MaxIter; // use const
    /// Stopping tolerance.
    double tolerance;
public:
    /// Constuctor.
    /// @param [in] mi maximum number of iterations allowed.
    /// @param [in] tol stopping tolerance.
    Algorithm(unsigned int mi,double tol) : MaxIter(mi),tolerance(tol) {};
    /// Backtracking line search. Halves step until simple decrease occurs
    /// or line-search's MaxIter reached.
    /// @param [in] x current iterate.
    /// @param [in] dir search direction.
    /// @param [in] obj objective object.
    /// @returns steplength \f$\alpha\f$ such that the objective decerases over line \f$x_k + \alpha d_k\f$.
    double lineSearch(const VectorXd& x,const VectorXd& dir,ObjectiveFunc obj);
    /// Indicates whether convergence has occurred.
    /// @param [in] deltaX the change in variable
    /// @param [in] deltaF the change in objective value.
    /// @param [in] obj objective object.
    /// @return true if convergence has occurred, false otherwise.
    bool hasConverged(const VectorXd& deltaX,const double& deltaF,const ObjectiveGrad& obj) const;
};

#endif /* qnClasses_h */
