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
    virtual void evaluate(Vector2d x);
    /// @return current objective value.
    /// @return current gradient value.
    double getFval() const {return fval;}
};

class ObjectiveGrad : public ObjectiveFunc {
private:
    /// Current gradient value.
    Vector2d grad;
public:
    /// Default constructor
    ObjectiveGrad(int N) : ObjectiveFunc(),grad(Vector2d::Zero(N)) {} 
    /// Evaluates objective and gradient.
    /// @param [in] x the current iterate.
    virtual void evaluate(Vector2d x) override;
    /// @return current gradient value.
    Vector2d getGrad() const {return grad;}
};

/// Class that represents the variable we are optimizing over.
class Variable {
private:
    /// Variable value of current iterate.
    Vector2d value;
public:
    /// Default and single-input constructor.
    /// @param [in] x0 Vector2d specifies the starting value of the variable.
    Variable(Vector2d x0) : value(x0) {};
    /// Updates the variable value.
    /// @param [in] deltaX Vector2d: search step.
    void update(const Vector2d& deltaX);
    /// @return value of variable at current iteration.
    Vector2d getVarValue() const {return value;}
};

/// Class that holds quasi-Newton algorithm quantities
class QuasiNewton {
private:
    /// QN matrix.
    Matrix2d matrix;
public:
    /// Default constructor.
    QuasiNewton(int N) : matrix(Matrix2d::Identity(N,N)) {}
    /// Updates QN matrix via de BFGS formula.
    /// @param [in] deltaX search step.
    /// @param [in] deltaGrad gradient change.
    void update(Vector2d deltaX, Vector2d deltaGrad);
    /// Computes search direction.
    /// @param [in] g current gradient.
    /// @returns search direction \f$d_k = - H_kg_k\f$, where \f$H_k\f$ is the QN matrix.
    Vector2d searchDirection(Vector2d g);
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
    double lineSearch(const Vector2d& x,const Vector2d& dir,ObjectiveFunc obj);
    /// Indicates whether convergence has occurred.
    /// @param [in] deltaX the change in variable
    /// @param [in] deltaF the change in objective value.
    /// @param [in] obj objective object.
    /// @return true if convergence has occurred, false otherwise.
    bool hasConverged(const Vector2d& deltaX,const double& deltaF,const ObjectiveGrad& obj) const;
};

#endif /* qnClasses_h */
