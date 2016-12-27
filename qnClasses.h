/// @file qnClasses.h
/// @brief Declaration of classes used in a quasi-Newton algorithm.
//
//  Created by MM on 12/20/16.
//  Copyright © 2016 MM. All rights reserved.
//

#ifndef qnClasses_h
#define qnClasses_h

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
    virtual void evaluate(double x);
    /// @return current objective value.
    /// @return current gradient value.
    double getFval() const {return fval;}
};

class ObjectiveGrad : public ObjectiveFunc {
private:
    /// Current gradient value.
    double grad;
public:
    /// Default constructor
    ObjectiveGrad() : ObjectiveFunc(),grad(0.0) {}
    /// Evaluates objective and gradient.
    /// @param [in] x the current iterate.
    virtual void evaluate(double x) override;
    /// @return current gradient value.
    double getGrad() const {return grad;}
};

/// Class that represents the variable we are optimizing over.
class Variable {
private:
    /// Variable value of current iterate.
    double value;
public:
    /// Default and single-input constructor.
    /// @param [in] x0 double specifies the starting value of the variable.
    Variable(double x0 = 0.0) : value(x0) {};
    /// Updates the variable value.
    /// @param [in] deltaX double: search step.
    void update(const double& deltaX);
    /// @return value of variable at current iteration.
    double getVarValue() const {return value;}
};

/// Class that holds quasi-Newton algorithm quantities
class QuasiNewton {
private:
    /// QN matrix.
    double matrix;
public:
    /// Default constructor.
    QuasiNewton() : matrix(1.0) {}
    /// Updates QN matrix via de BFGS formula.
    /// @param [in] deltaX search step.
    /// @param [in] deltaGrad gradient change.
    void update(double deltaX, double deltaGrad);
    /// Computes search direction.
    /// @param [in] g current gradient.
    /// @returns search direction \f$d_k = - H_kg_k\f$, where \f$H_k\f$ is the QN matrix.
    double searchDirection(double g);
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
    /// Line search.
    /// @param [in] x current iterate.
    /// @param [in] dir search direction.
    /// @param [in] obj objective object.
    /// @returns steplength \f$\alpha\f$ such that the objective decerases over line \f$x_k + \alpha d_k\f$.
    double lineSearch(const double& x,const double& dir,ObjectiveFunc obj);
    /// Indicates whether convergence has occurred.
    /// @param [in] deltaX the change in variable
    /// @param [in] deltaF the change in objective value.
    /// @param [in] obj objective object.
    /// @return true if convergence has occurred, false otherwise.
    bool hasConverged(const double& deltaX,const double& deltaF,const ObjectiveGrad& obj) const;
};

#endif /* qnClasses_h */
