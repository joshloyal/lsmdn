#include "procrustes.h"

namespace lsmdn {
    arma::mat center(const arma::mat &X) {
        arma::mat out = X.each_row() - arma::mean(X);

        return out;
    }

    arma::mat procrustes(const arma::mat &X, const arma::mat &Y) {
        // center X and Y
        arma::mat X_center = center(X);
        arma::mat Y_center = center(Y);

        arma::mat XtY = X_center.t() * Y_center;

        arma::mat U;
        arma::vec s;
        arma::mat V;
        arma::svd_econ(U, s, V, XtY);

        return Y * V * U.t();
    }
}
