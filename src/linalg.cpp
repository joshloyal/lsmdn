#include "linalg.h"

namespace lsmdn {

arma::mat flatten_cube(const arma::cube &X_cube) {
    arma::mat X_mat = X_cube.slice(0);
    for(int t = 1; t < X_cube.n_slices; ++t) {
        X_mat = arma::join_cols(X_mat, X_cube.slice(t));
    }

    return X_mat;
}

} // namespace lsmdn
