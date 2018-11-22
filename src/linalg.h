#pragma once

#include "RcppArmadillo.h"

namespace lsmdn {

arma::mat flatten_cube(const arma::cube &X_cube);

} // namespace lsmdn
