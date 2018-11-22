#pragma once

#include "RcppArmadillo.h"

namespace lsmdn {

    arma::mat procrustes(const arma::mat &X, const arma::mat &Y);

}
