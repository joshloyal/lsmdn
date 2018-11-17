#pragma once

#include "RcppArmadillo.h"

namespace lsmdn {

    // Samples from an inverse gamma distribution using the
    // representation:
    // y_i ~ gamma(shape, 1.0/scale)
    // 1 / y_i
    class InverseGammaSampler {
    public:
        InverseGammaSampler(const double shape,
                            const double scale,
                            std::mt19937_64 &random_state);

        arma::vec draw(unsigned int num_samples);
    private:
        std::gamma_distribution<double> gamma_dist_;
        std::mt19937_64 &random_state_;
    };

    // Samples from a dirichlet distribution using the
    // sum of gammas representation:
    // y_i ~ gamma(alpha, 1)
    // y_i / sum(y_i)
    class DirichletSampler {
    public:
        DirichletSampler(arma::vec &alphas,
                         std::mt19937_64 &random_state);
        arma::rowvec single_draw();
        arma::mat draw(unsigned int num_samples);
    private:
        std::vector<std::gamma_distribution<double>> gamma_dists_;
        unsigned int num_components_;
        std::mt19937_64 &random_state_;
    };

} // namespace lsmdn
