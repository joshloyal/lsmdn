#pragma once

#include "RcppArmadillo.h"

namespace lsmdn {

    class UniformSampler {
    public:
        UniformSampler(const double min,
                       const double max,
                       std::mt19937_64 &random_state);
        double single_sample();
        arma::vec sample(unsigned int num_samples);
    private:
        std::uniform_real_distribution<> uniform_dist_;
        std::mt19937_64 &random_state_;
    };

    class NormalSampler {
    public:
        NormalSampler(const double mean,
                      const double stdde,
                      std::mt19937_64 &random_state);
        double single_sample();
        arma::vec sample(unsigned int num_samples);
    private:
        std::normal_distribution<> normal_dist_;
        std::mt19937_64 &random_state_;
    };

    // Samples from an inverse gamma distribution using the
    // representation:
    // y_i ~ gamma(shape, 1.0/scale)
    // 1 / y_i
    class InverseGammaSampler {
    public:
        InverseGammaSampler(const double shape,
                            const double scale,
                            std::mt19937_64 &random_state);

        double single_sample();
        arma::vec sample(unsigned int num_samples);
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
        DirichletSampler(arma::rowvec &alphas,
                         std::mt19937_64 &random_state);
        arma::rowvec single_sample();
        arma::mat sample(unsigned int num_samples);
    private:
        std::vector<std::gamma_distribution<double>> gamma_dists_;
        unsigned int num_components_;
        std::mt19937_64 &random_state_;
    };

} // namespace lsmdn
