#include "RcppArmadillo.h"
#include "distributions.h"

namespace lsmdn {

    InverseGammaSampler::InverseGammaSampler(const double shape,
                                             const double scale,
                                             std::mt19937_64 &random_state) :
            gamma_dist_(shape, 1.0/scale),
            random_state_(random_state) {}

    arma::vec InverseGammaSampler::draw(unsigned int num_samples) {
        arma::vec samples(num_samples);
        for (int i = 0; i < num_samples; ++i) {
            samples(i) = 1.0 / gamma_dist_(random_state_);
        }

        return samples;
    }

    DirichletSampler::DirichletSampler(arma::vec &alphas,
                                       std::mt19937_64 &random_state) :
            num_components_(alphas.n_elem),
            random_state_(random_state) {

        for(int i = 0; i < num_components_; ++i) {
            gamma_dists_.emplace_back(alphas(i), 1.0);
        }

    }

    // Samples a dirichlet random variable using the sum of gamma
    // representation of a dirichlet distribution
    arma::rowvec DirichletSampler::single_draw() {
        arma::rowvec y(num_components_);
        for(int i = 0; i < num_components_; ++i) {
            y(i) = gamma_dists_.at(i)(random_state_);
        }
        return y / arma::sum(y);
    }

    arma::mat DirichletSampler::draw(unsigned int num_samples) {
        arma::mat samples(num_samples, num_components_, arma::fill::zeros);
        for (int i = 0; i < num_samples; ++i) {
            samples.row(i) = single_draw();
        }

        return samples;
    }

} // namespace lsmdn
