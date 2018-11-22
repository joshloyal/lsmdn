#pragma once

#include <vector>

#include "RcppArmadillo.h"

#include "distributions.h"

namespace lsmdn {


struct ModelParams {
    arma::vec tau_sq;           // variance of initial latent positions
    arma::vec sigma_sq;         // variance of transition equations
    arma::vec beta_in;          // popularity coefficient
    arma::vec beta_out;         // social activity coefficient
    arma::mat radius;           // social reach
    std::vector<arma::cube> X;  // latent positions
};


class DynamicLatentSpaceNetworkSampler {
public:
    DynamicLatentSpaceNetworkSampler(const arma::cube &Y,
                                     const arma::cube &X_init,
                                     const arma::vec &radii_init,
                                     const double beta_in,
                                     const double beta_out,
                                     const double nu_in,
                                     const double xi_in,
                                     const double nu_out,
                                     const double xi_out,
                                     const double tau_sq,
                                     const double tau_shape,
                                     const double tau_scale,
                                     const double sigma_sq,
                                     const double sigma_shape,
                                     const double sigma_scale,
                                     const unsigned int num_samples,
                                     const unsigned int num_burn_in,
                                     unsigned int seed);
    arma::cube sample_latent_positions();
    void sample();

private:
    const arma::cube &Y_;        // observed adjacency matrix per time point

    const unsigned int num_samples_;     // number of samples drawn
    const unsigned int num_burn_in_;     // burn in period

    const unsigned int num_nodes_;       // number of nodes in the graph
    const unsigned int num_dimensions_;  // number of latent dimensions
    const unsigned int num_time_steps_;  // number of time steps

    arma::vec tau_sq_;           // variance of the initial latent positions
    double tau_shape_;           // shape of inverse gamma prior
    double tau_scale_;           // scale of inverse gamma prior

    arma::vec sigma_sq_;         // variance of transition equations
    double sigma_shape_;         // shape of inverse gamma prior
    double sigma_scale_;         // scale of inverse gamma prior

    arma::vec beta_in_;          // popularity coefficient
    double nu_in_;               // mean of normal prior
    double xi_in_;               // variance of normal prior

    arma::vec beta_out_;         // social activity coefficient
    double nu_out_;              // mean of normal prior
    double xi_out_;              // variance of normal prior

    arma::mat radii_;            // social reaches

    std::vector<arma::cube> X_;  // latent positions

    // random number generators
    std::mt19937_64 random_state_;
    UniformSampler runif_;
    NormalSampler rnorm_;

    // step sizes for random walk metropolis
    double sigma_x_;

    // counters
    unsigned int sample_index_;

    // acceptance rates
    arma::mat X_acc_rate_;
    double beta_in_acc_rate_;
    double beta_out_acc_rate_;
    arma::vec radii_acc_rate_;
    double sigma_acc_rate_;
    double tau_acc_rate_;

};


} // namespace lsmdn
