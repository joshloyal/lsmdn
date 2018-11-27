#pragma once

#include <vector>

#include "RcppArmadillo.h"

#include "distributions.h"

namespace lsmdn {


struct ParamSamples {
    arma::vec tau_sq;           // variance of initial latent positions
    arma::vec sigma_sq;         // variance of transition equations
    arma::vec beta_in;          // popularity coefficient
    arma::vec beta_out;         // social activity coefficient
    arma::mat radii;           // social reach
    std::vector<arma::cube> X;  // latent positions
};


class DynamicLatentSpaceNetworkSampler {
public:
    DynamicLatentSpaceNetworkSampler(arma::cube &Y,
                                     const arma::mat &Y_miss,
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
                                     bool tune,
                                     const unsigned int tune_interval,
                                     const double step_size_x,
                                     const double step_size_beta,
                                     const double step_size_radii,
                                     unsigned int seed,
                                     bool debug=false);

    arma::cube sample_latent_positions(unsigned int sample_index);
    double sample_beta_in(unsigned int sample_index);
    double sample_beta_out(unsigned int sample_index);
    double sample_tau_sq(unsigned int sample_index);
    double sample_sigma_sq(unsigned int sample_index);
    arma::rowvec sample_radii(unsigned int sample_index);
    void sample_Y_miss(unsigned int sample_index);
    ParamSamples sample();

    // imputation
    arma::cube get_imputed_Y() { return Y_; }

    // step size tuning
    void tune_step_sizes();
    double get_step_size_x() { return step_size_x_; }
    double get_step_size_beta() { return step_size_beta_; }
    double get_step_size_radii() { return step_size_radii_; }

    // acceptance rates
    arma::mat get_X_acc_rate() { return X_acc_rate_ / num_samples_; }
    double get_beta_in_acc_rate() { return beta_in_acc_rate_ / num_samples_; }
    double get_beta_out_acc_rate() { return beta_out_acc_rate_ / num_samples_; }
    double get_radii_acc_rate() { return radii_acc_rate_ / num_samples_; }

    // getters
    arma::mat get_rnorms_X() const { return rnorms_X_; }
    arma::vec get_runifs_X() const { return runifs_X_; }
    arma::vec get_accept_ratio_X() const { return accept_ratio_X_; }

private:
    arma::cube &Y_;        // observed adjacency matrix per time point

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

    std::vector<arma::uvec> Y_miss_;  // indices of missing nodes in Y

    // random number generators
    std::mt19937_64 random_state_;
    UniformSampler runif_;
    NormalSampler rnorm_;

    // step size tuning
    bool tune_;
    unsigned int tune_interval_;
    unsigned int steps_until_tune_;

    // step sizes for random walk metropolis
    double step_size_x_;
    double step_size_beta_;
    double step_size_radii_;

    // acceptance rates
    arma::mat X_acc_rate_;
    double beta_in_acc_rate_;
    double beta_out_acc_rate_;
    double radii_acc_rate_;

    // debug statistics
    bool debug_;

    arma::mat rnorms_X_;
    arma::vec runifs_X_;
    arma::vec accept_ratio_X_;
};


} // namespace lsmdn
