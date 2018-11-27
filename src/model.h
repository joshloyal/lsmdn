#pragma once

#include "RcppArmadillo.h"

namespace lsmdn {

    class DynamicLatentSpaceNetwork {
    public:
        DynamicLatentSpaceNetwork(const arma::cube &Y, const arma::cube &X,
                                  const arma::vec &radii, double beta_in,
                                  double beta_out);

        double latent_distance(int i, int j, int t);
        double get_eta(double dx, int i, int j);
        double log_likelihood();
        arma::cube predict_proba();
        arma::vec grad_beta();

    private:
        const arma::cube &Y_;
        const arma::cube &X_;
        const arma::vec &radii_;
        double beta_in_;
        double beta_out_;

        unsigned int num_nodes_;
        unsigned int num_dimensions_;
        unsigned int num_time_steps_;
    };

}
