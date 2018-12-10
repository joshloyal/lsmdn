#include <cmath>

#include "model.h"

namespace lsmdn {

    DynamicLatentSpaceNetwork::DynamicLatentSpaceNetwork(
            const arma::cube &Y, const arma::cube &X, const arma::vec &radii,
            double beta_in, double beta_out) :
            Y_(Y), X_(X), radii_(radii), beta_in_(beta_in),
            beta_out_(beta_out), num_nodes_(X_.n_rows),
            num_dimensions_(X_.n_cols),
            num_time_steps_(X_.n_slices) {}

    double DynamicLatentSpaceNetwork::latent_distance(int i, int j, int t) {
        return arma::norm(X_.slice(t).row(i) - X_.slice(t).row(j), 2);
    }

    double DynamicLatentSpaceNetwork::get_eta(double dx, int i, int j) {
        return beta_in_ * (1 - dx / radii_(j)) +
                    beta_out_ * (1 - dx / radii_(i));
    }

    double DynamicLatentSpaceNetwork::log_likelihood() {
        double eta;
        double dx;
        double log_lik = 0;
        for(int t = 0; t < num_time_steps_; ++t) {
            for(int i = 0; i < num_nodes_; ++i) {
                for(int j = 0; j < num_nodes_; ++j) {
                    if(i != j) {
                        dx = latent_distance(i, j, t);
                        eta = get_eta(dx, i, j);
                        log_lik += Y_(i, j, t) * eta -
                            std::log(1 + std::exp(eta));
                    }
                }
            }
        }

        return log_lik;
    }

    arma::cube DynamicLatentSpaceNetwork::predict_proba() {
        double eta;
        double dx;
        arma::cube Y_proba(num_nodes_, num_nodes_, num_time_steps_,
                           arma::fill::zeros);

        for(int t = 0; t < num_time_steps_; ++t) {
            for(int i = 0; i < num_nodes_; ++i) {
                for(int j = 0; j < num_nodes_; ++j) {
                    if(i != j) {
                        dx = latent_distance(i, j, t);
                        eta = get_eta(dx, i, j);
                        Y_proba(i, j, t) = 1. / (1. + std::exp(-eta));
                    }
                }
            }
        }

        return Y_proba;
    }

    arma::cube DynamicLatentSpaceNetwork::sample(unsigned int seed) {
        double eta;
        double dx;
        double proba;
        double u;
        arma::cube Y(num_nodes_, num_nodes_, num_time_steps_,
                     arma::fill::zeros);

        std::mt19937_64 random_state(seed);
        UniformSampler runif(0.0, 1.0, random_state);
        for(int t = 0; t < num_time_steps_; ++t) {
            for(int i = 0; i < num_nodes_; ++i) {
                for(int j = 0; j < num_nodes_; ++j) {
                    if(i != j) {
                        dx = latent_distance(i, j, t);
                        eta = get_eta(dx, i, j);
                        proba = 1. / (1. + std::exp(-eta));
                        u = runif.single_sample();
                        if(u < proba) {
                            Y(i, j, t) = 1.0;
                        }
                    }
                }
            }
        }

        return Y;
    }

    arma::vec DynamicLatentSpaceNetwork::grad_beta() {
        double eta;
        double dx;
        arma::vec grad(2, arma::fill::zeros);
        for(int t = 0; t < num_time_steps_; ++t) {
            for(int i = 0; i < num_nodes_; ++i) {
                for(int j = 0; j < num_nodes_; ++j) {
                    if(i != j) {
                        dx = latent_distance(i, j, t);
                        eta = get_eta(dx, i, j);

                        // beta_in grad
                        grad(0) += (1 - dx / radii_(j)) *
                            (Y_(i, j, t) - 1 / (1 + std::exp(-eta)));

                        // beta_out grad
                        grad(1) += (1 - dx / radii_(i)) *
                            (Y_(i, j, t) - 1 / (1 + std::exp(-eta)));
                    }
                }
            }
        }

        return grad;
    }

} // namespace lsmdn
