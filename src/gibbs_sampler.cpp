#include "RcppArmadillo.h"

#include "distributions.h"
#include "gibbs_sampler.h"

namespace  lsmdn {

    DynamicLatentSpaceNetworkSampler::DynamicLatentSpaceNetworkSampler(
            const arma::cube &Y,
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
            unsigned int seed) :
            Y_(Y),
            num_samples_(num_samples),
            num_burn_in_(num_burn_in),
            num_nodes_(Y.n_rows),
            num_dimensions_(X_init.n_cols),
            num_time_steps_(Y.n_slices),
            tau_sq_(num_samples, arma::fill::zeros),
            tau_shape_(tau_shape),
            tau_scale_(tau_scale),
            sigma_sq_(num_samples, arma::fill::zeros),
            sigma_shape_(sigma_shape),
            sigma_scale_(sigma_scale),
            beta_in_(num_samples, arma::fill::zeros),
            nu_in_(nu_in),
            xi_in_(xi_in),
            beta_out_(num_samples, arma::fill::zeros),
            nu_out_(nu_out),
            xi_out_(xi_out),
            radii_(num_samples, Y_.n_rows, arma::fill::zeros),
            random_state_(seed),
            runif_(0.0, 1.0, random_state_),
            rnorm_(0.0, 1.0, random_state_),
            sigma_x_(0.0075),
            sample_index_(0),
            X_acc_rate_(num_nodes_, num_time_steps_, arma::fill::zeros),
            beta_in_acc_rate_(0),
            beta_out_acc_rate_(0),
            radii_acc_rate_(num_nodes_, arma::fill::zeros),
            sigma_acc_rate_(0),
            tau_acc_rate_(0) {

        tau_sq_(0) = tau_sq;
        sigma_sq_(0) = sigma_sq;
        beta_in_(0) = beta_in;
        beta_out_(0) = beta_out;
        radii_.row(0) = radii_init.t();

        X_.resize(num_samples);
        X_.push_back(X_init);
    }

    arma::cube DynamicLatentSpaceNetworkSampler::sample_latent_positions() {
        // samples latent positions for each time slice using a
        // random walk metropolis algorithm
        arma::cube X_new(X_.at(sample_index_ - 1)); // this should copy

        // previous and proposed latent position for node i
        arma::vec Xit_prev;
        arma::vec Xit_prop;

        // distance between node i and j in latent space
        double dij_prev;
        double dij_prop;

        // parameters from previous sample.
        // this is the first step in the metropolis-in-gibbs algorithm
        double beta_in = beta_in_(sample_index_ - 1);
        double beta_out = beta_out_(sample_index_ - 1);
        arma::rowvec radii = radii_.row(sample_index_ - 1);
        double tau_sq = tau_sq_(sample_index_ - 1);
        double sigma_sq = sigma_sq_(sample_index_ - 1);

        // random walk metropolis sampling
        double accept_ratio = 0;
        for(unsigned int t = 0; t < num_time_steps_; ++t) {
            for(unsigned int i = 0; i < num_nodes_; ++i) {
                // random walk proposal
                Xit_prev = X_.at(sample_index_ - 1).slice(t).row(i).t();
                Xit_prop = Xit_prev + sigma_x_ * rnorm_.sample(2);

                // calculate acceptance ratio (pi(Xit_prop)/pi(Xit_old))
                accept_ratio = 0;

                // p_ij * p_ji term (log(p_ij) + log(p_ji))
                for(unsigned int j = 0; j < num_nodes_; ++j) {
                    if (i != j) {
                        dij_prop = arma::norm(
                            Xit_prop - X_new.slice(t).row(j).t(), 2);
                        dij_prev = arma::norm(
                            Xit_prev - X_new.slice(t).row(j).t(), 2);

                        accept_ratio += (dij_prev - dij_prop) *
                            (Y_(j, i, t) *
                                (beta_in / radii(i) + beta_out / radii(j))) +
                            (Y_(i, j, t) *
                                (beta_in / radii(j) + beta_out / radii(i)));

                        accept_ratio += std::log(
                            1 + std::exp(beta_in * (1 - dij_prev / radii(i)) +
                                         beta_out * (1 - dij_prev / radii(j))));
                        accept_ratio += std::log(
                            1 + std::exp(beta_in * (1 - dij_prev / radii(j)) +
                                         beta_out * (1 - dij_prev / radii(i))));

                        accept_ratio -= std::log(
                            1 + std::exp(beta_in * (1 - dij_prop / radii(i)) +
                                         beta_out * (1 - dij_prop / radii(j))));
                        accept_ratio -= std::log(
                            1 + std::exp(beta_in * (1 - dij_prop / radii(j)) +
                                         beta_out * (1 - dij_prop / radii(i))));
                    }
                } // loop j

                // transition probabilities
                if (t == 0) { // pi(X_1 | X_0 = 0)
                    accept_ratio -=
                        arma::as_scalar((Xit_prop.t() * Xit_prop) / (2 * tau_sq));
                    accept_ratio +=
                        arma::as_scalar((Xit_prev.t() * Xit_prev) / (2 * tau_sq));
                } else { // pi(X_t | X_{t-1})
                    arma::vec Xit_1 = X_new.slice(t - 1).row(i).t();
                    accept_ratio  -=
                        arma::as_scalar((Xit_prop - Xit_1).t() * (Xit_prop - Xit_1) /
                            (2 * sigma_sq));
                    accept_ratio  +=
                        arma::as_scalar((Xit_prev - Xit_1).t() * (Xit_prev - Xit_1) /
                            (2 * sigma_sq));
                }

                // pi(X_{t+1} | X_t)
                if (t < num_time_steps_ - 1) {
                    arma::vec Xit_1 = X_new.slice(t + 1).row(i).t();
                    accept_ratio  -=
                        arma::as_scalar((Xit_1 - Xit_prop).t() * (Xit_1 - Xit_prop) /
                            (2 * sigma_sq));
                    accept_ratio  +=
                        arma::as_scalar((Xit_1 - Xit_prev).t() * (Xit_1 - Xit_prev) /
                            (2 * sigma_sq));

                }

                // accept / reject
                double u = runif_.single_sample();
                if(std::log(u) < accept_ratio) {
                    X_new.slice(t).row(i) = Xit_prop.t();
                    X_acc_rate_(i, t) += 1;
                }
            } // loop i
        } // loop t

        // center each time-slice
        for (int t = 0; t < num_time_steps_; ++t) {
            X_new.slice(t).each_row() -= arma::mean(X_new.slice(t));
        }

        // procrustes transformation if we are past the burn-in period
        //if (sample_index_ > num_burn_in_) {
        //    arma::mat X0 = flatten_cube(X_.at(num_burn_in_));
        //    arma::mat Xl = flatten_cube(X_new);
        //    arma::mat X_proc = procrustes(X0, Xl);
        //    for(unsigned int t = 0; t < num_time_steps_; ++t) {
        //        X_new.slice(t) = X_proc.rows(t * num_nodes_, (t + 1) * num_nodes_ - 1);
        //    }
        //}

        return X_new;
    }

    void DynamicLatentSpaceNetworkSampler::sample() {
        for(unsigned int i = 1; i < num_samples_; ++i) {
            sample_index_ = i;

            // latent positions random walk metropolis
            X_.push_back(sample_latent_positions());
        }
    }

} // namespace lsmdn
