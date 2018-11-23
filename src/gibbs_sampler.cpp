#include "RcppArmadillo.h"

#include "distributions.h"
#include "gibbs_sampler.h"
#include "procrustes.h"
#include "linalg.h"

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
            const double step_size_x,
            const double step_size_beta,
            const double step_size_radii,
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
            step_size_x_(step_size_x),
            step_size_beta_(step_size_beta),
            step_size_radii_(step_size_radii),
            sample_index_(0),
            X_acc_rate_(num_nodes_, num_time_steps_, arma::fill::zeros),
            beta_in_acc_rate_(0),
            beta_out_acc_rate_(0),
            radii_acc_rate_(0) {
        tau_sq_(0) = tau_sq;
        sigma_sq_(0) = sigma_sq;
        beta_in_(0) = beta_in;
        beta_out_(0) = beta_out;
        radii_.row(0) = radii_init.t();

        X_.resize(num_samples_);
        X_.at(0) = arma::cube(X_init);
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
                Xit_prop = Xit_prev + step_size_x_ * rnorm_.sample(2);

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

        // center across nodes and time steps
        //for (int t = 0; t < num_time_steps_; ++t) {
        //    X_new.slice(t).each_row() -= arma::mean(X_new.slice(t));
        //}
        // center across nodes and time steps
        arma::cube col_means =
            arma::sum(arma::sum(X_new, 0), 2) / (num_time_steps_ * num_nodes_);
        for (unsigned int t = 0; t < num_time_steps_; ++t) {
            X_new.slice(t).each_row() -= col_means.slice(0);
        }

        // procrustes transformation if we are past the burn-in period
        if (sample_index_ > num_burn_in_) {
            arma::mat X0 = flatten_cube(X_.at(num_burn_in_));
            arma::mat Xl = flatten_cube(X_new);
            arma::mat X_proc = procrustes(X0, Xl);
            for(unsigned int t = 0; t < num_time_steps_; ++t) {
                X_new.slice(t) = X_proc.rows(
                    t * num_nodes_, (t + 1) * num_nodes_ - 1);
            }
        }

        return X_new;
    }

    double DynamicLatentSpaceNetworkSampler::sample_beta_in() {
        double beta_in_prev = beta_in_(sample_index_ - 1);

        // current parameter values
        arma::cube X = X_.at(sample_index_);
        double beta_out = beta_out_(sample_index_ - 1);
        arma::rowvec radii = radii_.row(sample_index_ - 1);

        // random walk metropolis
        double beta_in_prop =
            beta_in_prev + step_size_beta_ * rnorm_.single_sample();

        // determine acceptance ratio
        double accept_ratio = 0;
        double dij = 0;
        double eta_prop = 0;
        double eta_prev = 0;

        // likelihood ratio
        for(unsigned int t = 0; t < num_time_steps_; ++t) {
            for(unsigned int i = 0; i < num_nodes_; ++i) {
                for(unsigned int j = 0; j < num_nodes_; ++j) {
                    if (i != j) {
                        dij = arma::norm(
                            X.slice(t).row(i) - X.slice(t).row(j), 2);
                        eta_prop = beta_in_prop * (1 - dij / radii(j)) +
                            beta_out * (1 - dij / radii(i));
                        eta_prev = beta_in_prev * (1 - dij / radii(j)) +
                            beta_out * (1 - dij / radii(i));

                        accept_ratio += Y_(i, j, t) *
                            (beta_in_prop - beta_in_prev) *
                                (1 - dij / radii(j)) -
                            std::log(1 + std::exp(eta_prop)) +
                            std::log(1 + std::exp(eta_prev));
                    }
                }
            }
        }

        // prior ratio
        accept_ratio -= std::pow(beta_in_prop - nu_in_, 2) / (2 * xi_in_);
        accept_ratio += std::pow(beta_in_prev - nu_in_, 2) / (2 * xi_in_);

        // accept / reject
        double u = runif_.single_sample();
        if(std::log(u) < accept_ratio) {
            beta_in_acc_rate_ += 1;
            return beta_in_prop;
        } else {
            return beta_in_prev;
        }
    }

    double DynamicLatentSpaceNetworkSampler::sample_beta_out() {
        double beta_out_prev = beta_out_(sample_index_ - 1);

        // current parameter values
        arma::cube X = X_.at(sample_index_);
        double beta_in = beta_in_(sample_index_);
        arma::rowvec radii = radii_.row(sample_index_ - 1);

        // random walk metropolis
        double beta_out_prop =
            beta_out_prev + step_size_beta_ * rnorm_.single_sample();

        // determine acceptance ratio
        double accept_ratio = 0;
        double dij = 0;
        double eta_prop = 0;
        double eta_prev = 0;

        // likelihood ratio
        for(unsigned int t = 0; t < num_time_steps_; ++t) {
            for(unsigned int i = 0; i < num_nodes_; ++i) {
                for(unsigned int j = 0; j < num_nodes_; ++j) {
                    if (i != j) {
                        dij = arma::norm(
                            X.slice(t).row(i) - X.slice(t).row(j), 2);
                        eta_prop = beta_in * (1 - dij / radii(j)) +
                            beta_out_prop * (1 - dij / radii(i));
                        eta_prev = beta_in * (1 - dij / radii(j)) +
                            beta_out_prev * (1 - dij / radii(i));

                        accept_ratio += Y_(i, j, t) *
                            (beta_out_prop - beta_out_prev) *
                                (1 - dij / radii(i)) -
                            std::log(1 + std::exp(eta_prop)) +
                            std::log(1 + std::exp(eta_prev));
                    }
                }
            }
        }

        // prior ratio
        accept_ratio -= std::pow(beta_out_prop - nu_out_, 2) / (2 * xi_out_);
        accept_ratio += std::pow(beta_out_prev - nu_out_, 2) / (2 * xi_out_);

        // accept / reject
        double u = runif_.single_sample();
        if(std::log(u) < accept_ratio) {
            beta_out_acc_rate_ += 1;
            return beta_out_prop;
        } else {
            return beta_out_prev;
        }
    }

    double DynamicLatentSpaceNetworkSampler::sample_tau_sq() {
        arma::mat X0 = X_.at(sample_index_).slice(0);

        // calculate scale of the inverse gamma distribution
        double invgamma_scale =
            tau_scale_ + 0.5 * (num_nodes_ * num_dimensions_);

        // calculate shape of the inverse gamma distribution
        double sq_norm_sum = 0.;
        for(unsigned int i = 0; i < num_nodes_; ++i) {
            sq_norm_sum += arma::as_scalar(X0.row(i) * X0.row(i).t());
        }
        double invgamma_shape = tau_shape_ + 0.5 * sq_norm_sum;

        // sample from the inverse gamma distribution
        InverseGammaSampler rinvgamma(invgamma_shape,
                                      invgamma_scale,
                                      random_state_);

        return rinvgamma.single_sample();
    }

    double DynamicLatentSpaceNetworkSampler::sample_sigma_sq() {
        arma::cube X = X_.at(sample_index_);

        // calculate the scale of the inverse gamma distribution
        double invgamma_scale =
            sigma_scale_ +
            0.5 * (num_nodes_ * num_dimensions_ * (num_time_steps_ - 1));

        // calculate the shape of the inverse gamma distribution
        double sq_norm_sum = 0.;
        arma::mat X_diffs;
        for(unsigned int t = 1; t < num_time_steps_; ++t) {
            X_diffs = X.slice(t) - X.slice(t - 1);
            for(unsigned int i = 0; i < num_nodes_; ++i) {
                sq_norm_sum += arma::as_scalar(
                    X_diffs.row(i) * X_diffs.row(i).t());
            }
        }

        double invgamma_shape = sigma_shape_ + 0.5 * sq_norm_sum;

        // sample from the inverse gamma distribution
        InverseGammaSampler rinvgamma(invgamma_shape,
                                      invgamma_scale,
                                      random_state_);
        return rinvgamma.single_sample();
    }

    arma::rowvec DynamicLatentSpaceNetworkSampler::sample_radii() {
        // previous radii
        arma::rowvec radii_prev = radii_.row(sample_index_ - 1);

        // extract current parameters of the model
        arma::cube X = X_.at(sample_index_);
        double beta_in = beta_in_(sample_index_);
        double beta_out = beta_out_(sample_index_);

        // distances and etas
        double dij;
        double eta_prev;
        double eta_prop;

        // dirichlet proposal
        arma::rowvec alphas = step_size_radii_ * radii_prev;
        DirichletSampler rdirichlet(alphas, random_state_);
        arma::rowvec radii_prop = rdirichlet.single_sample();

        // likelihood ratio
        double accept_ratio = 0.;
        for(unsigned int t = 0; t < num_time_steps_; ++t) {
            for(unsigned int i = 0; i < num_nodes_; ++i) {
                for(unsigned int j = 0; j < num_nodes_; ++j) {
                    if(i != j) {
                        dij = arma::norm(
                            X.slice(t).row(i) - X.slice(t).row(j), 2);
                        eta_prev = beta_in * (1 - dij / radii_prev(j)) +
                            beta_out * (1 - dij / radii_prev(i));
                        eta_prop = beta_in * (1 - dij / radii_prop(j)) +
                            beta_out * (1 - dij / radii_prop(i));

                        accept_ratio += Y_(i, j, t) * (eta_prop - eta_prev);
                        accept_ratio += std::log(1 + std::exp(eta_prev)) -
                            std::log(1 + std::exp(eta_prop));
                    }
                }
            }
        }

        // transition equation ratio
        for(unsigned int i = 0; i < num_nodes_; ++i) {
            accept_ratio +=
                (step_size_radii_ * radii_prop(i) - 1) * std::log(radii_prev(i)) -
                (step_size_radii_ * radii_prev(i) - 1) * std::log(radii_prop(i));

            // first-two terms taylor expansion (why use this?)
            //accept_ratio -=
            //    (step_size_radii_ * radii_prop(i) - 0.5) *
            //        std::log(step_size_radii_ * radii_prop(i)) -
            //        step_size_radii_ * radii_prop(i);
            //accept_ratio +=
            //    (step_size_radii_ * radii_prev(i) - 0.5) *
            //        std::log(step_size_radii_ * radii_prev(i)) -
            //        step_size_radii_ * radii_prev(i);

            // use actual log-gamma function for normalizing constant
            accept_ratio += std::lgamma(step_size_radii_ * radii_prev(i)) -
                std::lgamma(step_size_radii_ * radii_prop(i));
        }

        // accept / reject
        double u = runif_.single_sample();
        if(std::log(u) < accept_ratio) {
            radii_acc_rate_ += 1;
            return radii_prop;
        } else {
            return radii_prev;
        }
    }

    ParamSamples DynamicLatentSpaceNetworkSampler::sample() {
        for(unsigned int i = 1; i < num_samples_; ++i) {
            sample_index_ = i;

            // latent positions random walk metropolis
            X_.at(sample_index_) = sample_latent_positions();

            // beta_in / beta_out random walk metropolis
            beta_in_(sample_index_) = sample_beta_in();
            beta_out_(sample_index_) = sample_beta_out();

            // tau_sq / sigma_sq gibbs sample
            tau_sq_(sample_index_) = sample_tau_sq();
            sigma_sq_(sample_index_) = sample_sigma_sq();

            // radii metropolis-hastings with a nonsymmetric dirichlet proposal
            radii_.row(sample_index_) = sample_radii();
        }

        return { tau_sq_, sigma_sq_, beta_in_, beta_out_, radii_, X_ };
    }

} // namespace lsmdn
