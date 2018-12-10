#include "RcppArmadillo.h"

#include "distributions.h"
#include "gibbs_sampler.h"
#include "procrustes.h"
#include "linalg.h"

namespace  lsmdn {

    double tune_step_size(double step_size, double acc_rate) {
        if(acc_rate < 0.001) {
            step_size *= 0.1;
        } else if (acc_rate < 0.05) {
            step_size *= 0.5;
        } else if (acc_rate < 0.25) {
            step_size *= 0.9;
        } else if (acc_rate > 0.95) {
            step_size *= 10.0;
        } else if (acc_rate > 0.75) {
            step_size *= 2.0;
        } else if (acc_rate > 0.35) {
            step_size *= 1.1;
        }

        return step_size;
    }

    // Radii small step sizes => smaller acceptance
    double tune_step_size_radii(double step_size, double acc_rate) {
        if(acc_rate < 0.001) {
            step_size *= 10.0;
        } else if (acc_rate < 0.05) {
            step_size *= 2.0;
        } else if (acc_rate < 0.25) {
            step_size *= 1.1;
        } else if (acc_rate > 0.95) {
            step_size *= 0.1;
        } else if (acc_rate > 0.75) {
            step_size *= 0.5;
        } else if (acc_rate > 0.35) {
            step_size *= 0.9;
        }

        return step_size;
    }

    DynamicLatentSpaceNetworkSampler::DynamicLatentSpaceNetworkSampler(
            arma::cube &Y,
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
            tune_(tune),
            tune_interval_(tune_interval),
            steps_until_tune_(tune_interval),
            step_size_x_(step_size_x),
            step_size_beta_(step_size_beta),
            step_size_radii_(step_size_radii),
            X_acc_rate_(num_nodes_, num_time_steps_, arma::fill::zeros),
            beta_in_acc_rate_(0),
            beta_out_acc_rate_(0),
            radii_acc_rate_(0) {

        // set initial values
        tau_sq_(0) = tau_sq;
        sigma_sq_(0) = sigma_sq;
        beta_in_(0) = beta_in;
        beta_out_(0) = beta_out;
        radii_.row(0) = radii_init.t();
        X_.resize(num_samples_);
        X_.at(0) = arma::cube(X_init);

        // determine indices of missing vertices
        for(unsigned int t = 0; t < num_time_steps_; ++t) {
            Y_miss_.push_back(arma::find(Y_miss.row(t) != 0));
        }

        step_sizes_beta_.push_back(step_size_beta_);
    }

    void DynamicLatentSpaceNetworkSampler::tune_step_sizes() {
        // step size for X
        double acc_rate = arma::min(arma::vectorise(X_acc_rate_));
        step_size_x_ = tune_step_size(step_size_x_, acc_rate / tune_interval_);
        X_acc_rate_ = arma::mat(num_nodes_, num_time_steps_, arma::fill::zeros);

        // step size for beta_in
        acc_rate = std::min(beta_in_acc_rate_, beta_out_acc_rate_);
        step_size_beta_ = tune_step_size(
            step_size_beta_, acc_rate / tune_interval_);
        step_sizes_beta_.push_back(step_size_beta_);
        beta_in_acc_rate_ = 0.;
        beta_out_acc_rate_ = 0.;

        // step size for radii
        acc_rate = radii_acc_rate_;
        step_size_radii_ = tune_step_size_radii(
            step_size_radii_, acc_rate / tune_interval_),
        step_size_radii_ = std::min(step_size_radii_, 200000.);
        radii_acc_rate_ = 0.;
    }

    arma::cube DynamicLatentSpaceNetworkSampler::sample_latent_positions(
            unsigned int sample_index) {
        // samples latent positions for each time slice using a
        // random walk metropolis algorithm
        arma::cube X_new(X_.at(sample_index - 1)); // this should copy

        // previous and proposed latent position for node i
        arma::vec Xit_prev;
        arma::vec Xit_prop;

        // store normal draw for the random walk
        arma::vec epsilon;

        // distance between node i and j in latent space
        double dij_prev;
        double dij_prop;

        // etas
        double etaij_prop;
        double etaij_prev;
        double etaji_prop;
        double etaji_prev;

        // parameters from previous sample.
        // this is the first step in the metropolis-in-gibbs algorithm
        double beta_in = beta_in_(sample_index - 1);
        double beta_out = beta_out_(sample_index - 1);
        arma::rowvec radii = radii_.row(sample_index - 1);
        double tau_sq = tau_sq_(sample_index - 1);
        double sigma_sq = sigma_sq_(sample_index - 1);

        // random walk metropolis sampling
        double accept_ratio = 0.;
        for(unsigned int t = 0; t < num_time_steps_; ++t) {
            for(unsigned int i = 0; i < num_nodes_; ++i) {
                // random walk proposal
                Xit_prev = X_.at(sample_index - 1).slice(t).row(i).t();
                epsilon = rnorm_.sample(2);
                Xit_prop = Xit_prev + step_size_x_ * epsilon;

                // calculate acceptance ratio (pi(Xit_prop)/pi(Xit_old))
                accept_ratio = 0.;

                // p_ij * p_ji term (log(p_ij) + log(p_ji))
                for(unsigned int j = 0; j < num_nodes_; ++j) {
                    if (i != j) {
                        dij_prop = arma::norm(
                            Xit_prop - X_new.slice(t).row(j).t(), 2);
                        dij_prev = arma::norm(
                            Xit_prev - X_new.slice(t).row(j).t(), 2);

                        etaij_prop = beta_in * (1 - dij_prop / radii(j)) +
                                     beta_out * (1 - dij_prop / radii(i));
                        etaji_prop = beta_in * (1 - dij_prop / radii(i)) +
                                     beta_out * (1 - dij_prop / radii(j));

                        etaij_prev = beta_in * (1 - dij_prev / radii(j)) +
                                     beta_out * (1 - dij_prev / radii(i));
                        etaji_prev = beta_in * (1 - dij_prev / radii(i)) +
                                     beta_out * (1 - dij_prev / radii(j));

                        // pij
                        accept_ratio += Y_(i, j, t) * etaij_prop;
                        accept_ratio -= std::log(1 + std::exp(etaij_prop));

                        accept_ratio -= Y_(i, j, t) * etaij_prev;
                        accept_ratio += std::log(1 + std::exp(etaij_prev));

                        // pji
                        accept_ratio += Y_(j, i, t) * etaji_prop;
                        accept_ratio -= std::log(1 + std::exp(etaji_prop));

                        accept_ratio -= Y_(j, i, t) * etaji_prev;
                        accept_ratio += std::log(1 + std::exp(etaji_prev));
                    }
                } // loop j

                // transition probabilities
                if (t == 0) { // pi(X_1 | X_0 = 0)
                    accept_ratio -=
                        arma::as_scalar((Xit_prop.t() * Xit_prop) / (2 * tau_sq));
                    accept_ratio +=
                        arma::as_scalar((Xit_prev.t() * Xit_prev) / (2 * tau_sq));
                } else { // pi(X_t | X_{t-1})
                    arma::vec Xit_past = X_new.slice(t - 1).row(i).t();
                    accept_ratio  -=
                        arma::as_scalar((Xit_prop - Xit_past).t() * (Xit_prop - Xit_past) /
                            (2 * sigma_sq));
                    accept_ratio  +=
                        arma::as_scalar((Xit_prev - Xit_past).t() * (Xit_prev - Xit_past) /
                            (2 * sigma_sq));
                }

                // pi(X_{t+1} | X_t)
                if (t < num_time_steps_ - 1) {
                    arma::vec Xit_fut = X_new.slice(t + 1).row(i).t();
                    accept_ratio  -=
                        arma::as_scalar((Xit_fut - Xit_prop).t() * (Xit_fut - Xit_prop) /
                            (2 * sigma_sq));
                    accept_ratio  +=
                        arma::as_scalar((Xit_fut - Xit_prev).t() * (Xit_fut - Xit_prev) /
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
        arma::cube space_mean =
            arma::sum(arma::sum(X_new, 0), 2) / (num_time_steps_ * num_nodes_);
        for (unsigned int t = 0; t < num_time_steps_; ++t) {
            X_new.slice(t).each_row() -= space_mean.slice(0);
        }

        // procrustes transformation if we are past the burn-in period
        if (sample_index > num_burn_in_) {
            arma::mat X_ref = flatten_cube(X_.at(num_burn_in_));
            arma::mat X_new_flat = flatten_cube(X_new);
            arma::mat X_proc = procrustes(X_ref, X_new_flat);
            for(unsigned int t = 0; t < num_time_steps_; ++t) {
                X_new.slice(t) = X_proc.rows(
                    t * num_nodes_, (t + 1) * num_nodes_ - 1);
            }
        }

        return X_new;
    }

    double DynamicLatentSpaceNetworkSampler::sample_beta_in(
            unsigned int sample_index) {
        double beta_in_prev = beta_in_(sample_index - 1);

        // current parameter values
        arma::cube X = X_.at(sample_index);
        double beta_out = beta_out_(sample_index - 1);
        arma::rowvec radii = radii_.row(sample_index - 1);

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

                        accept_ratio += Y_(i, j, t) * eta_prop;
                        accept_ratio -= std::log(1 + std::exp(eta_prop));

                        accept_ratio -= Y_(i, j, t) * eta_prev;
                        accept_ratio += std::log(1 + std::exp(eta_prev));
                    }
                }
            }
        }

        // prior ratio
        accept_ratio -= std::pow(beta_in_prop - nu_in_, 2) / (2. * xi_in_);
        accept_ratio += std::pow(beta_in_prev - nu_in_, 2) / (2. * xi_in_);

        // accept / reject
        double u = runif_.single_sample();
        if(std::log(u) < accept_ratio) {
            beta_in_acc_rate_ += 1;
            return beta_in_prop;
        } else {
            return beta_in_prev;
        }
    }

    double DynamicLatentSpaceNetworkSampler::sample_beta_out(
            unsigned int sample_index) {
        double beta_out_prev = beta_out_(sample_index - 1);

        // current parameter values
        arma::cube X = X_.at(sample_index);
        double beta_in = beta_in_(sample_index);
        arma::rowvec radii = radii_.row(sample_index - 1);

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

                        accept_ratio += Y_(i, j, t) * eta_prop;
                        accept_ratio -= std::log(1 + std::exp(eta_prop));

                        accept_ratio -= Y_(i, j, t) * eta_prev;
                        accept_ratio += std::log(1 + std::exp(eta_prev));
                    }
                }
            }
        }

        // prior ratio
        accept_ratio -= std::pow(beta_out_prop - nu_out_, 2) / (2. * xi_out_);
        accept_ratio += std::pow(beta_out_prev - nu_out_, 2) / (2. * xi_out_);

        // accept / reject
        double u = runif_.single_sample();
        if(std::log(u) < accept_ratio) {
            beta_out_acc_rate_ += 1;
            return beta_out_prop;
        } else {
            return beta_out_prev;
        }
    }

    double DynamicLatentSpaceNetworkSampler::sample_tau_sq(
            unsigned int sample_index) {
        arma::mat X0 = X_.at(sample_index).slice(0);

        // calculate scale of the inverse gamma distribution
        double invgamma_shape =
            tau_shape_ + 0.5 * (num_nodes_ * num_dimensions_);

        // calculate shape of the inverse gamma distribution
        double sq_norm_sum = arma::sum(arma::vectorise(arma::square(X0)));
        double invgamma_scale = tau_scale_ + 0.5 * sq_norm_sum;

        // sample from the inverse gamma distribution
        InverseGammaSampler rinvgamma(invgamma_shape,
                                      invgamma_scale,
                                      random_state_);

        return rinvgamma.single_sample();
    }

    double DynamicLatentSpaceNetworkSampler::sample_sigma_sq(
            unsigned int sample_index) {
        arma::cube X = X_.at(sample_index);

        // calculate the scale of the inverse gamma distribution
        double invgamma_shape =
            sigma_shape_ +
            0.5 * (num_nodes_ * num_dimensions_ * (num_time_steps_ - 1));

        // calculate the shape of the inverse gamma distribution
        double sq_norm_sum = 0.;
        arma::mat X_diffs;
        for(unsigned int t = 1; t < num_time_steps_; ++t) {
            X_diffs = X.slice(t) - X.slice(t - 1);
            sq_norm_sum += arma::sum(arma::vectorise(arma::square(X_diffs)));
        }

        double invgamma_scale = sigma_scale_ + 0.5 * sq_norm_sum;

        // sample from the inverse gamma distribution
        InverseGammaSampler rinvgamma(invgamma_shape,
                                      invgamma_scale,
                                      random_state_);
        return rinvgamma.single_sample();
    }

    arma::rowvec DynamicLatentSpaceNetworkSampler::sample_radii(
            unsigned int sample_index) {
        // previous radii
        arma::rowvec radii_prev = radii_.row(sample_index - 1);

        // extract current parameters of the model
        arma::cube X = X_.at(sample_index);
        double beta_in = beta_in_(sample_index);
        double beta_out = beta_out_(sample_index);

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
                        eta_prop = beta_in * (1 - dij / radii_prop(j)) +
                                   beta_out * (1 - dij / radii_prop(i));
                        eta_prev = beta_in * (1 - dij / radii_prev(j)) +
                                   beta_out * (1 - dij / radii_prev(i));

                        accept_ratio += Y_(i, j, t) * eta_prop;
                        accept_ratio -= std::log(1 + std::exp(eta_prop));

                        accept_ratio -= Y_(i, j, t) * eta_prev;
                        accept_ratio += std::log(1 + std::exp(eta_prev));
                    }
                }
            }
        }

        // transition equation ratio
        for(unsigned int i = 0; i < num_nodes_; ++i) {
            // ratio of dirichlets
            accept_ratio +=
                (step_size_radii_ * radii_prop(i) - 1) * std::log(radii_prev(i));
            accept_ratio -=
                (step_size_radii_ * radii_prev(i) - 1) * std::log(radii_prop(i));

            // ratio of normalizing constansts. Note that sum of alpha
            // is the same for both transition equations so cancels out.
            accept_ratio -= std::lgamma(step_size_radii_ * radii_prop(i));
            accept_ratio += std::lgamma(step_size_radii_ * radii_prev(i));

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

    // assumes the whole row has missing values (true for dutch classroom)
    void DynamicLatentSpaceNetworkSampler::sample_Y_miss(
            unsigned int sample_index) {

        arma::cube X = X_.at(sample_index);
        double beta_in = beta_in_(sample_index);
        double beta_out = beta_out_(sample_index);
        arma::rowvec radii = radii_.row(sample_index);

        double dij = 0;
        double eta = 0;
        double Y_proba = 0;
        double u = 0;

        arma::uvec Y_miss_indices;
        unsigned int Y_miss_index;

        for(unsigned int t = 0; t < num_time_steps_; ++t) {
            Y_miss_indices = Y_miss_.at(t);
            if(Y_miss_indices.n_elem == 0) {
                continue;
            }

            for(unsigned int i = 0; i < Y_miss_indices.n_elem; ++i) {
                Y_miss_index = Y_miss_indices(i);
                for(unsigned int j = 0; j < num_nodes_; ++j) {
                    if(Y_miss_index != j) {
                        dij = arma::norm(
                            X.slice(t).row(Y_miss_index) - X.slice(t).row(j));
                        eta = beta_in * (1 - dij / radii(j)) +
                              beta_out * (1 - dij / radii(Y_miss_index));
                        Y_proba = 1 / (1 + std::exp(-eta));

                    }

                    double u = runif_.single_sample();
                    if(u < Y_proba) {
                        Y_(Y_miss_index, j, t) = 1.0;
                    } else {
                        Y_(Y_miss_index, j, t) = 0.0;
                    }
                }
            }
        }
    }

    ParamSamples DynamicLatentSpaceNetworkSampler::sample() {
        for(unsigned int i = 1; i < num_samples_; ++i) {
            // latent positions random walk metropolis
            X_.at(i) = sample_latent_positions(i);

            // tau_sq / sigma_sq gibbs sample
            tau_sq_(i) = sample_tau_sq(i);
            sigma_sq_(i) = sample_sigma_sq(i);

            // beta_in / beta_out random walk metropolis
            beta_in_(i) = sample_beta_in(i);
            beta_out_(i) = sample_beta_out(i);

            // radii metropolis-hastings with a nonsymmetric dirichlet proposal
            radii_.row(i) = sample_radii(i);

            // sample missing values
            sample_Y_miss(i);

            if(tune_ && (steps_until_tune_ == 0) && (i < num_burn_in_)) {
                tune_step_sizes();
                steps_until_tune_ = tune_interval_;
            } else {
                steps_until_tune_ -= 1;
            }
        }

        return { tau_sq_, sigma_sq_, beta_in_, beta_out_, radii_, X_ };
    }

} // namespace lsmdn
