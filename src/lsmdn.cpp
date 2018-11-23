#include "RcppArmadillo.h"

#include "distributions.h"
#include "procrustes.h"
#include "model.h"
#include "gibbs_sampler.h"

// [[Rcpp::depends(RcppArmadillo)]]

//' Fits a Procrustes Transformation
//' @export
// [[Rcpp::export]]
arma::mat procrustes(const arma::mat &X, const arma::mat &Y) {
    return lsmdn::procrustes(X, Y);
}

//' Samples from an Inverse Gamma Distribution.
//'
//' Note that you must change the seed per draw to get different values.
//' This is exposed for testing purposes and should not be used in R code.
//'
//' @export
//  [[Rcpp::export]]
arma::vec rinvgamma(const int num_samples,
                    const double shape,
                    const double scale=1,
                    const unsigned int seed=123) {

    std::mt19937_64 random_state(seed);
    auto sampler = lsmdn::InverseGammaSampler(shape, scale, random_state);

    return sampler.sample(num_samples);
}

//' Samples from a Dirichlet Distribution.
//'
//' Note that you must change the seed per draw to get different values.
//' This is exposed for testing purposes and should not be used in R code.
//'
//' @export
//  [[Rcpp::export]]
arma::mat rdirichlet(const int num_samples,
                     arma::vec &alphas,
                     const unsigned int seed=123) {
    std::mt19937_64 random_state(seed);
    arma::rowvec d_alphas = alphas.t();
    auto sampler = lsmdn::DirichletSampler(d_alphas, random_state);

    return sampler.sample(num_samples);
}

//' Calculates the log-likelihood of the latent space network model.
//'
//' @export
// [[Rcpp::export]]
double lsmdn_log_likelihood(const arma::cube &Y, const arma::cube &X,
                            const arma::vec &radii, double beta_in,
                            double beta_out) {
    lsmdn::DynamicLatentSpaceNetwork model(Y, X, radii, beta_in, beta_out);
    return model.log_likelihood();
}

//' Calculates the log-likelihood of the latent space network model.
//'
//' @export
// [[Rcpp::export]]
arma::vec lsmdn_grad_beta(const arma::cube &Y, const arma::cube &X,
                          const arma::vec &radii, double beta_in,
                          double beta_out) {
    lsmdn::DynamicLatentSpaceNetwork model(Y, X, radii, beta_in, beta_out);
    return model.grad_beta();
}

//' Fits a Latent Space Model for a Dynamic Network
//'
//' @param Y The adjacency matrix for each network,
//'  shape (n_time_points, n_nodes, n_nodes)
//' @return
//' @export
// [[Rcpp::export]]
Rcpp::List fit_latent_space_network(const arma::cube &Y,
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
                                    const int num_samples,
                                    const int num_burn_in,
                                    const double step_size_x,
                                    const double step_size_beta,
                                    const double step_size_radii,
                                    const unsigned int seed) {

    lsmdn::DynamicLatentSpaceNetworkSampler model(
        Y, X_init, radii_init, beta_in, beta_out, nu_in, xi_in, nu_out, xi_out,
        tau_sq, tau_shape, tau_scale, sigma_sq, sigma_shape, sigma_scale,
        num_samples, num_burn_in, step_size_x, step_size_beta, step_size_radii,
        seed);

    lsmdn::ParamSamples samples = model.sample();

    return Rcpp::List::create(
        Rcpp::Named("X") = samples.X,
        Rcpp::Named("tau_sq") = samples.tau_sq,
        Rcpp::Named("sigma_sq") = samples.sigma_sq,
        Rcpp::Named("beta_in") = samples.beta_in,
        Rcpp::Named("beta_out") = samples.beta_out,
        Rcpp::Named("radii") = samples.radii
    );
}
