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
                            double beta_out, double intercept) {
    lsmdn::DynamicLatentSpaceNetwork model(
        Y, X, radii, beta_in, beta_out, intercept);
    return model.log_likelihood();
}

//' Calculates the log-likelihood of the latent space network model.
//'
//' @export
// [[Rcpp::export]]
arma::vec lsmdn_grad_beta(const arma::cube &Y, const arma::cube &X,
                          const arma::vec &radii, double beta_in,
                          double beta_out, double intercept) {
    lsmdn::DynamicLatentSpaceNetwork model(
        Y, X, radii, beta_in, beta_out, intercept);
    return model.grad_beta();
}


//' Calculate the probabilities of an edge formation according to the model
//'
//' @export
// [[Rcpp::export]]
arma::cube lsmdn_predict_proba(const arma::cube &Y, const arma::cube &X,
                               const arma::vec &radii, double beta_in,
                               double beta_out) {
    lsmdn::DynamicLatentSpaceNetwork model(Y, X, radii, beta_in, beta_out);
    return model.predict_proba();
}

//' Sample a network from a lsmdn model
//'
//' @export
// [[Rcpp::export]]
arma::cube lsmdn_sample(const arma::cube &Y, const arma::cube &X,
                        const arma::vec &radii, double beta_in,
                        double beta_out, unsigned int seed) {
    lsmdn::DynamicLatentSpaceNetwork model(Y, X, radii, beta_in, beta_out);
    return model.sample(seed);
}

//' Sample a single latent position
//'
//Rcpp::List sample_latent_positions(const arma::cube &Y,
//                                   const arma::cube &X_init,
//                                   const arma::vec &radii_init,
//                                   const double beta_in,
//                                   const double beta_out,
//                                   const double nu_in,
//                                   const double xi_in,
//                                   const double nu_out,
//                                   const double xi_out,
//                                   const double tau_sq,
//                                   const double tau_shape,
//                                   const double tau_scale,
//                                   const double sigma_sq,
//                                   const double sigma_shape,
//                                   const double sigma_scale,
//                                   const int num_samples,
//                                   const int num_burn_in,
//                                   const double step_size_x,
//                                   const double step_size_beta,
//                                   const double step_size_radii,
//                                   const unsigned int seed) {
//    lsmdn::DynamicLatentSpaceNetworkSampler model(
//        Y, X_init, radii_init, beta_in, beta_out, nu_in, xi_in, nu_out, xi_out,
//        tau_sq, tau_shape, tau_scale, sigma_sq, sigma_shape, sigma_scale,
//        num_samples, num_burn_in, step_size_x, step_size_beta, step_size_radii,
//        seed, true);
//
//    arma::cube X = model.sample_latent_positions(1);
//
//    return Rcpp::List::create(
//        Rcpp::Named("X") = X,
//        Rcpp::Named("rnorms") = model.get_rnorms_X(),
//        Rcpp::Named("runifs") = model.get_runifs_X(),
//        Rcpp::Named("accept_ratio") = model.get_accept_ratio_X()
//    );
//}

//' Fits a Latent Space Model for a Dynamic Network
//'
//' @param Y The adjacency matrix for each network,
//'  shape (n_time_points, n_nodes, n_nodes)
//' @return
//' @export
// [[Rcpp::export]]
Rcpp::List fit_latent_space_network(arma::cube &Y,
                                    const arma::mat &Y_miss,
                                    const arma::cube &X_init,
                                    const arma::vec &radii_init,
                                    const double beta_in,
                                    const double beta_out,
                                    const double intercept,
                                    const double nu_in,
                                    const double xi_in,
                                    const double nu_out,
                                    const double xi_out,
                                    const double nu_intercept,
                                    const double xi_intercept,
                                    const double tau_sq,
                                    const double tau_shape,
                                    const double tau_scale,
                                    const double sigma_sq,
                                    const double sigma_shape,
                                    const double sigma_scale,
                                    const int num_samples,
                                    const int num_burn_in,
                                    const bool tune,
                                    unsigned int tune_interval,
                                    const double step_size_x,
                                    const double step_size_intercept,
                                    const double step_size_beta,
                                    const double step_size_radii,
                                    const unsigned int seed) {

    lsmdn::DynamicLatentSpaceNetworkSampler model(
        Y, Y_miss, X_init, radii_init, beta_in, beta_out, intercept,
        nu_in, xi_in, nu_out, xi_out, nu_intercept, xi_intercept,
        tau_sq, tau_shape, tau_scale, sigma_sq, sigma_shape, sigma_scale,
        num_samples, num_burn_in, tune, tune_interval,
        step_size_x, step_size_intercept, step_size_beta, step_size_radii, seed);

    lsmdn::ParamSamples samples = model.sample();

    return Rcpp::List::create(
        Rcpp::Named("Y") = model.get_imputed_Y(),
        Rcpp::Named("X") = samples.X,
        Rcpp::Named("X_acc_rate") = model.get_X_acc_rate(),
        Rcpp::Named("step_size_x") = model.get_step_size_x(),
        Rcpp::Named("tau_sq") = samples.tau_sq,
        Rcpp::Named("sigma_sq") = samples.sigma_sq,
        Rcpp::Named("intercept") = samples.intercept,
        Rcpp::Named("intercept_acc_rate") = model.get_intercept_acc_rate(),
        Rcpp::Named("step_size_intercept") = model.get_step_size_intercept(),
        Rcpp::Named("beta_in") = samples.beta_in,
        Rcpp::Named("beta_in_acc_rate") = model.get_beta_in_acc_rate(),
        Rcpp::Named("beta_out") = samples.beta_out,
        Rcpp::Named("beta_out_acc_rate") = model.get_beta_out_acc_rate(),
        Rcpp::Named("step_size_beta") =  model.get_step_size_beta(),
        Rcpp::Named("radii") = samples.radii,
        Rcpp::Named("radii_acc_rate") = model.get_radii_acc_rate(),
        Rcpp::Named("step_size_radii") = model.get_step_size_radii()
    );
}
