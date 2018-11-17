#include "RcppArmadillo.h"
#include "distributions.h"

// [[Rcpp::depends(RcppArmadillo)]]

//' Fits a Latent Space Model for a Dynamic Network
//' 
//' @param Y The adjacency matrix for each network, 
//'  shape (n_time_points, n_nodes, n_nodes)
//' @return
//' @export
// [[Rcpp::export]]
Rcpp::List fit_latent_space_network(const arma::cube &Y,
                                    const int num_dimensions,
                                    const int num_iter,
                                    const int num_burn_in,
                                    const unsigned int seed) {
    arma::mat first_slice = Y.slice(0);
    
    std::mt19937_64 random_state(seed);
    auto rinvgamma = lsmdn::InverseGammaSampler(100, 1, random_state);
    arma::vec gammas = rinvgamma.draw(5000);
    
    arma::vec alphas{2.0, 2.0, 2.0};
    auto rdirichlet = lsmdn::DirichletSampler(alphas, random_state);
    arma::mat dirichlets = rdirichlet.draw(500);
    
    return Rcpp::List::create(
        Rcpp::Named("out") = gammas,
        Rcpp::Named("dir") = dirichlets
    );
}