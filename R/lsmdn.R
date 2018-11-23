#' Latent Space Model for Dynamic Networks
#' 
#' @param Y The adjacency matrix for each network, 
#'  shape (n_time_points, n_nodes, n_nodes)
#' @return
#' @export
lsmdn <- function(Y, 
                  num_dimensions = 2, 
                  num_iter = 100000, 
                  num_burn_in = 15000, 
                  step_size_x = 0.0075,
                  step_size_beta = 0.1,
                  seed = 42) {
    
    params <- initialize_params(Y)
    
    out <- fit_latent_space_network(
        Y, params$X, params$radii, params$beta_in, params$beta_out,
        params$nu_in, params$xi_in, params$nu_out, params$xi_out,
        params$tau_sq, params$tau_shape, params$tau_scale, params$sigma_sq,
        params$sigma_shape, params$sigma_scale,
        num_iter, num_burn_in, step_size_x, step_size_beta, seed)
    
    out
}