#' Latent Space Model for Dynamic Networks
#' 
#' @param Y The adjacency matrix for each network, 
#'  shape (n_time_points, n_nodes, n_nodes)
#' @return
#' @export
lsmdn <- function(Y, 
                  num_dimensions = 2, 
                  num_samples = 100000, 
                  burn = 0.15, 
                  tune = TRUE,
                  tune_interval = 100,
                  step_size_x = 1.0,
                  step_size_beta = 1.0,
                  step_size_radii = 100,
                  seed = 42) {
    
    params <- initialize_params(Y)
    
    # specify fraction used for burn in
    if(0 < burn && burn < 1) {
        burn <- floor(burn * num_samples)
    }
    
    if (burn >= num_samples) {
        stop("Burn-in period longer than number of samples.")
    }
    
    mcmc_out <- fit_latent_space_network(
        Y, params$Y_miss, 
        params$X, params$radii, params$beta_in, params$beta_out,
        params$nu_in, params$xi_in, params$nu_out, params$xi_out,
        params$tau_sq, params$tau_shape, params$tau_scale, params$sigma_sq,
        params$sigma_shape, params$sigma_scale,
        num_samples, burn, tune, tune_interval,
        step_size_x, step_size_beta, step_size_radii, seed)
    
    # re-shape latent space samples
    latent_dim <- c(dim(Y)[1], num_dimensions, dim(Y)[3])
    mcmc_out$X <- lapply(mcmc_out$X, function(x) { array(x, latent_dim) })
    
    # remove burn-in samples since they are used to tune the step sizes
    samples = list(
        beta_in = mcmc_out$beta_in[-(1:burn)],
        beta_out = mcmc_out$beta_out[-(1:burn)],
        X = mcmc_out$X[-(1:burn)],
        radii = mcmc_out$radii[-(1:burn), ],
        tau_sq = mcmc_out$tau_sq[-(1:burn)],
        sigma_sq = mcmc_out$sigma_sq[-(1:burn)]
    )
    
    structure(
        list(
            Y = mcmc_out$Y,
            Y_miss = params$Y_miss,
            X = mean_latent_positions(samples$X),
            beta_in = mean(samples$beta_in),
            beta_out = mean(samples$beta_out),
            radii = colMeans(samples$radii),
            tau_sq = mean(samples$tau_sq),
            sigma_sq = mean(samples$sigma_sq),
            X_acc_rate = mcmc_out$X_acc_rate,
            step_size_x = mcmc_out$step_size_x,
            beta_in_acc_rate = mcmc_out$beta_in_acc_rate,
            beta_out_acc_rate = mcmc_out$beta_out_acc_rate,
            step_size_beta = mcmc_out$step_size_beta,
            radii_acc_rate = mcmc_out$radii_acc_rate,
            step_size_radii = mcmc_out$step_size_radii,
            samples = samples, 
            num_samples = length(samples$beta_in),
            burn = burn, 
            seed = seed
        ), 
        class = 'lsmdn'
    )
}

#' Compute edge probabilities for the fitted adjacency matrix
#' 
#' @export
predict.lsmdn <- function(model) {
    lsmdn_predict_proba(model$Y, model$X, model$radii, model$beta_in,
                        model$beta_out)
}

auc <- function (x, ...) {
    UseMethod("auc", x)
}

#' In-sample AUC
#' 
#' @export
auc.lsmdn <- function(model) {
    # probability of each edge 
    Y_proba <- predict(model)
    Y <- model$Y
    
    # flatten probabilities
    y_proba <- numeric(0)
    y <- numeric(0)
    num_time_steps <- dim(model$Y)[3]
    for(t in 1:num_time_steps) {
        Yt_proba <- Y_proba[, , num_time_steps]
        y_proba <- c(y_proba, Yt_proba[upper.tri(Yt_proba)])
        y_proba <- c(y_proba, Yt_proba[lower.tri(Yt_proba)])
        
        Yt <- Y[, , num_time_steps]
        y <- c(y, Yt[upper.tri(Yt)])
        y <- c(y, Yt[lower.tri(Yt)])
    }
    
    pROC::auc(y, y_proba)
}

update <- function (x, ...) {
    UseMethod("update", x)
}

#' Run the sampler for more iterations using the last values of the
#' chain as initial values.
#' 
#' @export
#update.lsmdn <- function(model, num_samples = 1000, seed = 123) {
#    num_samples <- model$num_samples
#    samples <- model$samples
#    
#    fit_latent_space_network(
#        Y, model$Y_miss, 
#        samples$X[num_samples], samples$radii[num_samples, ], 
#        samples$beta_in[num_samples], samples$beta_out[num_samples],
#        params$nu_in, params$xi_in, params$nu_out, params$xi_out,
#        params$tau_sq, params$tau_shape, params$tau_scale, params$sigma_sq,
#        params$sigma_shape, params$sigma_scale,
#        num_samples, burn, tune, tune_interval,
#        step_size_x, step_size_beta, step_size_radii, seed)
#}
