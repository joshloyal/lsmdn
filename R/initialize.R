initialize_params <- function(Y, num_dimensions = 2) {
    n_nodes <- dim(Y)[1]
    n_time_steps <- dim(Y)[3]
    
    # initialize radii
    radii <- initialize_radii(Y)
    
    # initialize latent positions
    X <- initialize_latent_positions(Y, num_dimensions)
    
    # intialize beta coefficients
    c(beta_in, beta_out) %<-% initialize_beta(Y, X, radii)
    
    # prior for beta_in ~ N(beta_in, 100)
    nu_in <- beta_in
    xi_in <- 100
    
    # prior for beta_out ~ N(beta_out, 100)
    nu_out <- beta_out
    xi_out <- 100
    
    # prior tau^2 ~ IG(shape, scale)
    tau_sq <- sum(X[, , 1] * X[, , 1]) / (n_nodes * num_dimensions)
    tau_shape <- 2.05
    tau_scale <- (tau_shape - 1) * tau_sq
    
    # prior sigma^2 ~ IG(shape, scale)
    sigma_sq <- 0.001
    sigma_shape <- 9
    sigma_scale <- 1.5
    
    list(X=X, radii=radii, beta_in=beta_in, beta_out=beta_out,
         nu_in=nu_in, xi_in=xi_in, nu_out=nu_out, xi_out=xi_out,
         tau_sq=tau_sq, tau_shape=tau_shape, tau_scale=tau_scale,
         sigma_sq=sigma_sq, sigma_shape=sigma_shape, sigma_scale=sigma_scale)
}

initialize_radii <- function(Y, eps = 1e-5) {
    n_nodes <- dim(Y)[1]
    n_time_steps <- dim(Y)[3]
    
    radii <- matrix(0, n_nodes)
    for (t in 1:n_time_steps) {
        radii <- radii + (apply(Y[, , t], 1, sum) + apply(Y[, , t], 2, sum)) / 2
    }
    radii <- radii / sum(Y) 
    
    # radii must be non-zero so add a fudge factor if any are zero
    if (sum(radii == 0) > 0) {
        radii <- radii + eps
        radii <- radii / sum(radii)
    }
    
    radii
}


shortest_path_dissimilarity <- function(Y) {
    n_time_steps = dim(Y)[3]
    
    # dissimilarty matrix
    D <- array(0, dim = dim(Y))
    
    # shortest path dissimilarity
    for (t in 1:n_time_steps) {
        Yt_graph <- igraph::graph.adjacency(Y[, , t])
        D[, , t] <- igraph::shortest.paths(graph = Yt_graph, mode="all")
    }
    
    # inpute infinities with the maximum shortest path in the graph
    D[which(D == Inf, arr.ind = TRUE)] <- max(
        c(D[which(D != Inf, arr.ind = TRUE)]))
    
    D
}


#' Initializes Latent Postitions using GMDS (Sarkar and Moore, 2005)
initialize_latent_positions <- function(Y, num_dimensions = 2, lambda = 10) {
    n_nodes <- dim(Y)[1]
    n_time_steps <- dim(Y)[3]
    
    # Dissimilarty between nodes is determined by their shortest path distance
    D <- shortest_path_dissimilarity(Y)
    
    # At t = 1 we use classical multi-dimensional scaling
    X <- array(0, c(n_nodes, num_dimensions, n_time_steps))
    X[, , 1] <- cmdscale(d = D[, , 1], k = num_dimensions)

    # The following minimizes the objective function found in Sarkar and Moore
    H <- matrix(-1 / n_nodes, n_nodes, n_nodes) + diag(n_nodes)
    for(t in 2:n_time_steps) {
        alpha <- 1 / (1 + lambda)
        beta <- lambda / (1 + lambda)
        XXt <- alpha * H %*% (-0.5 * D[, , t]^2) %*% H 
        XXt <- XXt + beta * (X[, , t-1] %*% t(X[, , t-1]))
      
        # The optimum is the eigendecomposition of XXt
        ev <- eigen(XXt)
        evectors <- ev$vectors[, 1:num_dimensions]
        evals <- ev$values[1:num_dimensions]^(0.5)
        X[, , t] <- sweep(evectors, 2, evals, "*")
        
        # procrustes transformation is used to solve the rotation degeneracy
        X[, , t] <- lsmdn::procrustes(X[, , t - 1], X[, , t])
    }
    
    # scale by number of nodes (same scale as the radii)
    X / n_nodes
}

initialize_beta <- function(Y, X, radii, eps = 1e-4) {
    neg_log_likelihood <- function(beta) {
        -lsmdn::lsmdn_log_likelihood(Y, X, radii, beta[1], beta[2])
    }
    
    grad <- function(beta) {
        -lsmdn::lsmdn_grad_beta(Y, X, radii, beta[1], beta[2])
    }
    
    result <- optim(par = c(1, 1), fn = neg_log_likelihood,
                    gr = grad, method = "BFGS")
    
    beta_in <- max(result$par[1], eps)
    beta_out <- max(result$par[2], eps)
    
    list(beta_in, beta_out)
}