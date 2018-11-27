context("test-lsmdn-fit")


test_that("latent samples correct", {
    num_dimensions = 2
    num_samples = 10000
    num_burn_in = 15000
    step_size_x = 0.0075
    step_size_beta = 0.1
    step_size_radii = 175000
    seed = 42
    
    Y <- load_dutch_classroom()
    
    params <- initialize_params(Y)
    
    mcmc_out <- sample_latent_positions(
        Y, params$X, params$radii, params$beta_in, params$beta_out,
        params$nu_in, params$xi_in, params$nu_out, params$xi_out,
        params$tau_sq, params$tau_shape, params$tau_scale, params$sigma_sq,
        params$sigma_shape, params$sigma_scale,
        num_samples, num_burn_in, step_size_x, step_size_beta, step_size_radii,
        seed)
})
