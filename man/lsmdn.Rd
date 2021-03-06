% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/lsmdn.R
\name{lsmdn}
\alias{lsmdn}
\title{Latent Space Model for Dynamic Networks}
\usage{
lsmdn(Y, num_dimensions = 2, num_samples = 1e+05, burn = 0.15,
  tune = TRUE, tune_interval = 100, step_size_x = 1,
  step_size_beta = 1, step_size_radii = 100, seed = 42)
}
\arguments{
\item{Y}{The adjacency matrix for each network, 
shape (n_time_points, n_nodes, n_nodes)}
}
\description{
Latent Space Model for Dynamic Networks
}
