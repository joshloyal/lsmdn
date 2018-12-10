library(networkDynamic)

data(newcomb)

num_time_steps <- length(newcomb)
num_nodes <- 17

newcomb_adj <- array(0, c(num_nodes, num_nodes, num_time_steps))
for(t in 1:num_time_steps) {
    newcomb_adj[, , t] <- (network::as.matrix.network.adjacency(
                           newcomb.rank[[t]], attrname = 'rank') <= 6) * 1
    diag(newcomb_adj[, , t]) <- 0
}

devtools::use_data(newcomb_adj, overwrite = TRUE)