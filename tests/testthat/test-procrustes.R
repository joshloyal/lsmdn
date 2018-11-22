context("test-procrustes")

test_that("Procrustes works with a known transformation", {
    X <- matrix(c(1, -1, -1, 1, 2, 2, -2, -2), nrow = 4, ncol = 2)
    R <- matrix(c(-0.866, -0.500, -0.500, 0.866), nrow = 2, ncol = 2)
    Y <- X %*% R
    
    Y_rot <- lsmdn::procrustes(X, Y)
    expect_equal(X, Y_rot, tolerance = 1e-3)
})
