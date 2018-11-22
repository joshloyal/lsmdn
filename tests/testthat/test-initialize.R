context("test-initialize")
library(xergm)
library(abind)
library(zeallot)

load_dutch_classroom <- function() {
    data(knecht)
    
    Y1 <- friendship$t1[-2, -2]
    Y2 <- friendship$t2[-2, -2]
    
    abind(Y1, Y2, along = 3)
}

test_that("initialization", {
    Y <- load_dutch_classroom()
    params <- initialize_params(Y)
    
    # check against reference values from the original LSMDN packag
    expected_X1 <- as.matrix(read.table('expected_X1.txt'))
    expect_equal(params$X[, , 1], expected_X1, tolerance = 1e-5, 
                 check.attributes = FALSE)
    
    expected_X2 <- as.matrix(read.table('expected_X2.txt'))
    expect_equal(params$X[, , 2], expected_X2, tolerance = 1e-5, 
                 check.attributes = FALSE)
    
    expected_radii <- as.matrix(c(
        0.03664921, 0.06282723, 0.04712042, 0.01832461, 0.07068063, 0.04188482,
        0.06806283, 0.03664921, 0.07068063, 0.03403141, 0.07329843, 0.03403141,
        0.03926702, 0.03403141, 0.01832461, 0.01308901, 0.05759162, 0.01832461,
        0.01832461, 0.03141361, 0.03403141, 0.04973822, 0.02617801, 0.02879581,
        0.03664921))
    expect_equal(params$radii, expected_radii)
    

    expect_equal(params$beta_in, 0.8630854, tolerance = 1e-5)
    expect_equal(params$beta_out, 0.6025094, tolerance = 1e-5)
    
    expect_equal(params$tau_sq, 0.003677462, tolerance = 1e-6)
    expect_equal(params$tau_scale, 0.003861335, tolerance = 1e-6)
})
