context("test-samplers")

test_that("rinvgamma(alphpa = 3, beta = 2) has correct moments", {
    x <- lsmdn::rinvgamma(5000, 3, 2)
    
    expect_equal(mean(x), 1, tolerance = 1e-1)
    expect_equal(var(x)[1], 1, tolerance = 1e-1)
})


test_that("rinvgamma(alpha = 3, beta = 0.5) has correct moments", {
    x <- lsmdn::rinvgamma(5000, 3, 0.5)
    
    expect_equal(mean(x), 0.25, tolerance = 1e-2)
    expect_equal(var(x)[1], 0.0625, tolerance = 1e-2)
})