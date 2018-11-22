context("test-lsmdn-fit")
library(xergm)
library(abind)
library(zeallot)

load_dutch_classroom <- function() {
    data(knecht)
    
    Y1 <- friendship$t1[-2, -2]
    Y2 <- friendship$t2[-2, -2]
    
    abind(Y1, Y2, along = 3)
}

test_that("Model fit", {
    Y <- load_dutch_classroom()
    lsmdn(Y)
})
