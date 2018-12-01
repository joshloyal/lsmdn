# adjacency matrices
drop_indices <- 21

Y1 <- as.matrix(read.table(here::here("data-raw", "klas12b-net-1.dat")))
colnames(Y1) <- NULL
Y1 <- Y1[-drop_indices, -drop_indices]
diag(Y1) <- 0

Y2 <- as.matrix(read.table(here::here("data-raw", "klas12b-net-2.dat")))
colnames(Y2) <- NULL
Y2 <- Y2[-drop_indices, -drop_indices]
diag(Y2) <- 0

Y3 <- as.matrix(read.table(here::here("data-raw", "klas12b-net-3.dat")))
colnames(Y3) <- NULL
Y3 <- Y3[-drop_indices, -drop_indices]
diag(Y3) <- 0

Y4 <- as.matrix(read.table(here::here("data-raw", "klas12b-net-4.dat")))
colnames(Y4) <- NULL
Y4 <- Y4[-drop_indices, -drop_indices]
diag(Y4) <- 0

Y <- abind::abind(Y1, Y2, Y3, Y4, along = 3)
Y[Y == 9] <- NA

# demographic information
demographics <- read.table(here::here("data-raw", "klas12b-demographics.dat"),
                           col.names = c('sex', 'age', 'ethnicity', 'religion'))
demographics <- demographics[-drop_indices, ]

# re-code sex variable
demographics$sex <- ifelse(demographics$sex == 1, 'girl', 'boy')
demographics$ethnicity <- unlist(sapply(
    demographics$ethnicity, function(x) { c('missing', 'dutch', 'other')[x + 1] }
))
demographics$religion <- unlist(sapply(
    demographics$religion, function(x) { c('missing', 'christian',
                                          'non-religious', 
                                          'non-christian-religion')[x + 1] }
))

dutch <- list(Y=Y, demographics=demographics)
devtools::use_data(dutch, overwrite = TRUE)
