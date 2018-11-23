#' Dutch Classrooms dataset
#' 
#' @export
load_dutch_classroom <- function() {
    data(knecht)
    
    drop_indices = c(2, 16, 19, 21)
    Y1 <- friendship$t1[-drop_indices, -drop_indices]
    Y2 <- friendship$t2[-drop_indices, -drop_indices]
    Y3 <- friendship$t3[-drop_indices, -drop_indices]
    Y4 <- friendship$t4[-drop_indices, -drop_indices]
    
    abind(Y1, Y2, Y3, Y4, along = 3)
}