plot_latent_positions <- function(X) {
    EX <- Reduce("+", X) / length(X)
    
    xl <- c(1.15*min(EX[,1,]),1.15*max(EX[,2,]))
    yl <- c(1.15*min(EX[,2,]),1.15*max(EX[,2,]))
    lims <- range(c(xl,yl))
    
    num_time_steps <- dim(EX)[3]
    for(tt in 1:num_time_steps){
        if (tt == 1) {
            plot(EX[,,tt],xlim=lims,ylim=lims,xlab="",ylab="",
                pch=16,cex=0.75,xaxt="n",yaxt="n",
                main="Posterior mean latent positions showing temporal dynamics")
        }
        if(tt>1) arrows(EX[,1,tt-1],EX[,2,tt-1],EX[,1,tt],EX[,2,tt],length=0.1)
        #  if(tt==1) textxy(EX[1,tt,],EX[2,tt,],labs=c(1:26)[-21]) #load "calibrate" package
        if(tt<num_time_steps) par(new=TRUE)
    }
}