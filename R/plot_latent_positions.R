mean_latent_positions <- function(X) {
   Reduce("+", X) / length(X)
}

plot_latent_positions <- function(EX) {
    xl <- c(1.15*min(EX[,1,]),1.15*max(EX[,1,]))
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

animate_positions <- function(EX) {
    xl <- c(1.15*min(EX[,1,]),1.15*max(EX[,1,]))
    yl <- c(1.15*min(EX[,2,]),1.15*max(EX[,2,]))
    lims <- range(c(xl,yl))
    require("animation")
    TT <- dim(EX)[3]
    p <- dim(EX)[2]
    n <- dim(EX)[1]
    L <- seq(from=1,to=0,length=51)[-51]
    oopt = ani.options(interval = 0.01, nmax = TT*length(L))
    Xtemp2 <- EX[,,1]
    for(tt in 2:TT)Xtemp2 <- rbind(Xtemp2,EX[,,tt])
    Xtemp2 <- Xtemp2 -
        kronecker(rep(1,n*TT),matrix(apply(Xtemp2,2,mean),1,p))
    
    ## use a loop to create images one by one
    saveHTML(
        {
            par(mar=c(2,2,2,2)+0.1)  
            for (tt in 1:(TT-1)) {
                for(l in 1:length(L)){
                    plot(L[l]*EX[,,tt]+(1-L[l])*EX[,,tt+1],xlab="",ylab="",xlim=lims,ylim=lims,
                         pch=16,xaxt="n",yaxt="n", cex=2)
                    ani.pause() ## pause for a while ('interval')
                }
            }
            
        },img.name="LSMDNPlot",htmlfile="LSMDN.html",autoplay=FALSE,
        interval=0.01,title="Latent Space Movements",
        imgdir="LSMDNimages",ani.height=800,ani.width=800,
        outdir="~/LSMDNMovie",
        description="Posterior means of the latent space positions, with the movements interpolated")
    
}