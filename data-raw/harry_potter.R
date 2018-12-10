# load first 4 harry-potter books
harry_potter <- array(0, c(64, 64, 4))
for (t in 1:4) {
    harry_potter[, , t] <- as.matrix(read.table(
        paste("http://myweb.uiowa.edu/dksewell/hpbook", t,".txt",sep='')))
    diag(harry_potter[, , t]) <- 0
}

attributes <- read.table("http://myweb.uiowa.edu/dksewell/hpattributes.txt", 
                         header=TRUE)
names = read.table("http://myweb.uiowa.edu/dksewell/hpnames.txt",
                   header=T,stringsAsFactors = F)[,2]

#combined <- rowSums(harry_potter, dims = 2)
#combined[combined > 0] <- 1
graph_full <- igraph::graph.adjacency(harry_potter[,,1], mode = 'directed')
deg <- igraph::degree(graph_full)
keep <- which(deg > 0)
graph_cut <- igraph::graph.adjacency(combined[keep, keep], mode = 'directed')

harry_potter <- harry_potter[keep, keep, ]

bkListNet = bkList = list()
hp <- array(0, c(37, 37, 6))
for(bb in 1:6){
    bkList[[bb]] =
        as.matrix(read.table(paste("http://myweb.uiowa.edu/dksewell/hpbook",bb,".txt",sep="")))[-ind2rm,-ind2rm]
    diag(bkList[[bb]])=0

    hp[,,bb] <- bkList[[bb]]
}