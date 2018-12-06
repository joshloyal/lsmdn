library(pscl)

s109 <- pscl::readKH("https://voteview.com/static/data/out/votes/S109_votes.ord",
                     desc = "109th U.S. Senate")

votes <- s109$votes

# yae = 1, nae = 0, missing = 0
votes[votes %in% 1:3] <- 1
votes[votes %in% 4:9] <- 0