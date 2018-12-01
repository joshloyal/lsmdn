library(readxl)
library(tidyverse)

trade_years <- list()
available_countries <- NULL
for(year in as.character(1964:1976)) {
    trade_year <- readxl::read_excel(here::here("data-raw", "dyadic-compact.xls"),
                                     sheet = year)
    trade_year <- as.matrix(trade_year[, 5:ncol(trade_year)])
    trade_year[is.na(trade_year)] <- 0
    trade_year[trade_year > 0] <- 1
    rownames(trade_year) <- colnames(trade_year)
    
    if(is.null(available_countries)) {
        available_countries <- colnames(trade_year)
    } else {
        keep <- available_countries %in% colnames(trade_year)
        available_countries <- available_countries[keep]
    }
    trade_years[[year]] <- trade_year
} 

# filter down countries and make a 3d array
n_countries <- length(available_countries)
Y <- array(0, dim = c(n_countries, n_countries, 13))
t <- 1
for(year in as.character(1964:1976)) {
   Y[, , t] <- trade_years[[year]][available_countries, available_countries]
   t <- t + 1
}

world_trade<- list(Y = Y, country_names = available_countries)
devtools::use_data(world_trade, overwrite = TRUE)