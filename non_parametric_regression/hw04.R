setwd("~/Desktop/engr421_dasc521_fall2019_hw04")

data_set <- read.csv("hw04_data_set.csv",header = FALSE)
training_set <- data_set[2:151,]
test_set <- data_set[152:273,]

# get x and y values
x_train <- as.numeric(as.character(training_set$V1))
y_train <- as.numeric(as.character(training_set$V2))
x_test <- as.numeric(as.character(test_set$V1))
y_test <- as.numeric(as.character(test_set$V2))

# get number of classes and number of samples
K <- max(y_train)
N <- length(y_train)

minimum_value <- 1.5
maximum_value <- 5.25
data_interval <- seq(from = minimum_value, to = maximum_value, by = 0.001)
bin_width <- 0.37
left_borders <- seq(from = minimum_value, to = maximum_value - bin_width, by = bin_width)
right_borders <- seq(from = minimum_value + bin_width, to = maximum_value, by = bin_width)

plotting <- function(){
  plot(x_train, y_train, type = "p", pch = 20,
       ylim = c(min(y_test), max(y_train)), xlim = c(minimum_value, maximum_value),
       ylab = "Waiting time to next eruption (min)", xlab = "Eruption time (min)", las = 1, main = sprintf("h = %g", bin_width), col="purple")
  legend("topleft", legend=c("training", "test"),
         col=c("purple", "red"), pch= 20, cex=0.7)
  points(x_test, y_test, col = "red", pch = 20)}

print_rmse <- function(p_hat,type) {
  rmse <- 0
  if(type == "Regressogram"){
    for (i in 1:length(right_borders)) {
      diff_i <- y_test[ x_test <= right_borders[i]  & left_borders[i] < x_test ] - p_hat[i]
      rmse <- rmse + sum(diff_i^2)
    }
    rmse <- sqrt(rmse/length(x_test))
    sprintf("%s =>RMSE is %s when h is %s", type, rmse, bin_width)
  }else{
    for (i in 1:length(x_test)) {
      diff_i <- y_test[i] - p_hat[(x_test[i]-minimum_value)*1000+1]
      rmse <- rmse + diff_i^2
    }
    rmse <- sqrt(rmse/length(x_test))
    sprintf("%s =>RMSE is %s when h is %s", type, trimws(format(round(rmse,3), nsmall=3)), bin_width)
  }
}

#####################
# Histogram estimator
#####################

p_hat <- sapply(1:length(left_borders), function(c) {sum(y_train[left_borders[c] < x_train & x_train <= right_borders[c]]) / sum(left_borders[c] < x_train & x_train <= right_borders[c])})
plotting()
for (b in 1:length(right_borders)) {
    lines(c(left_borders[b], right_borders[b]), c(p_hat[b], p_hat[b]), lwd = 2, col = "black")
    if (b < length(right_borders)) {
      lines(c(right_borders[b], right_borders[b]), c(p_hat[b], p_hat[b + 1]), lwd = 2, col = "black")}}
print_rmse(p_hat, "Regressogram")

#####################
# Mean Smoother
#####################

p_hat <- sapply(1:length(data_interval), function(c) { x<-abs((x_train - data_interval[c])/bin_width) <= 0.5
                                                      sum(y_train[x]) / sum(x)})
plotting()
lines(data_interval, p_hat, type = "l", lwd = 2, col = "black")
print_rmse(p_hat, "Running Mean Smoother")

#####################
# Kernel Estimator
#####################
p_hat <- sapply(data_interval, function(c) {x <- exp(-0.5*(c-x_train)^2/bin_width^2) * 1/sqrt(2*pi)
                                            sum(x*y_train) / sum(x)})
plotting()
lines(data_interval, p_hat, type = "l", lwd = 2, col = "black")
print_rmse(p_hat, "Kernel Smoother")
