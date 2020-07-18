setwd("~/Desktop/engr421_dasc521_fall2019_hw05")

# +++ ################ 1 ###################

data_set <- read.csv("hw05_data_set.csv",header = FALSE)

# +++ ################ 2 ###################

training_set <- data_set[2:151,]
test_set <- data_set[152:273,]

# get x and y values
x_train <- as.numeric(as.character(training_set$V1))
y_train <- as.numeric(as.character(training_set$V2))
x_test <- as.numeric(as.character(test_set$V1))
y_test <- as.numeric(as.character(test_set$V2))

# get number of classes and number of samples
N_train <- length(y_train)
N_test <- length(y_test)

# +++ ################ 3 ###################

dec_tree_reg <- function(P) {
  # create necessary data structures
  node_splits <- c()
  node_means <- c()
  
  # put all training instances into the root node
  node_indices <- list(1:N_train)
  is_terminal <- c(FALSE)
  need_split <- c(TRUE)
  
  # learning algorithm
  while (1) {
    # find nodes that need splitting
    split_nodes <- which(need_split)
    # check whether we reach all terminal nodes
    if (length(split_nodes) == 0) {
      break
    }
    # find best split positions for all nodes
    for (split_node in split_nodes) {
      data_indices <- node_indices[[split_node]]
      need_split[split_node] <- FALSE
      node_mean <- mean(y_train[data_indices])
      if (length(x_train[data_indices]) <= P) {
        is_terminal[split_node] <- TRUE
        node_means[split_node] <- node_mean
      } else {
        is_terminal[split_node] <- FALSE
        
        # Vectorized
        unique_values <- sort(unique(x_train[data_indices]))
        split_positions <- (unique_values[-1] + unique_values[-length(unique_values)]) / 2
        split_scores <- rep(0, length(split_positions))
        
        for (s in 1:length(split_positions)) {
          left_indices <- data_indices[which(x_train[data_indices] <= split_positions[s])]
          right_indices <- data_indices[which(x_train[data_indices] > split_positions[s])]
          left_length <- length(left_indices)
          right_length <- length(right_indices)
          e <- 0
          if (left_length>0) {
            e <- e + sum((y_train[left_indices] - mean(y_train[left_indices]))^2)
          }
          if (right_length>0) {
            e <- e + sum((y_train[right_indices] - mean(y_train[right_indices]))^2)
          }
          split_scores[s] <- e/(left_length+right_length)
        }
        node_splits[split_node] <- split_positions[which.min(split_scores)]
        
        # create left node using the selected split
        left_indices <- data_indices[which(x_train[data_indices] < split_positions[which.min(split_scores)])]
        node_indices[[2 * split_node]] <- left_indices
        is_terminal[2 * split_node] <- FALSE
        need_split[2 * split_node] <- TRUE
        
        # create right node using the selected split
        right_indices <- data_indices[which(x_train[data_indices] >= split_positions[which.min(split_scores)])]
        node_indices[[2 * split_node + 1]] <- right_indices
        is_terminal[2 * split_node + 1] <- FALSE
        need_split[2 * split_node + 1] <- TRUE
      }
    }
  }
  return(list("ns"=node_splits, "it"= is_terminal, "mn"= node_means))
}

# +++ ################ 4 ###################

# traverse tree for testing
predict <- function(n, ns, mn, is_terminal){
  y_predicted <- rep(0, N_test)
  for (i in 1:N_test) {
    index <- 1
    while (1) {
      if (is_terminal[index] == TRUE) {
        return(mn[index])
      } else {
        if (n <= ns[index]) {
          index <- index * 2
        } else {
          index <- index * 2 + 1}}}}}

# testing
P <- 25
res <- dec_tree_reg(P)

# plotting data points
minimum_value <- 1.5
maximum_value <- 5.5
plot(x_train, y_train, type = "p", pch = 20,
     ylim = c(min(y_test), max(y_train)), xlim = c(minimum_value, maximum_value),
     ylab = "Waiting time to next eruption (min)", xlab = "Eruption time (min)", las = 1, main = sprintf("P = %g", P), col="purple")
legend("topleft", legend=c("training", "test"),
       col=c("purple", "red"), pch= 20, cex=0.7)
points(x_test, y_test, col = "red", pch = 20)

#plot
plotting_fitting <- function(){
  grid <- 10e-3
  interval <- seq(from = minimum_value, to = maximum_value, by = grid)
  for (c in 1:length(interval)) {
    left <- interval[c]
    right <- interval[c+1]
    x <- predict(left, res$ns, res$mn, res$it)
    lines(c(left, right), c(x,x), lwd = 2, col = "black")
    if (c < length(interval)) {
      a <- predict(left, res$ns, res$mn, res$it)
      b <- predict(right, res$ns, res$mn, res$it)
      lines(c(right, right), c(a,b), lwd = 2, col = "black") 
    }
  }
}
plotting_fitting()

# +++ ################ 5 ###################

# find rmse 
y_test_predicted <- sapply(X=1:N_test, FUN = function(i) predict(x_test[i], res$ns , res$mn, res$it))
find_rmse <- function(y_test, y_test_predicted) {
  rmse <- sum((y_test - y_test_predicted) ^ 2)  
  return(sqrt(rmse/length(y_test)))
}

sprintf("RMSE is %s when P is %s", trimws(format(round(find_rmse(y_test ,y_test_predicted),4), nsmall=3)), P)

# +++ ################ 6 ###################

# plot of p-rmse 
p_to_rmse_values <- sapply(X=seq(from=5, to=50, by=5), FUN = function(P) {
  res <- dec_tree_reg(P)
  y_predicted <- sapply(X=1:N_test, FUN = function(i) predict(x_test[i],res$ns, res$mn, res$it))
  rmse <- sqrt(sum((y_test-y_predicted)^2)/length(y_test))
})

plot(seq(from=5, to=50, by=5), p_to_rmse_values,
     type = "o", lwd = 2, las = 1, pch = 19, lty = 1,
     xlab = "Pre-Pruning size (P)", ylab = "RMSE")
# Final version