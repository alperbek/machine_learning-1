setwd("~/Desktop/engr421_dasc521_fall2019_hw02")

data_set <- read.csv("hw02_images.csv",header = FALSE)
y <- read.csv("hw02_labels.csv",header = FALSE)

X <- data_set[1:500,]
X <- as.matrix(X)
y_truth <- y[1:500,]

K <- max(y_truth)
N <- length(y_truth)

Y_truth <- matrix(0, N, K)
Y_truth[cbind(1:N, y_truth)] <- 1

test_set <- data_set[501:1000,]
test_set <- as.matrix(test_set)
test_y <- y[501:1000,]
test_y <- as.matrix(test_y)

w <- read.csv("initial_w.csv",header = FALSE)
w <- as.matrix(w)
w0 <- read.csv("initial_w0.csv",header = FALSE)
w0 <- t(as.matrix(w0))

sigmoid <- function(X,w,w0) {
  return (1 / (1 + exp(-cbind(X, 1) %*% rbind(w, w0))))
}


gradient_W <- function(X, Y_truth, Y_predicted) {
  return (-sapply(X = 1:ncol(Y_truth), function(c) colSums(matrix((Y_truth[,c] - Y_predicted[,c]) * Y_predicted[,c] * (1 - Y_predicted[,c]), nrow = nrow(X), ncol = ncol(X), byrow = FALSE) * X)))
} 

gradient_w0 <- function(Y_truth, Y_predicted) {
  return (-colSums((Y_truth - Y_predicted) * Y_predicted * (1 -Y_predicted)))
}

eta <- 0.0001
epsilon <- 1e-3
max_iteration <- 500

iteration <- 1
objective_values <- c()

while (1) {
  Y_predicted <- sigmoid(X, w, w0)
  objective_values <- c(objective_values, sum((1/2) * (Y_truth - Y_predicted)^2))
 
  w_old <- w
  w0_old <- w0
  w <- w - eta * gradient_W(X, Y_truth, Y_predicted)
  w0 <- w0 - eta * gradient_w0(Y_truth, Y_predicted)
  
  if (sqrt(sum((w0 - w0_old)^2) + sum((w - w_old)^2)) < epsilon) {
    break
  }
  
  if (iteration >= max_iteration) {
    break
  }
  
  iteration <- iteration + 1
}

plot(1:iteration, objective_values,
     type = "l", lwd = 2, las = 1,
     xlab = "Iteration", ylab = "Error")
 
y_predicted <- apply(Y_predicted, MARGIN = 1, FUN = which.max)
confusion_matrix <- table(y_predicted, y_truth)
print(confusion_matrix)
 
test_predicted <- sigmoid(test_set, w, w0)
test_predicted <- apply(test_predicted, MARGIN = 1, FUN = which.max)
confusion_matrix <- table(test_predicted, test_y)
print(confusion_matrix)