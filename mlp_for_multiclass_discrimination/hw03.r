setwd("~/Desktop/machine_learning/mlp_for_multiclass_discrimination/")

data_set <- read.csv("hw03_images.csv",header = FALSE)
y <- read.csv("hw03_labels.csv",header = FALSE)

X <- data_set[1:500,]
X <- as.matrix(X)
y_truth <- y[1:500,]

K <- max(y_truth)
N <- length(y_truth)
D <- ncol(X)

Y_truth <- matrix(0, N, K)
Y_truth[cbind(1:N, y_truth)] <- 1

test_set <- data_set[501:1000,]
test_set <- as.matrix(test_set)
test_y <- y[501:1000,]
test_y <- as.matrix(test_y)

Y_test_truth <- matrix(0, N, K)
Y_test_truth[cbind(1:N, test_y)] <- 1

W <- read.csv("initial_W.csv",header = FALSE)
W <- as.matrix(W)
v <- read.csv("initial_V.csv",header = FALSE)
v <- as.matrix(v)

sigmoid <- function(a) {
  return (1 / (1 + exp(-a)))
}

safelog <- function(x) {
  return (log(x + 1e-100))
}

softmax <- function(Z, v) {
  scores <- cbind(1,Z) %*% v
  scores <- exp(scores - matrix(apply(scores, MARGIN = 1, FUN = max), nrow = nrow(scores), ncol = ncol(scores), byrow = FALSE))
  scores <- scores / matrix(rowSums(scores), nrow(scores), ncol(scores), byrow = FALSE)
  return (scores)
}

eta <- 0.0005
epsilon <- 1e-3
max_iteration <- 500
H <- 20

Z <- sigmoid(cbind(1, X) %*% W)
Y_predicted <- softmax(Z,v)
objective_values <- -sum(Y_truth * log(Y_predicted))

iteration <- 1
while (1) {
  delta_v <- matrix(0, nrow=nrow(v), ncol=ncol(v))
  delta_W <- matrix(0, nrow=nrow(W), ncol=ncol(W))
  for (i in 1:500) {
    Z[i,] <- sigmoid(c(1, X[i,]) %*% W)
    Y_predicted[i,] <- softmax(matrix(Z[i,], 1, H), v)

    for (k in 1:K) {
      delta_v[, k] <- delta_v[, k] + eta * (Y_truth[i, k] - Y_predicted[i, k]) * c(1, Z[i,])
    }
    
    for (h in 1:H) {
      delta_W[, h] <- delta_W[, h] + eta * sum((Y_truth[i,] - Y_predicted[i,]) * v[h+1,]) * Z[i, h] * (1 - Z[i, h]) * c(1, X[i,])
    }
  }
  
  v <- v + delta_v
  W <- W + delta_W 
  
  Z <- sigmoid(cbind(1, X) %*% W)
  Y_predicted <- softmax(Z,v)
  objective_values <- c(objective_values, -sum(Y_truth * log(Y_predicted)))

  
  if (abs(objective_values[iteration + 1] - objective_values[iteration]) < epsilon | iteration >= max_iteration) {
    break
  }
  
  if (iteration >= max_iteration) {
    break
  }

  iteration <- iteration + 1
}

plot(1:(iteration + 1), objective_values,
     type = "l", lwd = 2, las = 1,
     xlab = "Iteration", ylab = "Error")

confusion_matrix <- rowSums(sapply(X=1:5, FUN=function(c) { Y_truth[,c] * c }))
y_predicted <- apply(Y_predicted, 1, which.max)
confusion_matrix <- table(y_predicted, confusion_matrix)
print(confusion_matrix)

Z <- sigmoid(cbind(1, test_set) %*% W)
Y_test_predicted <- softmax(Z,v)
test_confusion_matrix <- rowSums(sapply(X=1:5, FUN=function(c) { Y_test_truth[,c] * c }))
y_test_predicted <- apply(Y_test_predicted, 1, which.max)
test_confusion_matrix <- table(y_test_predicted, test_confusion_matrix)
print(test_confusion_matrix)