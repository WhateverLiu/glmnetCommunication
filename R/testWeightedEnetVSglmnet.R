

set.seed(123)
Nrow = 1000
Ncol = 5000
X = matrix(runif(Nrow * Ncol), nrow = Nrow)


# Make X's columns somewhat correlated.
tmp = matrix(runif(Ncol * Ncol, -1, 1), Ncol)
system.time({X = X %*% tmp}); rm(tmp); gc()


# How many features (columns) are predictors ~ 50
NeffectiveFeature = max(1L, as.integer(round(Ncol * 0.01)))


# Use the selected predictors to construct y.
predInd = sample(Ncol, NeffectiveFeature)
randCoef = as.matrix(runif(NeffectiveFeature, -1, 1))
randNoise = rnorm(Nrow) * 0.001
system.time({y = X[, predInd, drop = F] %*% randCoef + randNoise})


# Train - validation split ratio
validRatio = 0.1
validInd = 1:max(2L, as.integer(round(Nrow * validRatio)))
Xvalid = X[validInd, , drop = F]
yvalid = y[validInd]
X = X[-validInd, , drop = F]
y = y[-validInd]


# Standardize data for making easy comparison
X = apply(X, 2, function(x) (x - mean(x)) / sqrt(mean((x - mean(x)) ^ 2)))
y = (y - mean(y)) / sqrt(mean((y - mean(y)) ^ 2))
trainW = rep(1, nrow(X))
  

# Validation data will not concern glmnet. 
Xvalid = apply(Xvalid, 2, function(x) (x - mean(x)) / sqrt(mean((x - mean(x)) ^ 2)) )
yvalid = (yvalid - mean(yvalid)) / sqrt(mean((yvalid - mean(yvalid)) ^ 2))
validW = rep(1, nrow(Xvalid))


Rcpp::sourceCpp("src/trainWeightedEnet006.cpp", verbose = T)


lambdaMinRatio = 0.01
Nlambda = 100


cat("My implementation time cost and training RMSE from the last model on LASSO path:\n")
alpha = 1
system.time({myRst = testWeightedEnetReturnAll(
  X, y, trainW, Xvalid, yvalid, validW,
  alpha = alpha, lambdaMinMaxRatio = lambdaMinRatio, lambdaPathLength = Nlambda,
  maxIter = 1000, eps = 1e-10,
  lambdaPathStopDeviance = 0.999,
  lambdaPathStopDevianceRelaIncrease = 1e-5)})
lastMdl = myRst[[length(myRst)]]
yhat = X %*% lastMdl[-length(lastMdl)] + lastMdl[length(lastMdl)]
sqrt(mean((yhat - y) ^ 2))




cat("glmnet time cost and training RMSE of the last model on LASSO path:")
system.time({glmnetRst = glmnet::glmnet(
  X, y, weights = NULL, family = "gaussian", alpha = alpha,
  type.gaussian = "naive", nlambda = Nlambda, 
  lambda.min.ratio = lambdaMinRatio,
  intercept = T)})
tmp = predict(glmnetRst, X)
glmnetRstLastBeta = glmnetRst$beta[, ncol(glmnetRst$beta)]
names(glmnetRstLastBeta) = NULL
sqrt(mean((tmp[, ncol(tmp)] - y) ^ 2))
glmnetRst$npasses
sum(glmnetRstLastBeta != 0)






