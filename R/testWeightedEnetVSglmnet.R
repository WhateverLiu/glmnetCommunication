

set.seed(123)


Nrow = 1000
Ncol = 5000
X = matrix(runif(Nrow * Ncol), nrow = Nrow)


# # Make X's columns somewhat correlated.
tmp = matrix(runif(Ncol * Ncol, -1, 1), Ncol)
# system.time({X = X %*% tmp}); rm(tmp); gc()
system.time({X = keyALGs::matmul(X, tmp, maxCore = 15) }); rm(tmp); gc()


# How many features (columns) are predictors ~ 1%
r = 0.01 # Larger `r` creates denser problem.
NeffectiveFeature = max(1L, as.integer(round(Ncol * r)))


# Use the selected predictors to construct y.
predInd = sample(Ncol, NeffectiveFeature)
randCoef = as.matrix(runif(NeffectiveFeature, -1, 1))
randNoise = rnorm(Nrow) * 0.01
system.time({y = X[, predInd, drop = F] %*% randCoef + randNoise})


# Train - validation split ratio
validRatio = 0.1
validInd = 1:max(2L, as.integer(round(Nrow * validRatio)))
Xvalid = X[validInd, , drop = F]
yvalid = y[validInd]
X = X[-validInd, , drop = F]
y = y[-validInd]


# Standardize data for easier comparison
X = apply(X, 2, function(x) (x - mean(x)) / sqrt(mean((x - mean(x)) ^ 2)))
y = (y - mean(y)) / sqrt(mean((y - mean(y)) ^ 2))
trainW = rep(1, nrow(X))
  

# Validation data will not concern glmnet. 
Xvalid = apply(Xvalid, 2, function(x) (x - mean(x)) / sqrt(mean((x - mean(x)) ^ 2)) )
yvalid = (yvalid - mean(yvalid)) / sqrt(mean((yvalid - mean(yvalid)) ^ 2))
validW = rep(1, nrow(Xvalid))


Rcpp::sourceCpp("src/trainWeightedEnet008.cpp", verbose = T, rebuild = F)


lambdaMinRatio = 0.01
Nlambda = 100




alpha = 1 # Lower alpha creates denser problem.


myTime = system.time({myRst = testWeightedEnetReturnAll(
  X, y, trainW, Xvalid, yvalid, validW,
  alpha = alpha, lambdaMinMaxRatio = lambdaMinRatio, lambdaPathLength = Nlambda,
  maxIter = 1000, eps = 1e-7,
  lambdaPathStopDeviance = 0.999,
  lambdaPathStopDevianceRelaIncrease = 1e-5)})
mineAllModelRmse = unlist(lapply(myRst, function(b)
{
  yhat = X %*% b[-length(b)] + b[length(b)]
  sqrt(mean((yhat - y) ^ 2))
}))




glmnetTime = system.time({glmnetRst = glmnet::glmnet(
  X, y, weights = NULL, family = "gaussian", alpha = alpha,
  type.gaussian = "naive", nlambda = Nlambda, 
  lambda.min.ratio = lambdaMinRatio,
  intercept = T)})
glmnetPreds = predict(glmnetRst, X)
glmnetAllModelRmse = apply(glmnetPreds, 2, function(x) 
  sqrt(mean((x - y) ^ 2)))[-1]
names(glmnetAllModelRmse) = NULL




cat(
  "My last model rmse =", round(mineAllModelRmse[length(mineAllModelRmse)], 4), ",",
  "My avg model rmse =", round(mean(mineAllModelRmse), 4), ",",
  "My time =", myTime[3], "\n",
  "glmnet last model rmse =", round(glmnetAllModelRmse[length(glmnetAllModelRmse)], 4), ",",
  "glmnet avg model rmse =", round(mean(glmnetAllModelRmse), 4), ",",
  "glmnet time =", glmnetTime[3], "\n",
  "glmnet npass =", glmnetRst$npasses, ",",
  "glmnet final model N(coef) =", sum(glmnetRst$beta[, ncol(glmnetRst$beta)] != 0), "\n"
)




















