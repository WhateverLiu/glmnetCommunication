// [[Rcpp::plugins(cpp17)]]
#include <Rcpp.h>
// #include <RcppEigen.h>
// // [[Rcpp::depends(RcppEigen)]]
using namespace Rcpp;
#ifndef vec
#define vec std::vector
#endif


#include "h/tiktok.hpp"
tiktok<std::chrono::microseconds> timer;


#include "h/helpers.hpp"


int npasses, timeForKKTrelated, timeForCoordinateDescent;


// X and y have been standardized.
template<typename ing, typename num>
struct NaiveCoordinateDescentSTD
{
  ing Xnrow, Xncol, Nvali; // Nvali = number of rows in validation data.
  int lambdaPathLength;
  double thisLambda, alpha;
  double ymu, ysd;
  num *X, *w, *y, *Xmu, *Xsd, *Xvali, *yvali; // w: weights on training data.
  vec<double> wx2divNrow; // the j-th element stores sum_{i=1}^{N}w_i x_{ij} ^ 2.
  vec<double> beta;
  vec<double> residual, priorResidual;
  vec<double> Xresi; // X * residual
  vec<double> XresiCumulativeChangeUpperBound; // If this entry is 0, then Xresi is up-to-date.
  double lambdaDescent;
  double lambdaPathStopDevianceRelaIncrease;
  double lambdaPathStopDeviance;
  
  
  // `lambdaPathStopDevianceRelaIncrease`: stop exploring lambda path once R2 (
  // coefficient of determination, deviance) on the training set decreases 
  // relatively by less than `lambdaPathStopDevianceRelaIncrease`.
  // R2 = 1 - mean squared error / variance of y.
  // `lambdaPathStopDeviance`: also stop exploring lambda path once R2 is greater
  // than this number, e.g., 0.999.
  void reset(num *X, num *w, num *Xmu, num *Xsd, 
             ing Xnrow, ing Xncol,
             num *y, double ymu, double ysd,
             num *Xvali, num *yvali,
             ing Nvali, double alpha,
             double lambdaMinMaxRatio, 
             int lambdaPathLength,
             double lambdaPathStopDeviance,
             double lambdaPathStopDevianceRelaIncrease)
  {
    this->w = w;
    this->alpha = alpha;
    this->lambdaPathStopDeviance = lambdaPathStopDeviance;
    this->lambdaPathStopDevianceRelaIncrease = 
      lambdaPathStopDevianceRelaIncrease;
    this->X = X;
    this->y = y;
    this->Xmu = Xmu;
    this->Xsd = Xsd;
    this->Xnrow = Xnrow;
    this->Xncol = Xncol;
    this->ymu = ymu;
    this->ysd = ysd;
    this->Xvali = Xvali;
    this->yvali = yvali;
    this->Nvali = Nvali;
    this->lambdaPathLength = lambdaPathLength;
    lambdaDescent = std::pow(
      lambdaMinMaxRatio, 1.0 / std::max(1, lambdaPathLength - 1));
    // Rcout << "lambdaDescent = " << lambdaDescent << "\n";
    
    
    wx2divNrow.resize(Xncol);
    for (ing j = 0; j < Xncol; ++j)
    {
      auto *Xj = X + int64_t(Xnrow) * j;
      wx2divNrow[j] = innerProd(Xj, Xj + Xnrow, Xj, w) / Xnrow;
    }
    
    
    residual.assign(y, y + Xnrow);
    Xresi.resize(Xncol);
    thisLambda = -1e300;
    for (ing j = 0; j < Xncol; ++j)
    {
      Xresi[j] = innerProd(&residual[0], &*residual.end(), 
                           X + j * int64_t(Xnrow)
                             , w
      );
      thisLambda = std::max<num> (thisLambda, std::abs(Xresi[j]));
    }
    XresiCumulativeChangeUpperBound.assign(Xncol, 0);
    priorResidual.assign(residual.begin(), residual.end());
    thisLambda /= Xnrow * alpha;
    
    
    everActive.resize(0);
    strong.resize(0);
    beta.assign(Xncol + 1, 0);
    
    
    // for (int i = 0, iend = Xnrow; i < iend; ++i)
    //   Rcout << "reset(), y[i] = " << y[i] << ", ";
    // Rcout << "\n";
  }
  
  
  NaiveCoordinateDescentSTD(num *X, num *w, num *Xmu, num *Xsd, ing Xnrow, ing Xncol,
                            num *y, double ymu, double ysd,
                            num *Xval, num *yval, ing Nvali,
                            double alpha,
                            double lambdaMinMaxRatio, int lambdaPathLength,
                            double lambdaPathStopDeviance,
                            double lambdaPathStopDevianceRelaIncrease)
  {
    reset(X, w, Xmu, Xsd, Xnrow, Xncol, y, ymu, ysd, Xval, yval, Nvali, alpha,
          lambdaMinMaxRatio, lambdaPathLength, 
          lambdaPathStopDeviance,
          lambdaPathStopDevianceRelaIncrease);
  }
  
  
  // Training data X and y have been standardized.
  // `j` points to the j-th column (feature). If `kktCore()` is being called,
  //   then currently the j-th column's coefficient (beta) is zero.
  // `ub` is the upper bound, usually `lambda * Xnrow`. Here, `Xnrow` is the
  // number of rows (observations) in `X`.
  // `j` will be pushed into `toBeAdded` if KKT says the coefficient for the j-th
  // column should not be zero.
  void kktCore(ing j, double ub, vec<ing> &toBeAdded)
  {
    double &cumChangeUB = XresiCumulativeChangeUpperBound[j]; // An upper bound
    // of cumulative changes in the innerproduct of j-th column and the residual
    // vector.
    
    
    if (cumChangeUB != 0) // Implies the innerproduct value Xresi[j] is NOT
      // up-to-date. Here, Xresi[j] stores the innerproduct of the X's j-th 
      // column and the residual vector.
    {
      
      auto bound = std::max(std::abs(Xresi[j] - cumChangeUB), 
                            std::abs(Xresi[j] + cumChangeUB));
      // Xresi[j] - cumChangeUB <= 
      //   actual innerproduct of j-th column and residual vector
      //     <= Xresi[j] + cumChangeUB
      
      
      if (bound > ub) // Upper bound of the innerproduct violates KKT, so we are
        // not sure if the actual innerproduct would violate it. Need to
        // recompute the innerproduct value.
      {
        num *v = X + j * int64_t(Xnrow);
        Xresi[j] = innerProd(v, v + Xnrow, &residual[0], w);
        cumChangeUB = 0; // zero it to pronounce `Xresi[j]` is up-to-date.
        if (std::abs(Xresi[j]) > ub) // KKT violated. Add `j` in.
          toBeAdded.push_back(j);
      }
      // else do nothing because the even the upper bound does not violate KKT.
    }
    else // Innerproduct Xresi[j] is up-to-date.
    {
      if (std::abs(Xresi[j]) > ub) toBeAdded.push_back(j);
    }
  }
  
  
  vec<ing> strong;
  void computeStrong(double lambda, double thisLambda)
  {
    strong.resize(0);
    double ub = (2 * lambda - thisLambda) * Xnrow * alpha;
    for (ing j = 0; j < Xncol; ++j)
    {
      kktCore(j, ub, strong);
    }
  }
  
  
  // Check if zeroed features in `strong` meet KKT given lambda.
  // If not, push the feature index into toBeAdded.
  bool kkt(vec<ing> &strong, double lambda, vec<ing> &toBeAdded)
  {
    double ub = Xnrow * lambda * alpha;
    toBeAdded.resize(0);
    for (ing u = 0, uend = strong.size(); u < uend; ++u)
    {
      ing j = strong[u];
      if (beta[j] != 0) continue;
      kktCore(j, ub, toBeAdded);
    }
    return toBeAdded.size() == 0;
  }
  
  
  // Check if all zeroed features satisfy KKT given lambda. 
  // If not, push the feature index into toBeAdded.
  bool kkt(double lambda, vec<ing> &toBeAdded)
  {
    double ub = Xnrow * lambda * alpha;
    toBeAdded.resize(0);
    for (ing j = 0; j < Xncol; ++j)
    {
      if (beta[j] != 0) continue;
      kktCore(j, ub, toBeAdded); 
    }
    return toBeAdded.size() == 0;
  }
  
  
  // fsubset stores the feature indices subject to exploration.
  // Return rmse.
  void solve(double eps, int maxIter, double lambda, vec<ing> &fsubset)
  {
    
    int iter = 0;
    // double priorRmse = std::sqrt(innerProd(
    //   &residual[0], &*residual.end(), &residual[0]) / Xnrow), rmse = 1e300;
    eps *= Xnrow * 2;
    
    
    double lambdaAlpha = lambda * alpha;
    // double priorRmse = 1e300;
    while (iter < maxIter)
    {
      npasses += 1;
      
      
      double maxChange = 0;
      for (ing u = 0, uend = fsubset.size(); u < uend; ++u)
      {
        ing j = fsubset[u];
        num *Xj = X + int64_t(Xnrow) * j;    
        
        
        double thetaj = 0;
        double pj = innerProd(Xj, Xj + Xnrow, &residual[0], w);
        // double pj = std::inner_product(Xj, Xj + Xnrow, &residual[0], 0.0);
        
        pj = pj / Xnrow + beta[j];
        
        
        if      (pj < -lambdaAlpha) thetaj = pj + lambdaAlpha;
        else if (pj >  lambdaAlpha) thetaj = pj - lambdaAlpha;
        thetaj /= wx2divNrow[j] + lambda * (1 - alpha);
        
        
        double deltaCoef = beta[j] - thetaj;
        if (std::abs(deltaCoef) >= 1e-15)
        {
          double change = 0;
          for (ing i = 0; i < Xnrow; ++i)
          {
            double c = deltaCoef * Xj[i];
            residual[i] += c;
            change += c * c * w[i];
          }
          maxChange = std::max(change, maxChange);
        }
        
        
        beta[j] = thetaj;
      }
      
      
      if (maxChange < eps) break;
      iter += 1;
    }
  }
  
  
  // B1, B2 are temporary buffers.
  vec<ing> everActive, B1, B2;
  double solveForOneLambda( // Return rmse.
      double eps, int maxIter, double lambda) // maxIter is the max
    // number of iterations fed to solve()
  {
    // Rcout << "thisLambda = " << thisLambda << ", ";
    // Rcout << "lambda = " << lambda << ", ";
    // Rcout << "lambda / thisLambda = " << lambda / thisLambda << ", ";
    // Rcout << "strong.size() = " << strong.size() << ", ";
    // Rcout << "everActive.size() = " << everActive.size() << "\n";
    
    
    while (true)
    {
      timer.tik();
      computeStrong(lambda, thisLambda);
      timeForKKTrelated += timer.tok();
      
      
      while (true)
      {
        
        timer.tik();
        solve(eps, maxIter, lambda, everActive);
        timeForCoordinateDescent += timer.tok();
        
        
        // Find the norm of changes in `residual`, and update 
        // `XresiCumulativeChangeUpperBound`.
        if (true)
        {
          // `mag` is the L2 norm of changes in residuals.
          // `squaredEuc` is squared Euclidean distance function.
          double mag = std::sqrt(squaredEuc(
            &residual[0], &*residual.end(), &priorResidual[0], w));
          std::copy(residual.begin(), residual.end(), priorResidual.begin());
          
          
          // Update the upper bound of innerproduct of every column and residuals.
          // This makes a loose but effective upper bound.
          for (ing j = 0; j < Xncol; ++j)
            XresiCumulativeChangeUpperBound[j] += mag;
        }
        
        
        // Check the KKT condition for every zeroed feature in strong. 
        // Any violation will push back the variable index into B1.
        timer.tik();
        bool met = kkt(strong, lambda, B1); 
        timeForKKTrelated += timer.tok();
        
        
        if (met) break; // No violations occured.
        
        
        // Merge elements in everActive and B1 together to B2:
        timer.tik();
        union2sortedVec(everActive, B1, B2);
        everActive.swap(B2);
        timeForKKTrelated += timer.tok();
      }
      
      
      // If it comes to this point, then all variables in `strong` satisfy
      // KKT, and we only need to check all the other zeroed variables.
      timer.tik();
      bool met = kkt(lambda, B1); 
      timeForKKTrelated += timer.tok();
      
      
      if (met) { break; } // No violations occured.
      
      
      timer.tik();
      union2sortedVec(everActive, B1, B2);
      everActive.swap(B2);
      timeForKKTrelated += timer.tok();
      
      
    }
    
    
    return std::sqrt(innerProd(
        &residual[0], &*residual.end(), &residual[0], w) / Xnrow);
  }
  
  
  // Stop once earlyStoppingRounds models are explored after the best model.
  SLMwindow<ing> slm;
  template<typename weightNum>
  SparseLinearModel<ing> * operator()(
      double eps, int maxIter, int earlyStoppingRounds, 
      weightNum *wVali = nullptr) // wVali is the weights on validation data.
  {
    // for (int i = 0, iend = Xnrow; i < iend; ++i)
    //   Rcout << "in operator()(), y[i] = " << y[i] << ", ";
    // Rcout << "\n\n";
    
    
    slm.reset(earlyStoppingRounds);
    double lambda = thisLambda * lambdaDescent;
    double priorR2 = 1e-16;
    int nlbda = 1;
    
    
    // tiktok<std::chrono::microseconds> tmptimer;
    // int tsolve1lambda = 0;
    // int tslmaddnew = 0;
    while (nlbda < lambdaPathLength)
    {
      
      bool shouldQuit = false;
      
      
      // tmptimer.tik();
      double rmse = solveForOneLambda(eps, maxIter, lambda);
      // tsolve1lambda += tmptimer.tok();
      
      
      // Rcout << "rmse = " << rmse << "\n";
      double R2 = 1 - rmse * rmse;
      shouldQuit |= R2 > lambdaPathStopDeviance;
      shouldQuit |= R2 / priorR2 < 1 + lambdaPathStopDevianceRelaIncrease;
      priorR2 = R2;
      
      
      // tmptimer.tik();
      shouldQuit |= slm.addNew(
        lambda, &beta[0], &*beta.end(), Xmu, Xsd, ymu, ysd, 
        Xvali, yvali, Nvali, wVali);
      // tslmaddnew += tmptimer.tok();
      
      
      if (shouldQuit) break;
      thisLambda = lambda;
      lambda *= lambdaDescent;
      nlbda += 1;
    }
    
    
    // Rcout << "tsolve1lambda = " << tsolve1lambda << "\n";
    // Rcout << "tslmaddnew = " << tslmaddnew << "\n";
    
    return &slm.bestModel;
  }
  
  
  
  // Return all the models.
  void operator()(double eps, int maxIter, 
                vec<vec<double> > &allBetas) // wVali is the weights on validation data.
  {
    double lambda = thisLambda * lambdaDescent;
    double priorR2 = 1e-16;
    int nlbda = 1;
    
    allBetas.reserve(lambdaPathLength);
    allBetas.resize(0);
    while (nlbda < lambdaPathLength)
    {
      
      bool shouldQuit = false;
      double rmse = solveForOneLambda(eps, maxIter, lambda);
      double R2 = 1 - rmse * rmse;
      shouldQuit |= R2 > lambdaPathStopDeviance;
      shouldQuit |= R2 / priorR2 < 1 + lambdaPathStopDevianceRelaIncrease;
      priorR2 = R2;
      
      
      allBetas.push_back(vec<double>(Xncol + 1));
      if (true)
      {
        double bias = ymu;
        for (ing j = 0; j < Xncol; ++j)
        {
          if (beta[j] == 0) continue;
          double c = beta[j] * ysd / Xsd[j];
          bias -= c * Xmu[j];
          allBetas.back()[j] = c;
        }
        allBetas.back().back() = bias;
      }
      
      
      if (shouldQuit) break;
      
      
      thisLambda = lambda;
      lambda *= lambdaDescent;
      nlbda += 1;
    }
  }
  
  
  
  
  
};




// [[Rcpp::export]]
List testWeightedEnet(
    NumericMatrix Xr, 
    NumericVector yr,
    NumericVector Wtrain,
    NumericMatrix XrVali, 
    NumericVector yrVali,
    NumericVector Wvalid,
    double alpha = 1.0,
    double lambdaMinMaxRatio = 0.01, 
    int lambdaPathLength = 100,
    int maxIter = 100, // Max iteration for solving every lambda.
    int earlyStoppingRounds = 10,
    double eps = 1e-10,
    double lambdaPathStopDeviance = 0.999,
    double lambdaPathStopDevianceRelaIncrease = 1e-5)
{
  
  
  int Nrow = yr.size(), Ncol = Xr.ncol();
  
  
#define dtype double
  
  
  vec<dtype> Xy(Xr.size() + yr.size() + Nrow + 2 * Ncol + 
    XrVali.size() + yrVali.size());
  dtype *X = &Xy[0], *y = X + Xr.size(), *w = y + Nrow,
    *Xmu = w + Nrow, *Xsd = Xmu + Ncol,
    *Xvali = Xsd + Ncol, *yvali = Xvali + XrVali.size();
    
    
    std::copy(XrVali.begin(), XrVali.end(), Xvali);
    std::copy(yrVali.begin(), yrVali.end(), yvali);
    
    
    double ymu, ysd;
    stdizeAll(&Xr[0], Nrow, Ncol, &yr[0], &Wtrain[0], X, 
              y, Xmu, Xsd, ymu, ysd, w);
    
    
    NaiveCoordinateDescentSTD<int, dtype> solver(
        X, w, Xmu, Xsd, Nrow, Ncol, y, ymu, ysd, Xvali, yvali, yrVali.size(), alpha,
        lambdaMinMaxRatio, lambdaPathLength,
        lambdaPathStopDeviance,
        lambdaPathStopDevianceRelaIncrease);
    
    
#undef dtype
    
    
    
    npasses = 0;
    timeForKKTrelated = 0;
    timeForCoordinateDescent = 0;
    
    
    auto *model = solver(eps, maxIter, earlyStoppingRounds, &Wvalid[0]);
    Rcout << "npasses = " << npasses << "\n";
    Rcout << "timeForKKTrelated = " << timeForKKTrelated << "ms\n";
    Rcout << "timeForCoordinateDescent = " << timeForCoordinateDescent << "ms\n";
    
    
    
    NumericVector yhat(yrVali.size());
    model->predict<double, double> (&XrVali[0], yrVali.size(), &yhat[0]);
    
    
    NumericVector coef(Ncol + 1);
    // Rcout << "model->ind.size() = " << model->ind.size() << "\n";
    for (int i = 0, iend = model->ind.size(); i < iend; ++i)
    {
      int j = model->ind[i];
      coef[j] = model->beta[i];
    }
    coef[coef.size() - 1] = model->beta.back();
    
    
    return List::create(Named("coef") = coef, Named("yhat") = yhat);
}




// [[Rcpp::export]]
std::vector<std::vector<double> > testWeightedEnetReturnAll(
    NumericMatrix Xr, 
    NumericVector yr,
    NumericVector Wtrain,
    NumericMatrix XrVali, 
    NumericVector yrVali,
    NumericVector Wvalid,
    double alpha = 1.0,
    double lambdaMinMaxRatio = 0.01, 
    int lambdaPathLength = 100,
    int maxIter = 100, // Max iteration for solving every lambda.
    double eps = 1e-10,
    double lambdaPathStopDeviance = 0.999,
    double lambdaPathStopDevianceRelaIncrease = 1e-5)
{
  
  
  int Nrow = yr.size(), Ncol = Xr.ncol();
  
  
#define dtype float
  
  
  vec<dtype> Xy(Xr.size() + yr.size() + Nrow + 2 * Ncol + 
    XrVali.size() + yrVali.size());
  dtype *X = &Xy[0], *y = X + Xr.size(), *w = y + Nrow,
    *Xmu = w + Nrow, *Xsd = Xmu + Ncol,
    *Xvali = Xsd + Ncol, *yvali = Xvali + XrVali.size();
    
    
    std::copy(XrVali.begin(), XrVali.end(), Xvali);
    std::copy(yrVali.begin(), yrVali.end(), yvali);
    
    
    double ymu, ysd;
    stdizeAll(&Xr[0], Nrow, Ncol, &yr[0], &Wtrain[0], X, 
              y, Xmu, Xsd, ymu, ysd, w);
    
    
    NaiveCoordinateDescentSTD<int, dtype> solver(
        X, w, Xmu, Xsd, Nrow, Ncol, y, ymu, ysd, Xvali, yvali, yrVali.size(), alpha,
        lambdaMinMaxRatio, lambdaPathLength,
        lambdaPathStopDeviance,
        lambdaPathStopDevianceRelaIncrease);
    
    
#undef dtype
    
    
    
    npasses = 0;
    timeForKKTrelated = 0;
    timeForCoordinateDescent = 0;
    
    
    // auto *model = solver(eps, maxIter, earlyStoppingRounds, &Wvalid[0]);
    vec<vec<double> > allbetas;
    solver(eps, maxIter, allbetas);
    Rcout << "npasses = " << npasses << "\n";
    Rcout << "timeForKKTrelated = " << timeForKKTrelated << "ms\n";
    Rcout << "timeForCoordinateDescent = " << timeForCoordinateDescent << "ms\n";
    return allbetas;
}





#ifdef vec
#undef vec
#endif


















