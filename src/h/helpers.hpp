// Weights must sum up to N that is the size of array.


// Unrolled version of inner-product. Showed at least 30% speedup against
// std::inner_product() for 32-bit floats.
// x and y can be the same.
template<typename num1, typename num2> 
inline double innerProd(num1 *x, num1 *xend, num2 *y)
{
  constexpr int ssize = 16;
  double ss[ssize];
  std::memset(ss, 0, ssize * sizeof(double));
  int i = 0;
  for (int iend = xend - x - ssize; i < iend; i += ssize)
  {
    for (int k = 0; k != ssize; ++k) 
      ss[k] += x[i + k] * (double)y[i + k];
  }
  double S = 0;
  for (int iend = xend - x; i < iend; ++i) S += x[i] * (double)y[i];
  for (i = 0; i != ssize; ++i) S += ss[i];
  return S;
}


template<typename num>
inline double sum(num *x, num *xend)
{
  constexpr int ssize = 16;
  double ss[ssize];
  std::memset(ss, 0, ssize * sizeof(double));
  int i = 0;
  for (int iend = xend - x - ssize; i < iend; i += ssize)
  {
    for (int k = 0; k != ssize; ++k) ss[k] += x[i + k];
  }
  double S = 0;
  for (int iend = xend - x; i < iend; ++i) S += x[i];
  for (i = 0; i != ssize; ++i) S += ss[i];
  return S;
}


template<typename num1, typename num2>
inline double sum(num1 *x, num1 *xend, num2 *w)
{
  return innerProd(x, xend, w);
}


template<typename num1, typename num2, typename num3> 
inline double innerProd(num1 *x, num1 *xend, num2 *y, num3 *w)
{
  constexpr const int ssize = 8;
  double ss[ssize];
  std::memset(ss, 0, ssize * sizeof(double));
  int i = 0;
  for (int iend = xend - x - ssize; i < iend; i += ssize)
  {
    for (int k = 0; k != ssize; ++k) 
      ss[k] += x[i + k] * (double)y[i + k] * w[i + k];
  }
  double S = 0;
  for (int iend = xend - x; i < iend; ++i) S += x[i] * (double)y[i] * w[i];
  for (i = 0; i != ssize; ++i) S += ss[i];
  return S;
}


template<typename num1, typename num2> 
inline double squaredEuc(num1 *x, num1 *xend, num2 *y)
{
  constexpr const int ssize = 16;
  double ss[ssize];
  std::memset(ss, 0, ssize * sizeof(double));
  int i = 0;
  for (int iend = xend - x - ssize; i < iend; i += ssize)
  {
    for (int k = 0; k != ssize; ++k) 
    {
      double tmp = x[i + k] - (double)y[i + k];
      ss[k] += tmp * tmp;
    }
  }
  double S = 0;
  for (int iend = xend - x; i < iend; ++i) 
  {
    double tmp = x[i] - (double)y[i];
    S += tmp * tmp;
  }
  for (i = 0; i != ssize; ++i) S += ss[i];
  return S;
}


template<typename num1, typename num2, typename num3> 
inline double squaredEuc(num1 *x, num1 *xend, num2 *y, num3 *w)
{
  constexpr int ssize = 16;
  double ss[ssize];
  std::memset(ss, 0, ssize * sizeof(double));
  int i = 0;
  for (int iend = xend - x - ssize; i < iend; i += ssize)
  {
    for (int k = 0; k != ssize; ++k) 
    {
      double tmp = x[i + k] - (double)y[i + k];
      ss[k] += tmp * tmp * w[i + k];
    }
  }
  double S = 0;
  for (int iend = xend - x; i < iend; ++i) 
  {
    double tmp = x[i] - (double)y[i];
    S += tmp * tmp * w[i];
  }
  for (i = 0; i != ssize; ++i) S += ss[i];
  return S;
}


template<typename num>
inline std::pair<double, double> meanSD(num *x, num *xend) // return (mean sd).
{
  constexpr int ssize = 8;
  double container[ssize * 2];
  std::memset(container, 0, ssize * 2 * sizeof(double));
  double *sum = container, *sumSquare = sum + ssize;
  int i = 0;
  for (int iend = xend - x - ssize; i < iend; i += ssize)
  {
    for (int k = 0; k != ssize; ++k)
    {
      double t = x[i + k];
      sum[k] += t;
      sumSquare[k] += t * t;
    }
  }
  double S = 0, SS = 0;
  for (int iend = xend - x; i < iend; ++i)
  {
    double t = x[i];
    S  += t;
    SS += t * t;
  }
  for (int k = 0; k != ssize; ++k) { S += sum[k]; SS += sumSquare[k]; }
  double mean = S / (xend - x);
  double sd = std::sqrt(SS / (xend - x) - mean * mean);
  // Rcout << "mean = " << mean << ", sd = " << sd << "\n";
  return std::pair<double, double>(mean, sd);
}


template<typename num1, typename num2>
inline std::pair<double, double> meanSD(num1 *x, num1 *xend, num2 *w) // return (mean sd).
{
  constexpr int ssize = 8;
  double container[ssize * 2];
  std::memset(container, 0, ssize * 2 * sizeof(double));
  double *sum = container, *sumSquare = sum + ssize;
  int i = 0;
  for (int iend = xend - x - ssize; i < iend; i += ssize)
  {
    for (int k = 0; k != ssize; ++k)
    {
      double t = x[i + k], tw = t * w[i + k];
      sum[k] += tw;
      sumSquare[k] += tw * t;
    }
  }
  double S = 0, SS = 0;
  for (int iend = xend - x; i < iend; ++i)
  {
    double t = x[i], tw = t * w[i];
    S  += tw;
    SS += tw * t;
  }
  for (int k = 0; k != ssize; ++k) { S += sum[k]; SS += sumSquare[k]; }
  double mean = S / (xend - x);
  double sd = std::sqrt(SS / (xend - x) - mean * mean);
  return std::pair<double, double>(mean, sd);
}


template<typename inNum, typename outNum, typename scalerNum>
inline void stdize(inNum *x, inNum *xend, outNum *rst, 
                   scalerNum &m, scalerNum &s)
{
  auto tmp = meanSD(x, xend);
  double &mean = tmp.first, &sd = tmp.second;
  for (int i = 0, iend = xend - x; i < iend; ++i)
    rst[i] = sd == 0 ? 0 : (x[i] - mean) / sd;
  m = mean;
  s = sd;
}


template<typename inNum, typename outNum, typename weightNum, typename scalerNum>
inline void stdize(inNum *x, inNum *xend, weightNum *w, outNum *rst, 
                   scalerNum &m, scalerNum &s)
{
  auto tmp = meanSD(x, xend, w);
  double &mean = tmp.first, &sd = tmp.second;
  for (int i = 0, iend = xend - x; i < iend; ++i)
    rst[i] = sd == 0 ? 0 : (x[i] - mean) / sd;
  m = mean;
  s = sd;
}


// Xoriginal, rstX can be equal; yOriginal and rsty can be equal.
template<typename inNum, typename outNum>
inline void stdizeAll(inNum *Xoriginal, int nrow, int ncol, inNum *yOriginal,
                      outNum *rstX, outNum *rsty, outNum *Xmu, outNum *Xsd,
                      double &ymu, double &ysd)
{
  for (int j = 0; j < ncol; ++j)
  {
    int64_t offset = int64_t(nrow) * j;
    stdize(Xoriginal + offset, Xoriginal + offset + nrow,
           rstX + offset, Xmu[j], Xsd[j]);
  }
  stdize(yOriginal, yOriginal + nrow, rsty, ymu, ysd);
}


// Xoriginal, rstX can be equal; yOriginal and rsty can be equal.
template<typename inNum, typename outNum>
inline void stdizeAll(inNum *Xoriginal, int nrow, int ncol, 
                      inNum *yOriginal, inNum *w, // Weights for the training data.
                      outNum *rstX, outNum *rsty, 
                      outNum *Xmu, outNum *Xsd,
                      double &ymu, double &ysd, 
                      outNum *rstW)
{
  double S = sum(w, w + nrow);
  double scaler = nrow / S;
  for (int i = 0; i < nrow; ++i) rstW[i] = w[i] * scaler;
  for (int j = 0; j < ncol; ++j)
  {
    int64_t offset = int64_t(nrow) * j;
    stdize(Xoriginal + offset, Xoriginal + offset + nrow, rstW, 
           rstX + offset, Xmu[j], Xsd[j]);
  }
  stdize(yOriginal, yOriginal + nrow, rstW, rsty, ymu, ysd);
}






// Return the end pointer.
template<typename ing>
inline ing *union2sortedVec(ing *x, ing *xend, ing *y, ing *yend, ing *rst)
{
  ing *rstBegin = rst;
  while (x < xend and y < yend)
  {
    if (*x < *y) { *rst = *x; ++x; }
    else         { *rst = *y; ++y; }
    // if rst > rstBegin and rst == rst[-1]: do not increment rst.
    rst += rst == rstBegin or *rst != rst[-1];
  }
  if (y < yend) { x = y; xend = yend; }
  for (; x < xend; ++x)
  {
    *rst = *x;
    rst += rst == rstBegin or *rst != rst[-1];
  }
  return rst;
}




// x, y will be changed only if x.size() == 0 or y.size() == 0.
template<typename ing>
inline void union2sortedVec(vec<ing> &x, vec<ing> &y, vec<ing> &rst)
{
  if (x.size() == 0) { rst.swap(y); return; } 
  if (y.size() == 0) { rst.swap(x); return; } 
  rst.resize(x.size() + y.size());
  ing *end = union2sortedVec(
    &x[0], &x[0] + x.size(), &y[0], &y[0] + y.size(), &rst[0]);
  rst.resize(end - &rst[0]);
}




template<typename ing>
struct SparseLinearModel
{
  double lambda; // The lambda used to obtain this linear regression. Just for record.
  vec<ing> ind;
  vec<double> beta;
  void swap(SparseLinearModel &x) 
  { 
    std::swap(lambda, x.lambda);
    ind.swap(x.ind); 
    beta.swap(x.beta); 
  }
  SparseLinearModel(){}
  
  
  template<typename num1, typename num2>
  void assign(double lambda, num1 *cf, num1 *cfend, // Raw coefficients.
              num2 *Xmu, num2 *Xsd, double ymu, double ysd)
  {
    // Rcout << "in assign(): cf = ";
    // for (ing j = 0, jend = cfend - cf; j < jend; ++j)
    // {
    //   Rcout << cf[j] << " ";
    // }
    this->lambda = lambda;
    ind.resize(0);
    beta.resize(0);
    double bias = ymu;
    for (ing j = 0, jend = cfend - cf; j < jend; ++j)
    {
      if (cf[j] == 0) continue;
      double c = cf[j] * ysd / Xsd[j];
      // Rcout << "cf[j], ysd, Xsd[j] = " << cf[j] << ", " << ysd << ", " << Xsd[j] << "\n";
      bias -= c * Xmu[j];
      ind.push_back(j);
      beta.push_back(c);
    }
    beta.push_back(bias);
    // Rcout << "\nBefore leaving assign(): ";
    // for (int j = 0, jend = ind.size(); j < jend; ++j)
    //   Rcout << ind[j] << " ";
    // Rcout << "\n";
    // for (int j = 0, jend = ind.size(); j < jend; ++j)
    //   Rcout << beta[j] << " ";
    // Rcout << "\n\n";
  }
  
  
  template<typename num1, typename num2>
  void predict(num1 *X, ing nrow, num2 *yhat) // Results stored in yhat.
  {
    for (ing i = 0; i < nrow; ++i) yhat[i] = beta.back();
    for (ing u = 0, uend = ind.size(); u < uend; ++u)
    {
      ing j = ind[u];
      auto Xj = X + int64_t(nrow) * j;
      for (ing i = 0; i < nrow; ++i) yhat[i] += beta[u] * Xj[i];
    }
  }
};




template<typename ing>
struct SLMwindow
{
  int counter;
  int NnewModelWorse;
  double bestErr;
  vec<SparseLinearModel<ing> > Q;
  SparseLinearModel<ing> bestModel;
  vec<double> yhat;
  
  
  template<typename num1, typename num2>
  double sumOfSquaredErr(num1 *X, ing nrow, num2 *ytruth, 
                         SparseLinearModel<ing> &model)
  {
    yhat.resize(nrow);
    model.predict(X, nrow, &yhat[0]);
    return squaredEuc(
      &yhat[0], &yhat[0] + nrow, ytruth);
  }
  
  
  template<typename num1, typename num2, typename num3>
  double sumOfSquaredErr(num1 *X, ing nrow, num2 *ytruth, 
                         SparseLinearModel<ing> &model, num3 *w)
  {
    yhat.resize(nrow);
    model.predict(X, nrow, &yhat[0]);
    return squaredEuc(
      &yhat[0], &yhat[0] + nrow, ytruth, w);
  }
  
  
  // n = window size.
  void reset(ing n) 
  { 
    Q.resize(n); 
    counter = 0; 
    NnewModelWorse = 0; 
    bestErr = 1e300;
  }
  
  
  // Return True if NerrorIncreaseStreak >= earlyStoppingRounds. 
  template<typename num1, typename num2, typename num3>
  bool addNew(double lambda, 
              num1 *cf,  num1 *cfend, // Raw coefficients.
              num2 *Xmu, num2 *Xsd, 
              double ymu, double ysd,
              num2 *Xvali, num2 *yvali, 
              ing Nvali, num3 *wVali) // Weight on validation error.
    // Xvali and yvali are used for computing validation errors.
  {
    int i = counter % Q.size();
    Q[i].assign(lambda, cf, cfend, Xmu, Xsd, ymu, ysd);
    
    
    auto newErr = sumOfSquaredErr(Xvali, Nvali, yvali, Q[i], wVali);
    if (newErr < bestErr)
    {
      NnewModelWorse = 0;
      bestModel.swap(Q[i]);
      bestErr = newErr;
    }
    else NnewModelWorse += 1;
    counter += 1;
    return NnewModelWorse >= (int)Q.size();
  }
  
  
  // Return True if NerrorIncreaseStreak >= earlyStoppingRounds. 
  template<typename num1, typename num2>
  bool addNew(double lambda, 
              num1 *cf,  num1 *cfend, // Raw coefficients.
              num2 *Xmu, num2 *Xsd, 
              double ymu, double ysd,
              num2 *Xvali, num2 *yvali, 
              ing Nvali) // Weight on validation error.
    // Xvali and yvali are used for computing validation errors.
  {
    int i = counter % Q.size();
    Q[i].assign(lambda, cf, cfend, Xmu, Xsd, ymu, ysd);
    
    
    auto newErr = sumOfSquaredErr(Xvali, Nvali, yvali, Q[i]);
    if (newErr < bestErr)
    {
      NnewModelWorse = 0;
      bestModel.swap(Q[i]);
      bestErr = newErr;
    }
    else NnewModelWorse += 1;
    counter += 1;
    return NnewModelWorse >= (int)Q.size();
  }
  
  
  /*
  SparseLinearModel<ing> *selectModel(int i)
  {
    int k = (counter - (Q.size() - i % Q.size())) % Q.size();
    return &Q[k];
  }
  
  
  SparseLinearModel<ing> *newestModel()
  {
    return &Q[(counter - 1) % Q.size()];
  }
  */
  
  
};







