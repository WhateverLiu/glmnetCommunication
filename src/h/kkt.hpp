
#ifndef vec
#define vec std::vector
#endif


// Nhistory should be small, e.g., 5
template<typename num>
struct KKT
{
  uint16_t latestHistID;
  int nrow, ncol;
  num *X;
  vec<vec<double> > RH; // Residual history
  vec<double> Xresi;
  vec<uint16_t> XresiID;
  
  
  // X and y have been standardized.
  void reset(num *X, num *y, num *beta, int nrow, int ncol, int Nhistory)
  {
    this->X = X;
    this->nrow = nrow;
    this->ncol = ncol;
    Nhistory = std::max(Nhistory, 3);
    RH.resize(Nhistory);
    
    
    // Compute the first residual in history.
    if (true)
    {
      RH.back().assign(nrow, 0);
      double *yhat = &RH.back()[0];
      for (int j = 0; j < ncol; ++j)
      {
        if (beta[j] == 0) continue;
        double *Xj = X + int64_t(nrow) * j;
        for (int i= 0; i < nrow; ++i)
          yhat[i] += beta[j] * Xj[i];
      }
      for (int i = 0; i < nrow; ++i)
        yhat[i] = y[i] - yhat[i];
      // yhat now stores the residual.
    }
    
    
    Xresi.resize(ncol);
    for (int j = 0; j < ncol; ++j)
      Xresi[j] = innerProd(&RH.back()[0], &RH.back()[0] + nrow, 
                           X + int64_t(nrow) * j);
    latestHistID = 0;
    XresiID.assign(ncol, 0);
  }
  
  
  void addHistory(vec<double> &residual)
  {
    
  }
  
  
  
  
};




#ifdef vec
#undef vec
#endif









