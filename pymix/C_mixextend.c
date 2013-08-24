
#include <Python.h>
#include <arrayobject.h> 

#include <stdlib.h>
#include <math.h>

#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_gamma.h>

#include "C_mixextend.h"

/* Define global gsl RNG */
const gsl_rng *RNG;

static PyMethodDef _C_mixextendMethods[] = { 
      {"test", test, METH_VARARGS},
      {"matrix_sum_logs", matrix_sum_logs, METH_VARARGS},
      {"get_normalized_posterior_matrix", get_normalized_posterior_matrix, METH_VARARGS},
      {"sum_logs", sum_logs, METH_VARARGS},
      {"wrap_gsl_dirichlet_lnpdf", wrap_gsl_dirichlet_lnpdf, METH_VARARGS},
      {"wrap_gsl_dirichlet_pdf", wrap_gsl_dirichlet_pdf, METH_VARARGS},
      {"wrap_gsl_ran_gaussian_pdf", wrap_gsl_ran_gaussian_pdf, METH_VARARGS},
      {"wrap_gsl_sf_gamma", wrap_gsl_sf_gamma, METH_VARARGS},
      {"wrap_gsl_sf_lngamma", wrap_gsl_sf_lngamma, METH_VARARGS},
      {"get_log_normal_inverse_gamma_prior_density", get_log_normal_inverse_gamma_prior_density, METH_VARARGS},
      {"get_two_largest_elements", get_two_largest_elements, METH_VARARGS},
      {"get_likelihoodbounds", get_likelihoodbounds, METH_VARARGS},
      {"update_two_largest_elements", update_two_largest_elements, METH_VARARGS},
      {"add_matrix_get_likelihoodbounds", add_matrix_get_likelihoodbounds, METH_VARARGS},
      {"add_matrix", add_matrix, METH_VARARGS},
      {"substract_matrix", substract_matrix, METH_VARARGS},      
      {"wrap_gsl_dirichlet_sample", wrap_gsl_dirichlet_sample, METH_VARARGS},     
      {"set_gsl_rng_seed", set_gsl_rng_seed, METH_VARARGS},     
      {NULL, NULL}     /* Sentinel - marks the end of this structure */
};



void init_C_mixextend(void)  {
   unsigned long randSeed;
   
   (void) Py_InitModule("_C_mixextend", _C_mixextendMethods);
   import_array();  /* Must be present for NumPy.  Called first after above line. */
   
    RNG = gsl_rng_alloc(gsl_rng_mt19937);
    srand(time(NULL));                    /* initialization for rand() */
    randSeed = rand();                    /* returns a non-negative integer */
    gsl_rng_set (RNG, randSeed);    /* seed the PRNG */

}


/*-------------------------------------------------------------------------------------------------------------------*/
/*                                               Numpy arrays                                                        */
/*-------------------------------------------------------------------------------------------------------------------*/

/* In order to access the entries of numpy arrays we need to construct pointers to C arrays of the appropriate type */
/*                                                                                                                  */
/* The functions in this section were initially copied from the Cookbook recipe on the SciPy website                */
/* (www.scipy.org/Cookbook/C_Extensions/NumPy_arrays) by Lou Pecora and slightly modified subsequently.             */

void init_numpy(void){
   import_array();  
} 

double *pyvector_to_Carrayptrs(PyArrayObject *arrayin)  {
    return (double *) arrayin->data;  /* pointer to arrayin data as double */
}

double **ptrvector(long n)  {
    double **v;
    v=(double **)malloc((size_t) (n*sizeof(double)));
    if (!v)   {
        printf("In **ptrvector. Allocation of memory for double array failed.");
        exit(0);  }
    return v;
}

double **pymatrix_to_Carrayptrs(PyArrayObject *arrayin)  {
     double **c, *a;
     int i,n,m;
     
     n=arrayin->dimensions[0];
     m=arrayin->dimensions[1];
     c=ptrvector(n);
     a=(double *) arrayin->data;  /* pointer to arrayin data as double */
     for ( i=0; i<n; i++)  {
         c[i]=a+i*m;  }
     return c;
}

void free_Carrayptrs(double **v)  {
    free((char*) v);
}


/*-------------------------------------------------------------------------------------------------------------------*/
/*                                               Utility functions                                                   */
/*-------------------------------------------------------------------------------------------------------------------*/

double* matrix_col_max(double** mat, int row, int col){
 int i,j;
 double* col_max = malloc(col*sizeof(double));
 
  for(i=0;i<col;i++) {
    col_max[i] = mat[0][i];
    //printf("col 0,%d, %f\n", i,mat[0][i] );
    
    for(j=1;j<row;j++) {
      if(mat[j][i] > col_max[i]){

        //printf("  %f > %f -> new max\n", mat[j][i], col_max[j]);

        col_max[i] = mat[j][i];
      }
    }  
  }
  return col_max;
}



/* same functionality as matrix_sum_logs but operating on C data structures, meant for internal use */
double* internal_matrix_sum_logs(double** mat, int row, int col){

  int i,j;
  double x, *col_max;
  double* res = malloc(col*sizeof(double));

  // printf("row, col: %d, %d\n",row,col);
  
  
  /* get column-wise maxima */
  col_max = matrix_col_max(mat, row, col);
    
  for(j=0;j<col;j++){  
    res[j] = 0.0;
    for(i=0;i<row;i++) {
    
      //printf("a[%d][%d] = %f\n",i,j,mat[i][j]);
    
      if (mat[i][j] >= col_max[j]) {
        res[j] += 1.0;
      }
      else {
        x = mat[i][j] - col_max[j];
        if (x < -1.0e-16) {
           res[j] += exp(x);
        }
        else {
          res[j] += 1.0;
        }
      }
    }
    res[j] = log(res[j]);
    res[j] += col_max[j];

    // printf("res[%d] = %f\n",j,res[j]);

  }

  free(col_max);
  return res;
  
} 

/*-------------------------------------------------------------------------------------------------------------------*/
/*                                            Python extension functions                                             */
/*-------------------------------------------------------------------------------------------------------------------*/





/*
   For a given numpy vector of log values log_p1, ..., log_pk, returns the log of the sum (p1, ..., pk)

*/
PyObject* sum_logs(PyObject *self, PyObject *args){
  PyObject *input1;

  PyArrayObject *pyvec;
  double  *cvec, max_val, result, x;
  int i,len;

  if (!PyArg_ParseTuple(args, "Od", &input1, &max_val)){
      return NULL;
  }     

  pyvec = (PyArrayObject *) PyArray_ContiguousFromObject(input1, PyArray_DOUBLE, 1, 1);
  if (pyvec == NULL){
    return NULL;
  }  
  if (pyvec->nd != 1 || pyvec->descr->type_num != PyArray_DOUBLE) {
      PyErr_SetString(PyExc_ValueError, "array must be one-dimensional and of type float");
      return NULL;
  }

  cvec = pyvector_to_Carrayptrs(pyvec); 
  len = pyvec->dimensions[0];
 
  result = 0.0;
  
  for(i=0;i<len;i++) {
    
    //printf("a[%d] = %f\n",i,arr[i]);
    
    if (cvec[i] >= max_val) {
      result += 1.0;
    }
    else {
      x = cvec[i] - max_val;
      if (x < -1.0e-16) {
         result += exp(x);
      }
      else {
        result += 1.0;
      }
    }
    
    //printf("result[%d] = %f\n",i,result);

  }
  result = log(result);
  result += max_val;

   /* deallocation */
  Py_DECREF(pyvec); 
  return PyFloat_FromDouble(result);
}

/* 
   For a given matrix (numpy 2D array) of log values, computes the column-wise log of the sum

   Returns a PyArrayObject
*/
PyObject* matrix_sum_logs(PyObject *self, PyObject *args){
  PyObject *input;

  PyArrayObject *pymat;
  PyArrayObject *pyres;

  int i,j, col, row, dim[1];
  double x, *col_max, *cres, **cmat;
 
  if (!PyArg_ParseTuple(args, "O", &input)){
      return NULL;
  }     
  

  pymat = (PyArrayObject *) PyArray_ContiguousFromObject(input, PyArray_DOUBLE, 2, 2);
  if (pymat == NULL){
    return NULL;
  }  
  
  if (pymat->nd != 2 || pymat->descr->type_num != PyArray_DOUBLE) {
      PyErr_SetString(PyExc_ValueError, "array must be two-dimensional and of type float");
      return NULL;
  }
  
  cmat = pymatrix_to_Carrayptrs(pymat);  
  
  row = pymat->dimensions[0];
  col = pymat->dimensions[1];
  
  // printf("row, col: %d, %d\n",row,col);
  
  /* create output PyArrayObject and corresponding C data struct pointer */
  dim[0] = col;
  /*pyres = (PyArrayObject *) PyArray_SimpleNew(1, dim, PyArray_DOUBLE);*/
  pyres = (PyArrayObject *) PyArray_FromDims(1, dim, PyArray_DOUBLE);
  
  cres = pyvector_to_Carrayptrs(pyres); 
  
  /* get column-wise maxima */
  col_max = matrix_col_max(cmat, row, col);
    
  for(j=0;j<col;j++){  
    cres[j] = 0.0;
    for(i=0;i<row;i++) {
    
      //printf("a[%d][%d] = %f\n",i,j,cmat[i][j]);
    
      if (cmat[i][j] >= col_max[j]) {
        cres[j] += 1.0;
      }
      else {
        x = cmat[i][j] - col_max[j];
        if (x < -1.0e-16) {
           cres[j] += exp(x);
        }
        else {
          cres[j] += 1.0;
        }
      }
    }
    cres[j] = log(cres[j]);
    cres[j] += col_max[j];
  }  

   /* deallocation */
  Py_DECREF(pymat); 
  free_Carrayptrs(cmat);
  free(col_max);
  return PyArray_Return(pyres);
} 


/* 
  Normalizes the given log posterior matrix in-place and returns the log-likelihood.

*/
PyObject* get_normalized_posterior_matrix(PyObject *self, PyObject *args){

 PyObject *input;
 PyArrayObject *pymat;
 double log_p, **cmat;
 int i, j, col, row;
 double* sum_logs;

  if (!PyArg_ParseTuple(args, "O", &input)){
      return NULL;
  }     

  pymat = (PyArrayObject *) PyArray_ContiguousFromObject(input, PyArray_DOUBLE, 2, 2);
  if (pymat == NULL){
    return NULL;
  }  

  if (pymat->nd != 2 || pymat->descr->type_num != PyArray_DOUBLE) {
      PyErr_SetString(PyExc_ValueError, "array must be two-dimensional and of type float");
      return NULL;
  }

  row = pymat->dimensions[0];
  col = pymat->dimensions[1];
  cmat = pymatrix_to_Carrayptrs(pymat);  

  sum_logs = internal_matrix_sum_logs(cmat, row, col);
  
  /*printf("  sum_logs=");
  for(j=0;j<col;j++) {
    printf("%f, ",sum_logs[j]);
  }
  printf("\n");  */

  log_p = 0.0;
  for(j=0;j<col;j++) {
    log_p += sum_logs[j];
    for(i=0;i<row;i++) {
      if(fpclassify(sum_logs[j]) == FP_INFINITE){
         cmat[i][j] = sum_logs[j];
      }
      else{
        cmat[i][j] = cmat[i][j] - sum_logs[j];
      }  
    }
  } 

  /* deallocation */
  Py_DECREF(pymat); 
  free_Carrayptrs(cmat);
  free(sum_logs);
  return PyFloat_FromDouble(log_p);
}

/* 
  For given vectors of mu and sigma parameters of univariate Gaussians, returns
  the log normal inverse-gamma prior density for paramters mu_p, kappa, dof, scale

*/
PyObject* get_log_normal_inverse_gamma_prior_density(PyObject *self, PyObject *args){
 PyObject *input1, *input2;
 PyArrayObject *pymu, *pysigma, *pyres;
 double mu_p, kappa, dof, scale, *cmu, *csigma, *cres;
 int i, len, dim[1];


  if (!PyArg_ParseTuple(args, "ddddOO",&mu_p, &kappa, &dof, &scale ,&input1, &input2)){
      return NULL;
  }     

  pymu = (PyArrayObject *) PyArray_ContiguousFromObject(input1, PyArray_DOUBLE, 1, 1);
  if (pymu == NULL){
    return NULL;
  }  
  if (pymu->nd != 1 || pymu->descr->type_num != PyArray_DOUBLE) {
      PyErr_SetString(PyExc_ValueError, "array must be one-dimensional and of type float");
      return NULL;
  }

  pysigma = (PyArrayObject *) PyArray_ContiguousFromObject(input2, PyArray_DOUBLE, 1, 1);
  if (pysigma == NULL){
    return NULL;
  }  
  if (pysigma->nd != 1 || pysigma->descr->type_num != PyArray_DOUBLE) {
      PyErr_SetString(PyExc_ValueError, "array must be one-dimensional and of type float");
      return NULL;
  }

  cmu = pyvector_to_Carrayptrs(pymu);  
  csigma = pyvector_to_Carrayptrs(pysigma);  
  len = pymu->dimensions[0];

  /* create output PyArrayObject and corresponding C data struct pointer */
  dim[0] = len;
  /*pyres = (PyArrayObject *) PyArray_SimpleNew(1, dim, PyArray_DOUBLE);*/
  pyres = (PyArrayObject *) PyArray_FromDims(1, dim, PyArray_DOUBLE);
  
  
  cres = pyvector_to_Carrayptrs(pyres); 



  for(i=0;i<len;i++) {
    /* p_sigma = (n.sigma**2) ** (-(self.dof+2) / 2 ) * math.exp(- self.scale / (2 * n.sigma**2) ) */

    //printf("%f\n",  pow(2.0, 2.0)   );
    
    //printf("%f\n",  csigma[i]**2.0   );
    
    //printf("%f\n",  (- (dof+2.0) / 2.0)  );
    
   // printf("%f\n", (csigma[i]**2.0)** (- (dof+2.0) / 2.0)  );
    
    
    // printf("%f\n",  exp( -scale / (2.0 * csigma[i]**2.0))  );
    
    cres[i] = log(  pow(( pow(csigma[i],2.0) ), (- (dof+2.0) / 2.0)) * exp( -scale / (2.0 * pow(csigma[i], 2.0)) )  );
    
    /*  math.sqrt(n.sigma**2 / self.kappa) */
    cres[i] = cres[i] + log( gsl_ran_gaussian_pdf( cmu[i]-mu_p, sqrt( pow(csigma[i], 2.0) / kappa) )  );

    //printf("%d: %f\n",i,cres[i]);

    if(fpclassify(cres[i]) == FP_INFINITE){
     /* printf("Gamma( %f | %f, %f) = %f\n",csigma[i], dof,scale, log(  pow(( pow(csigma[i],2.0) ), (- (dof+2.0) / 2.0)) * exp( -scale / (2.0 * pow(csigma[i], 2.0)) )  ));
      printf("N(%f | %f, %f) = %f\n",cmu[i]-mu_p, 0.0, sqrt( pow(csigma[i], 2.0) / kappa) , log( gsl_ran_gaussian_pdf( cmu[i]-mu_p, sqrt( pow(csigma[i], 2.0) / kappa) )  ) );*/
      
      PyErr_SetString(PyExc_ValueError, "Zero probability under Normal-Inverse-Gamma prior.\n");
      return NULL;
    }

  }

  /* deallocation */
  Py_DECREF(pymu); 
  Py_DECREF(pysigma); 

  return PyArray_Return(pyres);

}


/* 
  For a given matrix returns column-wise the largest two elements. 

*/
PyObject* get_two_largest_elements(PyObject *self, PyObject *args){
  PyObject *input;

  PyArrayObject *pymat;
  PyArrayObject *pyres;

  int i,j, col, row, dim[2];
  double x, **cres, **cmat;

  if (!PyArg_ParseTuple(args, "O", &input)){
      return NULL;
  }     

  pymat = (PyArrayObject *) PyArray_ContiguousFromObject(input, PyArray_DOUBLE, 2, 2);
  if (pymat == NULL){
    return NULL;
  }  

  if (pymat->nd != 2 || pymat->descr->type_num != PyArray_DOUBLE) {
      PyErr_SetString(PyExc_ValueError, "array must be two-dimensional and of type float");
      return NULL;
  }


  row = pymat->dimensions[0];
  col = pymat->dimensions[1];
  cmat = pymatrix_to_Carrayptrs(pymat);  

  /* create output PyArrayObject and corresponding C data struct pointer */
  dim[0] = 2;
  dim[1] = col;
  
  /*pyres = (PyArrayObject *) PyArray_SimpleNew(2, dim, PyArray_DOUBLE);*/
  pyres = (PyArrayObject *) PyArray_FromDims(2, dim, PyArray_DOUBLE);
  cres = pymatrix_to_Carrayptrs(pyres); 


  for(j=0;j<col;j++) {
    
    if (cmat[0][j] > cmat[1][j]){
       cres[0][j] = cmat[0][j];
       cres[1][j] = cmat[1][j];
    }
    else {
       cres[0][j] = cmat[1][j];
       cres[1][j] = cmat[0][j];
    }
    
    for(i=2;i<row;i++) {
       if (cmat[i][j] > cres[0][j]){
          x = cres[0][j];
          cres[0][j] = cmat[i][j];
          if (x > cres[1][j]){
            cres[1][j] = x;
          }
          continue;
       }
       if (cmat[i][j] > cres[1][j]){
         cres[1][j] = cmat[i][j];
       }
    }
  } 


  /* deallocation */
  Py_DECREF(pymat); 
  free_Carrayptrs(cmat);

  return PyArray_Return(pyres);


}


/* 
   XXX

*/
PyObject* get_likelihoodbounds(PyObject *self, PyObject *args){
  PyObject *input;

  PyArrayObject *pymat;
  PyArrayObject *pyres;

  int i,j, col, row, dim[1];
  double x, *cres, **cmat, fst_val, scd_val;

  if (!PyArg_ParseTuple(args, "O", &input)){
      return NULL;
  }     

  pymat = (PyArrayObject *) PyArray_ContiguousFromObject(input, PyArray_DOUBLE, 2, 2);
  if (pymat == NULL){
    return NULL;
  }  

  if (pymat->nd != 2 || pymat->descr->type_num != PyArray_DOUBLE) {
      PyErr_SetString(PyExc_ValueError, "array must be two-dimensional and of type float");
      return NULL;
  }


  row = pymat->dimensions[0];
  col = pymat->dimensions[1];
  cmat = pymatrix_to_Carrayptrs(pymat);  

  /* create output PyArrayObject and corresponding C data struct pointer */
  dim[0] = 2;
  
  /*pyres = (PyArrayObject *) PyArray_SimpleNew(1, dim, PyArray_DOUBLE);*/
  pyres = (PyArrayObject *) PyArray_FromDims(1, dim, PyArray_DOUBLE);
  
  cres = pyvector_to_Carrayptrs(pyres); 
  cres[0] = 0.0;
  cres[1] = 0.0;


  for(j=0;j<col;j++) {
    
    if (cmat[0][j] > cmat[1][j]){
       fst_val = cmat[0][j];
       scd_val = cmat[1][j];
    }
    else {
       fst_val = cmat[1][j];
       scd_val = cmat[0][j];
    }
    
    for(i=2;i<row;i++) {
       if (cmat[i][j] > fst_val){
          x = fst_val;
          fst_val = cmat[i][j];
          if (x > scd_val){
            scd_val = x;
          }
          continue;
       }
       if (cmat[i][j] > scd_val){
         scd_val = cmat[i][j];
       }
    }
    cres[0] = cres[0] + fst_val;
    /* numpy.log( 1+ (numpy.exp( res[1,jj] - res[0,jj] )*(self.G-1)) ) */
    cres[1] = cres[1] + log( 1 + exp( scd_val - fst_val)* (row-1) ); 
  } 
  cres[1] = cres[1] + cres[0];

  /* deallocation */
  Py_DECREF(pymat); 
  free_Carrayptrs(cmat);

  return PyArray_Return(pyres);


}


/* 
   Updates the posterior matrix (first argument) in-place by assigning the sum of the
   second and third arguments. Updates the log-max likelihood bounds in-place.

*/
PyObject* add_matrix_get_likelihoodbounds(PyObject *self, PyObject *args){
  PyObject *input1, *input2, *input3, *input4;

  PyArrayObject *py_g;
  PyArrayObject *py_g_wo_j;
  PyArrayObject *py_l_j_1;
  PyArrayObject *py_bounds;
    
  int i,j, col, row, dim[1];
  double x, *c_bounds, **c_g, **c_g_wo_j, **c_l_j_1, fst_val, scd_val;

  if (!PyArg_ParseTuple(args, "OOOO", &input1, &input2, &input3, &input4)){
      return NULL;
  }     

  py_g = (PyArrayObject *) PyArray_ContiguousFromObject(input1, PyArray_DOUBLE, 2, 2);
  if (py_g == NULL){
    return NULL;
  }  

  if (py_g->nd != 2 || py_g->descr->type_num != PyArray_DOUBLE) {
      PyErr_SetString(PyExc_ValueError, "array must be two-dimensional and of type float");
      return NULL;
  }

  py_g_wo_j = (PyArrayObject *) PyArray_ContiguousFromObject(input2, PyArray_DOUBLE, 2, 2);
  if (py_g_wo_j == NULL){
    return NULL;
  }  

  if (py_g_wo_j->nd != 2 || py_g_wo_j->descr->type_num != PyArray_DOUBLE) {
      PyErr_SetString(PyExc_ValueError, "array must be two-dimensional and of type float");
      return NULL;
  }
  
  py_l_j_1 = (PyArrayObject *) PyArray_ContiguousFromObject(input3, PyArray_DOUBLE, 2, 2);
  if (py_l_j_1 == NULL){
    return NULL;
  }  

  if (py_l_j_1->nd != 2 || py_l_j_1->descr->type_num != PyArray_DOUBLE) {
      PyErr_SetString(PyExc_ValueError, "array must be two-dimensional and of type float");
      return NULL;
  }

  py_bounds = (PyArrayObject *) PyArray_ContiguousFromObject(input4, PyArray_DOUBLE, 1, 1);
  if (py_bounds == NULL){
    return NULL;
  }  

  if (py_bounds->nd != 1 || py_bounds->descr->type_num != PyArray_DOUBLE) {
      PyErr_SetString(PyExc_ValueError, "array must be one-dimensional and of type float");
      return NULL;
  }
  

  row = py_g->dimensions[0];
  col = py_g->dimensions[1];
  c_g = pymatrix_to_Carrayptrs(py_g);  
  c_g_wo_j = pymatrix_to_Carrayptrs(py_g_wo_j);  
  c_l_j_1 = pymatrix_to_Carrayptrs(py_l_j_1);  


  /* create output PyArrayObject and corresponding C data struct pointer */
  dim[0] = 2;
  
  c_bounds = pyvector_to_Carrayptrs(py_bounds); 
  c_bounds[0] = 0.0;
  c_bounds[1] = 0.0;


  for(j=0;j<col;j++) {
    
    c_g[0][j] = c_g_wo_j[0][j] + c_l_j_1[0][j];
    c_g[1][j] = c_g_wo_j[1][j] + c_l_j_1[1][j];
    
    //printf("%d,%d: %f + %f = %f\n",0,j,  c_g_wo_j[0][j], c_l_j_1[0][j],c_g[0][j] );
    //printf("%d,%d: %f + %f = %f\n",1,j,  c_g_wo_j[1][j], c_l_j_1[1][j],c_g[1][j] );

    
    
    if (c_g[0][j] > c_g[1][j]){
       fst_val = c_g[0][j];
       scd_val = c_g[1][j];
    }
    else {
       fst_val = c_g[1][j];
       scd_val = c_g[0][j];
    }
    
    for(i=2;i<row;i++) {
       c_g[i][j] = c_g_wo_j[i][j] + c_l_j_1[i][j];
       
      // printf("%d,%d: %f + %f = %f\n",i,j,  c_g_wo_j[i][j], c_l_j_1[i][j],c_g[i][j] );
       
       if (c_g[i][j] > fst_val){
          x = fst_val;
          fst_val = c_g[i][j];
          if (x > scd_val){
            scd_val = x;
          }
          continue;
       }
       if (c_g[i][j] > scd_val){
         scd_val = c_g[i][j];
       }
    }
    c_bounds[0] = c_bounds[0] + fst_val;
    /* numpy.log( 1+ (numpy.exp( res[1,jj] - res[0,jj] )*(self.G-1)) ) */
    c_bounds[1] = c_bounds[1] + log( 1 + exp( scd_val - fst_val)* (row-1) ); 
    
    // TEST
    //c_bounds[1] = c_bounds[1] + exp( scd_val - fst_val)* (row-1) ; 
    
    //x = 1 + exp( scd_val - fst_val)* (row-1);
    //printf("\nx= %f\n",x);
    //printf("log(x)= %f\n",log(x));
    //printf("upper bound log(x)= %f\n", x-1);
    
  } 
  c_bounds[1] = c_bounds[1] + c_bounds[0];

  /* deallocation */
  Py_DECREF(py_g); 
  Py_DECREF(py_g_wo_j); 
  Py_DECREF(py_l_j_1); 
  Py_DECREF(py_bounds); 
  
  free_Carrayptrs(c_g);
  free_Carrayptrs(c_g_wo_j);
  free_Carrayptrs(c_l_j_1);  

  return PyFloat_FromDouble(0.0);


}


/* 
   Updates the posterior matrix (first argument) in-place by assigning the sum of the
   second and third arguments. Returns the log-max likelihood bounds.

*/
PyObject* add_matrix(PyObject *self, PyObject *args){
  PyObject *input1, *input2, *input3;

  PyArrayObject *py_g;
  PyArrayObject *py_g_wo_j;
  PyArrayObject *py_l_j_1;
    

  int i,j, col, row;
  double **c_g, **c_g_wo_j, **c_l_j_1;

  if (!PyArg_ParseTuple(args, "OOO", &input1, &input2, &input3)){
      return NULL;
  }     

  py_g = (PyArrayObject *) PyArray_ContiguousFromObject(input1, PyArray_DOUBLE, 2, 2);
  if (py_g == NULL){
    return NULL;
  }  

  if (py_g->nd != 2 || py_g->descr->type_num != PyArray_DOUBLE) {
      PyErr_SetString(PyExc_ValueError, "array must be two-dimensional and of type float");
      return NULL;
  }

  py_g_wo_j = (PyArrayObject *) PyArray_ContiguousFromObject(input2, PyArray_DOUBLE, 2, 2);
  if (py_g_wo_j == NULL){
    return NULL;
  }  

  if (py_g_wo_j->nd != 2 || py_g_wo_j->descr->type_num != PyArray_DOUBLE) {
      PyErr_SetString(PyExc_ValueError, "array must be two-dimensional and of type float");
      return NULL;
  }
  
  py_l_j_1 = (PyArrayObject *) PyArray_ContiguousFromObject(input3, PyArray_DOUBLE, 2, 2);
  if (py_l_j_1 == NULL){
    return NULL;
  }  

  if (py_l_j_1->nd != 2 || py_l_j_1->descr->type_num != PyArray_DOUBLE) {
      PyErr_SetString(PyExc_ValueError, "array must be two-dimensional and of type float");
      return NULL;
  }

  

  row = py_g->dimensions[0];
  col = py_g->dimensions[1];
  c_g = pymatrix_to_Carrayptrs(py_g);  
  c_g_wo_j = pymatrix_to_Carrayptrs(py_g_wo_j);  
  c_l_j_1 = pymatrix_to_Carrayptrs(py_l_j_1);  




  for(j=0;j<col;j++) {
    for(i=0;i<row;i++) {
       c_g[i][j] = c_g_wo_j[i][j] + c_l_j_1[i][j];
       
    }
  } 

  /* deallocation */
  Py_DECREF(py_g); 
  Py_DECREF(py_g_wo_j); 
  Py_DECREF(py_l_j_1); 
  
  free_Carrayptrs(c_g);
  free_Carrayptrs(c_g_wo_j);
  free_Carrayptrs(c_l_j_1);  

  return PyFloat_FromDouble(0.0);


}


/* 
   Substracts two numpy matrics.

*/
PyObject* substract_matrix(PyObject *self, PyObject *args){
  PyObject *input1, *input2;

  PyArrayObject *py_g;
  PyArrayObject *py_l_j;
  PyArrayObject *pyres;
    

  int i,j, col, row, dim[2];
  double **c_g, **cres, **c_l_j;

  if (!PyArg_ParseTuple(args, "OO", &input1, &input2)){
      return NULL;
  }     

  py_g = (PyArrayObject *) PyArray_ContiguousFromObject(input1, PyArray_DOUBLE, 2, 2);
  if (py_g == NULL){
    return NULL;
  }  

  if (py_g->nd != 2 || py_g->descr->type_num != PyArray_DOUBLE) {
      PyErr_SetString(PyExc_ValueError, "array must be two-dimensional and of type float");
      return NULL;
  }

 
  py_l_j = (PyArrayObject *) PyArray_ContiguousFromObject(input2, PyArray_DOUBLE, 2, 2);
  if (py_l_j == NULL){
    return NULL;
  }  

  if (py_l_j->nd != 2 || py_l_j->descr->type_num != PyArray_DOUBLE) {
      PyErr_SetString(PyExc_ValueError, "array must be two-dimensional and of type float");
      return NULL;
  }

  row = py_g->dimensions[0];
  col = py_g->dimensions[1];
  c_g = pymatrix_to_Carrayptrs(py_g);  
  c_l_j = pymatrix_to_Carrayptrs(py_l_j);  

  /* create output PyArrayObject and corresponding C data struct pointer */
  dim[0] = row;
  dim[1] = col;
  
  /*pyres = (PyArrayObject *) PyArray_SimpleNew(2, dim, PyArray_DOUBLE);*/
  pyres = (PyArrayObject *) PyArray_FromDims(2, dim, PyArray_DOUBLE);
  
  cres = pymatrix_to_Carrayptrs(pyres); 

  for(j=0;j<col;j++) {
    for(i=0;i<row;i++) {
       if 
       (fpclassify(c_g[i][j]) == FP_INFINITE && fpclassify(c_l_j[i][j]) == FP_INFINITE){
         
          cres[i][j] = c_g[i][j];
       }
       else{
         cres[i][j] = c_g[i][j] - c_l_j[i][j];
       }  
    }
  } 

  /* deallocation */
  Py_DECREF(py_g); 
  Py_DECREF(py_l_j); 
  
  free_Carrayptrs(c_g);
  free_Carrayptrs(c_l_j);  
  free_Carrayptrs(cres);  

  return PyArray_Return(pyres);

}


/* 
  For a given vector update top two elements (XXX ...)

*/
PyObject* update_two_largest_elements(PyObject *self, PyObject *args){
  PyObject *input1, *input2,  *input3;

  PyArrayObject *py_toptwo;
  PyArrayObject *py_g_wo_j;
  PyArrayObject *py_lrow;


  int j, col, row;
  double x, **c_toptwo, v, *c_g_wo_j, *c_lrow;

  if (!PyArg_ParseTuple(args, "OOO", &input1, &input2, &input3)){
      return NULL;
  }     

  py_toptwo = (PyArrayObject *) PyArray_ContiguousFromObject(input1, PyArray_DOUBLE, 2, 2);
  if (py_toptwo == NULL){
    return NULL;
  }  

  if (py_toptwo->nd != 2 || py_toptwo->descr->type_num != PyArray_DOUBLE) {
      PyErr_SetString(PyExc_ValueError, "array must be two-dimensional and of type float");
      return NULL;
  }

  py_g_wo_j = (PyArrayObject *) PyArray_ContiguousFromObject(input2, PyArray_DOUBLE, 1, 1);
  if (py_g_wo_j == NULL){
    return NULL;
  }  

  if (py_g_wo_j->nd != 1 || py_g_wo_j->descr->type_num != PyArray_DOUBLE) {
      PyErr_SetString(PyExc_ValueError, "array must be one-dimensional and of type float");
      return NULL;
  }

  py_lrow = (PyArrayObject *) PyArray_ContiguousFromObject(input3, PyArray_DOUBLE, 1, 1);
  if (py_lrow == NULL){
    return NULL;
  }  

  if (py_lrow->nd != 1 || py_lrow->descr->type_num != PyArray_DOUBLE) {
      PyErr_SetString(PyExc_ValueError, "array must be one-dimensional and of type float");
      return NULL;
  }


  row = 2;
  col = py_toptwo->dimensions[1];
  c_toptwo = pymatrix_to_Carrayptrs(py_toptwo);  

  c_g_wo_j = pyvector_to_Carrayptrs(py_g_wo_j); 
  c_lrow = pyvector_to_Carrayptrs(py_lrow); 




  for(j=0;j<col;j++) {
    v = c_g_wo_j[j] + c_lrow[j];
    
    if (v > c_toptwo[0][j]){
      x = c_toptwo[0][j];
      c_toptwo[0][j] = v;
      c_toptwo[1][j] = x;
    }
    else if ( v > c_toptwo[1][j] ) {
       c_toptwo[1][j] = v;
    }

  } 


  /* deallocation */
  Py_DECREF(py_toptwo); 
  free_Carrayptrs(c_toptwo);

  Py_DECREF(py_g_wo_j); 

  Py_DECREF(py_lrow); 

  return PyFloat_FromDouble(0.0);

}



/*-------------------------------------------------------------------------------------------------------------------*/
/*                                            GSL Wrapper functions                                             */
/*-------------------------------------------------------------------------------------------------------------------*/

/*
  Wrapps the GSL function gsl_dirichlet_lnpdf. 

*/
PyObject* wrap_gsl_dirichlet_lnpdf(PyObject *self, PyObject *args) {
  PyObject *input1, *input2;
  PyArrayObject *pydata, *pyalpha, *pyres;

  double *cres, *calpha, **cdata;
  int i, nr_samples[1], dim;

  if (!PyArg_ParseTuple(args, "OO", &input1, &input2)){
      return NULL;
  }     

  /* parse input arrays */
  pydata = (PyArrayObject *) PyArray_ContiguousFromObject(input2, PyArray_DOUBLE, 2, 2);
  if (pydata == NULL){
    return NULL;
  }  
  pyalpha = (PyArrayObject *) PyArray_ContiguousFromObject(input1, PyArray_DOUBLE, 1, 1);
  if (pyalpha == NULL){
    return NULL;
  }  

  /* check input validity */
  if (pydata->nd != 2 || pydata->descr->type_num != PyArray_DOUBLE) {
      PyErr_SetString(PyExc_ValueError, "array must be two-dimensional and of type float");
      return NULL;
  }
  if (pyalpha->nd != 1 || pyalpha->descr->type_num != PyArray_DOUBLE) {
      PyErr_SetString(PyExc_ValueError, "array must be one-dimensional and of type float");
      return NULL;
  }
  nr_samples[0] = pydata->dimensions[0];
  dim = pydata->dimensions[1];
  
  /* allocate numpy output object*/
  /*pyres = (PyArrayObject *) PyArray_SimpleNew(1, nr_samples, PyArray_DOUBLE);*/
  pyres = (PyArrayObject *) PyArray_FromDims(1, nr_samples, PyArray_DOUBLE);

  /* get C pointers to numpy data objects */
  cres = pyvector_to_Carrayptrs(pyres); 
  calpha = pyvector_to_Carrayptrs(pyalpha);  
  cdata = pymatrix_to_Carrayptrs(pydata);  

  for(i=0;i<nr_samples[0]; i++){
    /*  gsl_ran_dirichlet_lnpdf (size_t K, const double alpha[], const double theta[]) */
    cres[i] = gsl_ran_dirichlet_lnpdf( (size_t) dim, calpha, cdata[i]);

    if(fpclassify(cres[i]) == FP_INFINITE){
      PyErr_SetString(PyExc_ValueError, "Zero probability under Dirichlet prior.\n");
      return NULL;
    }
    
  }

  /* deallocation */
  Py_DECREF(pydata); 
  Py_DECREF(pyalpha);
  free_Carrayptrs(cdata);
  return PyArray_Return(pyres);
}

/*
  Wrapps the GSL function gsl_dirichlet_lnpdf. 

*/
PyObject* wrap_gsl_dirichlet_pdf(PyObject *self, PyObject *args) {
  PyObject *input1, *input2;
  PyArrayObject *pydata, *pyalpha, *pyres;

  double *cres, *calpha, **cdata;
  int i, nr_samples[1], dim;

  if (!PyArg_ParseTuple(args, "OO", &input1, &input2)){
      return NULL;
  }     

  /* parse input arrays */
  pydata = (PyArrayObject *) PyArray_ContiguousFromObject(input2, PyArray_DOUBLE, 2, 2);
  if (pydata == NULL){
    return NULL;
  }  
  pyalpha = (PyArrayObject *) PyArray_ContiguousFromObject(input1, PyArray_DOUBLE, 1, 1);
  if (pyalpha == NULL){
    return NULL;
  }  

  /* check input validity */
  if (pydata->nd != 2 || pydata->descr->type_num != PyArray_DOUBLE) {
      PyErr_SetString(PyExc_ValueError, "array must be two-dimensional and of type float");
      return NULL;
  }
  if (pyalpha->nd != 1 || pyalpha->descr->type_num != PyArray_DOUBLE) {
      PyErr_SetString(PyExc_ValueError, "array must be one-dimensional and of type float");
      return NULL;
  }
  nr_samples[0] = pydata->dimensions[0];
  dim = pydata->dimensions[1];
  
  /* allocate numpy output object*/
  /*pyres = (PyArrayObject *) PyArray_SimpleNew(1, nr_samples, PyArray_DOUBLE);*/
  pyres = (PyArrayObject *) PyArray_FromDims(1, nr_samples, PyArray_DOUBLE);

  /* get C pointers to numpy data objects */
  cres = pyvector_to_Carrayptrs(pyres); 
  calpha = pyvector_to_Carrayptrs(pyalpha);  
  cdata = pymatrix_to_Carrayptrs(pydata);  

  for(i=0;i<nr_samples[0]; i++){
    /*  gsl_ran_dirichlet_pdf (size_t K, const double alpha[], const double theta[]) */
    cres[i] = gsl_ran_dirichlet_pdf( (size_t) dim, calpha, cdata[i]);
  }

  /* deallocation */
  Py_DECREF(pydata); 
  Py_DECREF(pyalpha);
  free_Carrayptrs(cdata);
  return PyArray_Return(pyres);
}

/*
  Wrapps the GSL function gsl_dirichlet_lnpdf. 

*/
PyObject* wrap_gsl_dirichlet_sample(PyObject *self, PyObject *args) {
    PyObject *input1;
    PyArrayObject *pyalpha, *pyres;
    int M, dim[1];
    double *calpha, *cres;
   


  if (!PyArg_ParseTuple(args, "Oi", &input1, &M)){
      return NULL;
  }     
  /* parse input arrays */
  pyalpha = (PyArrayObject *) PyArray_ContiguousFromObject(input1, PyArray_DOUBLE, 1, 1);
  if (pyalpha == NULL){
    return NULL;
  }  

  /* check input validity */
  if (pyalpha->nd != 1 || pyalpha->descr->type_num != PyArray_DOUBLE) {
      PyErr_SetString(PyExc_ValueError, "array must be two-dimensional and of type float");
      return NULL;
  }
  /* allocate numpy output object*/
  dim[0] = M;
  /*pyres = (PyArrayObject *) PyArray_SimpleNew(1, dim, PyArray_DOUBLE);*/
  pyres = (PyArrayObject *) PyArray_FromDims(1, dim, PyArray_DOUBLE);

  /* get C pointers to numpy data objects */
  cres = pyvector_to_Carrayptrs(pyres); 
  calpha = pyvector_to_Carrayptrs(pyalpha); 

  gsl_ran_dirichlet(RNG,M,calpha,cres);
  
  //printf("(%g,%g): %g %g\n",a[0],a[1],b[0],b[1]);
  
  /* deallocation */
  Py_DECREF(pyalpha);

  return PyArray_Return(pyres);
}



/*
  Wrapps the GSL function gsl_ran_gaussian_pdf.
*/
PyObject* wrap_gsl_ran_gaussian_pdf(PyObject *self, PyObject *args) {
  PyObject *input;
  PyArrayObject *pydata, *pyres;

  double mu, sigma, *cdata, *cres;
  int i, nr_samples[1];
  
  if (!PyArg_ParseTuple(args, "ddO", &mu, &sigma, &input)){
      return NULL;
  }     

  /* parse input arrays */
  pydata = (PyArrayObject *) PyArray_ContiguousFromObject(input, PyArray_DOUBLE, 1, 1);
  if (pydata == NULL){
    return NULL;
  }      
   
  nr_samples[0] = pydata->dimensions[0];

  /* allocate numpy output object*/
  /*pyres = (PyArrayObject *) PyArray_SimpleNew(1, nr_samples, PyArray_DOUBLE);*/
  pyres = (PyArrayObject *) PyArray_FromDims(1, nr_samples, PyArray_DOUBLE);


  /* get C pointers to numpy data objects */
  cdata = pyvector_to_Carrayptrs(pydata);  
  cres = pyvector_to_Carrayptrs(pyres); 

  for(i=0;i<nr_samples[0]; i++){
    /* gsl_ran_gaussian_pdf (double x, double sigma) */
     cres[i] = gsl_ran_gaussian_pdf( cdata[i]-mu, sigma);
  }
  
   /* deallocation */
   Py_DECREF(pydata); 

  return PyArray_Return(pyres);
}

/*
  Wrapps the GSL function gsl_sf_gamma.
*/
PyObject* wrap_gsl_sf_gamma(PyObject *self, PyObject *args) {
  double x, res;
  
  if (!PyArg_ParseTuple(args, "d", &x)){
      return NULL;
  }     
 
  /* double gsl_sf_gamma (double x) */
  res = gsl_sf_gamma(x);
  
  return PyFloat_FromDouble(res);
}  
  
/*
  Wrapps the GSL function gsl_sf_lngamma.
*/
PyObject* wrap_gsl_sf_lngamma(PyObject *self, PyObject *args) {
  double x, res;
  
  if (!PyArg_ParseTuple(args, "d", &x)){
      return NULL;
  }     
 
  /* double gsl_sf_lngamma (double x) */
  res = gsl_sf_lngamma(x);
  
  return PyFloat_FromDouble(res);
  
}

PyObject* set_gsl_rng_seed(PyObject *self, PyObject *args) {
  int seed;
  
  if (!PyArg_ParseTuple(args, "i", &seed)){
      return NULL;
  } 
  
   gsl_rng_set (RNG, seed);    
   return PyFloat_FromDouble(0.0);
}




PyObject* test(PyObject *self, PyObject *args) {
  /* PyArrayObject *array = (PyArrayObject *) PyArray_ContiguousFromObject(args, PyArray_DOUBLE, 2, 2);*/
  
  PyArrayObject *array;
  double** cin;
  double sum;
  int i, n;
  
  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &array)){
      return NULL;
  } 
  
  if (array->nd != 2 || array->descr->type_num != PyArray_DOUBLE) {
      PyErr_SetString(PyExc_ValueError, "array must be two-dimensional and of type float");
      return NULL;
  }

  n = array->dimensions[0];
  if (n > array->dimensions[1]){
      n = array->dimensions[1];
  }

  cin= pymatrix_to_Carrayptrs(array);
  
  sum = 0.;
  for (i = 0; i < n; i++) {
      sum += *(double *)(array->data + i*array->strides[0] + i*array->strides[1]);
      cin[i][i] = 1.0;
  }
  free_Carrayptrs(cin);
  return PyFloat_FromDouble(sum);
}
