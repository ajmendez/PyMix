
PyObject* test(PyObject *self, PyObject *args);
PyObject* matrix_sum_logs(PyObject *self, PyObject *args);
PyObject* get_normalized_posterior_matrix(PyObject *self, PyObject *args);
PyObject* sum_logs(PyObject *self, PyObject *args);

/* TEST */
PyObject* get_two_largest_elements(PyObject *self, PyObject *args);
PyObject* get_likelihoodbounds(PyObject *self, PyObject *args);
PyObject* update_two_largest_elements(PyObject *self, PyObject *args);
PyObject* add_matrix_get_likelihoodbounds(PyObject *self, PyObject *args);

PyObject* add_matrix(PyObject *self, PyObject *args);
PyObject* substract_matrix(PyObject *self, PyObject *args);

/* GSL wrapper functions*/
PyObject* wrap_gsl_dirichlet_lnpdf(PyObject *self, PyObject *args);
PyObject* wrap_gsl_dirichlet_pdf(PyObject *self, PyObject *args);
PyObject* wrap_gsl_ran_gaussian_pdf(PyObject *self, PyObject *args);
PyObject* wrap_gsl_sf_gamma(PyObject *self, PyObject *args);
PyObject* wrap_gsl_sf_lngamma(PyObject *self, PyObject *args);
PyObject* get_log_normal_inverse_gamma_prior_density(PyObject *self, PyObject *args);
PyObject* wrap_gsl_dirichlet_sample(PyObject *self, PyObject *args);
PyObject* set_gsl_rng_seed(PyObject *self, PyObject *args);
