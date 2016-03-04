""" Mutual Information Analysis (MIA) attack using histogram for estimating distribution """

cimport cython
cimport numpy as np
import numpy as np
from cython cimport boundscheck, wraparound
from libc.math cimport log

from scaffold.ext import AnalysisExtension

# type definitions
ctypedef np.float64_t   C_TRACE_TYPE
ctypedef np.uint8_t     C_DATA_TYPE
ctypedef np.float64_t   C_INTERN_TYPE


class MIAHistogram(AnalysisExtension):
    """
    MiaHistogram module implemented using the AnalysisExtension interface.

    Computes Mutual Information Analysis (MIA) using histogram for estimating
    distribution between traces and target data set.
    For every target, the result consists of a single array of coefficients
    matching the trace size (i.e. a correlation trace).
    """

    def __init__(self, *args, **kwargs):
        """ Initialize the class members

        :param args:
        :param kwargs:
        :return:
        """
        super().__init__(*args, **kwargs)
        # 2-pass algorithm
        self.num_pass = 2
        # use double as internal type
        self._dtype = 'f8'
        self.data_type = 'u1'

    def _init_process(self, task):
        """ Initializes internal variables

        :param task:
        :return:
        """

        # Get the instance to store task variables and memory buffers
        my_task = _TaskVariables()

        # Initialize task members based on user input
        my_task.ui_nb_bit = task.parameters['parameters']['bit']
        my_task.ui_nb_bins = task.parameters['parameters']['nb_bins']
        my_task.ui_num_lk = 2 ** my_task.ui_nb_bit

        # Initialize memory buffers for processing
        my_task.buff_h_target_lk_samples = np.zeros(
            (task.target.num_val, my_task.ui_num_lk, task.num_samples),
            dtype=self._dtype)

        #
        my_task.buff_i_target_samples = np.zeros(
            (task.target.num_val, task.num_samples),
            dtype=self._dtype)

        #
        my_task.buff_app_target_lk = np.zeros(
            (task.target.num_val, my_task.ui_num_lk),
            dtype=self._dtype)

        #
        my_task.buff_pdf_samples = np.zeros(
            (my_task.ui_nb_bins + 1, task.num_samples),
            dtype=self._dtype)

        #
        my_task.buff_pdf_target_lk_samples = np.zeros(
            (task.target.num_val, my_task.ui_num_lk, my_task.ui_nb_bins + 1, task.num_samples),
            dtype=self._dtype)

        #
        my_task.buff_max_min_samples = np.zeros(
            (2, task.num_samples),
            dtype=self._dtype)

        # Storing the _TaskVariables in task.temp_vars so that other parts of code can access
        task.temp_vars = my_task

    def _process_chunk_pass_1(self, task, chunk):
        """ implementation for method _process_chunk_pass_1

        :param task:
        :param chunk:
        :return:
        """

        # perform max and min
        self._compute_max_min(task, chunk)

    def _process_chunk_pass_2(self, task, chunk):
        """ implementation for method _process_chunk_pass_2

        :param task:
        :param chunk:
        :return:
        """

        # perform histogram attack
        self._compute_attack(task, chunk)

    def _finalize_process(self, task):
        """ implementation for method _finalize_process

        :param task:
        :return:
        """

        # final pass
        self._compute_final_pass(task)

        # assign back the computed results from internal memory buffers to results
        my_task = task.temp_vars
        cdef np.ndarray[C_INTERN_TYPE, ndim = 2] I = my_task.buff_i_target_samples
        task.result = I

    @cython.profile(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def _compute_max_min(self, task, chunk):
        """ find the max and min of samples

        :param task:
        :param chunk:
        :return:
        """

        # get the index of the current chunk being processed by the module
        cdef Py_ssize_t c_counter = task.ccounter

        # recover internal temporary variable
        my_task = task.temp_vars

        # map ndarrays to c arrays
        cdef C_INTERN_TYPE [:,:] max_min = my_task.buff_max_min_samples

        # fetch samples and data
        cdef C_TRACE_TYPE [:,:] trc_chk = chunk['samples']

        cdef Py_ssize_t trc_shp0 = trc_chk.shape[0]
        cdef Py_ssize_t trc_shp1 = trc_chk.shape[1]
        cdef Py_ssize_t i, j, key

        # find max and min array
        # TODO: Use BLAS if possible
        if c_counter == 0:
            for j in range(trc_shp1):
                max_min[1, j] = trc_chk[0, j]
        for j in range(trc_shp1):
            for i in range(trc_shp0):
                if trc_chk[i, j] > max_min[0, j]:
                    max_min[0, j] = trc_chk[i, j]
                if trc_chk[i, j] < max_min[1, j]:
                    max_min[1, j] = trc_chk[i, j]

    @cython.profile(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def _compute_attack(self, task, chunk):
        """ perform histogram attack

        :param task:
        :param chunk:
        :return:
        """

        # recover internal temporary variable
        my_task = task.temp_vars

        # recover internal user input variables
        cdef Py_ssize_t ui_nb_bit = my_task.ui_nb_bit
        cdef Py_ssize_t ui_num_lk = my_task.ui_num_lk
        cdef Py_ssize_t ui_nb_bins = my_task.ui_nb_bins

        #python arrays to retreive from task
        cdef Py_ssize_t num_spl = task.num_samples
        cdef Py_ssize_t num_trc = self.num_traces
        cdef Py_ssize_t num_vals = task.target.num_val

        # recover internal memory buffers
        cdef C_INTERN_TYPE [:,:] max_min = my_task.buff_max_min_samples
        cdef C_INTERN_TYPE [:,:] pdf_trc = my_task.buff_pdf_samples
        # TODO: unused memory buffers if possible remove
        cdef C_INTERN_TYPE [:,:,:,:] pdf_trc_lk = my_task.buff_pdf_target_lk_samples
        cdef C_INTERN_TYPE [:,:] lk_app = my_task.buff_app_target_lk

        # fetch samples and data
        cdef C_TRACE_TYPE [:,:] trc_chk = chunk['samples']
        cdef C_DATA_TYPE [:,:] plt_chk = chunk['plaintext']


        #variables for iterating in C
        cdef Py_ssize_t i, j, key, ind, iv_ind, trc, shift_bits, mask_byte, iv_size, tttt

        #other variables
        iv_size = len(plt_chk)
        cdef int [:] iv_lk  = np.zeros(iv_size,dtype=np.int)
        cdef double lkc_ = 0
        cdef C_INTERN_TYPE d

        #pdf of the traces
        for i in range(num_spl):
            if max_min[0, i] == max_min[1, i]: #same value on every traces so first bin
                pdf_trc[0, i] = num_trc

        for i in range(trc_chk.shape[0]):
            for j in range(num_spl):
                if pdf_trc[0,j] != num_trc:
                    if trc_chk[i,j] == max_min[0,j]: #trc[j] = max, to avoid bugs separation of this case
                        pdf_trc[ui_nb_bins,j] += 1
                    elif trc_chk[i,j] == max_min[1,j]: #trc[j] = min, to avoid bugs separation of this case
                        pdf_trc[1,j] += 1
                    else :
                        #find the right bin for each trc_chk[i][j] to construct the histogramms
                        d = max_min[0,j] - max_min[1,j]
                        ind = <Py_ssize_t>((trc_chk[i,j]-max_min[1,j])*ui_nb_bins/d) + 1
                        pdf_trc[ind,j] += 1

        #pdf of the traces knowing the value of the leakage function
        #TODO: remove the hard coded length of bit array
        #shift_bits = 8-ui_nb_bit
        mask_byte = 31 #8-indice[0]
        for key in range(num_vals):
            iv = chunk.guess_data(task.target.str, key=key) # intermediate values
            for i in range(iv_size):
                #TODO: check how to replace python shift operator with C shift operator
                #iv_lk[i] = iv[i]>>shift_bits
                iv_lk[i] = (iv[i]&mask_byte)[0]

            for i in range(ui_num_lk):
                #compute the number of appearence of each possible value of the leakage model
                lkc_ = 0
                for j in range(iv_size):
                    if i == iv_lk[j]:
                        lkc_ += 1
                lk_app[key,i] += lkc_

            for i in range(trc_chk.shape[0]):
                for j in range(num_spl):
                    iv_ind = int(iv_lk[i])
                    if max_min[0,j] == max_min[1,j] :
                        pdf_trc_lk[key,iv_ind,0,j] += 1

                    elif trc_chk[i,j] == max_min[0,j]: #trc[j] = max, to avoid bugs separation of this case
                        pdf_trc_lk[key,iv_ind,ui_nb_bins,j] += 1

                    elif trc_chk[i,j] == max_min[1,j]: #trc[j] = min, to avoid bugs separation of this case
                        pdf_trc_lk[key,iv_ind,1,j] += 1

                    else :
                        d = max_min[0,j] - max_min[1,j]
                        ind = <int>((trc_chk[i,j]-max_min[1,j])*ui_nb_bins/d) + 1
                        pdf_trc_lk[key,iv_ind,ind,j] += 1

    @cython.profile(True)
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def _compute_final_pass(self, task):
        """ perform final pass of histogram attack

        :param task:
        :return:
        """

        my_task = task.temp_vars
        cdef Py_ssize_t ui_nb_bit = my_task.ui_nb_bit
        cdef Py_ssize_t ui_num_lk = my_task.ui_num_lk
        cdef Py_ssize_t ui_nb_bins = my_task.ui_nb_bins

        #python arrays to retreive from task
        cdef Py_ssize_t num_spl = task.num_samples
        cdef Py_ssize_t num_trc = self.num_traces
        cdef Py_ssize_t num_vals = task.target.num_val

        # recover internal variables
        cdef C_INTERN_TYPE [:,:,:] H_trc_lk = my_task.buff_h_target_lk_samples
        cdef np.ndarray[C_INTERN_TYPE, ndim = 2] I = my_task.buff_i_target_samples
        cdef C_INTERN_TYPE [:,:] pdf_trc = my_task.buff_pdf_samples
        cdef C_INTERN_TYPE [:,:,:,:] pdf_trc_lk = my_task.buff_pdf_target_lk_samples
        cdef C_INTERN_TYPE [:,:] lk_app = my_task.buff_app_target_lk

        #variables for iterating in C
        cdef Py_ssize_t i, j, b, key

        #other variables
        cdef C_INTERN_TYPE [:] H_trc, H_lk
        cdef C_INTERN_TYPE [:,:] I_mv = I
        cdef C_INTERN_TYPE _pdf_trc_val, _lk_prob

        #compute the probabilities of each bins for traces
        H_trc = np.zeros(num_spl)
        for i in range(ui_nb_bins+1):
            for j in range(num_spl):
                pdf_trc[i,j] /= num_trc
                _pdf_trc_val = pdf_trc[i,j]
                #compute the entropy of the traces
                if _pdf_trc_val != 0:
                    H_trc[j] += -_pdf_trc_val*(log(_pdf_trc_val)/log(2))

        #compute the entropy of the traces knowing the value of the leakage function
        H_lk = np.zeros(num_spl)
        for key in range(num_vals):
            # compute the entropy of the traces
            for i in range(ui_num_lk):
                if lk_app[key,i] != 0:
                    for b in range(ui_nb_bins+1):
                        for j in range(num_spl):
                            pdf_trc_lk[key,i,b,j] /= lk_app[key,i]
                            _pdf_trc_val = pdf_trc_lk[key,i,b,j]
                            if _pdf_trc_val != 0:
                                 H_trc_lk[key,i,j] += -_pdf_trc_val*(log(_pdf_trc_val)/log(2))
            #
            for i in range(ui_num_lk):
                #compute the probabilities of each value of the leakage function
                _lk_prob = lk_app[key,i]/num_trc
                #compute the entropy of the traces variable with key guess based on histograms
                for j in range(num_spl):
                    if i == 0: #simple reset to reuse the H_lk array for every new key
                        H_lk[j] = _lk_prob* H_trc_lk[key,i,j]
                    else:
                        H_lk[j] += _lk_prob* H_trc_lk[key,i,j]

            #TODO: can different metric be used over here?
            #mutual information computation
            for j in range(num_spl):
                I_mv[key,j] = H_trc[j] - H_lk[j]

class _TaskVariables():
    """ Stores temporary variables required by each task of MiaHistogram

    :var ui_nb_bit:                user input - number of bits to check
    :var ui_nb_bins:               user input - number of bins to use
    :var ui_num_lk                 possible combinations of ui_nb_bit - 2**ui_nb_bit
    """

    ui_nb_bit = 0
    ui_nb_bins = 0
    ui_num_lk = 0
    buff_h_target_lk_samples = None
    buff_i_target_samples = None
    buff_app_target_lk = None
    buff_pdf_samples = None
    buff_pdf_target_lk_samples = None
    buff_max_min_samples = None

