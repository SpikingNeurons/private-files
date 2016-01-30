

import numpy as np
from sys import getsizeof



def _print_pointer(data):
    import ctypes
    print('.........................................')
    print(data.ctypes.data_as(ctypes.c_void_p))
    print('Size: ' + str(getsizeof(data)))
    print('dtype: ' + str(data.dtype))
    print('shape: ' + str(data.shape))
    print('strides: ' + str(data.strides))
    print('')

g_num_traces = 10000
g_num_elements_per_trace = 16/4
g_plain_text = np.random.randint(0, np.iinfo(np.uint16).max, (g_num_traces, g_num_elements_per_trace))
_print_pointer(g_plain_text)
print(g_plain_text[0,:])
g_plain_text.dtype = np.uint8
g_plain_text.shape = (g_plain_text.shape[0]*g_plain_text.shape[1])
_print_pointer(g_plain_text)

