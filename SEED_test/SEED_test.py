

import numpy as np
from sys import getsizeof

g_num_traces = 10000
g_num_bytes_per_trace = 128/8
g_plain_text = None
g_truth_cipher_text = None
g_keys = None
g_key = None


def _print_pointer(data):
    import ctypes
    print('.........................................')
    print(data.ctypes.data_as(ctypes.c_void_p))
    print('Size: ' + str(getsizeof(data)))
    print('dtype: ' + str(data.dtype))
    print('shape: ' + str(data.shape))
    print('strides: ' + str(data.strides))
    print('')
    print('')
    print('')


def _print_data_to_hexstr(data):
    print('b\'' + '\\'.join([hex(i)[1:] for i in data]) + '\'')


def _generate_fake_dataset():
    global g_plain_text, g_truth_cipher_text, g_keys, g_key

    print('_generate_fake_dataset')

    # seed
    np.random.seed(1234567)

    # plain text
    str_temp = np.random.bytes(g_num_traces*g_num_bytes_per_trace)
    g_plain_text = np.fromstring(str_temp, dtype=np.uint8)
    print('')
    print('....................................................................')
    print('')
    print('plain_text')
    _print_data_to_hexstr(g_plain_text[0:32])

    # keys
    str_temp = np.random.bytes(g_num_traces*g_num_bytes_per_trace)
    g_keys = np.fromstring(str_temp, dtype=np.uint8)
    print('keys')
    _print_data_to_hexstr(g_keys[0:32])

    # key
    str_temp = np.random.bytes(g_num_bytes_per_trace)
    g_key = np.fromstring(str_temp, dtype=np.uint8)
    print('key')
    _print_data_to_hexstr(g_key)
    print('')
    print('....................................................................')
    print('')
    print('')
    print('')


def ____code_to_blabla():
    np.random.seed(1234567)
    g_plain_text = np.random.randint(0, np.iinfo(np.uint16).max, (g_num_traces, g_num_words_per_trace))
    _print_pointer(g_plain_text)
    print(g_plain_text[0,:])
    # way to reshape give a try
    g_plain_text.dtype = np.uint8
    g_plain_text.shape = (g_plain_text.shape[0]*g_plain_text.shape[1])
    _print_pointer(g_plain_text)
    print(g_plain_text[0:32])

    ###
    g_plain_text.tobytes('C') == g_plain_text.tobytes()


class ThirdPartyCryptography():

    @staticmethod
    def encrypt_seed(plain_text, key):
        from cryptography.hazmat.backends.openssl.backend import backend
        from cryptography.hazmat.primitives.ciphers import algorithms, base, modes
        cipher = base.Cipher(
            algorithms.SEED(key),
            modes.ECB(),
            backend
        )
        encryptor = cipher.encryptor()
        ct = encryptor.update(plain_text)
        ct += encryptor.finalize()
        return ct

    @staticmethod
    def generate_cipher(plain_text, keys, algorithm):
        if algorithm == 'SEED':
            ct = ThirdPartyCryptography.encrypt_seed(plain_text.tobytes('C'), keys.tobytes('C'))
            return np.fromstring(ct, dtype=np.uint8)
        return None


def main():
    global g_plain_text, g_truth_cipher_text, g_keys, g_key
    import sys
    print(sys.version)
    _generate_fake_dataset()
    g_truth_cipher_text = ThirdPartyCryptography.generate_cipher(
        g_plain_text,
        g_key,
        'SEED')
    _print_data_to_hexstr(g_truth_cipher_text[0:32])


if __name__ == "__main__":
    main()


