# -*- coding: utf-8 -*-

import binascii
import numpy as np
import pyximport
from numpy.f2py.crackfortran import include_paths
import cProfile

pyximport.install(inplace=False,
                  setup_args={"include_dirs": np.get_include()},
                  reload_support=True)
from Cython.Build import cythonize
cythonize('SEED_cy.pyx', annotate=True)

from cryptography.hazmat.backends.openssl.backend import backend
from cryptography.hazmat.primitives.ciphers import algorithms, base, modes
from SEED_tables import SS0, SS1, SS2, SS3, KC


def encrypt_third_party(mode, key, plaintext):
    cipher = base.Cipher(
        algorithms.SEED(binascii.unhexlify(key)),
        mode(),
        backend
    )
    encryptor = cipher.encryptor()
    ct = encryptor.update(binascii.unhexlify(plaintext))
    ct += encryptor.finalize()
    return binascii.hexlify(ct)


def build_vectors(mode, filename):
    with open(filename, "r") as f:
        vector_file = f.read().splitlines()

    count = 0
    output = []
    key = None
    iv = None
    plaintext = None
    for line in vector_file:
        line = line.strip()
        if line.startswith("KEY"):
            if count != 0:
                output.append("CIPHERTEXT = {0}".format(
                    encrypt_third_party(mode, key, iv, plaintext))
                )
            output.append("\nCOUNT = {0}".format(count))
            count += 1
            name, key = line.split(" = ")
            output.append("KEY = {0}".format(key))
        elif line.startswith("IV"):
            name, iv = line.split(" = ")
            output.append("IV = {0}".format(iv))
        elif line.startswith("PLAINTEXT"):
            name, plaintext = line.split(" = ")
            output.append("PLAINTEXT = {0}".format(plaintext))

    output.append("CIPHERTEXT = {0}".format(encrypt_third_party(mode, key, plaintext)))
    return "\n".join(output)


def write_file(data, filename):
    with open(filename, "w") as f:
        f.write(data)


def _print_hex(arr):
    fill = 0
    if arr.dtype == np.uint8:
        fill = 2
    elif arr.dtype == np.uint16:
        fill = 4
    elif arr.dtype == np.uint32:
        fill = 8
    elif arr.dtype == np.uint64:
        fill = 16
    for elem in arr:
        s = hex(elem)[2:].zfill(fill)
        print(s, end=' ')
    print('\n')


def _print_pointer(data):
    import ctypes
    import sys
    print('.........................................')
    print(data.ctypes.data_as(ctypes.c_void_p))
    print('Size: ' + str(sys.getsizeof(data)))
    print('dtype: ' + str(data.dtype))
    print('shape: ' + str(data.shape))
    print('strides: ' + str(data.strides))
    print('')
    print('')
    print('')


def _third_party_encrypt_seed(plain_text, key):
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


def _third_party_encrypt_aes(plain_text, key):
    from cryptography.hazmat.backends.openssl.backend import backend
    from cryptography.hazmat.primitives.ciphers import algorithms, base, modes
    cipher = base.Cipher(
        algorithms.AES(key),
        modes.ECB(),
        backend
    )
    encryptor = cipher.encryptor()
    ct = encryptor.update(plain_text)
    ct += encryptor.finalize()
    return ct


def _third_party_encrypt_third_party(plain_text, key, algorithm):
    if algorithm == 'SEED':
        ct = _third_party_encrypt_seed(plain_text.tobytes('C'), key.tobytes('C'))
        return np.fromstring(ct, dtype=np.uint8)
    if algorithm == 'AES':
        ct = _third_party_encrypt_aes(plain_text.tobytes('C'), key.tobytes('C'))
        return np.fromstring(ct, dtype=np.uint8)
    return None


def _mask_with_8f(var):
    #return np.bitwise_and(var, 0xffffffff)
    return var


def _g_func(var):
    index_l00 = np.bitwise_and(var, 0xff)
    index_l08 = np.bitwise_and(np.right_shift(var, 8), 0xff)
    index_l16 = np.bitwise_and(np.right_shift(var, 16), 0xff)
    index_l24 = np.bitwise_and(np.right_shift(var, 24), 0xff)
    return np.bitwise_xor(
        np.bitwise_xor(SS0[index_l00], SS1[index_l08]),
        np.bitwise_xor(SS2[index_l16], SS3[index_l24])
    )


def _right8_left24_update(var1, var2):
    return _mask_with_8f(np.bitwise_xor(
        np.right_shift(var1, 8),
        np.left_shift(var2, 24)
    ))


def _left8_right24_update(var1, var2):
    return _mask_with_8f(np.bitwise_xor(
        np.left_shift(var1, 8),
        np.right_shift(var2, 24)
    ))


def _fragment_block_to_words(var):
    var_ = var[::-1].copy()
    var_.dtype = np.uint32
    var__ = var_[::-1]
    x1 = var__[::4].copy()
    x2 = var__[1::4].copy()
    x3 = var__[2::4].copy()
    x4 = var__[3::4].copy()
    return x1, x2, x3, x4


def _fuse_words_to_block(x1, x2, x3, x4):
    pass

def generate_key_schedule(keys, rnd):
    x1, x2, x3, x4 = _fragment_block_to_words(keys)

    key_schedule_0 = None
    key_schedule_1 = None
    t0 = None

    # round 0 update
    if rnd == 0:
        key_schedule_0 = _g_func(_mask_with_8f(x1 + x3 - KC[0]))
        key_schedule_1 = _g_func(_mask_with_8f(x2 - x4 + KC[0]))

    # round 1 update
    t0 = x1
    x1 = _right8_left24_update(x1, x2)
    x2 = _right8_left24_update(x2, t0)
    if rnd == 1:
        key_schedule_0 = _g_func(_mask_with_8f(x1 + x3 - KC[1]))
        key_schedule_1 = _g_func(_mask_with_8f(x2 + KC[1] - x4))

    # round 2 ... 16
    for ii in np.arange(16)[2::2]:
        t0 = x3
        x3 = _left8_right24_update(x3, x4)
        x4 = _left8_right24_update(x4, t0)
        if rnd == ii:
            key_schedule_0 = _g_func(_mask_with_8f(x1 + x3 - KC[ii]))
            key_schedule_1 = _g_func(_mask_with_8f(x2 + KC[ii] - x4))
            break
        t0 = x1
        x1 = _right8_left24_update(x1, x2)
        x2 = _right8_left24_update(x2, t0)
        if rnd == ii + 1:
            key_schedule_0 = _g_func(_mask_with_8f(x1 + x3 - KC[ii + 1]))
            key_schedule_1 = _g_func(_mask_with_8f(x2 + KC[ii + 1] - x4))
            break

    return key_schedule_0, key_schedule_1


def encrypt_seed(plain_text, keys, rnd):
    x1, x2, x3, x4 = _fragment_block_to_words(plain_text)
    a1 = x1
    a2 = x2
    a3 = x3
    a4 = x4
    for ii in np.arange(16):
        print('.......................' + str(ii))
        _print_hex(a1)
        _print_hex(a2)
        _print_hex(a3)
        _print_hex(a4)

        ks_0, ks_1 = generate_key_schedule(keys, ii)
        t0 = np.bitwise_xor(a3, ks_0)
        t1 = np.bitwise_xor(a4, ks_1)
        np.bitwise_xor(t1, t0, out=t1)
        t1 = _g_func(t1)
        t0 = _mask_with_8f(t0 + t1)
        t0 = _g_func(t0)
        t1 = _mask_with_8f(t1 + t0)
        t1 = _g_func(t1)
        t0 = _mask_with_8f(t0 + t1)
        np.bitwise_xor(a1, t0, out=a1)
        np.bitwise_xor(a2, t1, out=a2)

        if ii % 2 == 0:
            a1 = x3
            a2 = x4
            a3 = x1
            a4 = x2
        else:
            a1 = x1
            a2 = x2
            a3 = x3
            a4 = x4




def check_test_vectors():
    from SEED_tables import test_vectors
    for test in test_vectors:
        ct = _third_party_encrypt_third_party(test.ptx, test.key, 'SEED')
        print("Arrays are equal = " + str(np.array_equal(test.ctx, ct)))
        _print_hex(test.ctx)
        _print_hex(ct)


def create_fake_trace_file():
    from scaffold.core import TraceFile, TraceFileHandler

    file_name = 'M:\\NXP_workspace\\traces\\seed8bit_fixed_key_10k.bin'
    tf = TraceFile(file_name)
    tfh = TraceFileHandler(tf, algo='SEED')
    keys = tf['key'].astype(np.uint8)
    plaintext = tfh['plaintext'].astype(np.uint8)
    ciphertext = tfh['ciphertext'].astype(np.uint8)
    bla = tfh['5:SEED']
    print(bla[0])

    plaintext.dtype = np.uint64
    keys.dtype = np.uint64
    ciphertext.dtype = np.uint64

    np.savetxt('del_dump_plaintext.txt', plaintext, fmt='%u')
    np.savetxt('del_dump_keys.txt', keys, fmt='%u')
    np.savetxt('del_dump_ciphertext.txt', ciphertext, fmt='%u')
    plaintext = np.loadtxt('del_dump_plaintext.txt', dtype=np.uint64)
    keys = np.loadtxt('del_dump_keys.txt', dtype=np.uint64)
    ciphertext = np.loadtxt('del_dump_ciphertext.txt', dtype=np.uint64)

    print('\n\n---------------------------')
    ct_seed = _third_party_encrypt_third_party(plaintext, keys, 'SEED')
    ct_aes = _third_party_encrypt_third_party(plaintext, keys, 'AES')
    ct_seed.dtype = np.uint64
    ct_aes.dtype = np.uint64
    _print_hex(keys)
    _print_hex(plaintext.flatten())
    _print_hex(ciphertext.flatten())
    _print_hex(ct_aes)
    _print_hex(ct_seed)
    print('---------------------------\n\n')



def _test_c_types():
    aa = np.arange(500).reshape(100, 5)
    print(aa[:, 0].flags['C_CONTIGUOUS'])
    print(aa[0, :].flags['C_CONTIGUOUS'])

    tt = aa[:,0]
    _print_pointer(tt)
    tt = aa[:,1]
    _print_pointer(tt)
    tt = aa[0,:]
    _print_pointer(tt)
    tt = aa[1,:]
    _print_pointer(aa[1,:])

    print(aa.ctypes.shape)
    print(aa.ctypes.strides)
    print(aa.ctypes.strides)


if __name__ == '__main__':
    import sys
    from SEED_tables import test_vector_big, test_vectors
    #np.__config__.show()
    #sys.stdout = open('del_console_stdout.txt', 'w')
    print('Main of SEED.py')
    #test_c_types()
    #create_fake_trace_file()
    #check_test_vectors()
    #ECB_PATH = "seed_data\seed-ecb.txt"
    #write_file(build_vectors(modes.ECB, ECB_PATH), "seed_data\seed-ecb-temp.txt")

    #generate_key_schedule(test_vectors[3].key)
    #generate_key_schedule(test_vector_big.key[48:64])
    #for i in np.arange(16):
        #generate_key_schedule(test_vector_big.key, i)
    #encrypt_seed(test_vector_big.ptx, test_vector_big.key, 0)


    from SEED_cy import SEEDAlgorithm

    tt = np.tile(test_vector_big.ptx, 1)
    kk = np.tile(test_vector_big.key, 1)
    tt = test_vectors[3].ptx
    kk = test_vectors[3].key

    a = SEEDAlgorithm()
    #a.py_encrypt_seed(test_vector_big.ptx, test_vector_big.key, 0, 0)
    a.call_cython(tt, kk, 0, 0)

    #cProfile.run('a.call_cython(tt, kk, 0, 0)')


