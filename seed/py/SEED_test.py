# -*- coding: utf-8 -*-

import binascii
import numpy as np
import pyximport
from numpy.f2py.crackfortran import include_paths
import cProfile
from collections import namedtuple

pyximport.install(inplace=False,
                  setup_args={"include_dirs": np.get_include()},
                  reload_support=True)
from Cython.Build import cythonize
# TODO: check with multi thread
cythonize('SEED_cy.pyx', annotate=True, nthreads=8)

from cryptography.hazmat.backends.openssl.backend import backend
from cryptography.hazmat.primitives.ciphers import algorithms, base, modes




test_vector = namedtuple('test_vector', 'key ptx ctx')

test_vectors_sksp = [
    test_vector(
        key=np.asarray([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                       np.uint8),
        ptx=np.asarray([0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F],
                       np.uint8),
        ctx=np.asarray([0x5E, 0xBA, 0xC6, 0xE0, 0x05, 0x4E, 0x16, 0x68, 0x19, 0xAF, 0xF1, 0xCC, 0x6D, 0x34, 0x6C, 0xDB],
                       np.uint8)
    ),
    test_vector(
        key=np.asarray([0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F],
                       np.uint8),
        ptx=np.asarray([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
                       np.uint8),
        ctx=np.asarray([0xC1, 0x1F, 0x22, 0xF2, 0x01, 0x40, 0x50, 0x50, 0x84, 0x48, 0x35, 0x97, 0xE4, 0x37, 0x0F, 0x43],
                       np.uint8)
    ),
    test_vector(
        key=np.asarray([0x47, 0x06, 0x48, 0x08, 0x51, 0xE6, 0x1B, 0xE8, 0x5D, 0x74, 0xBF, 0xB3, 0xFD, 0x95, 0x61, 0x85],
                       np.uint8),
        ptx=np.asarray([0x83, 0xA2, 0xF8, 0xA2, 0x88, 0x64, 0x1F, 0xB9, 0xA4, 0xE9, 0xA5, 0xCC, 0x2F, 0x13, 0x1C, 0x7D],
                       np.uint8),
        ctx=np.asarray([0xEE, 0x54, 0xD1, 0x3E, 0xBC, 0xAE, 0x70, 0x6D, 0x22, 0x6B, 0xC3, 0x14, 0x2C, 0xD4, 0x0D, 0x4A],
                       np.uint8)
    ),
    test_vector(
        key=np.asarray([0x28, 0xDB, 0xC3, 0xBC, 0x49, 0xFF, 0xD8, 0x7D, 0xCF, 0xA5, 0x09, 0xB1, 0x1D, 0x42, 0x2B, 0xE7],
                       np.uint8),
        ptx=np.asarray([0xB4, 0x1E, 0x6B, 0xE2, 0xEB, 0xA8, 0x4A, 0x14, 0x8E, 0x2E, 0xED, 0x84, 0x59, 0x3C, 0x5E, 0xC7],
                       np.uint8),
        ctx=np.asarray([0x9B, 0x9B, 0x7B, 0xFC, 0xD1, 0x81, 0x3C, 0xB9, 0x5D, 0x0B, 0x36, 0x18, 0xF4, 0x0F, 0x51, 0x22],
                       np.uint8)
    )
]

test_vector_mkmp = test_vector(
    key=np.asarray([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
                    0x47, 0x06, 0x48, 0x08, 0x51, 0xE6, 0x1B, 0xE8, 0x5D, 0x74, 0xBF, 0xB3, 0xFD, 0x95, 0x61, 0x85,
                    0x28, 0xDB, 0xC3, 0xBC, 0x49, 0xFF, 0xD8, 0x7D, 0xCF, 0xA5, 0x09, 0xB1, 0x1D, 0x42, 0x2B, 0xE7],
                   np.uint8),
    ptx=np.asarray([0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
                    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                    0x83, 0xA2, 0xF8, 0xA2, 0x88, 0x64, 0x1F, 0xB9, 0xA4, 0xE9, 0xA5, 0xCC, 0x2F, 0x13, 0x1C, 0x7D,
                    0xB4, 0x1E, 0x6B, 0xE2, 0xEB, 0xA8, 0x4A, 0x14, 0x8E, 0x2E, 0xED, 0x84, 0x59, 0x3C, 0x5E, 0xC7],
                   np.uint8),
    ctx=np.asarray([0x5E, 0xBA, 0xC6, 0xE0, 0x05, 0x4E, 0x16, 0x68, 0x19, 0xAF, 0xF1, 0xCC, 0x6D, 0x34, 0x6C, 0xDB,
                    0xC1, 0x1F, 0x22, 0xF2, 0x01, 0x40, 0x50, 0x50, 0x84, 0x48, 0x35, 0x97, 0xE4, 0x37, 0x0F, 0x43,
                    0xEE, 0x54, 0xD1, 0x3E, 0xBC, 0xAE, 0x70, 0x6D, 0x22, 0x6B, 0xC3, 0x14, 0x2C, 0xD4, 0x0D, 0x4A,
                    0x9B, 0x9B, 0x7B, 0xFC, 0xD1, 0x81, 0x3C, 0xB9, 0x5D, 0x0B, 0x36, 0x18, 0xF4, 0x0F, 0x51, 0x22],
                   np.uint8)
)

test_big_size = 1000000
test_vector_big_skmp = test_vector(
    key=np.asarray(
        [0x28, 0xDB, 0xC3, 0xBC, 0x49, 0xFF, 0xD8, 0x7D, 0xCF, 0xA5, 0x09, 0xB1, 0x1D, 0x42, 0x2B, 0xE7],
        np.uint8),
    ptx=np.tile(
        np.asarray(
            [0xB4, 0x1E, 0x6B, 0xE2, 0xEB, 0xA8, 0x4A, 0x14, 0x8E, 0x2E, 0xED, 0x84, 0x59, 0x3C, 0x5E, 0xC7],
            np.uint8
        ),
        test_big_size),
    ctx=np.tile(
        np.asarray(
            [0x9B, 0x9B, 0x7B, 0xFC, 0xD1, 0x81, 0x3C, 0xB9, 0x5D, 0x0B, 0x36, 0x18, 0xF4, 0x0F, 0x51, 0x22],
            np.uint8
        ),
        test_big_size),
)

test_vector_big_mkmp = test_vector(
    key=np.tile(
        np.asarray(
            [0x28, 0xDB, 0xC3, 0xBC, 0x49, 0xFF, 0xD8, 0x7D, 0xCF, 0xA5, 0x09, 0xB1, 0x1D, 0x42, 0x2B, 0xE7],
            np.uint8),
        test_big_size),
    ptx=np.tile(
        np.asarray(
            [0xB4, 0x1E, 0x6B, 0xE2, 0xEB, 0xA8, 0x4A, 0x14, 0x8E, 0x2E, 0xED, 0x84, 0x59, 0x3C, 0x5E, 0xC7],
            np.uint8
        ),
        test_big_size),
    ctx=np.tile(
        np.asarray(
            [0x9B, 0x9B, 0x7B, 0xFC, 0xD1, 0x81, 0x3C, 0xB9, 0x5D, 0x0B, 0x36, 0x18, 0xF4, 0x0F, 0x51, 0x22],
            np.uint8),
        test_big_size),
)


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


def _third_party_encrypt(plain_text, key, algorithm):
    if algorithm == 'SEED':
        ct = _third_party_encrypt_seed(plain_text.tobytes('C'), key.tobytes('C'))
        return np.fromstring(ct, dtype=np.uint8)
    if algorithm == 'AES':
        ct = _third_party_encrypt_aes(plain_text.tobytes('C'), key.tobytes('C'))
        return np.fromstring(ct, dtype=np.uint8)
    return None


def create_fake_trace_file():
    from scaffold.core import TraceFile, TraceFileHandler

    file_name = 'M:\\NXP_workspace\\traces\\seed8bit_fixed_key_10k.bin'
    tf = TraceFile(file_name)
    tfh = TraceFileHandler(tf, algo='SEED')
    keys = tf['key'].astype(np.uint8)
    plaintext = tfh['plaintext'].astype(np.uint8)
    ciphertext_aes = tfh['ciphertext'].astype(np.uint8)
    #keys = tf['key']
    #plaintext = tfh['plaintext']
    #ciphertext_aes = tfh['ciphertext']
    #bla = tfh['5:SEED']
    #print(bla[0])

    plaintext.dtype = np.uint64
    keys.dtype = np.uint64
    ciphertext_aes.dtype = np.uint64
    ciphertext_aes = ciphertext_aes.flatten()

    dump = False
    if dump == True:
        np.savetxt('del_dump_plaintext.txt', plaintext, fmt='%u')
        np.savetxt('del_dump_keys.txt', keys, fmt='%u')
        np.savetxt('del_dump_ciphertext.txt', ciphertext_aes, fmt='%u')
        plaintext = np.loadtxt('del_dump_plaintext.txt', dtype=np.uint64)
        keys = np.loadtxt('del_dump_keys.txt', dtype=np.uint64)
        ciphertext = np.loadtxt('del_dump_ciphertext.txt', dtype=np.uint64)

    print('\n\n---------------------------')
    ct_seed = _third_party_encrypt(plaintext, keys, 'SEED')
    ct_aes = _third_party_encrypt(plaintext, keys, 'AES')
    ct_seed.dtype = np.uint64
    ct_aes.dtype = np.uint64
    #_print_hex(keys)
    #_print_hex(plaintext.flatten())
    #_print_hex(ciphertext_aes.flatten())
    #_print_hex(ct_aes)
    #_print_hex(ct_seed)
    for ii in range(ciphertext_aes.shape[0]):
        if ciphertext_aes[ii] != ct_aes[ii]:
            print('bad :(')
    print('---------------------------\n\n')
    print(tfh['ciphertext'])
    ct_aes.dtype = np.uint8
    print(ct_aes)

    # dump the seed results for verification
    ct_seed.dtype = np.uint32
    np.save('seed_data\ciphertext_seed', ct_seed)


def run_test_cases():
    from SEED_cy import SEEDAlgorithmCy

    # check third part algorithm is correct or not
    for test in test_vectors_sksp:
        ct = _third_party_encrypt(test.ptx, test.key, 'SEED')
        print("Third party algorithm sksp correct = " + str(np.array_equal(test.ctx, ct)))
        #_print_hex(test.ctx)
        #_print_hex(ct)


if __name__ == '__main__':
    import sys
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

    #run_test_cases()

    from SEED_cy import SEEDAlgorithmCy, STEPS_PROVIDED
    from SEED_py import SEEDAlgorithmPy


    #tt = np.tile(test_vectors_sksp.ptx, 1)
    #kk = np.tile(test_vectors_sksp[3].key, 1000000)


    # four ptxt and four key
    ptx = test_vector_big_mkmp.ptx
    key = test_vector_big_mkmp.key
    ctx = test_vector_big_mkmp.ctx

    acy = SEEDAlgorithmCy()


    #for rnd in range(1,17)[::-1]:
        #res = acy.encrypt(tt, kk, rnd, STEPS_PROVIDED.RoundKey_64)
        #res = acy.encrypt(tt, kk, rnd, STEPS_PROVIDED.Right_64)
        #res = acy.encrypt(tt, kk, rnd, STEPS_PROVIDED.AddRoundKey_64)
        #res = acy.encrypt(tt, kk, rnd, STEPS_PROVIDED.GDa_32)
        #res = acy.encrypt(tt, kk, rnd, STEPS_PROVIDED.GC_32)
        #res = acy.encrypt(tt, kk, rnd, STEPS_PROVIDED.GDb_32)
        #res = acy.encrypt(tt, kk, rnd, STEPS_PROVIDED.F_64)
        #res = acy.encrypt(tt, kk, rnd, STEPS_PROVIDED.Output_128)
        #print('Round ............... '+str(rnd))
        #print(res)
        #for rr in res:
        #    for r in rr:
        #        print(hex(r))



    cProfile.run('acy.encrypt(ptx, key, 16, STEPS_PROVIDED.Output_128)')
    res = acy.encrypt(ptx, key, 16, STEPS_PROVIDED.Output_128)
    #res = np.asarray(res)
    #res.dtype = np.uint64
    print('Round ............... ')





