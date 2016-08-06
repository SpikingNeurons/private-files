# -*- coding: utf-8 -*-

# Generic libraries import
import numpy as np
import time, os, sys
from collections import namedtuple
import unittest

# import SEED and TraceFileHandler modules
import SEED_helper


# some config params
_PRINT = False

# some global params
_G_custom_stream = sys.stdout

# Test vectors to check SEED encryption and decryption
test_vector = namedtuple('test_vector', 'key, ptx, ctx')

test_vectors_sksp = [
    test_vector(
        key=np.asarray([[0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                         0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
                        ], np.uint8),
        ptx=np.asarray([[0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                         0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F]
                        ], np.uint8),
        ctx=np.asarray([[0x5E, 0xBA, 0xC6, 0xE0, 0x05, 0x4E, 0x16, 0x68,
                         0x19, 0xAF, 0xF1, 0xCC, 0x6D, 0x34, 0x6C, 0xDB]
                        ], np.uint8)
    ),
    test_vector(
        key=np.asarray([[0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                         0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F]
                        ], np.uint8),
        ptx=np.asarray([[0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                         0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
                        ], np.uint8),
        ctx=np.asarray([[0xC1, 0x1F, 0x22, 0xF2, 0x01, 0x40, 0x50, 0x50,
                         0x84, 0x48, 0x35, 0x97, 0xE4, 0x37, 0x0F, 0x43]
                        ], np.uint8)
    ),
    test_vector(
        key=np.asarray([[0x47, 0x06, 0x48, 0x08, 0x51, 0xE6, 0x1B, 0xE8,
                         0x5D, 0x74, 0xBF, 0xB3, 0xFD, 0x95, 0x61, 0x85]
                        ], np.uint8),
        ptx=np.asarray([[0x83, 0xA2, 0xF8, 0xA2, 0x88, 0x64, 0x1F, 0xB9,
                         0xA4, 0xE9, 0xA5, 0xCC, 0x2F, 0x13, 0x1C, 0x7D]
                        ], np.uint8),
        ctx=np.asarray([[0xEE, 0x54, 0xD1, 0x3E, 0xBC, 0xAE, 0x70, 0x6D,
                         0x22, 0x6B, 0xC3, 0x14, 0x2C, 0xD4, 0x0D, 0x4A]
                        ], np.uint8)
    ),
    test_vector(
        key=np.asarray([[0x28, 0xDB, 0xC3, 0xBC, 0x49, 0xFF, 0xD8, 0x7D,
                         0xCF, 0xA5, 0x09, 0xB1, 0x1D, 0x42, 0x2B, 0xE7]
                        ], np.uint8),
        ptx=np.asarray([[0xB4, 0x1E, 0x6B, 0xE2, 0xEB, 0xA8, 0x4A, 0x14,
                         0x8E, 0x2E, 0xED, 0x84, 0x59, 0x3C, 0x5E, 0xC7]
                        ], np.uint8),
        ctx=np.asarray([[0x9B, 0x9B, 0x7B, 0xFC, 0xD1, 0x81, 0x3C, 0xB9,
                         0x5D, 0x0B, 0x36, 0x18, 0xF4, 0x0F, 0x51, 0x22]
                        ], np.uint8)
    )
]

test_vector_mkmp = test_vector(
    key=np.asarray([
        [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
         0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
        [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
         0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F],
        [0x47, 0x06, 0x48, 0x08, 0x51, 0xE6, 0x1B, 0xE8,
         0x5D, 0x74, 0xBF, 0xB3, 0xFD, 0x95, 0x61, 0x85],
        [0x28, 0xDB, 0xC3, 0xBC, 0x49, 0xFF, 0xD8, 0x7D,
         0xCF, 0xA5, 0x09, 0xB1, 0x1D, 0x42, 0x2B, 0xE7]
    ], np.uint8),
    ptx=np.asarray([
        [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
         0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F],
        [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
         0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
        [0x83, 0xA2, 0xF8, 0xA2, 0x88, 0x64, 0x1F, 0xB9,
         0xA4, 0xE9, 0xA5, 0xCC, 0x2F, 0x13, 0x1C, 0x7D],
        [0xB4, 0x1E, 0x6B, 0xE2, 0xEB, 0xA8, 0x4A, 0x14,
         0x8E, 0x2E, 0xED, 0x84, 0x59, 0x3C, 0x5E, 0xC7]
    ], np.uint8),
    ctx=np.asarray([
        [0x5E, 0xBA, 0xC6, 0xE0, 0x05, 0x4E, 0x16, 0x68,
         0x19, 0xAF, 0xF1, 0xCC, 0x6D, 0x34, 0x6C, 0xDB],
        [0xC1, 0x1F, 0x22, 0xF2, 0x01, 0x40, 0x50, 0x50,
         0x84, 0x48, 0x35, 0x97, 0xE4, 0x37, 0x0F, 0x43],
        [0xEE, 0x54, 0xD1, 0x3E, 0xBC, 0xAE, 0x70, 0x6D,
         0x22, 0x6B, 0xC3, 0x14, 0x2C, 0xD4, 0x0D, 0x4A],
        [0x9B, 0x9B, 0x7B, 0xFC, 0xD1, 0x81, 0x3C, 0xB9,
         0x5D, 0x0B, 0x36, 0x18, 0xF4, 0x0F, 0x51, 0x22]
    ], np.uint8)
)


def get_test_vector_big_skmp(test_big_size):
    return test_vector(
        key=np.asarray(
            [0x28, 0xDB, 0xC3, 0xBC, 0x49, 0xFF, 0xD8, 0x7D,
             0xCF, 0xA5, 0x09, 0xB1, 0x1D, 0x42, 0x2B, 0xE7],
            np.uint8),
        ptx=np.repeat(
            np.asarray(
                [[0xB4, 0x1E, 0x6B, 0xE2, 0xEB, 0xA8, 0x4A, 0x14,
                  0x8E, 0x2E, 0xED, 0x84, 0x59, 0x3C, 0x5E, 0xC7]],
                np.uint8
            ),
            test_big_size, axis=0),
        ctx=np.repeat(
            np.asarray(
                [[0x9B, 0x9B, 0x7B, 0xFC, 0xD1, 0x81, 0x3C, 0xB9,
                  0x5D, 0x0B, 0x36, 0x18, 0xF4, 0x0F, 0x51, 0x22]],
                np.uint8
            ),
            test_big_size, axis=0)
    )


def get_test_vector_big_mkmp(test_big_size):
    return test_vector(
        key=np.repeat(
            np.asarray(
                [[0x28, 0xDB, 0xC3, 0xBC, 0x49, 0xFF, 0xD8, 0x7D,
                  0xCF, 0xA5, 0x09, 0xB1, 0x1D, 0x42, 0x2B, 0xE7]],
                np.uint8
            ),
            test_big_size, axis=0),
        ptx=np.repeat(
            np.asarray(
                [[0xB4, 0x1E, 0x6B, 0xE2, 0xEB, 0xA8, 0x4A, 0x14,
                  0x8E, 0x2E, 0xED, 0x84, 0x59, 0x3C, 0x5E, 0xC7]],
                np.uint8
            ),
            test_big_size, axis=0),
        ctx=np.repeat(
            np.asarray(
                [[0x9B, 0x9B, 0x7B, 0xFC, 0xD1, 0x81, 0x3C, 0xB9,
                  0x5D, 0x0B, 0x36, 0x18, 0xF4, 0x0F, 0x51, 0x22]],
                np.uint8
            ),
            test_big_size, axis=0)
    )


def print_utility(time_taken, data_size, method_name):

    if not _PRINT:
        return

    _G_custom_stream.write('\n..... ' + str(time_taken) + '\t sec taken by ' +
                           method_name + ' for data size of ' +
                           str(data_size))
    _G_custom_stream.write('\n..... ')
    pass


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


def _third_party_decrypt_seed(cipher_text, key):
    from cryptography.hazmat.backends.openssl.backend import backend
    from cryptography.hazmat.primitives.ciphers import algorithms, base, modes
    cipher = base.Cipher(
        algorithms.SEED(key),
        modes.ECB(),
        backend
    )
    decryptor = cipher.decryptor()
    ct = decryptor.update(cipher_text)
    ct += decryptor.finalize()
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


def _third_party_decrypt_aes(cipher_text, key):
    from cryptography.hazmat.backends.openssl.backend import backend
    from cryptography.hazmat.primitives.ciphers import algorithms, base, modes
    cipher = base.Cipher(
        algorithms.AES(key),
        modes.ECB(),
        backend
    )
    decryptor = cipher.decryptor()
    ct = decryptor.update(cipher_text)
    ct += decryptor.finalize()
    return ct


def third_party_encrypt(plain_text, key, algorithm):
    if algorithm == 'SEED':
        ct = _third_party_encrypt_seed(
            plain_text.tobytes('C'), key.tobytes('C'))
        return np.fromstring(ct, dtype=np.uint8)
    if algorithm == 'AES':
        ct = _third_party_encrypt_aes(
            plain_text.tobytes('C'), key.tobytes('C'))
        return np.fromstring(ct, dtype=np.uint8)
    return None


def third_party_decrypt(cipher_text, key, algorithm):
    if algorithm == 'SEED':
        ct = _third_party_decrypt_seed(
            cipher_text.tobytes('C'), key.tobytes('C'))
        return np.fromstring(ct, dtype=np.uint8)
    if algorithm == 'AES':
        ct = _third_party_decrypt_aes(
            cipher_text.tobytes('C'), key.tobytes('C'))
        return np.fromstring(ct, dtype=np.uint8)
    return None


class TimingTest(unittest.TestCase):
    """
    Tests related to timing performance of SEED module.
    Note that the data is generated in memory using fake data.
    """
    def setUp(self):
        self.big_size = 10000
        self.threads_to_use = 1
        pass

    def tearDown(self):
        pass

    def test_encrypt_multiple_key(self):
        """
        Check for cython implementation encryption of SEED (with multiple keys)
        """
        cy_solver = SEED_helper.SEEDAlgorithmCy(decrypt=False)
        vec = get_test_vector_big_mkmp(self.big_size)
        start = time.time()
        res = cy_solver.execute(
            val_text=vec.ptx,
            keys=vec.key,
            rnd=16,
            step='Output',
            decrypt=False,
            threads_to_use=self.threads_to_use)
        end = time.time()
        print_utility(end-start, self.big_size, self._testMethodName)
        equality_check = np.array_equal(res, vec.ctx)
        self.assertTrue(
            equality_check, 'SEED_helper_encrypt_multiple_key failed')

    def test_decrypt_multiple_key(self):
        """
        Check for cython implementation decryption of SEED (with multiple keys)
        """
        cy_solver = SEED_helper.SEEDAlgorithmCy(decrypt=True)
        vec = get_test_vector_big_mkmp(self.big_size)
        start = time.time()
        res = cy_solver.execute(
            val_text=vec.ctx,
            keys=vec.key,
            rnd=16,
            step='Output',
            decrypt=True,
            threads_to_use=self.threads_to_use)
        end = time.time()
        print_utility(end-start, self.big_size, self._testMethodName)
        equality_check = np.array_equal(res, vec.ptx)
        self.assertTrue(
            equality_check, 'SEED_helper_decrypt_multiple_key failed')

    def test_encrypt_single_key(self):
        """
        Check for cython implementation encryption of SEED (with single keys)
        """
        cy_solver = SEED_helper.SEEDAlgorithmCy(decrypt=False)
        test_vec = get_test_vector_big_skmp(self.big_size)
        start = time.time()
        res = cy_solver.execute(
            val_text=test_vec.ptx,
            keys=test_vec.key,
            rnd=16,
            step='Output',
            decrypt=False,
            threads_to_use=self.threads_to_use)
        end = time.time()
        print_utility(end-start, self.big_size, self._testMethodName)
        equality_check = np.array_equal(res, test_vec.ctx)
        self.assertTrue(
            equality_check, 'SEED_helper_encrypt_single_key failed')

    def test_decrypt_single_key(self):
        """
        Check for cython implementation decryption of SEED (with single keys)
        """
        cy_solver = SEED_helper.SEEDAlgorithmCy(decrypt=True)
        test_vec = get_test_vector_big_skmp(self.big_size)
        start = time.time()
        res = cy_solver.execute(
            val_text=test_vec.ctx,
            keys=test_vec.key,
            rnd=16,
            step='Output',
            decrypt=True,
            threads_to_use=self.threads_to_use)
        end = time.time()
        print_utility(end-start, self.big_size, self._testMethodName)
        equality_check = np.array_equal(res, test_vec.ptx)
        self.assertTrue(
            equality_check, 'SEED_helper_decrypt_single_key failed')


class ThirdPartyTest(unittest.TestCase):
    """
    This is the third party SEED module for verifying the encryption and
    decryption. You can safely ignore these tests.
    """
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_encrypt(self):
        """
        Check for third party encryption of SEED
        """
        try:
            equality_check = True
            for ptx, ctx, key in zip(
                    test_vector_mkmp.ptx,
                    test_vector_mkmp.ctx,
                    test_vector_mkmp.key):
                ctx_ret = third_party_encrypt(ptx, key, 'SEED')
                equality_check = \
                    equality_check and np.array_equal(ctx, ctx_ret)
            self.assertTrue(
                equality_check, 'Third party SEED encryption failed')
        except ImportError as e:
            self.skipTest('Third party SEED library missing: ' + str(e))

    def test_decrypt(self):
        """
        Check for third party decryption of SEED
        """
        try:
            equality_check = True
            for ptx, ctx, key in zip(
                    test_vector_mkmp.ptx,
                    test_vector_mkmp.ctx,
                    test_vector_mkmp.key):
                ptx_ret = third_party_decrypt(ctx, key, 'SEED')
                equality_check = \
                    equality_check and np.array_equal(ptx, ptx_ret)
            self.assertTrue(
                equality_check, 'Third party SEED decryption failed')
        except ImportError as e:
            self.skipTest('Third party SEED library missing: ' + str(e))


class StandAloneTest(unittest.TestCase):
    """
    Test all basic functions using fake test vectors.
    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_encrypt_multi_key(self):
        """
        Check for cython implementation encryption of SEED (with multiple keys)
        """
        cy_solver = SEED_helper.SEEDAlgorithmCy(decrypt=False)
        res = cy_solver.execute(
            val_text=test_vector_mkmp.ptx,
            keys=test_vector_mkmp.key,
            rnd=16,
            step='Output',
            decrypt=False,
            threads_to_use=1)
        equality_check = np.array_equal(res, test_vector_mkmp.ctx)
        self.assertTrue(
            equality_check, 'SEED_helper_encrypt_multiple_key failed')

    def test_decrypt_multi_key(self):
        """
        Check for cython implementation decryption of SEED (with multiple keys)
        """
        cy_solver = SEED_helper.SEEDAlgorithmCy(decrypt=True)
        res = cy_solver.execute(
            val_text=test_vector_mkmp.ctx,
            keys=test_vector_mkmp.key,
            rnd=16,
            step='Output',
            decrypt=True,
            threads_to_use=1)
        equality_check = np.array_equal(res, test_vector_mkmp.ptx)
        self.assertTrue(
            equality_check, 'SEED_helper_decrypt_multiple_key failed')

    def test_encrypt_single_key(self):
        """
        Check for cython implementation encryption of SEED (with single keys)
        """
        test_vec = get_test_vector_big_skmp(10)
        cy_solver = SEED_helper.SEEDAlgorithmCy(decrypt=False)
        res = cy_solver.execute(
            val_text=test_vec.ptx,
            keys=test_vec.key,
            rnd=16,
            step='Output',
            decrypt=False,
            threads_to_use=1)
        equality_check = np.array_equal(res, test_vec.ctx)
        self.assertTrue(
            equality_check, 'SEED_helper_encrypt_single_key failed')

    def test_decrypt_single_key(self):
        """
        Check for cython implementation decryption of SEED (with single keys)
        """
        test_vec = get_test_vector_big_skmp(10)
        cy_solver = SEED_helper.SEEDAlgorithmCy(decrypt=True)
        res = cy_solver.execute(
            val_text=test_vec.ctx,
            keys=test_vec.key,
            rnd=16,
            step='Output',
            decrypt=True,
            threads_to_use=1)
        equality_check = np.array_equal(res, test_vec.ptx)
        self.assertTrue(
            equality_check, 'SEED_helper_decrypt_single_key failed')

    def test_encrypt_key_schedule(self):
        """
        Test of key schedule generation logic while encryption
        """
        pass

    def test_decrypt_key_schedule(self):
        """
        Test of key schedule generation logic while decryption
        """
        pass

    def test_encrypt_intermediate_F(self):
        """
        We only test output of function F which is intermediate value.
        We check for all rounds.
        """
        pass

    def test_decrypt_intermediate_F(self):
        """
        We only test output of function F which is intermediate value.
        We check for all rounds.
        """
        pass


def all_tests():

    test_loader = unittest.TestLoader()
    suite1 = test_loader.loadTestsFromTestCase(TimingTest)
    suite2 = test_loader.loadTestsFromTestCase(StandAloneTest)
    # suite = unittest.TestSuite()
    # suite.addTest(suite1)
    # suite.addTest(suite2)
    _G_custom_stream.write('\n\n\n|||||||||||||||||||||||||||||||||||||||||||'
                           '|||||||||||||||||||||||||||||||||||\n\n')
    s = unittest.TextTestRunner(stream=_G_custom_stream, descriptions=True,
                                verbosity=3).run(suite1)
    _G_custom_stream.write('\n|||||||||||||||||||||||||||||||||||||||||||||||'
                           '|||||||||||||||||||||||||||||||\n')
    _G_custom_stream.write('|||||||||||||||||||||||||||||||||||||||||||||||||'
                           '|||||||||||||||||||||||||||||\n\n')
    s = unittest.TextTestRunner(stream=_G_custom_stream, descriptions=True,
                                verbosity=3).run(suite2)
    _G_custom_stream.write('\n|||||||||||||||||||||||||||||||||||||||||||||||'
                           '|||||||||||||||||||||||||||||||\n\n')


def one_test():

    test_loader = unittest.TestLoader()
    suite1 = test_loader.loadTestsFromTestCase(StandAloneTest)
    _G_custom_stream.write('\n\n\n|||||||||||||||||||||||||||||||||||||||||||'
                           '|||||||||||||||||||||||||||||||||||\n\n')
    s = unittest.TextTestRunner(stream=_G_custom_stream, descriptions=True,
                                verbosity=3).run(suite1)
    _G_custom_stream.write('\n|||||||||||||||||||||||||||||||||||||||||||||||'
                           '|||||||||||||||||||||||||||||||\n')
    pass

if __name__ == '__main__':

    #all_tests()
    one_test()
    #unittest.main()








