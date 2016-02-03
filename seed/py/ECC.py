#from math import floor, ceil
import numpy as np
try:
    import gmpy2
except ImportError:
    pass

#from IPython.core.debugger import Tracer

from . import CryptoDataProcessor, CryptoData


def array_to_int(array, base=256):
    res = 0
    l = len(array)
    for i in range(l):
        res *= base
        res += int(array[-(i+1)])
    return res

def int_to_array(x, len, dtype='u1'):
    if dtype != 'u1':
        raise NotImplementedError()
    base = 0x100
    #l = floor(np.log(x)/np.log(base))+1 if (len is None) else len
    l = len
    a = x
    r = np.zeros(l, dtype=dtype)
    for i in range(l):
        r[l-i-1] = a % base
        a = a // base
    return r

class SMDigits(CryptoData):
    
    str = 'smdigits'

    def __init__(self, **kwargs):

        l = kwargs['bitlen']
        n = kwargs['field_p']

        if l%16 != 0:
            raise ValueError('bitlen must be a multiple of 16')

        self.max_range = l/2
        self.num_val = 4
        
        self.bitlen = l
        self.N = n
        self.Q = (2**l) %n
        self.Q_ = int(gmpy2.invert(self.Q, n))
        
    #@classmethod
    def calc(self, crypto_data, guess, trace_range, data_range):
        # assume rng data is stored in the key field
        if 'key' in guess:
            rng_data = guess['key']
        else:
            rng_data = self._fetch(crypto_data, 'key', trace_range, slice(None))
            #Tracer()()
        return self.sm_bit_select( self.compute_eph_key( rng_data ) )[:,data_range]

    def compute_eph_key(self, rng_data):

        l0 = len(rng_data)
        l1 = self.bitlen // 8

        r = np.empty((l0, l1), dtype='u1')

        for i in range(l0):

            rand_c = array_to_int(rng_data[i,0:32])
            c      = rand_c >> 31
            rand_s = array_to_int(rng_data[i,32:56])
            s      = (((rand_s*self.Q_) %self.N)%(self.N-1)) +1
            t      = array_to_int(rng_data[i,56:64])
            phi    = array_to_int(rng_data[i,64:72]) | 1

      
            u = t*(self.N-1) + s
            #r = c + u
            #r2 = r % (self.N-1)
            r2 = (c+u) % (self.N-1)
            v = int(gmpy2.invert(phi, self.N))
            R = r2*v
            R2 = (R - s*v) % ((self.N-1)*v)
            #D = R2 + v
            #D2 = D + u
            #d3 = D2 % self.N
            d3 = (R2 + v + u) % self.N
            d2 = (d3 + t - s) % self.N

            #print('c  = ' + hex(c) )
            #print('s  = ' + hex(s) )
            #print('t  = ' + hex(t) )
            #print('phi= ' + hex(phi))
            
            #print('u  = ' + hex(u) )
            ##print('r  = ' + hex(r) )
            #print('r2 = ' + hex(r2))
            #print('v  = ' + hex(v) )
            #print('R  = ' + hex(R) )
            #print('R2 = ' + hex(R2))
            ##print('D  = ' + hex(D) )
            ##print('D2 = ' + hex(D2))
            #print('d3 = ' + hex(d3))
            #print('d\' = ' + hex(d2))

            r[i] = int_to_array(d2, l1)

        return r

    def sm_bit_select(self, key):
        '''
        Secure scalar multiplication comb digit scanning

        digit[i] = 2*bit[l-i-1] + bit[l/2-i-1] with i ranging from 0 to l/2

        where first null digits are skipped (array is rotated to the left)
        '''
        l0, l1 = key.shape
        d = l1//2
        l2 = 4*l1
        r = np.zeros((l0, l2), dtype='u1')

        # digit computation loop
        for i in range(d):
            #for j, mask in enumerate([0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01]):
            #    r[:,i+j] = (((key[:,i] & mask) >> (7-j)) << 1) + ((key[:,i+d] & mask) >> (7-j))
            r[:,8*i]   = ((key[:,i] & 0x80) >> 6) + ((key[:,i+d] & 0x80) >> 7)
            r[:,8*i+1] = ((key[:,i] & 0x40) >> 5) + ((key[:,i+d] & 0x40) >> 6)
            r[:,8*i+2] = ((key[:,i] & 0x20) >> 4) + ((key[:,i+d] & 0x20) >> 5)
            r[:,8*i+3] = ((key[:,i] & 0x10) >> 3) + ((key[:,i+d] & 0x10) >> 4)
            r[:,8*i+4] = ((key[:,i] & 0x08) >> 2) + ((key[:,i+d] & 0x08) >> 3)
            r[:,8*i+5] = ((key[:,i] & 0x04) >> 1) + ((key[:,i+d] & 0x04) >> 2)
            r[:,8*i+6] =  (key[:,i] & 0x02)       + ((key[:,i+d] & 0x02) >> 1)
            r[:,8*i+7] = ((key[:,i] & 0x01) << 1) +  (key[:,i+d] & 0x01)

        # rotate digits left in order to have r[i,0] not null
        for i in range(l0):
            fnz = 0
            while fnz<l2 and r[i,fnz] == 0:
                fnz += 1
            if fnz > 0 and fnz < l2-1:
                r[i,:l2-fnz] = r[i,fnz:]
                r[i,l2-fnz:] = 0
            
        #Tracer()()
        return r

class ECC_CLv3(CryptoDataProcessor):
    """
    Handles intermediate data for ECC (CL v.3)
    """
    algo_name = 'ECC_CLv3'
    cryptodata_list = [SMDigits]
    parameters_list = ['field_p', 'bitlen']


if __name__ == '__main__':

    from scaffold.core.trace_file import TraceFile
    from scaffold.core.trace_file_handler import TraceFileHandler
    np.set_printoptions(formatter={'int':hex})
    tf = TraceFile('D:\\ecc_scalar_mult\\XfYrZfWfMr\\EccGfp_KeyGen_RANOM MASKC2.trs', data_format='K 72 T 32 I 40')
    ecc160 = {'field_p': 0x0100000000000000000001B8FA16DFAB9ACA16B6B3,
              'bitlen': 160}

    tfh = TraceFileHandler(tf, algo='ECC_CLv3', algo_param=ecc160)

    #n = 0x0100000000000000000001B8FA16DFAB9ACA16B6B3
    #d' = 0xf87b706045bd186d4b1b7900e62b987ef7d46074

    smd = SMDigits(n, 192)
    d2 = smd.compute_eph_key(tf)

    print('d\' = ' + hex(d2))

    
# required by __init__.py to register CryptoDataProcessors
crypto_data_processors = {"ECC_CLv3": ECC_CLv3}