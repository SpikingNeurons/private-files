
import numpy as np
from SEED_tables import SS0, SS1, SS2, SS3, KC


class SEEDAlgorithmPy:

    def _py_mask_with_8f(self, var):
        #return np.bitwise_and(var, 0xffffffff)
        return var


    def _py_g_func(self, var):
        index_l00 = np.bitwise_and(var, 0xff)
        index_l08 = np.bitwise_and(np.right_shift(var, 8), 0xff)
        index_l16 = np.bitwise_and(np.right_shift(var, 16), 0xff)
        index_l24 = np.bitwise_and(np.right_shift(var, 24), 0xff)
        return np.bitwise_xor(
            np.bitwise_xor(SS0[index_l00], SS1[index_l08]),
            np.bitwise_xor(SS2[index_l16], SS3[index_l24])
        )


    def _py_right8_left24_update(self, var1, var2):
        return self._py_mask_with_8f(np.bitwise_xor(
            np.right_shift(var1, 8),
            np.left_shift(var2, 24)
        ))


    def _py_left8_right24_update(self, var1, var2):
        return self._py_mask_with_8f(np.bitwise_xor(
            np.left_shift(var1, 8),
            np.right_shift(var2, 24)
        ))


    def _py_generate_key_schedule(self, keys, rnd):
        x1, x2, x3, x4 = self._py_fragment_block_to_words(keys)

        key_schedule_0 = None
        key_schedule_1 = None
        t0 = None

        # round 0 update
        if rnd == 0:
            key_schedule_0 = self._py_g_func(self._py_mask_with_8f(x1 + x3 - KC[0]))
            key_schedule_1 = self._py_g_func(self._py_mask_with_8f(x2 - x4 + KC[0]))

        # round 1 update
        t0 = x1
        x1 = self._py_right8_left24_update(x1, x2)
        x2 = self._py_right8_left24_update(x2, t0)
        if rnd == 1:
            key_schedule_0 = self._py_g_func(self._py_mask_with_8f(x1 + x3 - KC[1]))
            key_schedule_1 = self._py_g_func(self._py_mask_with_8f(x2 + KC[1] - x4))

        # round 2 ... 16
        for ii in np.arange(16)[2::2]:
            t0 = x3
            x3 = self._py_left8_right24_update(x3, x4)
            x4 = self._py_left8_right24_update(x4, t0)
            if rnd == ii:
                key_schedule_0 = self._py_g_func(self._py_mask_with_8f(x1 + x3 - KC[ii]))
                key_schedule_1 = self._py_g_func(self._py_mask_with_8f(x2 + KC[ii] - x4))
                break
            t0 = x1
            x1 = self._py_right8_left24_update(x1, x2)
            x2 = self._py_right8_left24_update(x2, t0)
            if rnd == ii + 1:
                key_schedule_0 = self._py_g_func(self._py_mask_with_8f(x1 + x3 - KC[ii + 1]))
                key_schedule_1 = self._py_g_func(self._py_mask_with_8f(x2 + KC[ii + 1] - x4))
                break

        return key_schedule_0, key_schedule_1


    def _py_print_hex(self, arr):
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
            print(s)
        print('\n')



    def _py_fragment_block_to_words(self, var):
        var_ = var[::-1].copy()
        var_.dtype = np.uint32
        var__ = var_[::-1]
        x1 = var__[::4].copy()
        x2 = var__[1::4].copy()
        x3 = var__[2::4].copy()
        x4 = var__[3::4].copy()
        return x1, x2, x3, x4


    def encrypt(self, plain_text, keys, rnd, step):
        x1, x2, x3, x4 = self._py_fragment_block_to_words(plain_text)

        for ii in np.arange(16):

            if ii % 2 == 0:
                a1 = x1
                a2 = x2
                a3 = x3
                a4 = x4
            else:
                a1 = x3
                a2 = x4
                a3 = x1
                a4 = x2

            #print('.......................' + str(ii))
            #self._py_print_hex(a1)
            #self._py_print_hex(a2)
            #self._py_print_hex(a3)
            #self._py_print_hex(a4)

            ks_0, ks_1 = self._py_generate_key_schedule(keys, ii)
            t0 = np.bitwise_xor(a3, ks_0)
            t1 = np.bitwise_xor(a4, ks_1)
            np.bitwise_xor(t1, t0, out=t1)
            t1 = self._py_g_func(t1)
            t0 = self._py_mask_with_8f(t0 + t1)
            t0 = self._py_g_func(t0)
            t1 = self._py_mask_with_8f(t1 + t0)
            t1 = self._py_g_func(t1)
            t0 = self._py_mask_with_8f(t0 + t1)
            np.bitwise_xor(a1, t0, out=a1)
            np.bitwise_xor(a2, t1, out=a2)