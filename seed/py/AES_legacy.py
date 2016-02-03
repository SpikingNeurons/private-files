# what is this? key length in bytes?
AES128_BYT_LEN = 16

# and what is this?
BYTE_NUM_VAL = 256

def fast_AES_algo(input, key):

    for round in range(10):    
        input=fast_AES_round(key,input,round==9)   
        key = round_key(key,round+1)
    return [input[byte]^key[byte] for byte in range(16)]    

      
def AES_algo(input, key):
    rounds_int_values=[]
    round_results=dict()
    for round in range(10):
        # why call AES_round(key, input, last_round) instead of AES_round(input, key, last_round)
        # ????
        round_results['ARK'], round_results['SR'], round_results['SB'], round_results['MC'], input = AES_round(key, input, round==9)
        key = round_results['RK'] = round_key(key,round+1)
        rounds_int_values.append(round_results.copy()) 
    return [input[byte]^key[byte] for byte in range(16)], rounds_int_values    


class AesSbinR1(CryptoData):
    """AES 1st round SBox input"""
    max_range = AES128_BYT_LEN
    num_val = BYTE_NUM_VAL
    str = 'sbinr1'
    label = 'AddRoundKey'
    round = 0
    known_input = 'plaintext'

    def __init__(self, **parameters):
        self.parameters = parameters
    def calc(self, crypto_data, guess, trace_range, data_range):
        plaintext = self._fetch(crypto_data, 'plaintext', trace_range, data_range)
        if 'key' in guess:
            key = guess['key']
        else:
            key = self._fetch(crypto_data, 'key', trace_range, data_range)
        return np.bitwise_xor( plaintext, key )
    
    def calc_from_value(self, plaintext, key):
        return np.bitwise_xor( plaintext, key )


class AesSboutR1(CryptoData):
    """AES 1st round SBox output"""
    max_range = AES128_BYT_LEN
    num_val = BYTE_NUM_VAL
    label = 'S-Box'
    round = 0
    str = 'sboutr1'

    known_input = 'plaintext'
    def __init__(self, **parameters):
        self.parameters = parameters
    def calc(self, crypto_data, guess, trace_range, data_range):
        plaintext = self._fetch(crypto_data, 'plaintext', trace_range, data_range)
        if 'key' in guess:
            key = guess['key']
        else:
            key = self._fetch(crypto_data, 'key', trace_range, data_range)
        return apply_sbox(np.bitwise_xor( plaintext, key ))
    
    def calc_from_value(self, plaintext, key):
        return apply_sbox(np.bitwise_xor( plaintext, key ))


# OLD IMPLEMENTATION!
class AES128CryptoDataProcessor(CryptoDataProcessor):
    """
    Handles intermediate data for AES 128
    """
    algo_name = 'AES128'
    cryptodata_list = [AesSbinR1, AesSboutR1]
    parameters_list = []
    plaintext_len = 16
    ciphertext_len = 16
    key_len = 16

def round_key(key,round):

    rk=[S_Box[key[-3]]^Rcon[round]^key[-16],\
        S_Box[key[-2]]^key[-15],\
        S_Box[key[-1]]^key[-14],\
        S_Box[key[-4]]^key[-13]]
        
    tmp=rk[-4:]
    rk+=[tmp[0]^key[-12],\
         tmp[1]^key[-11],\
         tmp[2]^key[-10],\
         tmp[3]^key[-9]]
                
    tmp=rk[-4:]
    rk+=[tmp[0]^key[-8],\
         tmp[1]^key[-7],\
         tmp[2]^key[-6],\
         tmp[3]^key[-5]]

    tmp=rk[-4:]
    rk+=[tmp[0]^key[-4],\
         tmp[1]^key[-3],\
         tmp[2]^key[-2],\
         tmp[3]^key[-1]]
                
    return np.array(rk, dtype=np.uint8)


def fast_AES_round(input, key, last_round):

    ciphertext=[S_Box[input[0] ^ key[0]],\
        S_Box[input[5] ^  key[5]],\
        S_Box[input[10] ^ key[10]],\
        S_Box[input[15] ^ key[15]],\
        S_Box[input[4]  ^ key[4]],\
        S_Box[input[9]  ^ key[9]],\
        S_Box[input[14] ^ key[14]],\
        S_Box[input[3]  ^ key[3]],\
        S_Box[input[8]  ^ key[8]],\
        S_Box[input[13] ^ key[13]],\
        S_Box[input[2]  ^ key[2]],\
        S_Box[input[7]  ^ key[7]],\
        S_Box[input[12] ^ key[12]],\
        S_Box[input[1]  ^ key[1]],\
        S_Box[input[6]  ^ key[6]],\
        S_Box[input[11] ^ key[11]]]

    # if last_round != 1:
    if not last_round:
        ciphertext=[T_tables[0][ciphertext[0]] ^ T_tables[1][ciphertext[1]] ^ T_tables[2][ciphertext[2]] ^ T_tables[3][ciphertext[3]],
        T_tables[0][ciphertext[4]] ^ T_tables[1][ciphertext[5]] ^ T_tables[2][ciphertext[6]] ^ T_tables[3][ciphertext[7]],
        T_tables[0][ciphertext[8]] ^ T_tables[1][ciphertext[9]] ^ T_tables[2][ciphertext[10]] ^ T_tables[3][ciphertext[11]],
        T_tables[0][ciphertext[12]] ^ T_tables[1][ciphertext[13]] ^ T_tables[2][ciphertext[14]] ^ T_tables[3][ciphertext[15]]]
            
        ciphertext = [  ciphertext[0]>>24, (ciphertext[0]&0xff0000)>>16, (ciphertext[0]&0xff00)>>8, ciphertext[0]&0xff,\
                        ciphertext[1]>>24, (ciphertext[1]&0xff0000)>>16, (ciphertext[1]&0xff00)>>8, ciphertext[1]&0xff,\
                        ciphertext[2]>>24, (ciphertext[2]&0xff0000)>>16, (ciphertext[2]&0xff00)>>8, ciphertext[2]&0xff,\
                        ciphertext[3]>>24, (ciphertext[3]&0xff0000)>>16, (ciphertext[3]&0xff00)>>8, ciphertext[3]&0xff]
       
    return ciphertext
      
def AES_round(input, key, last_round):

    ARK=[input[i] ^ key[i] for i in range(16)]
    SR=[ARK[i] for i in [0,5,10,15,4,9,14,3,8,13,2,7,12,1,6,11]]
    SB=[S_Box[SR[i]] for i in range(16)]

    # why?!
    #if last_round != 1:
    if not last_round:
        MC=[T_tables[0][SB[0]] ^ T_tables[1][SB[1]] ^ T_tables[2][SB[2]] ^ T_tables[3][SB[3]],
        T_tables[0][SB[4]] ^ T_tables[1][SB[5]] ^ T_tables[2][SB[6]] ^ T_tables[3][SB[7]],
        T_tables[0][SB[8]] ^ T_tables[1][SB[9]] ^ T_tables[2][SB[10]] ^ T_tables[3][SB[11]],
        T_tables[0][SB[12]] ^ T_tables[1][SB[13]] ^ T_tables[2][SB[14]] ^ T_tables[3][SB[15]]]
      
        MC = [  MC[0]>>24, (MC[0]&0xff0000)>>16, (MC[0]&0xff00)>>8, MC[0]&0xff,\
                        MC[1]>>24, (MC[1]&0xff0000)>>16, (MC[1]&0xff00)>>8, MC[1]&0xff,\
                        MC[2]>>24, (MC[2]&0xff0000)>>16, (MC[2]&0xff00)>>8, MC[2]&0xff,\
                        MC[3]>>24, (MC[3]&0xff0000)>>16, (MC[3]&0xff00)>>8, MC[3]&0xff
        ]
    else:
        MC=SB

    # why return MC twice?
    return ARK,SR,SB,MC,MC
