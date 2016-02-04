
T_tables=[
    [0x00000000,0x02010103,0x04020206,0x06030305,0x0804040c,0x0a05050f,0x0c06060a,0x0e070709,0x10080818,0x1209091b,0x140a0a1e,0x160b0b1d,0x180c0c14,0x1a0d0d17,0x1c0e0e12,0x1e0f0f11,
    0x20101030,0x22111133,0x24121236,0x26131335,0x2814143c,0x2a15153f,0x2c16163a,0x2e171739,0x30181828,0x3219192b,0x341a1a2e,0x361b1b2d,0x381c1c24,0x3a1d1d27,0x3c1e1e22,0x3e1f1f21,
    0x40202060,0x42212163,0x44222266,0x46232365,0x4824246c,0x4a25256f,0x4c26266a,0x4e272769,0x50282878,0x5229297b,0x542a2a7e,0x562b2b7d,0x582c2c74,0x5a2d2d77,0x5c2e2e72,0x5e2f2f71,
    0x60303050,0x62313153,0x64323256,0x66333355,0x6834345c,0x6a35355f,0x6c36365a,0x6e373759,0x70383848,0x7239394b,0x743a3a4e,0x763b3b4d,0x783c3c44,0x7a3d3d47,0x7c3e3e42,0x7e3f3f41,
    0x804040c0,0x824141c3,0x844242c6,0x864343c5,0x884444cc,0x8a4545cf,0x8c4646ca,0x8e4747c9,0x904848d8,0x924949db,0x944a4ade,0x964b4bdd,0x984c4cd4,0x9a4d4dd7,0x9c4e4ed2,0x9e4f4fd1,
    0xa05050f0,0xa25151f3,0xa45252f6,0xa65353f5,0xa85454fc,0xaa5555ff,0xac5656fa,0xae5757f9,0xb05858e8,0xb25959eb,0xb45a5aee,0xb65b5bed,0xb85c5ce4,0xba5d5de7,0xbc5e5ee2,0xbe5f5fe1,
    0xc06060a0,0xc26161a3,0xc46262a6,0xc66363a5,0xc86464ac,0xca6565af,0xcc6666aa,0xce6767a9,0xd06868b8,0xd26969bb,0xd46a6abe,0xd66b6bbd,0xd86c6cb4,0xda6d6db7,0xdc6e6eb2,0xde6f6fb1,
    0xe0707090,0xe2717193,0xe4727296,0xe6737395,0xe874749c,0xea75759f,0xec76769a,0xee777799,0xf0787888,0xf279798b,0xf47a7a8e,0xf67b7b8d,0xf87c7c84,0xfa7d7d87,0xfc7e7e82,0xfe7f7f81,
    0x1b80809b,0x19818198,0x1f82829d,0x1d83839e,0x13848497,0x11858594,0x17868691,0x15878792,0x0b888883,0x09898980,0x0f8a8a85,0x0d8b8b86,0x038c8c8f,0x018d8d8c,0x078e8e89,0x058f8f8a,
    0x3b9090ab,0x399191a8,0x3f9292ad,0x3d9393ae,0x339494a7,0x319595a4,0x379696a1,0x359797a2,0x2b9898b3,0x299999b0,0x2f9a9ab5,0x2d9b9bb6,0x239c9cbf,0x219d9dbc,0x279e9eb9,0x259f9fba,
    0x5ba0a0fb,0x59a1a1f8,0x5fa2a2fd,0x5da3a3fe,0x53a4a4f7,0x51a5a5f4,0x57a6a6f1,0x55a7a7f2,0x4ba8a8e3,0x49a9a9e0,0x4faaaae5,0x4dababe6,0x43acacef,0x41adadec,0x47aeaee9,0x45afafea,
    0x7bb0b0cb,0x79b1b1c8,0x7fb2b2cd,0x7db3b3ce,0x73b4b4c7,0x71b5b5c4,0x77b6b6c1,0x75b7b7c2,0x6bb8b8d3,0x69b9b9d0,0x6fbabad5,0x6dbbbbd6,0x63bcbcdf,0x61bdbddc,0x67bebed9,0x65bfbfda,
    0x9bc0c05b,0x99c1c158,0x9fc2c25d,0x9dc3c35e,0x93c4c457,0x91c5c554,0x97c6c651,0x95c7c752,0x8bc8c843,0x89c9c940,0x8fcaca45,0x8dcbcb46,0x83cccc4f,0x81cdcd4c,0x87cece49,0x85cfcf4a,
    0xbbd0d06b,0xb9d1d168,0xbfd2d26d,0xbdd3d36e,0xb3d4d467,0xb1d5d564,0xb7d6d661,0xb5d7d762,0xabd8d873,0xa9d9d970,0xafdada75,0xaddbdb76,0xa3dcdc7f,0xa1dddd7c,0xa7dede79,0xa5dfdf7a,
    0xdbe0e03b,0xd9e1e138,0xdfe2e23d,0xdde3e33e,0xd3e4e437,0xd1e5e534,0xd7e6e631,0xd5e7e732,0xcbe8e823,0xc9e9e920,0xcfeaea25,0xcdebeb26,0xc3ecec2f,0xc1eded2c,0xc7eeee29,0xc5efef2a,
    0xfbf0f00b,0xf9f1f108,0xfff2f20d,0xfdf3f30e,0xf3f4f407,0xf1f5f504,0xf7f6f601,0xf5f7f702,0xebf8f813,0xe9f9f910,0xeffafa15,0xedfbfb16,0xe3fcfc1f,0xe1fdfd1c,0xe7fefe19,0xe5ffff1a],
    [0x00000000,0x03020101,0x06040202,0x05060303,0x0c080404,0x0f0a0505,0x0a0c0606,0x090e0707,0x18100808,0x1b120909,0x1e140a0a,0x1d160b0b,0x14180c0c,0x171a0d0d,0x121c0e0e,0x111e0f0f,
    0x30201010,0x33221111,0x36241212,0x35261313,0x3c281414,0x3f2a1515,0x3a2c1616,0x392e1717,0x28301818,0x2b321919,0x2e341a1a,0x2d361b1b,0x24381c1c,0x273a1d1d,0x223c1e1e,0x213e1f1f,
    0x60402020,0x63422121,0x66442222,0x65462323,0x6c482424,0x6f4a2525,0x6a4c2626,0x694e2727,0x78502828,0x7b522929,0x7e542a2a,0x7d562b2b,0x74582c2c,0x775a2d2d,0x725c2e2e,0x715e2f2f,
    0x50603030,0x53623131,0x56643232,0x55663333,0x5c683434,0x5f6a3535,0x5a6c3636,0x596e3737,0x48703838,0x4b723939,0x4e743a3a,0x4d763b3b,0x44783c3c,0x477a3d3d,0x427c3e3e,0x417e3f3f,
    0xc0804040,0xc3824141,0xc6844242,0xc5864343,0xcc884444,0xcf8a4545,0xca8c4646,0xc98e4747,0xd8904848,0xdb924949,0xde944a4a,0xdd964b4b,0xd4984c4c,0xd79a4d4d,0xd29c4e4e,0xd19e4f4f,
    0xf0a05050,0xf3a25151,0xf6a45252,0xf5a65353,0xfca85454,0xffaa5555,0xfaac5656,0xf9ae5757,0xe8b05858,0xebb25959,0xeeb45a5a,0xedb65b5b,0xe4b85c5c,0xe7ba5d5d,0xe2bc5e5e,0xe1be5f5f,
    0xa0c06060,0xa3c26161,0xa6c46262,0xa5c66363,0xacc86464,0xafca6565,0xaacc6666,0xa9ce6767,0xb8d06868,0xbbd26969,0xbed46a6a,0xbdd66b6b,0xb4d86c6c,0xb7da6d6d,0xb2dc6e6e,0xb1de6f6f,
    0x90e07070,0x93e27171,0x96e47272,0x95e67373,0x9ce87474,0x9fea7575,0x9aec7676,0x99ee7777,0x88f07878,0x8bf27979,0x8ef47a7a,0x8df67b7b,0x84f87c7c,0x87fa7d7d,0x82fc7e7e,0x81fe7f7f,
    0x9b1b8080,0x98198181,0x9d1f8282,0x9e1d8383,0x97138484,0x94118585,0x91178686,0x92158787,0x830b8888,0x80098989,0x850f8a8a,0x860d8b8b,0x8f038c8c,0x8c018d8d,0x89078e8e,0x8a058f8f,
    0xab3b9090,0xa8399191,0xad3f9292,0xae3d9393,0xa7339494,0xa4319595,0xa1379696,0xa2359797,0xb32b9898,0xb0299999,0xb52f9a9a,0xb62d9b9b,0xbf239c9c,0xbc219d9d,0xb9279e9e,0xba259f9f,
    0xfb5ba0a0,0xf859a1a1,0xfd5fa2a2,0xfe5da3a3,0xf753a4a4,0xf451a5a5,0xf157a6a6,0xf255a7a7,0xe34ba8a8,0xe049a9a9,0xe54faaaa,0xe64dabab,0xef43acac,0xec41adad,0xe947aeae,0xea45afaf,
    0xcb7bb0b0,0xc879b1b1,0xcd7fb2b2,0xce7db3b3,0xc773b4b4,0xc471b5b5,0xc177b6b6,0xc275b7b7,0xd36bb8b8,0xd069b9b9,0xd56fbaba,0xd66dbbbb,0xdf63bcbc,0xdc61bdbd,0xd967bebe,0xda65bfbf,
    0x5b9bc0c0,0x5899c1c1,0x5d9fc2c2,0x5e9dc3c3,0x5793c4c4,0x5491c5c5,0x5197c6c6,0x5295c7c7,0x438bc8c8,0x4089c9c9,0x458fcaca,0x468dcbcb,0x4f83cccc,0x4c81cdcd,0x4987cece,0x4a85cfcf,
    0x6bbbd0d0,0x68b9d1d1,0x6dbfd2d2,0x6ebdd3d3,0x67b3d4d4,0x64b1d5d5,0x61b7d6d6,0x62b5d7d7,0x73abd8d8,0x70a9d9d9,0x75afdada,0x76addbdb,0x7fa3dcdc,0x7ca1dddd,0x79a7dede,0x7aa5dfdf,
    0x3bdbe0e0,0x38d9e1e1,0x3ddfe2e2,0x3edde3e3,0x37d3e4e4,0x34d1e5e5,0x31d7e6e6,0x32d5e7e7,0x23cbe8e8,0x20c9e9e9,0x25cfeaea,0x26cdebeb,0x2fc3ecec,0x2cc1eded,0x29c7eeee,0x2ac5efef,
    0x0bfbf0f0,0x08f9f1f1,0x0dfff2f2,0x0efdf3f3,0x07f3f4f4,0x04f1f5f5,0x01f7f6f6,0x02f5f7f7,0x13ebf8f8,0x10e9f9f9,0x15effafa,0x16edfbfb,0x1fe3fcfc,0x1ce1fdfd,0x19e7fefe,0x1ae5ffff],
    [0x00000000,0x01030201,0x02060402,0x03050603,0x040c0804,0x050f0a05,0x060a0c06,0x07090e07,0x08181008,0x091b1209,0x0a1e140a,0x0b1d160b,0x0c14180c,0x0d171a0d,0x0e121c0e,0x0f111e0f,
    0x10302010,0x11332211,0x12362412,0x13352613,0x143c2814,0x153f2a15,0x163a2c16,0x17392e17,0x18283018,0x192b3219,0x1a2e341a,0x1b2d361b,0x1c24381c,0x1d273a1d,0x1e223c1e,0x1f213e1f,
    0x20604020,0x21634221,0x22664422,0x23654623,0x246c4824,0x256f4a25,0x266a4c26,0x27694e27,0x28785028,0x297b5229,0x2a7e542a,0x2b7d562b,0x2c74582c,0x2d775a2d,0x2e725c2e,0x2f715e2f,
    0x30506030,0x31536231,0x32566432,0x33556633,0x345c6834,0x355f6a35,0x365a6c36,0x37596e37,0x38487038,0x394b7239,0x3a4e743a,0x3b4d763b,0x3c44783c,0x3d477a3d,0x3e427c3e,0x3f417e3f,
    0x40c08040,0x41c38241,0x42c68442,0x43c58643,0x44cc8844,0x45cf8a45,0x46ca8c46,0x47c98e47,0x48d89048,0x49db9249,0x4ade944a,0x4bdd964b,0x4cd4984c,0x4dd79a4d,0x4ed29c4e,0x4fd19e4f,
    0x50f0a050,0x51f3a251,0x52f6a452,0x53f5a653,0x54fca854,0x55ffaa55,0x56faac56,0x57f9ae57,0x58e8b058,0x59ebb259,0x5aeeb45a,0x5bedb65b,0x5ce4b85c,0x5de7ba5d,0x5ee2bc5e,0x5fe1be5f,
    0x60a0c060,0x61a3c261,0x62a6c462,0x63a5c663,0x64acc864,0x65afca65,0x66aacc66,0x67a9ce67,0x68b8d068,0x69bbd269,0x6abed46a,0x6bbdd66b,0x6cb4d86c,0x6db7da6d,0x6eb2dc6e,0x6fb1de6f,
    0x7090e070,0x7193e271,0x7296e472,0x7395e673,0x749ce874,0x759fea75,0x769aec76,0x7799ee77,0x7888f078,0x798bf279,0x7a8ef47a,0x7b8df67b,0x7c84f87c,0x7d87fa7d,0x7e82fc7e,0x7f81fe7f,
    0x809b1b80,0x81981981,0x829d1f82,0x839e1d83,0x84971384,0x85941185,0x86911786,0x87921587,0x88830b88,0x89800989,0x8a850f8a,0x8b860d8b,0x8c8f038c,0x8d8c018d,0x8e89078e,0x8f8a058f,
    0x90ab3b90,0x91a83991,0x92ad3f92,0x93ae3d93,0x94a73394,0x95a43195,0x96a13796,0x97a23597,0x98b32b98,0x99b02999,0x9ab52f9a,0x9bb62d9b,0x9cbf239c,0x9dbc219d,0x9eb9279e,0x9fba259f,
    0xa0fb5ba0,0xa1f859a1,0xa2fd5fa2,0xa3fe5da3,0xa4f753a4,0xa5f451a5,0xa6f157a6,0xa7f255a7,0xa8e34ba8,0xa9e049a9,0xaae54faa,0xabe64dab,0xacef43ac,0xadec41ad,0xaee947ae,0xafea45af,
    0xb0cb7bb0,0xb1c879b1,0xb2cd7fb2,0xb3ce7db3,0xb4c773b4,0xb5c471b5,0xb6c177b6,0xb7c275b7,0xb8d36bb8,0xb9d069b9,0xbad56fba,0xbbd66dbb,0xbcdf63bc,0xbddc61bd,0xbed967be,0xbfda65bf,
    0xc05b9bc0,0xc15899c1,0xc25d9fc2,0xc35e9dc3,0xc45793c4,0xc55491c5,0xc65197c6,0xc75295c7,0xc8438bc8,0xc94089c9,0xca458fca,0xcb468dcb,0xcc4f83cc,0xcd4c81cd,0xce4987ce,0xcf4a85cf,
    0xd06bbbd0,0xd168b9d1,0xd26dbfd2,0xd36ebdd3,0xd467b3d4,0xd564b1d5,0xd661b7d6,0xd762b5d7,0xd873abd8,0xd970a9d9,0xda75afda,0xdb76addb,0xdc7fa3dc,0xdd7ca1dd,0xde79a7de,0xdf7aa5df,
    0xe03bdbe0,0xe138d9e1,0xe23ddfe2,0xe33edde3,0xe437d3e4,0xe534d1e5,0xe631d7e6,0xe732d5e7,0xe823cbe8,0xe920c9e9,0xea25cfea,0xeb26cdeb,0xec2fc3ec,0xed2cc1ed,0xee29c7ee,0xef2ac5ef,
    0xf00bfbf0,0xf108f9f1,0xf20dfff2,0xf30efdf3,0xf407f3f4,0xf504f1f5,0xf601f7f6,0xf702f5f7,0xf813ebf8,0xf910e9f9,0xfa15effa,0xfb16edfb,0xfc1fe3fc,0xfd1ce1fd,0xfe19e7fe,0xff1ae5ff],
    [0x00000000,0x01010302,0x02020604,0x03030506,0x04040c08,0x05050f0a,0x06060a0c,0x0707090e,0x08081810,0x09091b12,0x0a0a1e14,0x0b0b1d16,0x0c0c1418,0x0d0d171a,0x0e0e121c,0x0f0f111e,
    0x10103020,0x11113322,0x12123624,0x13133526,0x14143c28,0x15153f2a,0x16163a2c,0x1717392e,0x18182830,0x19192b32,0x1a1a2e34,0x1b1b2d36,0x1c1c2438,0x1d1d273a,0x1e1e223c,0x1f1f213e,
    0x20206040,0x21216342,0x22226644,0x23236546,0x24246c48,0x25256f4a,0x26266a4c,0x2727694e,0x28287850,0x29297b52,0x2a2a7e54,0x2b2b7d56,0x2c2c7458,0x2d2d775a,0x2e2e725c,0x2f2f715e,
    0x30305060,0x31315362,0x32325664,0x33335566,0x34345c68,0x35355f6a,0x36365a6c,0x3737596e,0x38384870,0x39394b72,0x3a3a4e74,0x3b3b4d76,0x3c3c4478,0x3d3d477a,0x3e3e427c,0x3f3f417e,
    0x4040c080,0x4141c382,0x4242c684,0x4343c586,0x4444cc88,0x4545cf8a,0x4646ca8c,0x4747c98e,0x4848d890,0x4949db92,0x4a4ade94,0x4b4bdd96,0x4c4cd498,0x4d4dd79a,0x4e4ed29c,0x4f4fd19e,
    0x5050f0a0,0x5151f3a2,0x5252f6a4,0x5353f5a6,0x5454fca8,0x5555ffaa,0x5656faac,0x5757f9ae,0x5858e8b0,0x5959ebb2,0x5a5aeeb4,0x5b5bedb6,0x5c5ce4b8,0x5d5de7ba,0x5e5ee2bc,0x5f5fe1be,
    0x6060a0c0,0x6161a3c2,0x6262a6c4,0x6363a5c6,0x6464acc8,0x6565afca,0x6666aacc,0x6767a9ce,0x6868b8d0,0x6969bbd2,0x6a6abed4,0x6b6bbdd6,0x6c6cb4d8,0x6d6db7da,0x6e6eb2dc,0x6f6fb1de,
    0x707090e0,0x717193e2,0x727296e4,0x737395e6,0x74749ce8,0x75759fea,0x76769aec,0x777799ee,0x787888f0,0x79798bf2,0x7a7a8ef4,0x7b7b8df6,0x7c7c84f8,0x7d7d87fa,0x7e7e82fc,0x7f7f81fe,
    0x80809b1b,0x81819819,0x82829d1f,0x83839e1d,0x84849713,0x85859411,0x86869117,0x87879215,0x8888830b,0x89898009,0x8a8a850f,0x8b8b860d,0x8c8c8f03,0x8d8d8c01,0x8e8e8907,0x8f8f8a05,
    0x9090ab3b,0x9191a839,0x9292ad3f,0x9393ae3d,0x9494a733,0x9595a431,0x9696a137,0x9797a235,0x9898b32b,0x9999b029,0x9a9ab52f,0x9b9bb62d,0x9c9cbf23,0x9d9dbc21,0x9e9eb927,0x9f9fba25,
    0xa0a0fb5b,0xa1a1f859,0xa2a2fd5f,0xa3a3fe5d,0xa4a4f753,0xa5a5f451,0xa6a6f157,0xa7a7f255,0xa8a8e34b,0xa9a9e049,0xaaaae54f,0xababe64d,0xacacef43,0xadadec41,0xaeaee947,0xafafea45,
    0xb0b0cb7b,0xb1b1c879,0xb2b2cd7f,0xb3b3ce7d,0xb4b4c773,0xb5b5c471,0xb6b6c177,0xb7b7c275,0xb8b8d36b,0xb9b9d069,0xbabad56f,0xbbbbd66d,0xbcbcdf63,0xbdbddc61,0xbebed967,0xbfbfda65,
    0xc0c05b9b,0xc1c15899,0xc2c25d9f,0xc3c35e9d,0xc4c45793,0xc5c55491,0xc6c65197,0xc7c75295,0xc8c8438b,0xc9c94089,0xcaca458f,0xcbcb468d,0xcccc4f83,0xcdcd4c81,0xcece4987,0xcfcf4a85,
    0xd0d06bbb,0xd1d168b9,0xd2d26dbf,0xd3d36ebd,0xd4d467b3,0xd5d564b1,0xd6d661b7,0xd7d762b5,0xd8d873ab,0xd9d970a9,0xdada75af,0xdbdb76ad,0xdcdc7fa3,0xdddd7ca1,0xdede79a7,0xdfdf7aa5,
    0xe0e03bdb,0xe1e138d9,0xe2e23ddf,0xe3e33edd,0xe4e437d3,0xe5e534d1,0xe6e631d7,0xe7e732d5,0xe8e823cb,0xe9e920c9,0xeaea25cf,0xebeb26cd,0xecec2fc3,0xeded2cc1,0xeeee29c7,0xefef2ac5,
    0xf0f00bfb,0xf1f108f9,0xf2f20dff,0xf3f30efd,0xf4f407f3,0xf5f504f1,0xf6f601f7,0xf7f702f5,0xf8f813eb,0xf9f910e9,0xfafa15ef,0xfbfb16ed,0xfcfc1fe3,0xfdfd1ce1,0xfefe19e7,0xffff1ae5]]