/* dhparam.c: initial DH parameters for stunnel */
#include "common.h"
#ifndef OPENSSL_NO_DH
#define DN_new DH_new
DH *get_dh2048(void)
{
    static unsigned char dhp_2048[] = {
        0x9E, 0xC9, 0x92, 0x71, 0x85, 0x44, 0x10, 0x42, 0xFD, 0x41,
        0x75, 0x2F, 0x1F, 0xC6, 0xA6, 0x05, 0x3A, 0x7B, 0x01, 0x8E,
        0x69, 0xCD, 0xEB, 0xB3, 0x39, 0x38, 0xD1, 0x0D, 0x2B, 0x3C,
        0xE3, 0x97, 0x8C, 0xD7, 0x58, 0x8C, 0x50, 0x12, 0x5A, 0xDA,
        0xBA, 0x52, 0x6E, 0x1E, 0x05, 0xB5, 0xE1, 0x70, 0xFD, 0x8C,
        0xF5, 0x3C, 0x8F, 0xE5, 0x8E, 0x1E, 0x5D, 0x98, 0x2D, 0xF8,
        0x92, 0x64, 0xF4, 0xA5, 0x5B, 0xC8, 0x25, 0x2B, 0x78, 0x68,
        0x9F, 0x37, 0xC9, 0xCA, 0x79, 0xAA, 0x22, 0xE7, 0x93, 0xDC,
        0x58, 0xAA, 0x65, 0xF5, 0x48, 0x2C, 0x42, 0x2F, 0x12, 0x6A,
        0x10, 0xB1, 0x19, 0x77, 0xE0, 0xB2, 0x8B, 0xB6, 0x8B, 0x03,
        0x62, 0x64, 0x6A, 0x29, 0x0F, 0x42, 0x08, 0xEB, 0x71, 0xCB,
        0xE3, 0x13, 0x09, 0x0A, 0x4D, 0x4F, 0xCC, 0x50, 0x2C, 0xAA,
        0xA4, 0x11, 0x1B, 0xC6, 0x14, 0xD4, 0x79, 0x4D, 0x2E, 0x83,
        0x06, 0x8F, 0x32, 0xFA, 0xE1, 0x0F, 0xF4, 0x1D, 0x57, 0x25,
        0x00, 0x13, 0x18, 0x61, 0x44, 0x83, 0xDC, 0x52, 0x57, 0x41,
        0xB9, 0xEA, 0x9B, 0xAD, 0xCA, 0xEE, 0x31, 0x0C, 0x25, 0x17,
        0xE2, 0xF8, 0xBE, 0xC3, 0xF3, 0xBF, 0x62, 0x30, 0x08, 0x5C,
        0x99, 0x25, 0x03, 0x74, 0xAF, 0xDB, 0x1E, 0xC5, 0x9E, 0x5A,
        0x40, 0x42, 0x15, 0x67, 0x0A, 0x0F, 0x6B, 0x34, 0x4D, 0x4B,
        0x07, 0x2B, 0xAB, 0xA4, 0x11, 0xE7, 0x3E, 0x35, 0x4A, 0x75,
        0xF6, 0x20, 0xA8, 0x9A, 0x5E, 0x29, 0xEC, 0x0F, 0x29, 0xA3,
        0x5D, 0x05, 0x58, 0xA9, 0xA7, 0x5E, 0x38, 0x7B, 0x95, 0x2D,
        0x6C, 0xC9, 0x69, 0x4E, 0xF8, 0x8A, 0x6A, 0x5E, 0x34, 0x91,
        0x47, 0xF6, 0xA8, 0x16, 0xD5, 0xEA, 0x5D, 0x3E, 0xEE, 0x04,
        0x20, 0xF7, 0x12, 0x52, 0x3C, 0x93, 0x05, 0x2E, 0x36, 0xE4,
        0x05, 0x10, 0x98, 0xE0, 0x5A, 0x9B
    };
    static unsigned char dhg_2048[] = {
        0x02
    };
    DH *dh = DH_new();
    BIGNUM *p, *g;

    if (dh == NULL)
        return NULL;
    p = BN_bin2bn(dhp_2048, sizeof(dhp_2048), NULL);
    g = BN_bin2bn(dhg_2048, sizeof(dhg_2048), NULL);
    if (p == NULL || g == NULL
            || !DH_set0_pqg(dh, p, NULL, g)) {
        DH_free(dh);
        BN_free(p);
        BN_free(g);
        return NULL;
    }
    return dh;
}
#endif /* OPENSSL_NO_DH */
/* built for stunnel 5.63 */
