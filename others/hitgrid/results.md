### comparison
fp 32
```c
htime = (a>0.f);
```

sass
```bash
        /*0170*/                   BAR.SYNC 0x0;                     /* 0xf0a81b8000070000 */
        /*0178*/                   CS2R RZ, SR_CLOCKLO;              /* 0x50c80000050700ff */
                                                                     /* 0x007fbc03fde01fef */
        /*0188*/                   CS2R R7, SR_CLOCKLO;              /* 0x50c8000005070007 */
        /*0190*/                   MOV R7, R7;                       /* 0x5c98078000770007 */
        /*0198*/                   MOV R7, R7;                       /* 0x5c98078000770007 */
                                                                     /* 0x00643c03fde01fef */
        /*01a8*/                   FSETP.GT.AND P0, PT, R0, RZ, PT;  /* 0x5bb403800ff70007 */
        /*01b0*/                   SEL R0, RZ, 0x1, !P0;             /* 0x38a004000017ff00 */
        /*01b8*/                   I2F.F32.S32 R0, R0;               /* 0x5cb8000000072a00 */
                                                                     /* 0x007fbc00fe201fef */
        /*01c8*/                   MOV R0, R0;                       /* 0x5c98078000070000 */
        /*01d0*/                   CS2R RZ, SR_CLOCKLO;              /* 0x50c80000050700ff */
        /*01d8*/                   CS2R R8, SR_CLOCKLO;              /* 0x50c8000005070008 */
```

device 0 : Tesla P100-SXM2-16GB
(fp32 cmp) clocks: 128

fp 16
```c
result = __float2half(float(__hge(a_half, zero_half)));
```

sass
```bash
        /*01a8*/                   CS2R RZ, SR_CLOCKLO;                           /* 0x50c80000050700ff */
        /*01b0*/                   CS2R R6, SR_CLOCKLO;                           /* 0x50c8000005070006 */
        /*01b8*/                   MOV R6, R6;                                    /* 0x5c98078000670006 */
                                                                                  /* 0x007fbc0321e01fef */
        /*01c8*/                   MOV R6, R6;                                    /* 0x5c98078000670006 */
        /*01d0*/                   HSETP2.GE.AND P0, PT, R7.H0_H0, R7.H1_H1, PT;  /* 0x5d2103b030770707 */
        /*01d8*/                   SEL R7, RZ, 0x1, !P0;                          /* 0x38a004000017ff07 */
                                                                                  /* 0x007fbc0321e0190f */
        /*01e8*/                   I2I.S32.S16 R8, RZ;                            /* 0x5ce000000ff73608 */
        /*01f0*/                   I2I.S32.S16 R7, R7;                            /* 0x5ce0000000773607 */
        /*01f8*/                   ISETP.NE.AND P0, PT, R7, R8, PT;               /* 0x5b6b038000870707 */
                                                                                  /* 0x00643c0321e01fef */
        /*0208*/                   SEL R7, RZ, 0x1, !P0;                          /* 0x38a004000017ff07 */
        /*0210*/                   I2F.F32.S16 R7, R7;                            /* 0x5cb8000000772607 */
        /*0218*/                   F2F.F16.F32 R7, R7;                            /* 0x5ca8000000770907 */
                                                                                  /* 0x007fbc03fde007f1 */
        /*0228*/                   CS2R RZ, SR_CLOCKLO;                           /* 0x50c80000050700ff */
        /*0230*/                   CS2R R8, SR_CLOCKLO;                           /* 0x50c8000005070008 */
```

device 0 : Tesla P100-SXM2-16GB
(fp16 cmp) clocks: 128
