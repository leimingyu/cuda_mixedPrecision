### comparison
fp 32
```c
htime = (a>=0.f);
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
        /*01a8*/                   FSETP.GE.AND P0, PT, R0, RZ, PT;  /* 0x5bb603800ff70007 */
        /*01b0*/                   SEL R0, RZ, 0x1, !P0;             /* 0x38a004000017ff00 */
        /*01b8*/                   I2F.F32.S32 R0, R0;               /* 0x5cb8000000072a00 */
                                                                     /* 0x007fbc00fe201fef */
        /*01c8*/                   MOV R0, R0;                       /* 0x5c98078000070000 */
        /*01d0*/                   CS2R RZ, SR_CLOCKLO;              /* 0x50c80000050700ff */
        /*01d8*/                   CS2R R8, SR_CLOCKLO;              /* 0x50c8000005070008 */
                                                                     /* 0x0067bc03fde01fef */
        /*01e8*/                   MOV R8, R8;                       /* 0x5c98078000870008 */
        /*01f0*/                   MOV R8, R8;                       /* 0x5c98078000870008 */
        /*01f8*/                   BAR.SYNC 0x0;                     /* 0xf0a81b8000070000 */
```



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
* (fp32 cmp) clocks: 128
* (fp16 cmp) clocks: 128


### floor
fp 32
```c
htime = floorf(a);
```

sass
```bash
        /*0148*/                   BAR.SYNC 0x0;                     /* 0xf0a81b8000070000 */
        /*0150*/                   CS2R RZ, SR_CLOCKLO;              /* 0x50c80000050700ff */
        /*0158*/                   CS2R R7, SR_CLOCKLO;              /* 0x50c8000005070007 */
                                                                     /* 0x00643c03fde01fef */
        /*0168*/                   MOV R7, R7;                       /* 0x5c98078000770007 */
        /*0170*/                   MOV R7, R7;                       /* 0x5c98078000770007 */
        /*0178*/                   F2F.F32.F32.FLOOR R0, R0;         /* 0x5ca8048000070a00 */
                                                                     /* 0x007fbc00fe201fef */
        /*0188*/                   MOV R0, R0;                       /* 0x5c98078000070000 */
        /*0190*/                   CS2R RZ, SR_CLOCKLO;              /* 0x50c80000050700ff */
        /*0198*/                   CS2R R8, SR_CLOCKLO;              /* 0x50c8000005070008 */
                                                                     /* 0x0067bc03fde01fef */
        /*01a8*/                   MOV R8, R8;                       /* 0x5c98078000870008 */
        /*01b0*/                   MOV R8, R8;                       /* 0x5c98078000870008 */
        /*01b8*/                   BAR.SYNC 0x0;                     /* 0xf0a81b8000070000 */
```

fp 16

```c
htime = hfloor(a_half);
```

sass
```bash
        /*0150*/                   BAR.SYNC 0x0;                     /* 0xf0a81b8000070000 */
        /*0158*/                   CS2R RZ, SR_CLOCKLO;              /* 0x50c80000050700ff */
                                                                     /* 0x007fbc03fde01fef */
        /*0168*/                   CS2R R7, SR_CLOCKLO;              /* 0x50c8000005070007 */
        /*0170*/                   MOV R7, R7;                       /* 0x5c98078000770007 */
        /*0178*/                   MOV R7, R7;                       /* 0x5c98078000770007 */
                                                                     /* 0x007fbc00fe20190f */
        /*0188*/                   F2F.F16.F16.FLOOR R0, R0;         /* 0x5ca8048000070500 */
        /*0190*/                   CS2R RZ, SR_CLOCKLO;              /* 0x50c80000050700ff */
        /*0198*/                   CS2R R8, SR_CLOCKLO;              /* 0x50c8000005070008 */
                                                                     /* 0x0067bc03fde01fef */
        /*01a8*/                   MOV R8, R8;                       /* 0x5c98078000870008 */
        /*01b0*/                   MOV R8, R8;                       /* 0x5c98078000870008 */
        /*01b8*/                   BAR.SYNC 0x0;                     /* 0xf0a81b8000070000 */
```

device 0 : Tesla P100-SXM2-16GB
* (fp32 floor) clocks: 128
* (fp16 floor) clocks: 128


### add
fp 32

Note: directly using v = t1 + t2 won't generate the desired sass code. Here, use inline ptx instead.
```c
        asm volatile (
                        "add.f32 %0, %1, %2;\n\t" : "=f"(v) : "f"(t1), "f"(t2)
                     );

```

```bash
        /*0228*/                   BAR.SYNC 0x0;                      /* 0xf0a81b8000070000 */
        /*0230*/                   CS2R RZ, SR_CLOCKLO;               /* 0x50c80000050700ff */
        /*0238*/                   CS2R R9, SR_CLOCKLO;               /* 0x50c8000005070009 */
                                                                      /* 0x007fbc03fde01fef */
        /*0248*/                   MOV R9, R9;                        /* 0x5c98078000970009 */
        /*0250*/                   MOV R9, R9;                        /* 0x5c98078000970009 */
        /*0258*/                   FADD R0, R0, R8;                   /* 0x5c58000000870000 */
                                                                      /* 0x007fbc00fe201fef */
        /*0268*/                   MOV R0, R0;                        /* 0x5c98078000070000 */
        /*0270*/                   CS2R RZ, SR_CLOCKLO;               /* 0x50c80000050700ff */
        /*0278*/                   CS2R R8, SR_CLOCKLO;               /* 0x50c8000005070008 */
                                                                      /* 0x0067bc03fde01fef */
        /*0288*/                   MOV R8, R8;                        /* 0x5c98078000870008 */
        /*0290*/                   MOV R8, R8;                        /* 0x5c98078000870008 */
        /*0298*/                   BAR.SYNC 0x0;                      /* 0xf0a81b8000070000 */
```


fp 16

```c
result = __hadd(a_half, b_half);
```

```bash
        /*0188*/                   BAR.SYNC 0x0;                     /* 0xf0a81b8000070000 */
        /*0190*/                   CS2R RZ, SR_CLOCKLO;              /* 0x50c80000050700ff */
        /*0198*/                   CS2R R9, SR_CLOCKLO;              /* 0x50c8000005070009 */
                                                                     /* 0x00643c03fde01fef */
        /*01a8*/                   MOV R9, R9;                       /* 0x5c98078000970009 */
        /*01b0*/                   MOV R9, R9;                       /* 0x5c98078000970009 */
        /*01b8*/                   HADD2 R0, R0.H0_H0, R0.H1_H1;     /* 0x5d11000030070000 */
                                                                     /* 0x007fbc03fde007f1 */
        /*01c8*/                   CS2R RZ, SR_CLOCKLO;              /* 0x50c80000050700ff */
        /*01d0*/                   CS2R R10, SR_CLOCKLO;             /* 0x50c800000507000a */
        /*01d8*/                   MOV R10, R10;                     /* 0x5c98078000a7000a */
                                                                     /* 0x007fbc033de01fef */
        /*01e8*/                   MOV R10, R10;                     /* 0x5c98078000a7000a */
        /*01f0*/                   BAR.SYNC 0x0;                     /* 0xf0a81b8000070000 */

```

device 0 : Tesla P100-SXM2-16GB
* (fp32 add) clocks: 128
* (fp16 add) clocks: 128


### multiplication
fp 32
```c        
asm volatile ("mul.f32 %0, %1, %2;\n\t"
               : "=f"(v) : "f"(t1) , "f"(t2));
```

```bash
        /*0228*/                   BAR.SYNC 0x0;                      /* 0xf0a81b8000070000 */
        /*0230*/                   CS2R RZ, SR_CLOCKLO;               /* 0x50c80000050700ff */
        /*0238*/                   CS2R R9, SR_CLOCKLO;               /* 0x50c8000005070009 */
                                                                      /* 0x007fbc03fde01fef */
        /*0248*/                   MOV R9, R9;                        /* 0x5c98078000970009 */
        /*0250*/                   MOV R9, R9;                        /* 0x5c98078000970009 */
        /*0258*/                   FMUL R0, R0, R8;                   /* 0x5c68000000870000 */
                                                                      /* 0x007fbc00fe201fef */
        /*0268*/                   MOV R0, R0;                        /* 0x5c98078000070000 */
        /*0270*/                   CS2R RZ, SR_CLOCKLO;               /* 0x50c80000050700ff */
        /*0278*/                   CS2R R8, SR_CLOCKLO;               /* 0x50c8000005070008 */
                                                                      /* 0x0067bc03fde01fef */
        /*0288*/                   MOV R8, R8;                        /* 0x5c98078000870008 */
        /*0290*/                   MOV R8, R8;                        /* 0x5c98078000870008 */
        /*0298*/                   BAR.SYNC 0x0;                      /* 0xf0a81b8000070000 */

```

fp 16
```c
result = __hmul(a_half, b_half);
```


```bash
        /*0188*/                   BAR.SYNC 0x0;                     /* 0xf0a81b8000070000 */
        /*0190*/                   CS2R RZ, SR_CLOCKLO;              /* 0x50c80000050700ff */
        /*0198*/                   CS2R R9, SR_CLOCKLO;              /* 0x50c8000005070009 */
                                                                     /* 0x00643c03fde01fef */
        /*01a8*/                   MOV R9, R9;                       /* 0x5c98078000970009 */
        /*01b0*/                   MOV R9, R9;                       /* 0x5c98078000970009 */
        /*01b8*/                   HMUL2 R0, R0.H0_H0, R0.H1_H1;     /* 0x5d09000030070000 */
                                                                     /* 0x007fbc03fde007f1 */
        /*01c8*/                   CS2R RZ, SR_CLOCKLO;              /* 0x50c80000050700ff */
        /*01d0*/                   CS2R R10, SR_CLOCKLO;             /* 0x50c800000507000a */
        /*01d8*/                   MOV R10, R10;                     /* 0x5c98078000a7000a */
                                                                     /* 0x007fbc033de01fef */
        /*01e8*/                   MOV R10, R10;                     /* 0x5c98078000a7000a */
        /*01f0*/                   BAR.SYNC 0x0;                     /* 0xf0a81b8000070000 */
```


device 0 : Tesla P100-SXM2-16GB
* (fp32 mul) clocks: 128
* (fp16 mul) clocks: 128
