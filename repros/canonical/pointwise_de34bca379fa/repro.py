"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s3_g60
Pattern hash: de34bca379fa
Shape hash: 54e6cd3d
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[30522, 768]", arg1_1: "f32[512, 768]", arg2_1: "f32[1024, 768]", arg3_1: "f32[1024, 768]", arg4_1: "f32[1024, 768]", arg5_1: "f32[1024, 768]", arg6_1: "f32[2, 768]", arg7_1: "f32[768]", arg8_1: "f32[768]", arg9_1: "f32[768, 768]", arg10_1: "f32[768]", arg11_1: "f32[768, 768]", arg12_1: "f32[768]", arg13_1: "f32[768, 768]", arg14_1: "f32[768]", arg15_1: "f32[768, 768]", arg16_1: "f32[768]", arg17_1: "f32[768]", arg18_1: "f32[768]", arg19_1: "f32[3072, 768]", arg20_1: "f32[3072]", arg21_1: "f32[768, 3072]", arg22_1: "f32[768]", arg23_1: "f32[768]", arg24_1: "f32[768]", arg25_1: "f32[768, 768]", arg26_1: "f32[768]", arg27_1: "f32[768, 768]", arg28_1: "f32[768]", arg29_1: "f32[768, 768]", arg30_1: "f32[768]", arg31_1: "f32[768, 768]", arg32_1: "f32[768]", arg33_1: "f32[768]", arg34_1: "f32[768]", arg35_1: "f32[3072, 768]", arg36_1: "f32[3072]", arg37_1: "f32[768, 3072]", arg38_1: "f32[768]", arg39_1: "f32[768]", arg40_1: "f32[768]", arg41_1: "f32[768, 768]", arg42_1: "f32[768]", arg43_1: "f32[768, 768]", arg44_1: "f32[768]", arg45_1: "f32[768, 768]", arg46_1: "f32[768]", arg47_1: "f32[768, 768]", arg48_1: "f32[768]", arg49_1: "f32[768]", arg50_1: "f32[768]", arg51_1: "f32[3072, 768]", arg52_1: "f32[3072]", arg53_1: "f32[768, 3072]", arg54_1: "f32[768]", arg55_1: "f32[768]", arg56_1: "f32[768]", arg57_1: "f32[768, 768]", arg58_1: "f32[768]", arg59_1: "f32[768, 768]", arg60_1: "f32[768]", arg61_1: "f32[768, 768]", arg62_1: "f32[768]", arg63_1: "f32[768, 768]", arg64_1: "f32[768]", arg65_1: "f32[768]", arg66_1: "f32[768]", arg67_1: "f32[3072, 768]", arg68_1: "f32[3072]", arg69_1: "f32[768, 3072]", arg70_1: "f32[768]", arg71_1: "f32[768]", arg72_1: "f32[768]", arg73_1: "f32[768, 768]", arg74_1: "f32[768]", arg75_1: "f32[768, 768]", arg76_1: "f32[768]", arg77_1: "f32[768, 768]", arg78_1: "f32[768]", arg79_1: "f32[768, 768]", arg80_1: "f32[768]", arg81_1: "f32[768]", arg82_1: "f32[768]", arg83_1: "f32[3072, 768]", arg84_1: "f32[3072]", arg85_1: "f32[768, 3072]", arg86_1: "f32[768]", arg87_1: "f32[768]", arg88_1: "f32[768]", arg89_1: "f32[768, 768]", arg90_1: "f32[768]", arg91_1: "f32[768, 768]", arg92_1: "f32[768]", arg93_1: "f32[768, 768]", arg94_1: "f32[768]", arg95_1: "f32[768, 768]", arg96_1: "f32[768]", arg97_1: "f32[768]", arg98_1: "f32[768]", arg99_1: "f32[3072, 768]", arg100_1: "f32[3072]", arg101_1: "f32[768, 3072]", arg102_1: "f32[768]", arg103_1: "f32[768]", arg104_1: "f32[768]", arg105_1: "f32[768, 768]", arg106_1: "f32[768]", arg107_1: "f32[768, 768]", arg108_1: "f32[768]", arg109_1: "f32[768, 768]", arg110_1: "f32[768]", arg111_1: "f32[768, 768]", arg112_1: "f32[768]", arg113_1: "f32[768]", arg114_1: "f32[768]", arg115_1: "f32[3072, 768]", arg116_1: "f32[3072]", arg117_1: "f32[768, 3072]", arg118_1: "f32[768]", arg119_1: "f32[768]", arg120_1: "f32[768]", arg121_1: "f32[768, 768]", arg122_1: "f32[768]", arg123_1: "f32[768, 768]", arg124_1: "f32[768]", arg125_1: "f32[768, 768]", arg126_1: "f32[768]", arg127_1: "f32[768, 768]", arg128_1: "f32[768]", arg129_1: "f32[768]", arg130_1: "f32[768]", arg131_1: "f32[3072, 768]", arg132_1: "f32[3072]", arg133_1: "f32[768, 3072]", arg134_1: "f32[768]", arg135_1: "f32[768]", arg136_1: "f32[768]", arg137_1: "f32[768, 768]", arg138_1: "f32[768]", arg139_1: "f32[768, 768]", arg140_1: "f32[768]", arg141_1: "f32[768, 768]", arg142_1: "f32[768]", arg143_1: "f32[768, 768]", arg144_1: "f32[768]", arg145_1: "f32[768]", arg146_1: "f32[768]", arg147_1: "f32[3072, 768]", arg148_1: "f32[3072]", arg149_1: "f32[768, 3072]", arg150_1: "f32[768]", arg151_1: "f32[768]", arg152_1: "f32[768]", arg153_1: "f32[768, 768]", arg154_1: "f32[768]", arg155_1: "f32[768, 768]", arg156_1: "f32[768]", arg157_1: "f32[768, 768]", arg158_1: "f32[768]", arg159_1: "f32[768, 768]", arg160_1: "f32[768]", arg161_1: "f32[768]", arg162_1: "f32[768]", arg163_1: "f32[3072, 768]", arg164_1: "f32[3072]", arg165_1: "f32[768, 3072]", arg166_1: "f32[768]", arg167_1: "f32[768]", arg168_1: "f32[768]", arg169_1: "f32[768, 768]", arg170_1: "f32[768]", arg171_1: "f32[768, 768]", arg172_1: "f32[768]", arg173_1: "f32[768, 768]", arg174_1: "f32[768]", arg175_1: "f32[768, 768]", arg176_1: "f32[768]", arg177_1: "f32[768]", arg178_1: "f32[768]", arg179_1: "f32[3072, 768]", arg180_1: "f32[3072]", arg181_1: "f32[768, 3072]", arg182_1: "f32[768]", arg183_1: "f32[768]", arg184_1: "f32[768]", arg185_1: "f32[768, 768]", arg186_1: "f32[768]", arg187_1: "f32[768, 768]", arg188_1: "f32[768]", arg189_1: "f32[768, 768]", arg190_1: "f32[768]", arg191_1: "f32[768, 768]", arg192_1: "f32[768]", arg193_1: "f32[768]", arg194_1: "f32[768]", arg195_1: "f32[3072, 768]", arg196_1: "f32[3072]", arg197_1: "f32[768, 3072]", arg198_1: "f32[768]", arg199_1: "f32[768]", arg200_1: "f32[768]", arg201_1: "f32[30522]", arg202_1: "f32[768, 768]", arg203_1: "f32[768]", arg204_1: "f32[768]", arg205_1: "f32[768]", getitem_412: "f32[30522, 768]", getitem_413: "f32[512, 768]", getitem_414: "f32[1024, 768]", getitem_415: "f32[1024, 768]", getitem_416: "f32[1024, 768]", getitem_417: "f32[1024, 768]", getitem_418: "f32[2, 768]", getitem_419: "f32[768]", getitem_420: "f32[768]", getitem_421: "f32[768, 768]", getitem_422: "f32[768]", getitem_423: "f32[768, 768]", getitem_424: "f32[768]", getitem_425: "f32[768, 768]", getitem_426: "f32[768]", getitem_427: "f32[768, 768]", getitem_428: "f32[768]", getitem_429: "f32[768]", getitem_430: "f32[768]", getitem_431: "f32[3072, 768]", getitem_432: "f32[3072]", getitem_433: "f32[768, 3072]", getitem_434: "f32[768]", getitem_435: "f32[768]", getitem_436: "f32[768]", getitem_437: "f32[768, 768]", getitem_438: "f32[768]", getitem_439: "f32[768, 768]", getitem_440: "f32[768]", getitem_441: "f32[768, 768]", getitem_442: "f32[768]", getitem_443: "f32[768, 768]", getitem_444: "f32[768]", getitem_445: "f32[768]", getitem_446: "f32[768]", getitem_447: "f32[3072, 768]", getitem_448: "f32[3072]", getitem_449: "f32[768, 3072]", getitem_450: "f32[768]", getitem_451: "f32[768]", getitem_452: "f32[768]", getitem_453: "f32[768, 768]", getitem_454: "f32[768]", getitem_455: "f32[768, 768]", getitem_456: "f32[768]", getitem_457: "f32[768, 768]", getitem_458: "f32[768]", getitem_459: "f32[768, 768]", getitem_460: "f32[768]", getitem_461: "f32[768]", getitem_462: "f32[768]", getitem_463: "f32[3072, 768]", getitem_464: "f32[3072]", getitem_465: "f32[768, 3072]", getitem_466: "f32[768]", getitem_467: "f32[768]", getitem_468: "f32[768]", getitem_469: "f32[768, 768]", getitem_470: "f32[768]", getitem_471: "f32[768, 768]", getitem_472: "f32[768]", getitem_473: "f32[768, 768]", getitem_474: "f32[768]", getitem_475: "f32[768, 768]", getitem_476: "f32[768]", getitem_477: "f32[768]", getitem_478: "f32[768]", getitem_479: "f32[3072, 768]", getitem_480: "f32[3072]", getitem_481: "f32[768, 3072]", getitem_482: "f32[768]", getitem_483: "f32[768]", getitem_484: "f32[768]", getitem_485: "f32[768, 768]", getitem_486: "f32[768]", getitem_487: "f32[768, 768]", getitem_488: "f32[768]", getitem_489: "f32[768, 768]", getitem_490: "f32[768]", getitem_491: "f32[768, 768]", getitem_492: "f32[768]", getitem_493: "f32[768]", getitem_494: "f32[768]", getitem_495: "f32[3072, 768]", getitem_496: "f32[3072]", getitem_497: "f32[768, 3072]", getitem_498: "f32[768]", getitem_499: "f32[768]", getitem_500: "f32[768]", getitem_501: "f32[768, 768]", getitem_502: "f32[768]", getitem_503: "f32[768, 768]", getitem_504: "f32[768]", getitem_505: "f32[768, 768]", getitem_506: "f32[768]", getitem_507: "f32[768, 768]", getitem_508: "f32[768]", getitem_509: "f32[768]", getitem_510: "f32[768]", getitem_511: "f32[3072, 768]", getitem_512: "f32[3072]", getitem_513: "f32[768, 3072]", getitem_514: "f32[768]", getitem_515: "f32[768]", getitem_516: "f32[768]", getitem_517: "f32[768, 768]", getitem_518: "f32[768]", getitem_519: "f32[768, 768]", getitem_520: "f32[768]", getitem_521: "f32[768, 768]", getitem_522: "f32[768]", getitem_523: "f32[768, 768]", getitem_524: "f32[768]", getitem_525: "f32[768]", getitem_526: "f32[768]", getitem_527: "f32[3072, 768]", getitem_528: "f32[3072]", getitem_529: "f32[768, 3072]", getitem_530: "f32[768]", getitem_531: "f32[768]", getitem_532: "f32[768]", getitem_533: "f32[768, 768]", getitem_534: "f32[768]", getitem_535: "f32[768, 768]", getitem_536: "f32[768]", getitem_537: "f32[768, 768]", getitem_538: "f32[768]", getitem_539: "f32[768, 768]", getitem_540: "f32[768]", getitem_541: "f32[768]", getitem_542: "f32[768]", getitem_543: "f32[3072, 768]", getitem_544: "f32[3072]", getitem_545: "f32[768, 3072]", getitem_546: "f32[768]", getitem_547: "f32[768]", getitem_548: "f32[768]", getitem_549: "f32[768, 768]", getitem_550: "f32[768]", getitem_551: "f32[768, 768]", getitem_552: "f32[768]", getitem_553: "f32[768, 768]", getitem_554: "f32[768]", getitem_555: "f32[768, 768]", getitem_556: "f32[768]", getitem_557: "f32[768]", getitem_558: "f32[768]", getitem_559: "f32[3072, 768]", getitem_560: "f32[3072]", getitem_561: "f32[768, 3072]", getitem_562: "f32[768]", getitem_563: "f32[768]", getitem_564: "f32[768]", getitem_565: "f32[768, 768]", getitem_566: "f32[768]", getitem_567: "f32[768, 768]", getitem_568: "f32[768]", getitem_569: "f32[768, 768]", getitem_570: "f32[768]", getitem_571: "f32[768, 768]", getitem_572: "f32[768]", getitem_573: "f32[768]", getitem_574: "f32[768]", getitem_575: "f32[3072, 768]", getitem_576: "f32[3072]", getitem_577: "f32[768, 3072]", getitem_578: "f32[768]", getitem_579: "f32[768]", getitem_580: "f32[768]", getitem_581: "f32[768, 768]", getitem_582: "f32[768]", getitem_583: "f32[768, 768]", getitem_584: "f32[768]", getitem_585: "f32[768, 768]", getitem_586: "f32[768]", getitem_587: "f32[768, 768]", getitem_588: "f32[768]", getitem_589: "f32[768]", getitem_590: "f32[768]", getitem_591: "f32[3072, 768]", getitem_592: "f32[3072]", getitem_593: "f32[768, 3072]", getitem_594: "f32[768]", getitem_595: "f32[768]", getitem_596: "f32[768]", getitem_597: "f32[768, 768]", getitem_598: "f32[768]", getitem_599: "f32[768, 768]", getitem_600: "f32[768]", getitem_601: "f32[768, 768]", getitem_602: "f32[768]", getitem_603: "f32[768, 768]", getitem_604: "f32[768]", getitem_605: "f32[768]", getitem_606: "f32[768]", getitem_607: "f32[3072, 768]", getitem_608: "f32[3072]", getitem_609: "f32[768, 3072]", getitem_610: "f32[768]", getitem_611: "f32[768]", getitem_612: "f32[768]", getitem_613: "f32[30522]", getitem_614: "f32[768, 768]", getitem_615: "f32[768]", getitem_616: "f32[768]", getitem_617: "f32[768]", getitem_3296: "f32[30522, 768]", getitem_3297: "f32[512, 768]", getitem_3298: "f32[1024, 768]", getitem_3299: "f32[1024, 768]", getitem_3300: "f32[1024, 768]", getitem_3301: "f32[1024, 768]", getitem_3302: "f32[2, 768]", getitem_3303: "f32[768]", getitem_3304: "f32[768]", getitem_3305: "f32[768, 768]", getitem_3306: "f32[768]", getitem_3307: "f32[768, 768]", getitem_3308: "f32[768]", getitem_3309: "f32[768, 768]", getitem_3310: "f32[768]", getitem_3311: "f32[768, 768]", getitem_3312: "f32[768]", getitem_3313: "f32[768]", getitem_3314: "f32[768]", getitem_3315: "f32[3072, 768]", getitem_3316: "f32[3072]", getitem_3317: "f32[768, 3072]", getitem_3318: "f32[768]", getitem_3319: "f32[768]", getitem_3320: "f32[768]", getitem_3321: "f32[768, 768]", getitem_3322: "f32[768]", getitem_3323: "f32[768, 768]", getitem_3324: "f32[768]", getitem_3325: "f32[768, 768]", getitem_3326: "f32[768]", getitem_3327: "f32[768, 768]", getitem_3328: "f32[768]", getitem_3329: "f32[768]", getitem_3330: "f32[768]", getitem_3331: "f32[3072, 768]", getitem_3332: "f32[3072]", getitem_3333: "f32[768, 3072]", getitem_3334: "f32[768]", getitem_3335: "f32[768]", getitem_3336: "f32[768]", getitem_3337: "f32[768, 768]", getitem_3338: "f32[768]", getitem_3339: "f32[768, 768]", getitem_3340: "f32[768]", getitem_3341: "f32[768, 768]", getitem_3342: "f32[768]", getitem_3343: "f32[768, 768]", getitem_3344: "f32[768]", getitem_3345: "f32[768]", getitem_3346: "f32[768]", getitem_3347: "f32[3072, 768]", getitem_3348: "f32[3072]", getitem_3349: "f32[768, 3072]", getitem_3350: "f32[768]", getitem_3351: "f32[768]", getitem_3352: "f32[768]", getitem_3353: "f32[768, 768]", getitem_3354: "f32[768]", getitem_3355: "f32[768, 768]", getitem_3356: "f32[768]", getitem_3357: "f32[768, 768]", getitem_3358: "f32[768]", getitem_3359: "f32[768, 768]", getitem_3360: "f32[768]", getitem_3361: "f32[768]", getitem_3362: "f32[768]", getitem_3363: "f32[3072, 768]", getitem_3364: "f32[3072]", getitem_3365: "f32[768, 3072]", getitem_3366: "f32[768]", getitem_3367: "f32[768]", getitem_3368: "f32[768]", getitem_3369: "f32[768, 768]", getitem_3370: "f32[768]", getitem_3371: "f32[768, 768]", getitem_3372: "f32[768]", getitem_3373: "f32[768, 768]", getitem_3374: "f32[768]", getitem_3375: "f32[768, 768]", getitem_3376: "f32[768]", getitem_3377: "f32[768]", getitem_3378: "f32[768]", getitem_3379: "f32[3072, 768]", getitem_3380: "f32[3072]", getitem_3381: "f32[768, 3072]", getitem_3382: "f32[768]", getitem_3383: "f32[768]", getitem_3384: "f32[768]", getitem_3385: "f32[768, 768]", getitem_3386: "f32[768]", getitem_3387: "f32[768, 768]", getitem_3388: "f32[768]", getitem_3389: "f32[768, 768]", getitem_3390: "f32[768]", getitem_3391: "f32[768, 768]", getitem_3392: "f32[768]", getitem_3393: "f32[768]", getitem_3394: "f32[768]", getitem_3395: "f32[3072, 768]", getitem_3396: "f32[3072]", getitem_3397: "f32[768, 3072]", getitem_3398: "f32[768]", getitem_3399: "f32[768]", getitem_3400: "f32[768]", getitem_3401: "f32[768, 768]", getitem_3402: "f32[768]", getitem_3403: "f32[768, 768]", getitem_3404: "f32[768]", getitem_3405: "f32[768, 768]", getitem_3406: "f32[768]", getitem_3407: "f32[768, 768]", getitem_3408: "f32[768]", getitem_3409: "f32[768]", getitem_3410: "f32[768]", getitem_3411: "f32[3072, 768]", getitem_3412: "f32[3072]", getitem_3413: "f32[768, 3072]", getitem_3414: "f32[768]", getitem_3415: "f32[768]", getitem_3416: "f32[768]", getitem_3417: "f32[768, 768]", getitem_3418: "f32[768]", getitem_3419: "f32[768, 768]", getitem_3420: "f32[768]", getitem_3421: "f32[768, 768]", getitem_3422: "f32[768]", getitem_3423: "f32[768, 768]", getitem_3424: "f32[768]", getitem_3425: "f32[768]", getitem_3426: "f32[768]", getitem_3427: "f32[3072, 768]", getitem_3428: "f32[3072]", getitem_3429: "f32[768, 3072]", getitem_3430: "f32[768]", getitem_3431: "f32[768]", getitem_3432: "f32[768]", getitem_3433: "f32[768, 768]", getitem_3434: "f32[768]", getitem_3435: "f32[768, 768]", getitem_3436: "f32[768]", getitem_3437: "f32[768, 768]", getitem_3438: "f32[768]", getitem_3439: "f32[768, 768]", getitem_3440: "f32[768]", getitem_3441: "f32[768]", getitem_3442: "f32[768]", getitem_3443: "f32[3072, 768]", getitem_3444: "f32[3072]", getitem_3445: "f32[768, 3072]", getitem_3446: "f32[768]", getitem_3447: "f32[768]", getitem_3448: "f32[768]", getitem_3449: "f32[768, 768]", getitem_3450: "f32[768]", getitem_3451: "f32[768, 768]", getitem_3452: "f32[768]", getitem_3453: "f32[768, 768]", getitem_3454: "f32[768]", getitem_3455: "f32[768, 768]", getitem_3456: "f32[768]", getitem_3457: "f32[768]", getitem_3458: "f32[768]", getitem_3459: "f32[3072, 768]", getitem_3460: "f32[3072]", getitem_3461: "f32[768, 3072]", getitem_3462: "f32[768]", getitem_3463: "f32[768]", getitem_3464: "f32[768]", getitem_3465: "f32[768, 768]", getitem_3466: "f32[768]", getitem_3467: "f32[768, 768]", getitem_3468: "f32[768]", getitem_3469: "f32[768, 768]", getitem_3470: "f32[768]", getitem_3471: "f32[768, 768]", getitem_3472: "f32[768]", getitem_3473: "f32[768]", getitem_3474: "f32[768]", getitem_3475: "f32[3072, 768]", getitem_3476: "f32[3072]", getitem_3477: "f32[768, 3072]", getitem_3478: "f32[768]", getitem_3479: "f32[768]", getitem_3480: "f32[768]", getitem_3481: "f32[768, 768]", getitem_3482: "f32[768]", getitem_3483: "f32[768, 768]", getitem_3484: "f32[768]", getitem_3485: "f32[768, 768]", getitem_3486: "f32[768]", getitem_3487: "f32[768, 768]", getitem_3488: "f32[768]", getitem_3489: "f32[768]", getitem_3490: "f32[768]", getitem_3491: "f32[3072, 768]", getitem_3492: "f32[3072]", getitem_3493: "f32[768, 3072]", getitem_3494: "f32[768]", getitem_3495: "f32[768]", getitem_3496: "f32[768]", getitem_3497: "f32[30522]", getitem_3498: "f32[768, 768]", getitem_3499: "f32[768]", getitem_3500: "f32[768]", getitem_3501: "f32[768]"):
        # No stacktrace found for following nodes
        _foreach_addcdiv_scalar = torch.ops.aten._foreach_addcdiv.Scalar([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1], [getitem_412, getitem_413, getitem_414, getitem_415, getitem_416, getitem_417, getitem_418, getitem_419, getitem_420, getitem_421, getitem_422, getitem_423, getitem_424, getitem_425, getitem_426, getitem_427, getitem_428, getitem_429, getitem_430, getitem_431, getitem_432, getitem_433, getitem_434, getitem_435, getitem_436, getitem_437, getitem_438, getitem_439, getitem_440, getitem_441, getitem_442, getitem_443, getitem_444, getitem_445, getitem_446, getitem_447, getitem_448, getitem_449, getitem_450, getitem_451, getitem_452, getitem_453, getitem_454, getitem_455, getitem_456, getitem_457, getitem_458, getitem_459, getitem_460, getitem_461, getitem_462, getitem_463, getitem_464, getitem_465, getitem_466, getitem_467, getitem_468, getitem_469, getitem_470, getitem_471, getitem_472, getitem_473, getitem_474, getitem_475, getitem_476, getitem_477, getitem_478, getitem_479, getitem_480, getitem_481, getitem_482, getitem_483, getitem_484, getitem_485, getitem_486, getitem_487, getitem_488, getitem_489, getitem_490, getitem_491, getitem_492, getitem_493, getitem_494, getitem_495, getitem_496, getitem_497, getitem_498, getitem_499, getitem_500, getitem_501, getitem_502, getitem_503, getitem_504, getitem_505, getitem_506, getitem_507, getitem_508, getitem_509, getitem_510, getitem_511, getitem_512, getitem_513, getitem_514, getitem_515, getitem_516, getitem_517, getitem_518, getitem_519, getitem_520, getitem_521, getitem_522, getitem_523, getitem_524, getitem_525, getitem_526, getitem_527, getitem_528, getitem_529, getitem_530, getitem_531, getitem_532, getitem_533, getitem_534, getitem_535, getitem_536, getitem_537, getitem_538, getitem_539, getitem_540, getitem_541, getitem_542, getitem_543, getitem_544, getitem_545, getitem_546, getitem_547, getitem_548, getitem_549, getitem_550, getitem_551, getitem_552, getitem_553, getitem_554, getitem_555, getitem_556, getitem_557, getitem_558, getitem_559, getitem_560, getitem_561, getitem_562, getitem_563, getitem_564, getitem_565, getitem_566, getitem_567, getitem_568, getitem_569, getitem_570, getitem_571, getitem_572, getitem_573, getitem_574, getitem_575, getitem_576, getitem_577, getitem_578, getitem_579, getitem_580, getitem_581, getitem_582, getitem_583, getitem_584, getitem_585, getitem_586, getitem_587, getitem_588, getitem_589, getitem_590, getitem_591, getitem_592, getitem_593, getitem_594, getitem_595, getitem_596, getitem_597, getitem_598, getitem_599, getitem_600, getitem_601, getitem_602, getitem_603, getitem_604, getitem_605, getitem_606, getitem_607, getitem_608, getitem_609, getitem_610, getitem_611, getitem_612, getitem_613, getitem_614, getitem_615, getitem_616, getitem_617], [getitem_3296, getitem_3297, getitem_3298, getitem_3299, getitem_3300, getitem_3301, getitem_3302, getitem_3303, getitem_3304, getitem_3305, getitem_3306, getitem_3307, getitem_3308, getitem_3309, getitem_3310, getitem_3311, getitem_3312, getitem_3313, getitem_3314, getitem_3315, getitem_3316, getitem_3317, getitem_3318, getitem_3319, getitem_3320, getitem_3321, getitem_3322, getitem_3323, getitem_3324, getitem_3325, getitem_3326, getitem_3327, getitem_3328, getitem_3329, getitem_3330, getitem_3331, getitem_3332, getitem_3333, getitem_3334, getitem_3335, getitem_3336, getitem_3337, getitem_3338, getitem_3339, getitem_3340, getitem_3341, getitem_3342, getitem_3343, getitem_3344, getitem_3345, getitem_3346, getitem_3347, getitem_3348, getitem_3349, getitem_3350, getitem_3351, getitem_3352, getitem_3353, getitem_3354, getitem_3355, getitem_3356, getitem_3357, getitem_3358, getitem_3359, getitem_3360, getitem_3361, getitem_3362, getitem_3363, getitem_3364, getitem_3365, getitem_3366, getitem_3367, getitem_3368, getitem_3369, getitem_3370, getitem_3371, getitem_3372, getitem_3373, getitem_3374, getitem_3375, getitem_3376, getitem_3377, getitem_3378, getitem_3379, getitem_3380, getitem_3381, getitem_3382, getitem_3383, getitem_3384, getitem_3385, getitem_3386, getitem_3387, getitem_3388, getitem_3389, getitem_3390, getitem_3391, getitem_3392, getitem_3393, getitem_3394, getitem_3395, getitem_3396, getitem_3397, getitem_3398, getitem_3399, getitem_3400, getitem_3401, getitem_3402, getitem_3403, getitem_3404, getitem_3405, getitem_3406, getitem_3407, getitem_3408, getitem_3409, getitem_3410, getitem_3411, getitem_3412, getitem_3413, getitem_3414, getitem_3415, getitem_3416, getitem_3417, getitem_3418, getitem_3419, getitem_3420, getitem_3421, getitem_3422, getitem_3423, getitem_3424, getitem_3425, getitem_3426, getitem_3427, getitem_3428, getitem_3429, getitem_3430, getitem_3431, getitem_3432, getitem_3433, getitem_3434, getitem_3435, getitem_3436, getitem_3437, getitem_3438, getitem_3439, getitem_3440, getitem_3441, getitem_3442, getitem_3443, getitem_3444, getitem_3445, getitem_3446, getitem_3447, getitem_3448, getitem_3449, getitem_3450, getitem_3451, getitem_3452, getitem_3453, getitem_3454, getitem_3455, getitem_3456, getitem_3457, getitem_3458, getitem_3459, getitem_3460, getitem_3461, getitem_3462, getitem_3463, getitem_3464, getitem_3465, getitem_3466, getitem_3467, getitem_3468, getitem_3469, getitem_3470, getitem_3471, getitem_3472, getitem_3473, getitem_3474, getitem_3475, getitem_3476, getitem_3477, getitem_3478, getitem_3479, getitem_3480, getitem_3481, getitem_3482, getitem_3483, getitem_3484, getitem_3485, getitem_3486, getitem_3487, getitem_3488, getitem_3489, getitem_3490, getitem_3491, getitem_3492, getitem_3493, getitem_3494, getitem_3495, getitem_3496, getitem_3497, getitem_3498, getitem_3499, getitem_3500, getitem_3501]);  arg0_1 = arg1_1 = arg2_1 = arg3_1 = arg4_1 = arg5_1 = arg6_1 = arg7_1 = arg8_1 = arg9_1 = arg10_1 = arg11_1 = arg12_1 = arg13_1 = arg14_1 = arg15_1 = arg16_1 = arg17_1 = arg18_1 = arg19_1 = arg20_1 = arg21_1 = arg22_1 = arg23_1 = arg24_1 = arg25_1 = arg26_1 = arg27_1 = arg28_1 = arg29_1 = arg30_1 = arg31_1 = arg32_1 = arg33_1 = arg34_1 = arg35_1 = arg36_1 = arg37_1 = arg38_1 = arg39_1 = arg40_1 = arg41_1 = arg42_1 = arg43_1 = arg44_1 = arg45_1 = arg46_1 = arg47_1 = arg48_1 = arg49_1 = arg50_1 = arg51_1 = arg52_1 = arg53_1 = arg54_1 = arg55_1 = arg56_1 = arg57_1 = arg58_1 = arg59_1 = arg60_1 = arg61_1 = arg62_1 = arg63_1 = arg64_1 = arg65_1 = arg66_1 = arg67_1 = arg68_1 = arg69_1 = arg70_1 = arg71_1 = arg72_1 = arg73_1 = arg74_1 = arg75_1 = arg76_1 = arg77_1 = arg78_1 = arg79_1 = arg80_1 = arg81_1 = arg82_1 = arg83_1 = arg84_1 = arg85_1 = arg86_1 = arg87_1 = arg88_1 = arg89_1 = arg90_1 = arg91_1 = arg92_1 = arg93_1 = arg94_1 = arg95_1 = arg96_1 = arg97_1 = arg98_1 = arg99_1 = arg100_1 = arg101_1 = arg102_1 = arg103_1 = arg104_1 = arg105_1 = arg106_1 = arg107_1 = arg108_1 = arg109_1 = arg110_1 = arg111_1 = arg112_1 = arg113_1 = arg114_1 = arg115_1 = arg116_1 = arg117_1 = arg118_1 = arg119_1 = arg120_1 = arg121_1 = arg122_1 = arg123_1 = arg124_1 = arg125_1 = arg126_1 = arg127_1 = arg128_1 = arg129_1 = arg130_1 = arg131_1 = arg132_1 = arg133_1 = arg134_1 = arg135_1 = arg136_1 = arg137_1 = arg138_1 = arg139_1 = arg140_1 = arg141_1 = arg142_1 = arg143_1 = arg144_1 = arg145_1 = arg146_1 = arg147_1 = arg148_1 = arg149_1 = arg150_1 = arg151_1 = arg152_1 = arg153_1 = arg154_1 = arg155_1 = arg156_1 = arg157_1 = arg158_1 = arg159_1 = arg160_1 = arg161_1 = arg162_1 = arg163_1 = arg164_1 = arg165_1 = arg166_1 = arg167_1 = arg168_1 = arg169_1 = arg170_1 = arg171_1 = arg172_1 = arg173_1 = arg174_1 = arg175_1 = arg176_1 = arg177_1 = arg178_1 = arg179_1 = arg180_1 = arg181_1 = arg182_1 = arg183_1 = arg184_1 = arg185_1 = arg186_1 = arg187_1 = arg188_1 = arg189_1 = arg190_1 = arg191_1 = arg192_1 = arg193_1 = arg194_1 = arg195_1 = arg196_1 = arg197_1 = arg198_1 = arg199_1 = arg200_1 = arg201_1 = arg202_1 = arg203_1 = arg204_1 = arg205_1 = getitem_412 = getitem_413 = getitem_414 = getitem_415 = getitem_416 = getitem_417 = getitem_418 = getitem_419 = getitem_420 = getitem_421 = getitem_422 = getitem_423 = getitem_424 = getitem_425 = getitem_426 = getitem_427 = getitem_428 = getitem_429 = getitem_430 = getitem_431 = getitem_432 = getitem_433 = getitem_434 = getitem_435 = getitem_436 = getitem_437 = getitem_438 = getitem_439 = getitem_440 = getitem_441 = getitem_442 = getitem_443 = getitem_444 = getitem_445 = getitem_446 = getitem_447 = getitem_448 = getitem_449 = getitem_450 = getitem_451 = getitem_452 = getitem_453 = getitem_454 = getitem_455 = getitem_456 = getitem_457 = getitem_458 = getitem_459 = getitem_460 = getitem_461 = getitem_462 = getitem_463 = getitem_464 = getitem_465 = getitem_466 = getitem_467 = getitem_468 = getitem_469 = getitem_470 = getitem_471 = getitem_472 = getitem_473 = getitem_474 = getitem_475 = getitem_476 = getitem_477 = getitem_478 = getitem_479 = getitem_480 = getitem_481 = getitem_482 = getitem_483 = getitem_484 = getitem_485 = getitem_486 = getitem_487 = getitem_488 = getitem_489 = getitem_490 = getitem_491 = getitem_492 = getitem_493 = getitem_494 = getitem_495 = getitem_496 = getitem_497 = getitem_498 = getitem_499 = getitem_500 = getitem_501 = getitem_502 = getitem_503 = getitem_504 = getitem_505 = getitem_506 = getitem_507 = getitem_508 = getitem_509 = getitem_510 = getitem_511 = getitem_512 = getitem_513 = getitem_514 = getitem_515 = getitem_516 = getitem_517 = getitem_518 = getitem_519 = getitem_520 = getitem_521 = getitem_522 = getitem_523 = getitem_524 = getitem_525 = getitem_526 = getitem_527 = getitem_528 = getitem_529 = getitem_530 = getitem_531 = getitem_532 = getitem_533 = getitem_534 = getitem_535 = getitem_536 = getitem_537 = getitem_538 = getitem_539 = getitem_540 = getitem_541 = getitem_542 = getitem_543 = getitem_544 = getitem_545 = getitem_546 = getitem_547 = getitem_548 = getitem_549 = getitem_550 = getitem_551 = getitem_552 = getitem_553 = getitem_554 = getitem_555 = getitem_556 = getitem_557 = getitem_558 = getitem_559 = getitem_560 = getitem_561 = getitem_562 = getitem_563 = getitem_564 = getitem_565 = getitem_566 = getitem_567 = getitem_568 = getitem_569 = getitem_570 = getitem_571 = getitem_572 = getitem_573 = getitem_574 = getitem_575 = getitem_576 = getitem_577 = getitem_578 = getitem_579 = getitem_580 = getitem_581 = getitem_582 = getitem_583 = getitem_584 = getitem_585 = getitem_586 = getitem_587 = getitem_588 = getitem_589 = getitem_590 = getitem_591 = getitem_592 = getitem_593 = getitem_594 = getitem_595 = getitem_596 = getitem_597 = getitem_598 = getitem_599 = getitem_600 = getitem_601 = getitem_602 = getitem_603 = getitem_604 = getitem_605 = getitem_606 = getitem_607 = getitem_608 = getitem_609 = getitem_610 = getitem_611 = getitem_612 = getitem_613 = getitem_614 = getitem_615 = getitem_616 = getitem_617 = getitem_3296 = getitem_3297 = getitem_3298 = getitem_3299 = getitem_3300 = getitem_3301 = getitem_3302 = getitem_3303 = getitem_3304 = getitem_3305 = getitem_3306 = getitem_3307 = getitem_3308 = getitem_3309 = getitem_3310 = getitem_3311 = getitem_3312 = getitem_3313 = getitem_3314 = getitem_3315 = getitem_3316 = getitem_3317 = getitem_3318 = getitem_3319 = getitem_3320 = getitem_3321 = getitem_3322 = getitem_3323 = getitem_3324 = getitem_3325 = getitem_3326 = getitem_3327 = getitem_3328 = getitem_3329 = getitem_3330 = getitem_3331 = getitem_3332 = getitem_3333 = getitem_3334 = getitem_3335 = getitem_3336 = getitem_3337 = getitem_3338 = getitem_3339 = getitem_3340 = getitem_3341 = getitem_3342 = getitem_3343 = getitem_3344 = getitem_3345 = getitem_3346 = getitem_3347 = getitem_3348 = getitem_3349 = getitem_3350 = getitem_3351 = getitem_3352 = getitem_3353 = getitem_3354 = getitem_3355 = getitem_3356 = getitem_3357 = getitem_3358 = getitem_3359 = getitem_3360 = getitem_3361 = getitem_3362 = getitem_3363 = getitem_3364 = getitem_3365 = getitem_3366 = getitem_3367 = getitem_3368 = getitem_3369 = getitem_3370 = getitem_3371 = getitem_3372 = getitem_3373 = getitem_3374 = getitem_3375 = getitem_3376 = getitem_3377 = getitem_3378 = getitem_3379 = getitem_3380 = getitem_3381 = getitem_3382 = getitem_3383 = getitem_3384 = getitem_3385 = getitem_3386 = getitem_3387 = getitem_3388 = getitem_3389 = getitem_3390 = getitem_3391 = getitem_3392 = getitem_3393 = getitem_3394 = getitem_3395 = getitem_3396 = getitem_3397 = getitem_3398 = getitem_3399 = getitem_3400 = getitem_3401 = getitem_3402 = getitem_3403 = getitem_3404 = getitem_3405 = getitem_3406 = getitem_3407 = getitem_3408 = getitem_3409 = getitem_3410 = getitem_3411 = getitem_3412 = getitem_3413 = getitem_3414 = getitem_3415 = getitem_3416 = getitem_3417 = getitem_3418 = getitem_3419 = getitem_3420 = getitem_3421 = getitem_3422 = getitem_3423 = getitem_3424 = getitem_3425 = getitem_3426 = getitem_3427 = getitem_3428 = getitem_3429 = getitem_3430 = getitem_3431 = getitem_3432 = getitem_3433 = getitem_3434 = getitem_3435 = getitem_3436 = getitem_3437 = getitem_3438 = getitem_3439 = getitem_3440 = getitem_3441 = getitem_3442 = getitem_3443 = getitem_3444 = getitem_3445 = getitem_3446 = getitem_3447 = getitem_3448 = getitem_3449 = getitem_3450 = getitem_3451 = getitem_3452 = getitem_3453 = getitem_3454 = getitem_3455 = getitem_3456 = getitem_3457 = getitem_3458 = getitem_3459 = getitem_3460 = getitem_3461 = getitem_3462 = getitem_3463 = getitem_3464 = getitem_3465 = getitem_3466 = getitem_3467 = getitem_3468 = getitem_3469 = getitem_3470 = getitem_3471 = getitem_3472 = getitem_3473 = getitem_3474 = getitem_3475 = getitem_3476 = getitem_3477 = getitem_3478 = getitem_3479 = getitem_3480 = getitem_3481 = getitem_3482 = getitem_3483 = getitem_3484 = getitem_3485 = getitem_3486 = getitem_3487 = getitem_3488 = getitem_3489 = getitem_3490 = getitem_3491 = getitem_3492 = getitem_3493 = getitem_3494 = getitem_3495 = getitem_3496 = getitem_3497 = getitem_3498 = getitem_3499 = getitem_3500 = getitem_3501 = None
        getitem: "f32[30522, 768]" = _foreach_addcdiv_scalar[0]
        getitem_3502: "f32[512, 768]" = _foreach_addcdiv_scalar[1]
        getitem_3503: "f32[1024, 768]" = _foreach_addcdiv_scalar[2]
        getitem_3504: "f32[1024, 768]" = _foreach_addcdiv_scalar[3]
        getitem_3505: "f32[1024, 768]" = _foreach_addcdiv_scalar[4]
        getitem_3506: "f32[1024, 768]" = _foreach_addcdiv_scalar[5]
        getitem_3507: "f32[2, 768]" = _foreach_addcdiv_scalar[6]
        getitem_3508: "f32[768]" = _foreach_addcdiv_scalar[7]
        getitem_3509: "f32[768]" = _foreach_addcdiv_scalar[8]
        getitem_3510: "f32[768, 768]" = _foreach_addcdiv_scalar[9]
        getitem_3511: "f32[768]" = _foreach_addcdiv_scalar[10]
        getitem_3512: "f32[768, 768]" = _foreach_addcdiv_scalar[11]
        getitem_3513: "f32[768]" = _foreach_addcdiv_scalar[12]
        getitem_3514: "f32[768, 768]" = _foreach_addcdiv_scalar[13]
        getitem_3515: "f32[768]" = _foreach_addcdiv_scalar[14]
        getitem_3516: "f32[768, 768]" = _foreach_addcdiv_scalar[15]
        getitem_3517: "f32[768]" = _foreach_addcdiv_scalar[16]
        getitem_3518: "f32[768]" = _foreach_addcdiv_scalar[17]
        getitem_3519: "f32[768]" = _foreach_addcdiv_scalar[18]
        getitem_3520: "f32[3072, 768]" = _foreach_addcdiv_scalar[19]
        getitem_3521: "f32[3072]" = _foreach_addcdiv_scalar[20]
        getitem_3522: "f32[768, 3072]" = _foreach_addcdiv_scalar[21]
        getitem_3523: "f32[768]" = _foreach_addcdiv_scalar[22]
        getitem_3524: "f32[768]" = _foreach_addcdiv_scalar[23]
        getitem_3525: "f32[768]" = _foreach_addcdiv_scalar[24]
        getitem_3526: "f32[768, 768]" = _foreach_addcdiv_scalar[25]
        getitem_3527: "f32[768]" = _foreach_addcdiv_scalar[26]
        getitem_3528: "f32[768, 768]" = _foreach_addcdiv_scalar[27]
        getitem_3529: "f32[768]" = _foreach_addcdiv_scalar[28]
        getitem_3530: "f32[768, 768]" = _foreach_addcdiv_scalar[29]
        getitem_3531: "f32[768]" = _foreach_addcdiv_scalar[30]
        getitem_3532: "f32[768, 768]" = _foreach_addcdiv_scalar[31]
        getitem_3533: "f32[768]" = _foreach_addcdiv_scalar[32]
        getitem_3534: "f32[768]" = _foreach_addcdiv_scalar[33]
        getitem_3535: "f32[768]" = _foreach_addcdiv_scalar[34]
        getitem_3536: "f32[3072, 768]" = _foreach_addcdiv_scalar[35]
        getitem_3537: "f32[3072]" = _foreach_addcdiv_scalar[36]
        getitem_3538: "f32[768, 3072]" = _foreach_addcdiv_scalar[37]
        getitem_3539: "f32[768]" = _foreach_addcdiv_scalar[38]
        getitem_3540: "f32[768]" = _foreach_addcdiv_scalar[39]
        getitem_3541: "f32[768]" = _foreach_addcdiv_scalar[40]
        getitem_3542: "f32[768, 768]" = _foreach_addcdiv_scalar[41]
        getitem_3543: "f32[768]" = _foreach_addcdiv_scalar[42]
        getitem_3544: "f32[768, 768]" = _foreach_addcdiv_scalar[43]
        getitem_3545: "f32[768]" = _foreach_addcdiv_scalar[44]
        getitem_3546: "f32[768, 768]" = _foreach_addcdiv_scalar[45]
        getitem_3547: "f32[768]" = _foreach_addcdiv_scalar[46]
        getitem_3548: "f32[768, 768]" = _foreach_addcdiv_scalar[47]
        getitem_3549: "f32[768]" = _foreach_addcdiv_scalar[48]
        getitem_3550: "f32[768]" = _foreach_addcdiv_scalar[49]
        getitem_3551: "f32[768]" = _foreach_addcdiv_scalar[50]
        getitem_3552: "f32[3072, 768]" = _foreach_addcdiv_scalar[51]
        getitem_3553: "f32[3072]" = _foreach_addcdiv_scalar[52]
        getitem_3554: "f32[768, 3072]" = _foreach_addcdiv_scalar[53]
        getitem_3555: "f32[768]" = _foreach_addcdiv_scalar[54]
        getitem_3556: "f32[768]" = _foreach_addcdiv_scalar[55]
        getitem_3557: "f32[768]" = _foreach_addcdiv_scalar[56]
        getitem_3558: "f32[768, 768]" = _foreach_addcdiv_scalar[57]
        getitem_3559: "f32[768]" = _foreach_addcdiv_scalar[58]
        getitem_3560: "f32[768, 768]" = _foreach_addcdiv_scalar[59]
        getitem_3561: "f32[768]" = _foreach_addcdiv_scalar[60]
        getitem_3562: "f32[768, 768]" = _foreach_addcdiv_scalar[61]
        getitem_3563: "f32[768]" = _foreach_addcdiv_scalar[62]
        getitem_3564: "f32[768, 768]" = _foreach_addcdiv_scalar[63]
        getitem_3565: "f32[768]" = _foreach_addcdiv_scalar[64]
        getitem_3566: "f32[768]" = _foreach_addcdiv_scalar[65]
        getitem_3567: "f32[768]" = _foreach_addcdiv_scalar[66]
        getitem_3568: "f32[3072, 768]" = _foreach_addcdiv_scalar[67]
        getitem_3569: "f32[3072]" = _foreach_addcdiv_scalar[68]
        getitem_3570: "f32[768, 3072]" = _foreach_addcdiv_scalar[69]
        getitem_3571: "f32[768]" = _foreach_addcdiv_scalar[70]
        getitem_3572: "f32[768]" = _foreach_addcdiv_scalar[71]
        getitem_3573: "f32[768]" = _foreach_addcdiv_scalar[72]
        getitem_3574: "f32[768, 768]" = _foreach_addcdiv_scalar[73]
        getitem_3575: "f32[768]" = _foreach_addcdiv_scalar[74]
        getitem_3576: "f32[768, 768]" = _foreach_addcdiv_scalar[75]
        getitem_3577: "f32[768]" = _foreach_addcdiv_scalar[76]
        getitem_3578: "f32[768, 768]" = _foreach_addcdiv_scalar[77]
        getitem_3579: "f32[768]" = _foreach_addcdiv_scalar[78]
        getitem_3580: "f32[768, 768]" = _foreach_addcdiv_scalar[79]
        getitem_3581: "f32[768]" = _foreach_addcdiv_scalar[80]
        getitem_3582: "f32[768]" = _foreach_addcdiv_scalar[81]
        getitem_3583: "f32[768]" = _foreach_addcdiv_scalar[82]
        getitem_3584: "f32[3072, 768]" = _foreach_addcdiv_scalar[83]
        getitem_3585: "f32[3072]" = _foreach_addcdiv_scalar[84]
        getitem_3586: "f32[768, 3072]" = _foreach_addcdiv_scalar[85]
        getitem_3587: "f32[768]" = _foreach_addcdiv_scalar[86]
        getitem_3588: "f32[768]" = _foreach_addcdiv_scalar[87]
        getitem_3589: "f32[768]" = _foreach_addcdiv_scalar[88]
        getitem_3590: "f32[768, 768]" = _foreach_addcdiv_scalar[89]
        getitem_3591: "f32[768]" = _foreach_addcdiv_scalar[90]
        getitem_3592: "f32[768, 768]" = _foreach_addcdiv_scalar[91]
        getitem_3593: "f32[768]" = _foreach_addcdiv_scalar[92]
        getitem_3594: "f32[768, 768]" = _foreach_addcdiv_scalar[93]
        getitem_3595: "f32[768]" = _foreach_addcdiv_scalar[94]
        getitem_3596: "f32[768, 768]" = _foreach_addcdiv_scalar[95]
        getitem_3597: "f32[768]" = _foreach_addcdiv_scalar[96]
        getitem_3598: "f32[768]" = _foreach_addcdiv_scalar[97]
        getitem_3599: "f32[768]" = _foreach_addcdiv_scalar[98]
        getitem_3600: "f32[3072, 768]" = _foreach_addcdiv_scalar[99]
        getitem_3601: "f32[3072]" = _foreach_addcdiv_scalar[100]
        getitem_3602: "f32[768, 3072]" = _foreach_addcdiv_scalar[101]
        getitem_3603: "f32[768]" = _foreach_addcdiv_scalar[102]
        getitem_3604: "f32[768]" = _foreach_addcdiv_scalar[103]
        getitem_3605: "f32[768]" = _foreach_addcdiv_scalar[104]
        getitem_3606: "f32[768, 768]" = _foreach_addcdiv_scalar[105]
        getitem_3607: "f32[768]" = _foreach_addcdiv_scalar[106]
        getitem_3608: "f32[768, 768]" = _foreach_addcdiv_scalar[107]
        getitem_3609: "f32[768]" = _foreach_addcdiv_scalar[108]
        getitem_3610: "f32[768, 768]" = _foreach_addcdiv_scalar[109]
        getitem_3611: "f32[768]" = _foreach_addcdiv_scalar[110]
        getitem_3612: "f32[768, 768]" = _foreach_addcdiv_scalar[111]
        getitem_3613: "f32[768]" = _foreach_addcdiv_scalar[112]
        getitem_3614: "f32[768]" = _foreach_addcdiv_scalar[113]
        getitem_3615: "f32[768]" = _foreach_addcdiv_scalar[114]
        getitem_3616: "f32[3072, 768]" = _foreach_addcdiv_scalar[115]
        getitem_3617: "f32[3072]" = _foreach_addcdiv_scalar[116]
        getitem_3618: "f32[768, 3072]" = _foreach_addcdiv_scalar[117]
        getitem_3619: "f32[768]" = _foreach_addcdiv_scalar[118]
        getitem_3620: "f32[768]" = _foreach_addcdiv_scalar[119]
        getitem_3621: "f32[768]" = _foreach_addcdiv_scalar[120]
        getitem_3622: "f32[768, 768]" = _foreach_addcdiv_scalar[121]
        getitem_3623: "f32[768]" = _foreach_addcdiv_scalar[122]
        getitem_3624: "f32[768, 768]" = _foreach_addcdiv_scalar[123]
        getitem_3625: "f32[768]" = _foreach_addcdiv_scalar[124]
        getitem_3626: "f32[768, 768]" = _foreach_addcdiv_scalar[125]
        getitem_3627: "f32[768]" = _foreach_addcdiv_scalar[126]
        getitem_3628: "f32[768, 768]" = _foreach_addcdiv_scalar[127]
        getitem_3629: "f32[768]" = _foreach_addcdiv_scalar[128]
        getitem_3630: "f32[768]" = _foreach_addcdiv_scalar[129]
        getitem_3631: "f32[768]" = _foreach_addcdiv_scalar[130]
        getitem_3632: "f32[3072, 768]" = _foreach_addcdiv_scalar[131]
        getitem_3633: "f32[3072]" = _foreach_addcdiv_scalar[132]
        getitem_3634: "f32[768, 3072]" = _foreach_addcdiv_scalar[133]
        getitem_3635: "f32[768]" = _foreach_addcdiv_scalar[134]
        getitem_3636: "f32[768]" = _foreach_addcdiv_scalar[135]
        getitem_3637: "f32[768]" = _foreach_addcdiv_scalar[136]
        getitem_3638: "f32[768, 768]" = _foreach_addcdiv_scalar[137]
        getitem_3639: "f32[768]" = _foreach_addcdiv_scalar[138]
        getitem_3640: "f32[768, 768]" = _foreach_addcdiv_scalar[139]
        getitem_3641: "f32[768]" = _foreach_addcdiv_scalar[140]
        getitem_3642: "f32[768, 768]" = _foreach_addcdiv_scalar[141]
        getitem_3643: "f32[768]" = _foreach_addcdiv_scalar[142]
        getitem_3644: "f32[768, 768]" = _foreach_addcdiv_scalar[143]
        getitem_3645: "f32[768]" = _foreach_addcdiv_scalar[144]
        getitem_3646: "f32[768]" = _foreach_addcdiv_scalar[145]
        getitem_3647: "f32[768]" = _foreach_addcdiv_scalar[146]
        getitem_3648: "f32[3072, 768]" = _foreach_addcdiv_scalar[147]
        getitem_3649: "f32[3072]" = _foreach_addcdiv_scalar[148]
        getitem_3650: "f32[768, 3072]" = _foreach_addcdiv_scalar[149]
        getitem_3651: "f32[768]" = _foreach_addcdiv_scalar[150]
        getitem_3652: "f32[768]" = _foreach_addcdiv_scalar[151]
        getitem_3653: "f32[768]" = _foreach_addcdiv_scalar[152]
        getitem_3654: "f32[768, 768]" = _foreach_addcdiv_scalar[153]
        getitem_3655: "f32[768]" = _foreach_addcdiv_scalar[154]
        getitem_3656: "f32[768, 768]" = _foreach_addcdiv_scalar[155]
        getitem_3657: "f32[768]" = _foreach_addcdiv_scalar[156]
        getitem_3658: "f32[768, 768]" = _foreach_addcdiv_scalar[157]
        getitem_3659: "f32[768]" = _foreach_addcdiv_scalar[158]
        getitem_3660: "f32[768, 768]" = _foreach_addcdiv_scalar[159]
        getitem_3661: "f32[768]" = _foreach_addcdiv_scalar[160]
        getitem_3662: "f32[768]" = _foreach_addcdiv_scalar[161]
        getitem_3663: "f32[768]" = _foreach_addcdiv_scalar[162]
        getitem_3664: "f32[3072, 768]" = _foreach_addcdiv_scalar[163]
        getitem_3665: "f32[3072]" = _foreach_addcdiv_scalar[164]
        getitem_3666: "f32[768, 3072]" = _foreach_addcdiv_scalar[165]
        getitem_3667: "f32[768]" = _foreach_addcdiv_scalar[166]
        getitem_3668: "f32[768]" = _foreach_addcdiv_scalar[167]
        getitem_3669: "f32[768]" = _foreach_addcdiv_scalar[168]
        getitem_3670: "f32[768, 768]" = _foreach_addcdiv_scalar[169]
        getitem_3671: "f32[768]" = _foreach_addcdiv_scalar[170]
        getitem_3672: "f32[768, 768]" = _foreach_addcdiv_scalar[171]
        getitem_3673: "f32[768]" = _foreach_addcdiv_scalar[172]
        getitem_3674: "f32[768, 768]" = _foreach_addcdiv_scalar[173]
        getitem_3675: "f32[768]" = _foreach_addcdiv_scalar[174]
        getitem_3676: "f32[768, 768]" = _foreach_addcdiv_scalar[175]
        getitem_3677: "f32[768]" = _foreach_addcdiv_scalar[176]
        getitem_3678: "f32[768]" = _foreach_addcdiv_scalar[177]
        getitem_3679: "f32[768]" = _foreach_addcdiv_scalar[178]
        getitem_3680: "f32[3072, 768]" = _foreach_addcdiv_scalar[179]
        getitem_3681: "f32[3072]" = _foreach_addcdiv_scalar[180]
        getitem_3682: "f32[768, 3072]" = _foreach_addcdiv_scalar[181]
        getitem_3683: "f32[768]" = _foreach_addcdiv_scalar[182]
        getitem_3684: "f32[768]" = _foreach_addcdiv_scalar[183]
        getitem_3685: "f32[768]" = _foreach_addcdiv_scalar[184]
        getitem_3686: "f32[768, 768]" = _foreach_addcdiv_scalar[185]
        getitem_3687: "f32[768]" = _foreach_addcdiv_scalar[186]
        getitem_3688: "f32[768, 768]" = _foreach_addcdiv_scalar[187]
        getitem_3689: "f32[768]" = _foreach_addcdiv_scalar[188]
        getitem_3690: "f32[768, 768]" = _foreach_addcdiv_scalar[189]
        getitem_3691: "f32[768]" = _foreach_addcdiv_scalar[190]
        getitem_3692: "f32[768, 768]" = _foreach_addcdiv_scalar[191]
        getitem_3693: "f32[768]" = _foreach_addcdiv_scalar[192]
        getitem_3694: "f32[768]" = _foreach_addcdiv_scalar[193]
        getitem_3695: "f32[768]" = _foreach_addcdiv_scalar[194]
        getitem_3696: "f32[3072, 768]" = _foreach_addcdiv_scalar[195]
        getitem_3697: "f32[3072]" = _foreach_addcdiv_scalar[196]
        getitem_3698: "f32[768, 3072]" = _foreach_addcdiv_scalar[197]
        getitem_3699: "f32[768]" = _foreach_addcdiv_scalar[198]
        getitem_3700: "f32[768]" = _foreach_addcdiv_scalar[199]
        getitem_3701: "f32[768]" = _foreach_addcdiv_scalar[200]
        getitem_3702: "f32[30522]" = _foreach_addcdiv_scalar[201]
        getitem_3703: "f32[768, 768]" = _foreach_addcdiv_scalar[202]
        getitem_3704: "f32[768]" = _foreach_addcdiv_scalar[203]
        getitem_3705: "f32[768]" = _foreach_addcdiv_scalar[204]
        getitem_3706: "f32[768]" = _foreach_addcdiv_scalar[205];  _foreach_addcdiv_scalar = None
        return (getitem, getitem_3502, getitem_3503, getitem_3504, getitem_3505, getitem_3506, getitem_3507, getitem_3508, getitem_3509, getitem_3510, getitem_3511, getitem_3512, getitem_3513, getitem_3514, getitem_3515, getitem_3516, getitem_3517, getitem_3518, getitem_3519, getitem_3520, getitem_3521, getitem_3522, getitem_3523, getitem_3524, getitem_3525, getitem_3526, getitem_3527, getitem_3528, getitem_3529, getitem_3530, getitem_3531, getitem_3532, getitem_3533, getitem_3534, getitem_3535, getitem_3536, getitem_3537, getitem_3538, getitem_3539, getitem_3540, getitem_3541, getitem_3542, getitem_3543, getitem_3544, getitem_3545, getitem_3546, getitem_3547, getitem_3548, getitem_3549, getitem_3550, getitem_3551, getitem_3552, getitem_3553, getitem_3554, getitem_3555, getitem_3556, getitem_3557, getitem_3558, getitem_3559, getitem_3560, getitem_3561, getitem_3562, getitem_3563, getitem_3564, getitem_3565, getitem_3566, getitem_3567, getitem_3568, getitem_3569, getitem_3570, getitem_3571, getitem_3572, getitem_3573, getitem_3574, getitem_3575, getitem_3576, getitem_3577, getitem_3578, getitem_3579, getitem_3580, getitem_3581, getitem_3582, getitem_3583, getitem_3584, getitem_3585, getitem_3586, getitem_3587, getitem_3588, getitem_3589, getitem_3590, getitem_3591, getitem_3592, getitem_3593, getitem_3594, getitem_3595, getitem_3596, getitem_3597, getitem_3598, getitem_3599, getitem_3600, getitem_3601, getitem_3602, getitem_3603, getitem_3604, getitem_3605, getitem_3606, getitem_3607, getitem_3608, getitem_3609, getitem_3610, getitem_3611, getitem_3612, getitem_3613, getitem_3614, getitem_3615, getitem_3616, getitem_3617, getitem_3618, getitem_3619, getitem_3620, getitem_3621, getitem_3622, getitem_3623, getitem_3624, getitem_3625, getitem_3626, getitem_3627, getitem_3628, getitem_3629, getitem_3630, getitem_3631, getitem_3632, getitem_3633, getitem_3634, getitem_3635, getitem_3636, getitem_3637, getitem_3638, getitem_3639, getitem_3640, getitem_3641, getitem_3642, getitem_3643, getitem_3644, getitem_3645, getitem_3646, getitem_3647, getitem_3648, getitem_3649, getitem_3650, getitem_3651, getitem_3652, getitem_3653, getitem_3654, getitem_3655, getitem_3656, getitem_3657, getitem_3658, getitem_3659, getitem_3660, getitem_3661, getitem_3662, getitem_3663, getitem_3664, getitem_3665, getitem_3666, getitem_3667, getitem_3668, getitem_3669, getitem_3670, getitem_3671, getitem_3672, getitem_3673, getitem_3674, getitem_3675, getitem_3676, getitem_3677, getitem_3678, getitem_3679, getitem_3680, getitem_3681, getitem_3682, getitem_3683, getitem_3684, getitem_3685, getitem_3686, getitem_3687, getitem_3688, getitem_3689, getitem_3690, getitem_3691, getitem_3692, getitem_3693, getitem_3694, getitem_3695, getitem_3696, getitem_3697, getitem_3698, getitem_3699, getitem_3700, getitem_3701, getitem_3702, getitem_3703, getitem_3704, getitem_3705, getitem_3706)


def _default_make_inputs():
    return [
    torch.randn([30522, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([30522], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([30522, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([30522], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([30522, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([30522], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
