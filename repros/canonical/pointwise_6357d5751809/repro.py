"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s9_g77
Pattern hash: 6357d5751809
Shape hash: 5db07020
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg304_1: "f32[768]", arg302_1: "f32[50, 768]", arg606_1: "f32[768, 512]", arg607_1: "f32[768, 3, 32, 32]", arg608_1: "f32[768]", arg609_1: "f32[768]", arg610_1: "f32[2304, 768]", arg611_1: "f32[2304]", arg612_1: "f32[768, 768]", arg613_1: "f32[768]", arg614_1: "f32[3072, 768]", arg615_1: "f32[3072]", arg616_1: "f32[768, 3072]", arg617_1: "f32[768]", arg618_1: "f32[768]", arg619_1: "f32[768]", arg620_1: "f32[768]", arg621_1: "f32[768]", arg622_1: "f32[2304, 768]", arg623_1: "f32[2304]", arg624_1: "f32[768, 768]", arg625_1: "f32[768]", arg626_1: "f32[3072, 768]", arg627_1: "f32[3072]", arg628_1: "f32[768, 3072]", arg629_1: "f32[768]", arg630_1: "f32[768]", arg631_1: "f32[768]", arg632_1: "f32[768]", arg633_1: "f32[768]", arg634_1: "f32[2304, 768]", arg635_1: "f32[2304]", arg636_1: "f32[768, 768]", arg637_1: "f32[768]", arg638_1: "f32[3072, 768]", arg639_1: "f32[3072]", arg640_1: "f32[768, 3072]", arg641_1: "f32[768]", arg642_1: "f32[768]", arg643_1: "f32[768]", arg644_1: "f32[768]", arg645_1: "f32[768]", arg646_1: "f32[2304, 768]", arg647_1: "f32[2304]", arg648_1: "f32[768, 768]", arg649_1: "f32[768]", arg650_1: "f32[3072, 768]", arg651_1: "f32[3072]", arg652_1: "f32[768, 3072]", arg653_1: "f32[768]", arg654_1: "f32[768]", arg655_1: "f32[768]", arg656_1: "f32[768]", arg657_1: "f32[768]", arg658_1: "f32[2304, 768]", arg659_1: "f32[2304]", arg660_1: "f32[768, 768]", arg661_1: "f32[768]", arg662_1: "f32[3072, 768]", arg663_1: "f32[3072]", arg664_1: "f32[768, 3072]", arg665_1: "f32[768]", arg666_1: "f32[768]", arg667_1: "f32[768]", arg668_1: "f32[768]", arg669_1: "f32[768]", arg670_1: "f32[2304, 768]", arg671_1: "f32[2304]", arg672_1: "f32[768, 768]", arg673_1: "f32[768]", arg674_1: "f32[3072, 768]", arg675_1: "f32[3072]", arg676_1: "f32[768, 3072]", arg677_1: "f32[768]", arg678_1: "f32[768]", arg679_1: "f32[768]", arg680_1: "f32[768]", arg681_1: "f32[768]", arg682_1: "f32[2304, 768]", arg683_1: "f32[2304]", arg684_1: "f32[768, 768]", arg685_1: "f32[768]", arg686_1: "f32[3072, 768]", arg687_1: "f32[3072]", arg688_1: "f32[768, 3072]", arg689_1: "f32[768]", arg690_1: "f32[768]", arg691_1: "f32[768]", arg692_1: "f32[768]", arg693_1: "f32[768]", arg694_1: "f32[2304, 768]", arg695_1: "f32[2304]", arg696_1: "f32[768, 768]", arg697_1: "f32[768]", arg698_1: "f32[3072, 768]", arg699_1: "f32[3072]", arg700_1: "f32[768, 3072]", arg701_1: "f32[768]", arg702_1: "f32[768]", arg703_1: "f32[768]", arg704_1: "f32[768]", arg705_1: "f32[768]", arg706_1: "f32[2304, 768]", arg707_1: "f32[2304]", arg708_1: "f32[768, 768]", arg709_1: "f32[768]", arg710_1: "f32[3072, 768]", arg711_1: "f32[3072]", arg712_1: "f32[768, 3072]", arg713_1: "f32[768]", arg714_1: "f32[768]", arg715_1: "f32[768]", arg716_1: "f32[768]", arg717_1: "f32[768]", arg718_1: "f32[2304, 768]", arg719_1: "f32[2304]", arg720_1: "f32[768, 768]", arg721_1: "f32[768]", arg722_1: "f32[3072, 768]", arg723_1: "f32[3072]", arg724_1: "f32[768, 3072]", arg725_1: "f32[768]", arg726_1: "f32[768]", arg727_1: "f32[768]", arg728_1: "f32[768]", arg729_1: "f32[768]", arg730_1: "f32[2304, 768]", arg731_1: "f32[2304]", arg732_1: "f32[768, 768]", arg733_1: "f32[768]", arg734_1: "f32[3072, 768]", arg735_1: "f32[3072]", arg736_1: "f32[768, 3072]", arg737_1: "f32[768]", arg738_1: "f32[768]", arg739_1: "f32[768]", arg740_1: "f32[768]", arg741_1: "f32[768]", arg742_1: "f32[2304, 768]", arg743_1: "f32[2304]", arg744_1: "f32[768, 768]", arg745_1: "f32[768]", arg746_1: "f32[3072, 768]", arg747_1: "f32[3072]", arg748_1: "f32[768, 3072]", arg749_1: "f32[768]", arg750_1: "f32[768]", arg751_1: "f32[768]", arg752_1: "f32[768]", arg753_1: "f32[768]", arg754_1: "f32[768]", arg755_1: "f32[768]", arg756_1: "f32[77, 512]", arg757_1: "f32[49408, 512]", arg758_1: "f32[1536, 512]", arg759_1: "f32[1536]", arg760_1: "f32[512, 512]", arg761_1: "f32[512]", arg762_1: "f32[2048, 512]", arg763_1: "f32[2048]", arg764_1: "f32[512, 2048]", arg765_1: "f32[512]", arg766_1: "f32[512]", arg767_1: "f32[512]", arg768_1: "f32[512]", arg769_1: "f32[512]", arg770_1: "f32[1536, 512]", arg771_1: "f32[1536]", arg772_1: "f32[512, 512]", arg773_1: "f32[512]", arg774_1: "f32[2048, 512]", arg775_1: "f32[2048]", arg776_1: "f32[512, 2048]", arg777_1: "f32[512]", arg778_1: "f32[512]", arg779_1: "f32[512]", arg780_1: "f32[512]", arg781_1: "f32[512]", arg782_1: "f32[1536, 512]", arg783_1: "f32[1536]", arg784_1: "f32[512, 512]", arg785_1: "f32[512]", arg786_1: "f32[2048, 512]", arg787_1: "f32[2048]", arg788_1: "f32[512, 2048]", arg789_1: "f32[512]", arg790_1: "f32[512]", arg791_1: "f32[512]", arg792_1: "f32[512]", arg793_1: "f32[512]", arg794_1: "f32[1536, 512]", arg795_1: "f32[1536]", arg796_1: "f32[512, 512]", arg797_1: "f32[512]", arg798_1: "f32[2048, 512]", arg799_1: "f32[2048]", arg800_1: "f32[512, 2048]", arg801_1: "f32[512]", arg802_1: "f32[512]", arg803_1: "f32[512]", arg804_1: "f32[512]", arg805_1: "f32[512]", arg806_1: "f32[1536, 512]", arg807_1: "f32[1536]", arg808_1: "f32[512, 512]", arg809_1: "f32[512]", arg810_1: "f32[2048, 512]", arg811_1: "f32[2048]", arg812_1: "f32[512, 2048]", arg813_1: "f32[512]", arg814_1: "f32[512]", arg815_1: "f32[512]", arg816_1: "f32[512]", arg817_1: "f32[512]", arg818_1: "f32[1536, 512]", arg819_1: "f32[1536]", arg820_1: "f32[512, 512]", arg821_1: "f32[512]", arg822_1: "f32[2048, 512]", arg823_1: "f32[2048]", arg824_1: "f32[512, 2048]", arg825_1: "f32[512]", arg826_1: "f32[512]", arg827_1: "f32[512]", arg828_1: "f32[512]", arg829_1: "f32[512]", arg830_1: "f32[1536, 512]", arg831_1: "f32[1536]", arg832_1: "f32[512, 512]", arg833_1: "f32[512]", arg834_1: "f32[2048, 512]", arg835_1: "f32[2048]", arg836_1: "f32[512, 2048]", arg837_1: "f32[512]", arg838_1: "f32[512]", arg839_1: "f32[512]", arg840_1: "f32[512]", arg841_1: "f32[512]", arg842_1: "f32[1536, 512]", arg843_1: "f32[1536]", arg844_1: "f32[512, 512]", arg845_1: "f32[512]", arg846_1: "f32[2048, 512]", arg847_1: "f32[2048]", arg848_1: "f32[512, 2048]", arg849_1: "f32[512]", arg850_1: "f32[512]", arg851_1: "f32[512]", arg852_1: "f32[512]", arg853_1: "f32[512]", arg854_1: "f32[1536, 512]", arg855_1: "f32[1536]", arg856_1: "f32[512, 512]", arg857_1: "f32[512]", arg858_1: "f32[2048, 512]", arg859_1: "f32[2048]", arg860_1: "f32[512, 2048]", arg861_1: "f32[512]", arg862_1: "f32[512]", arg863_1: "f32[512]", arg864_1: "f32[512]", arg865_1: "f32[512]", arg866_1: "f32[1536, 512]", arg867_1: "f32[1536]", arg868_1: "f32[512, 512]", arg869_1: "f32[512]", arg870_1: "f32[2048, 512]", arg871_1: "f32[2048]", arg872_1: "f32[512, 2048]", arg873_1: "f32[512]", arg874_1: "f32[512]", arg875_1: "f32[512]", arg876_1: "f32[512]", arg877_1: "f32[512]", arg878_1: "f32[1536, 512]", arg879_1: "f32[1536]", arg880_1: "f32[512, 512]", arg881_1: "f32[512]", arg882_1: "f32[2048, 512]", arg883_1: "f32[2048]", arg884_1: "f32[512, 2048]", arg885_1: "f32[512]", arg886_1: "f32[512]", arg887_1: "f32[512]", arg888_1: "f32[512]", arg889_1: "f32[512]", arg890_1: "f32[1536, 512]", arg891_1: "f32[1536]", arg892_1: "f32[512, 512]", arg893_1: "f32[512]", arg894_1: "f32[2048, 512]", arg895_1: "f32[2048]", arg896_1: "f32[512, 2048]", arg897_1: "f32[512]", arg898_1: "f32[512]", arg899_1: "f32[512]", arg900_1: "f32[512]", arg901_1: "f32[512]", arg902_1: "f32[512]", arg903_1: "f32[512]", arg904_1: "f32[512, 512]", getitem_301: "f32[768]", getitem_302: "f32[50, 768]", getitem_303: "f32[768, 512]", getitem_304: "f32[768, 3, 32, 32]", getitem_305: "f32[768]", getitem_306: "f32[768]", getitem_307: "f32[2304, 768]", getitem_308: "f32[2304]", getitem_309: "f32[768, 768]", getitem_310: "f32[768]", getitem_311: "f32[3072, 768]", getitem_312: "f32[3072]", getitem_313: "f32[768, 3072]", getitem_314: "f32[768]", getitem_315: "f32[768]", getitem_316: "f32[768]", getitem_317: "f32[768]", getitem_318: "f32[768]", getitem_319: "f32[2304, 768]", getitem_320: "f32[2304]", getitem_321: "f32[768, 768]", getitem_322: "f32[768]", getitem_323: "f32[3072, 768]", getitem_324: "f32[3072]", getitem_325: "f32[768, 3072]", getitem_326: "f32[768]", getitem_327: "f32[768]", getitem_328: "f32[768]", getitem_329: "f32[768]", getitem_330: "f32[768]", getitem_331: "f32[2304, 768]", getitem_332: "f32[2304]", getitem_333: "f32[768, 768]", getitem_334: "f32[768]", getitem_335: "f32[3072, 768]", getitem_336: "f32[3072]", getitem_337: "f32[768, 3072]", getitem_338: "f32[768]", getitem_339: "f32[768]", getitem_340: "f32[768]", getitem_341: "f32[768]", getitem_342: "f32[768]", getitem_343: "f32[2304, 768]", getitem_344: "f32[2304]", getitem_345: "f32[768, 768]", getitem_346: "f32[768]", getitem_347: "f32[3072, 768]", getitem_348: "f32[3072]", getitem_349: "f32[768, 3072]", getitem_350: "f32[768]", getitem_351: "f32[768]", getitem_352: "f32[768]", getitem_353: "f32[768]", getitem_354: "f32[768]", getitem_355: "f32[2304, 768]", getitem_356: "f32[2304]", getitem_357: "f32[768, 768]", getitem_358: "f32[768]", getitem_359: "f32[3072, 768]", getitem_360: "f32[3072]", getitem_361: "f32[768, 3072]", getitem_362: "f32[768]", getitem_363: "f32[768]", getitem_364: "f32[768]", getitem_365: "f32[768]", getitem_366: "f32[768]", getitem_367: "f32[2304, 768]", getitem_368: "f32[2304]", getitem_369: "f32[768, 768]", getitem_370: "f32[768]", getitem_371: "f32[3072, 768]", getitem_372: "f32[3072]", getitem_373: "f32[768, 3072]", getitem_374: "f32[768]", getitem_375: "f32[768]", getitem_376: "f32[768]", getitem_377: "f32[768]", getitem_378: "f32[768]", getitem_379: "f32[2304, 768]", getitem_380: "f32[2304]", getitem_381: "f32[768, 768]", getitem_382: "f32[768]", getitem_383: "f32[3072, 768]", getitem_384: "f32[3072]", getitem_385: "f32[768, 3072]", getitem_386: "f32[768]", getitem_387: "f32[768]", getitem_388: "f32[768]", getitem_389: "f32[768]", getitem_390: "f32[768]", getitem_391: "f32[2304, 768]", getitem_392: "f32[2304]", getitem_393: "f32[768, 768]", getitem_394: "f32[768]", getitem_395: "f32[3072, 768]", getitem_396: "f32[3072]", getitem_397: "f32[768, 3072]", getitem_398: "f32[768]", getitem_399: "f32[768]", getitem_400: "f32[768]", getitem_401: "f32[768]", getitem_402: "f32[768]", getitem_403: "f32[2304, 768]", getitem_404: "f32[2304]", getitem_405: "f32[768, 768]", getitem_406: "f32[768]", getitem_407: "f32[3072, 768]", getitem_408: "f32[3072]", getitem_409: "f32[768, 3072]", getitem_410: "f32[768]", getitem_411: "f32[768]", getitem_412: "f32[768]", getitem_413: "f32[768]", getitem_414: "f32[768]", getitem_415: "f32[2304, 768]", getitem_416: "f32[2304]", getitem_417: "f32[768, 768]", getitem_418: "f32[768]", getitem_419: "f32[3072, 768]", getitem_420: "f32[3072]", getitem_421: "f32[768, 3072]", getitem_422: "f32[768]", getitem_423: "f32[768]", getitem_424: "f32[768]", getitem_425: "f32[768]", getitem_426: "f32[768]", getitem_427: "f32[2304, 768]", getitem_428: "f32[2304]", getitem_429: "f32[768, 768]", getitem_430: "f32[768]", getitem_431: "f32[3072, 768]", getitem_432: "f32[3072]", getitem_433: "f32[768, 3072]", getitem_434: "f32[768]", getitem_435: "f32[768]", getitem_436: "f32[768]", getitem_437: "f32[768]", getitem_438: "f32[768]", getitem_439: "f32[2304, 768]", getitem_440: "f32[2304]", getitem_441: "f32[768, 768]", getitem_442: "f32[768]", getitem_443: "f32[3072, 768]", getitem_444: "f32[3072]", getitem_445: "f32[768, 3072]", getitem_446: "f32[768]", getitem_447: "f32[768]", getitem_448: "f32[768]", getitem_449: "f32[768]", getitem_450: "f32[768]", getitem_451: "f32[768]", getitem_452: "f32[768]", getitem_453: "f32[77, 512]", getitem_454: "f32[49408, 512]", getitem_455: "f32[1536, 512]", getitem_456: "f32[1536]", getitem_457: "f32[512, 512]", getitem_458: "f32[512]", getitem_459: "f32[2048, 512]", getitem_460: "f32[2048]", getitem_461: "f32[512, 2048]", getitem_462: "f32[512]", getitem_463: "f32[512]", getitem_464: "f32[512]", getitem_465: "f32[512]", getitem_466: "f32[512]", getitem_467: "f32[1536, 512]", getitem_468: "f32[1536]", getitem_469: "f32[512, 512]", getitem_470: "f32[512]", getitem_471: "f32[2048, 512]", getitem_472: "f32[2048]", getitem_473: "f32[512, 2048]", getitem_474: "f32[512]", getitem_475: "f32[512]", getitem_476: "f32[512]", getitem_477: "f32[512]", getitem_478: "f32[512]", getitem_479: "f32[1536, 512]", getitem_480: "f32[1536]", getitem_481: "f32[512, 512]", getitem_482: "f32[512]", getitem_483: "f32[2048, 512]", getitem_484: "f32[2048]", getitem_485: "f32[512, 2048]", getitem_486: "f32[512]", getitem_487: "f32[512]", getitem_488: "f32[512]", getitem_489: "f32[512]", getitem_490: "f32[512]", getitem_491: "f32[1536, 512]", getitem_492: "f32[1536]", getitem_493: "f32[512, 512]", getitem_494: "f32[512]", getitem_495: "f32[2048, 512]", getitem_496: "f32[2048]", getitem_497: "f32[512, 2048]", getitem_498: "f32[512]", getitem_499: "f32[512]", getitem_500: "f32[512]", getitem_501: "f32[512]", getitem_502: "f32[512]", getitem_503: "f32[1536, 512]", getitem_504: "f32[1536]", getitem_505: "f32[512, 512]", getitem_506: "f32[512]", getitem_507: "f32[2048, 512]", getitem_508: "f32[2048]", getitem_509: "f32[512, 2048]", getitem_510: "f32[512]", getitem_511: "f32[512]", getitem_512: "f32[512]", getitem_513: "f32[512]", getitem_514: "f32[512]", getitem_515: "f32[1536, 512]", getitem_516: "f32[1536]", getitem_517: "f32[512, 512]", getitem_518: "f32[512]", getitem_519: "f32[2048, 512]", getitem_520: "f32[2048]", getitem_521: "f32[512, 2048]", getitem_522: "f32[512]", getitem_523: "f32[512]", getitem_524: "f32[512]", getitem_525: "f32[512]", getitem_526: "f32[512]", getitem_527: "f32[1536, 512]", getitem_528: "f32[1536]", getitem_529: "f32[512, 512]", getitem_530: "f32[512]", getitem_531: "f32[2048, 512]", getitem_532: "f32[2048]", getitem_533: "f32[512, 2048]", getitem_534: "f32[512]", getitem_535: "f32[512]", getitem_536: "f32[512]", getitem_537: "f32[512]", getitem_538: "f32[512]", getitem_539: "f32[1536, 512]", getitem_540: "f32[1536]", getitem_541: "f32[512, 512]", getitem_542: "f32[512]", getitem_543: "f32[2048, 512]", getitem_544: "f32[2048]", getitem_545: "f32[512, 2048]", getitem_546: "f32[512]", getitem_547: "f32[512]", getitem_548: "f32[512]", getitem_549: "f32[512]", getitem_550: "f32[512]", getitem_551: "f32[1536, 512]", getitem_552: "f32[1536]", getitem_553: "f32[512, 512]", getitem_554: "f32[512]", getitem_555: "f32[2048, 512]", getitem_556: "f32[2048]", getitem_557: "f32[512, 2048]", getitem_558: "f32[512]", getitem_559: "f32[512]", getitem_560: "f32[512]", getitem_561: "f32[512]", getitem_562: "f32[512]", getitem_563: "f32[1536, 512]", getitem_564: "f32[1536]", getitem_565: "f32[512, 512]", getitem_566: "f32[512]", getitem_567: "f32[2048, 512]", getitem_568: "f32[2048]", getitem_569: "f32[512, 2048]", getitem_570: "f32[512]", getitem_571: "f32[512]", getitem_572: "f32[512]", getitem_573: "f32[512]", getitem_574: "f32[512]", getitem_575: "f32[1536, 512]", getitem_576: "f32[1536]", getitem_577: "f32[512, 512]", getitem_578: "f32[512]", getitem_579: "f32[2048, 512]", getitem_580: "f32[2048]", getitem_581: "f32[512, 2048]", getitem_582: "f32[512]", getitem_583: "f32[512]", getitem_584: "f32[512]", getitem_585: "f32[512]", getitem_586: "f32[512]", getitem_587: "f32[1536, 512]", getitem_588: "f32[1536]", getitem_589: "f32[512, 512]", getitem_590: "f32[512]", getitem_591: "f32[2048, 512]", getitem_592: "f32[2048]", getitem_593: "f32[512, 2048]", getitem_594: "f32[512]", getitem_595: "f32[512]", getitem_596: "f32[512]", getitem_597: "f32[512]", getitem_598: "f32[512]", getitem_599: "f32[512]", getitem_600: "f32[512]", getitem_601: "f32[512, 512]", getitem_4515: "f32[768]", getitem_4516: "f32[50, 768]", getitem_4517: "f32[768, 512]", getitem_4518: "f32[768, 3, 32, 32]", getitem_4519: "f32[768]", getitem_4520: "f32[768]", getitem_4521: "f32[2304, 768]", getitem_4522: "f32[2304]", getitem_4523: "f32[768, 768]", getitem_4524: "f32[768]", getitem_4525: "f32[3072, 768]", getitem_4526: "f32[3072]", getitem_4527: "f32[768, 3072]", getitem_4528: "f32[768]", getitem_4529: "f32[768]", getitem_4530: "f32[768]", getitem_4531: "f32[768]", getitem_4532: "f32[768]", getitem_4533: "f32[2304, 768]", getitem_4534: "f32[2304]", getitem_4535: "f32[768, 768]", getitem_4536: "f32[768]", getitem_4537: "f32[3072, 768]", getitem_4538: "f32[3072]", getitem_4539: "f32[768, 3072]", getitem_4540: "f32[768]", getitem_4541: "f32[768]", getitem_4542: "f32[768]", getitem_4543: "f32[768]", getitem_4544: "f32[768]", getitem_4545: "f32[2304, 768]", getitem_4546: "f32[2304]", getitem_4547: "f32[768, 768]", getitem_4548: "f32[768]", getitem_4549: "f32[3072, 768]", getitem_4550: "f32[3072]", getitem_4551: "f32[768, 3072]", getitem_4552: "f32[768]", getitem_4553: "f32[768]", getitem_4554: "f32[768]", getitem_4555: "f32[768]", getitem_4556: "f32[768]", getitem_4557: "f32[2304, 768]", getitem_4558: "f32[2304]", getitem_4559: "f32[768, 768]", getitem_4560: "f32[768]", getitem_4561: "f32[3072, 768]", getitem_4562: "f32[3072]", getitem_4563: "f32[768, 3072]", getitem_4564: "f32[768]", getitem_4565: "f32[768]", getitem_4566: "f32[768]", getitem_4567: "f32[768]", getitem_4568: "f32[768]", getitem_4569: "f32[2304, 768]", getitem_4570: "f32[2304]", getitem_4571: "f32[768, 768]", getitem_4572: "f32[768]", getitem_4573: "f32[3072, 768]", getitem_4574: "f32[3072]", getitem_4575: "f32[768, 3072]", getitem_4576: "f32[768]", getitem_4577: "f32[768]", getitem_4578: "f32[768]", getitem_4579: "f32[768]", getitem_4580: "f32[768]", getitem_4581: "f32[2304, 768]", getitem_4582: "f32[2304]", getitem_4583: "f32[768, 768]", getitem_4584: "f32[768]", getitem_4585: "f32[3072, 768]", getitem_4586: "f32[3072]", getitem_4587: "f32[768, 3072]", getitem_4588: "f32[768]", getitem_4589: "f32[768]", getitem_4590: "f32[768]", getitem_4591: "f32[768]", getitem_4592: "f32[768]", getitem_4593: "f32[2304, 768]", getitem_4594: "f32[2304]", getitem_4595: "f32[768, 768]", getitem_4596: "f32[768]", getitem_4597: "f32[3072, 768]", getitem_4598: "f32[3072]", getitem_4599: "f32[768, 3072]", getitem_4600: "f32[768]", getitem_4601: "f32[768]", getitem_4602: "f32[768]", getitem_4603: "f32[768]", getitem_4604: "f32[768]", getitem_4605: "f32[2304, 768]", getitem_4606: "f32[2304]", getitem_4607: "f32[768, 768]", getitem_4608: "f32[768]", getitem_4609: "f32[3072, 768]", getitem_4610: "f32[3072]", getitem_4611: "f32[768, 3072]", getitem_4612: "f32[768]", getitem_4613: "f32[768]", getitem_4614: "f32[768]", getitem_4615: "f32[768]", getitem_4616: "f32[768]", getitem_4617: "f32[2304, 768]", getitem_4618: "f32[2304]", getitem_4619: "f32[768, 768]", getitem_4620: "f32[768]", getitem_4621: "f32[3072, 768]", getitem_4622: "f32[3072]", getitem_4623: "f32[768, 3072]", getitem_4624: "f32[768]", getitem_4625: "f32[768]", getitem_4626: "f32[768]", getitem_4627: "f32[768]", getitem_4628: "f32[768]", getitem_4629: "f32[2304, 768]", getitem_4630: "f32[2304]", getitem_4631: "f32[768, 768]", getitem_4632: "f32[768]", getitem_4633: "f32[3072, 768]", getitem_4634: "f32[3072]", getitem_4635: "f32[768, 3072]", getitem_4636: "f32[768]", getitem_4637: "f32[768]", getitem_4638: "f32[768]", getitem_4639: "f32[768]", getitem_4640: "f32[768]", getitem_4641: "f32[2304, 768]", getitem_4642: "f32[2304]", getitem_4643: "f32[768, 768]", getitem_4644: "f32[768]", getitem_4645: "f32[3072, 768]", getitem_4646: "f32[3072]", getitem_4647: "f32[768, 3072]", getitem_4648: "f32[768]", getitem_4649: "f32[768]", getitem_4650: "f32[768]", getitem_4651: "f32[768]", getitem_4652: "f32[768]", getitem_4653: "f32[2304, 768]", getitem_4654: "f32[2304]", getitem_4655: "f32[768, 768]", getitem_4656: "f32[768]", getitem_4657: "f32[3072, 768]", getitem_4658: "f32[3072]", getitem_4659: "f32[768, 3072]", getitem_4660: "f32[768]", getitem_4661: "f32[768]", getitem_4662: "f32[768]", getitem_4663: "f32[768]", getitem_4664: "f32[768]", getitem_4665: "f32[768]", getitem_4666: "f32[768]", getitem_4667: "f32[77, 512]", getitem_4668: "f32[49408, 512]", getitem_4669: "f32[1536, 512]", getitem_4670: "f32[1536]", getitem_4671: "f32[512, 512]", getitem_4672: "f32[512]", getitem_4673: "f32[2048, 512]", getitem_4674: "f32[2048]", getitem_4675: "f32[512, 2048]", getitem_4676: "f32[512]", getitem_4677: "f32[512]", getitem_4678: "f32[512]", getitem_4679: "f32[512]", getitem_4680: "f32[512]", getitem_4681: "f32[1536, 512]", getitem_4682: "f32[1536]", getitem_4683: "f32[512, 512]", getitem_4684: "f32[512]", getitem_4685: "f32[2048, 512]", getitem_4686: "f32[2048]", getitem_4687: "f32[512, 2048]", getitem_4688: "f32[512]", getitem_4689: "f32[512]", getitem_4690: "f32[512]", getitem_4691: "f32[512]", getitem_4692: "f32[512]", getitem_4693: "f32[1536, 512]", getitem_4694: "f32[1536]", getitem_4695: "f32[512, 512]", getitem_4696: "f32[512]", getitem_4697: "f32[2048, 512]", getitem_4698: "f32[2048]", getitem_4699: "f32[512, 2048]", getitem_4700: "f32[512]", getitem_4701: "f32[512]", getitem_4702: "f32[512]", getitem_4703: "f32[512]", getitem_4704: "f32[512]", getitem_4705: "f32[1536, 512]", getitem_4706: "f32[1536]", getitem_4707: "f32[512, 512]", getitem_4708: "f32[512]", getitem_4709: "f32[2048, 512]", getitem_4710: "f32[2048]", getitem_4711: "f32[512, 2048]", getitem_4712: "f32[512]", getitem_4713: "f32[512]", getitem_4714: "f32[512]", getitem_4715: "f32[512]", getitem_4716: "f32[512]", getitem_4717: "f32[1536, 512]", getitem_4718: "f32[1536]", getitem_4719: "f32[512, 512]", getitem_4720: "f32[512]", getitem_4721: "f32[2048, 512]", getitem_4722: "f32[2048]", getitem_4723: "f32[512, 2048]", getitem_4724: "f32[512]", getitem_4725: "f32[512]", getitem_4726: "f32[512]", getitem_4727: "f32[512]", getitem_4728: "f32[512]", getitem_4729: "f32[1536, 512]", getitem_4730: "f32[1536]", getitem_4731: "f32[512, 512]", getitem_4732: "f32[512]", getitem_4733: "f32[2048, 512]", getitem_4734: "f32[2048]", getitem_4735: "f32[512, 2048]", getitem_4736: "f32[512]", getitem_4737: "f32[512]", getitem_4738: "f32[512]", getitem_4739: "f32[512]", getitem_4740: "f32[512]", getitem_4741: "f32[1536, 512]", getitem_4742: "f32[1536]", getitem_4743: "f32[512, 512]", getitem_4744: "f32[512]", getitem_4745: "f32[2048, 512]", getitem_4746: "f32[2048]", getitem_4747: "f32[512, 2048]", getitem_4748: "f32[512]", getitem_4749: "f32[512]", getitem_4750: "f32[512]", getitem_4751: "f32[512]", getitem_4752: "f32[512]", getitem_4753: "f32[1536, 512]", getitem_4754: "f32[1536]", getitem_4755: "f32[512, 512]", getitem_4756: "f32[512]", getitem_4757: "f32[2048, 512]", getitem_4758: "f32[2048]", getitem_4759: "f32[512, 2048]", getitem_4760: "f32[512]", getitem_4761: "f32[512]", getitem_4762: "f32[512]", getitem_4763: "f32[512]", getitem_4764: "f32[512]", getitem_4765: "f32[1536, 512]", getitem_4766: "f32[1536]", getitem_4767: "f32[512, 512]", getitem_4768: "f32[512]", getitem_4769: "f32[2048, 512]", getitem_4770: "f32[2048]", getitem_4771: "f32[512, 2048]", getitem_4772: "f32[512]", getitem_4773: "f32[512]", getitem_4774: "f32[512]", getitem_4775: "f32[512]", getitem_4776: "f32[512]", getitem_4777: "f32[1536, 512]", getitem_4778: "f32[1536]", getitem_4779: "f32[512, 512]", getitem_4780: "f32[512]", getitem_4781: "f32[2048, 512]", getitem_4782: "f32[2048]", getitem_4783: "f32[512, 2048]", getitem_4784: "f32[512]", getitem_4785: "f32[512]", getitem_4786: "f32[512]", getitem_4787: "f32[512]", getitem_4788: "f32[512]", getitem_4789: "f32[1536, 512]", getitem_4790: "f32[1536]", getitem_4791: "f32[512, 512]", getitem_4792: "f32[512]", getitem_4793: "f32[2048, 512]", getitem_4794: "f32[2048]", getitem_4795: "f32[512, 2048]", getitem_4796: "f32[512]", getitem_4797: "f32[512]", getitem_4798: "f32[512]", getitem_4799: "f32[512]", getitem_4800: "f32[512]", getitem_4801: "f32[1536, 512]", getitem_4802: "f32[1536]", getitem_4803: "f32[512, 512]", getitem_4804: "f32[512]", getitem_4805: "f32[2048, 512]", getitem_4806: "f32[2048]", getitem_4807: "f32[512, 2048]", getitem_4808: "f32[512]", getitem_4809: "f32[512]", getitem_4810: "f32[512]", getitem_4811: "f32[512]", getitem_4812: "f32[512]", getitem_4813: "f32[512]", getitem_4814: "f32[512]", getitem_4815: "f32[512, 512]", getitem_3311: "f32[]", getitem_3312: "f32[]", getitem_3313: "f32[]", getitem_3314: "f32[]", getitem_3315: "f32[]", getitem_3316: "f32[]", getitem_3317: "f32[]", getitem_3318: "f32[]", getitem_3319: "f32[]", getitem_3320: "f32[]", getitem_3321: "f32[]", getitem_3322: "f32[]", getitem_3323: "f32[]", getitem_3324: "f32[]", getitem_3325: "f32[]", getitem_3326: "f32[]", getitem_3327: "f32[]", getitem_3328: "f32[]", getitem_3329: "f32[]", getitem_3330: "f32[]", getitem_3331: "f32[]", getitem_3332: "f32[]", getitem_3333: "f32[]", getitem_3334: "f32[]", getitem_3335: "f32[]", getitem_3336: "f32[]", getitem_3337: "f32[]", getitem_3338: "f32[]", getitem_3339: "f32[]", getitem_3340: "f32[]", getitem_3341: "f32[]", getitem_3342: "f32[]", getitem_3343: "f32[]", getitem_3344: "f32[]", getitem_3345: "f32[]", getitem_3346: "f32[]", getitem_3347: "f32[]", getitem_3348: "f32[]", getitem_3349: "f32[]", getitem_3350: "f32[]", getitem_3351: "f32[]", getitem_3352: "f32[]", getitem_3353: "f32[]", getitem_3354: "f32[]", getitem_3355: "f32[]", getitem_3356: "f32[]", getitem_3357: "f32[]", getitem_3358: "f32[]", getitem_3359: "f32[]", getitem_3360: "f32[]", getitem_3361: "f32[]", getitem_3362: "f32[]", getitem_3363: "f32[]", getitem_3364: "f32[]", getitem_3365: "f32[]", getitem_3366: "f32[]", getitem_3367: "f32[]", getitem_3368: "f32[]", getitem_3369: "f32[]", getitem_3370: "f32[]", getitem_3371: "f32[]", getitem_3372: "f32[]", getitem_3373: "f32[]", getitem_3374: "f32[]", getitem_3375: "f32[]", getitem_3376: "f32[]", getitem_3377: "f32[]", getitem_3378: "f32[]", getitem_3379: "f32[]", getitem_3380: "f32[]", getitem_3381: "f32[]", getitem_3382: "f32[]", getitem_3383: "f32[]", getitem_3384: "f32[]", getitem_3385: "f32[]", getitem_3386: "f32[]", getitem_3387: "f32[]", getitem_3388: "f32[]", getitem_3389: "f32[]", getitem_3390: "f32[]", getitem_3391: "f32[]", getitem_3392: "f32[]", getitem_3393: "f32[]", getitem_3394: "f32[]", getitem_3395: "f32[]", getitem_3396: "f32[]", getitem_3397: "f32[]", getitem_3398: "f32[]", getitem_3399: "f32[]", getitem_3400: "f32[]", getitem_3401: "f32[]", getitem_3402: "f32[]", getitem_3403: "f32[]", getitem_3404: "f32[]", getitem_3405: "f32[]", getitem_3406: "f32[]", getitem_3407: "f32[]", getitem_3408: "f32[]", getitem_3409: "f32[]", getitem_3410: "f32[]", getitem_3411: "f32[]", getitem_3412: "f32[]", getitem_3413: "f32[]", getitem_3414: "f32[]", getitem_3415: "f32[]", getitem_3416: "f32[]", getitem_3417: "f32[]", getitem_3418: "f32[]", getitem_3419: "f32[]", getitem_3420: "f32[]", getitem_3421: "f32[]", getitem_3422: "f32[]", getitem_3423: "f32[]", getitem_3424: "f32[]", getitem_3425: "f32[]", getitem_3426: "f32[]", getitem_3427: "f32[]", getitem_3428: "f32[]", getitem_3429: "f32[]", getitem_3430: "f32[]", getitem_3431: "f32[]", getitem_3432: "f32[]", getitem_3433: "f32[]", getitem_3434: "f32[]", getitem_3435: "f32[]", getitem_3436: "f32[]", getitem_3437: "f32[]", getitem_3438: "f32[]", getitem_3439: "f32[]", getitem_3440: "f32[]", getitem_3441: "f32[]", getitem_3442: "f32[]", getitem_3443: "f32[]", getitem_3444: "f32[]", getitem_3445: "f32[]", getitem_3446: "f32[]", getitem_3447: "f32[]", getitem_3448: "f32[]", getitem_3449: "f32[]", getitem_3450: "f32[]", getitem_3451: "f32[]", getitem_3452: "f32[]", getitem_3453: "f32[]", getitem_3454: "f32[]", getitem_3455: "f32[]", getitem_3456: "f32[]", getitem_3457: "f32[]", getitem_3458: "f32[]", getitem_3459: "f32[]", getitem_3460: "f32[]", getitem_3461: "f32[]", getitem_3462: "f32[]", getitem_3463: "f32[]", getitem_3464: "f32[]", getitem_3465: "f32[]", getitem_3466: "f32[]", getitem_3467: "f32[]", getitem_3468: "f32[]", getitem_3469: "f32[]", getitem_3470: "f32[]", getitem_3471: "f32[]", getitem_3472: "f32[]", getitem_3473: "f32[]", getitem_3474: "f32[]", getitem_3475: "f32[]", getitem_3476: "f32[]", getitem_3477: "f32[]", getitem_3478: "f32[]", getitem_3479: "f32[]", getitem_3480: "f32[]", getitem_3481: "f32[]", getitem_3482: "f32[]", getitem_3483: "f32[]", getitem_3484: "f32[]", getitem_3485: "f32[]", getitem_3486: "f32[]", getitem_3487: "f32[]", getitem_3488: "f32[]", getitem_3489: "f32[]", getitem_3490: "f32[]", getitem_3491: "f32[]", getitem_3492: "f32[]", getitem_3493: "f32[]", getitem_3494: "f32[]", getitem_3495: "f32[]", getitem_3496: "f32[]", getitem_3497: "f32[]", getitem_3498: "f32[]", getitem_3499: "f32[]", getitem_3500: "f32[]", getitem_3501: "f32[]", getitem_3502: "f32[]", getitem_3503: "f32[]", getitem_3504: "f32[]", getitem_3505: "f32[]", getitem_3506: "f32[]", getitem_3507: "f32[]", getitem_3508: "f32[]", getitem_3509: "f32[]", getitem_3510: "f32[]", getitem_3511: "f32[]", getitem_3512: "f32[]", getitem_3513: "f32[]", getitem_3514: "f32[]", getitem_3515: "f32[]", getitem_3516: "f32[]", getitem_3517: "f32[]", getitem_3518: "f32[]", getitem_3519: "f32[]", getitem_3520: "f32[]", getitem_3521: "f32[]", getitem_3522: "f32[]", getitem_3523: "f32[]", getitem_3524: "f32[]", getitem_3525: "f32[]", getitem_3526: "f32[]", getitem_3527: "f32[]", getitem_3528: "f32[]", getitem_3529: "f32[]", getitem_3530: "f32[]", getitem_3531: "f32[]", getitem_3532: "f32[]", getitem_3533: "f32[]", getitem_3534: "f32[]", getitem_3535: "f32[]", getitem_3536: "f32[]", getitem_3537: "f32[]", getitem_3538: "f32[]", getitem_3539: "f32[]", getitem_3540: "f32[]", getitem_3541: "f32[]", getitem_3542: "f32[]", getitem_3543: "f32[]", getitem_3544: "f32[]", getitem_3545: "f32[]", getitem_3546: "f32[]", getitem_3547: "f32[]", getitem_3548: "f32[]", getitem_3549: "f32[]", getitem_3550: "f32[]", getitem_3551: "f32[]", getitem_3552: "f32[]", getitem_3553: "f32[]", getitem_3554: "f32[]", getitem_3555: "f32[]", getitem_3556: "f32[]", getitem_3557: "f32[]", getitem_3558: "f32[]", getitem_3559: "f32[]", getitem_3560: "f32[]", getitem_3561: "f32[]", getitem_3562: "f32[]", getitem_3563: "f32[]", getitem_3564: "f32[]", getitem_3565: "f32[]", getitem_3566: "f32[]", getitem_3567: "f32[]", getitem_3568: "f32[]", getitem_3569: "f32[]", getitem_3570: "f32[]", getitem_3571: "f32[]", getitem_3572: "f32[]", getitem_3573: "f32[]", getitem_3574: "f32[]", getitem_3575: "f32[]", getitem_3576: "f32[]", getitem_3577: "f32[]", getitem_3578: "f32[]", getitem_3579: "f32[]", getitem_3580: "f32[]", getitem_3581: "f32[]", getitem_3582: "f32[]", getitem_3583: "f32[]", getitem_3584: "f32[]", getitem_3585: "f32[]", getitem_3586: "f32[]", getitem_3587: "f32[]", getitem_3588: "f32[]", getitem_3589: "f32[]", getitem_3590: "f32[]", getitem_3591: "f32[]", getitem_3592: "f32[]", getitem_3593: "f32[]", getitem_3594: "f32[]", getitem_3595: "f32[]", getitem_3596: "f32[]", getitem_3597: "f32[]", getitem_3598: "f32[]", getitem_3599: "f32[]", getitem_3600: "f32[]", getitem_3601: "f32[]", getitem_3602: "f32[]", getitem_3603: "f32[]", getitem_3604: "f32[]", getitem_3605: "f32[]", getitem_3606: "f32[]", getitem_3607: "f32[]", getitem_3608: "f32[]", getitem_3609: "f32[]", getitem_3610: "f32[]", getitem_3611: "f32[]"):
        # No stacktrace found for following nodes
        full_default: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[50, 768]" = torch.ops.aten.full.default([50, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "f32[768, 512]" = torch.ops.aten.full.default([768, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_3: "f32[768, 3, 32, 32]" = torch.ops.aten.full.default([768, 3, 32, 32], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_4: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_5: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_6: "f32[2304, 768]" = torch.ops.aten.full.default([2304, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_7: "f32[2304]" = torch.ops.aten.full.default([2304], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_8: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_9: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_10: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_11: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_12: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_13: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_14: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_15: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_16: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_17: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_18: "f32[2304, 768]" = torch.ops.aten.full.default([2304, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_19: "f32[2304]" = torch.ops.aten.full.default([2304], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_20: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_21: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_22: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_23: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_24: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_25: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_26: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_27: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_28: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_29: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_30: "f32[2304, 768]" = torch.ops.aten.full.default([2304, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_31: "f32[2304]" = torch.ops.aten.full.default([2304], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_32: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_33: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_34: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_35: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_36: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_37: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_38: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_39: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_40: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_41: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_42: "f32[2304, 768]" = torch.ops.aten.full.default([2304, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_43: "f32[2304]" = torch.ops.aten.full.default([2304], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_44: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_45: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_46: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_47: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_48: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_49: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_50: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_51: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_52: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_53: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_54: "f32[2304, 768]" = torch.ops.aten.full.default([2304, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_55: "f32[2304]" = torch.ops.aten.full.default([2304], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_56: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_57: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_58: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_59: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_60: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_61: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_62: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_63: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_64: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_65: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_66: "f32[2304, 768]" = torch.ops.aten.full.default([2304, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_67: "f32[2304]" = torch.ops.aten.full.default([2304], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_68: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_69: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_70: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_71: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_72: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_73: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_74: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_75: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_76: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_77: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_78: "f32[2304, 768]" = torch.ops.aten.full.default([2304, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_79: "f32[2304]" = torch.ops.aten.full.default([2304], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_80: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_81: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_82: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_83: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_84: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_85: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_86: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_87: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_88: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_89: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_90: "f32[2304, 768]" = torch.ops.aten.full.default([2304, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_91: "f32[2304]" = torch.ops.aten.full.default([2304], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_92: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_93: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_94: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_95: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_96: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_97: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_98: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_99: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_100: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_101: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_102: "f32[2304, 768]" = torch.ops.aten.full.default([2304, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_103: "f32[2304]" = torch.ops.aten.full.default([2304], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_104: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_105: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_106: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_107: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_108: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_109: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_110: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_111: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_112: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_113: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_114: "f32[2304, 768]" = torch.ops.aten.full.default([2304, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_115: "f32[2304]" = torch.ops.aten.full.default([2304], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_116: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_117: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_118: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_119: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_120: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_121: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_122: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_123: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_124: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_125: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_126: "f32[2304, 768]" = torch.ops.aten.full.default([2304, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_127: "f32[2304]" = torch.ops.aten.full.default([2304], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_128: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_129: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_130: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_131: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_132: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_133: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_134: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_135: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_136: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_137: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_138: "f32[2304, 768]" = torch.ops.aten.full.default([2304, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_139: "f32[2304]" = torch.ops.aten.full.default([2304], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_140: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_141: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_142: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_143: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_144: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_145: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_146: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_147: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_148: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_149: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_150: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_151: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_152: "f32[77, 512]" = torch.ops.aten.full.default([77, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_153: "f32[49408, 512]" = torch.ops.aten.full.default([49408, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_154: "f32[1536, 512]" = torch.ops.aten.full.default([1536, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_155: "f32[1536]" = torch.ops.aten.full.default([1536], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_156: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_157: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_158: "f32[2048, 512]" = torch.ops.aten.full.default([2048, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_159: "f32[2048]" = torch.ops.aten.full.default([2048], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_160: "f32[512, 2048]" = torch.ops.aten.full.default([512, 2048], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_161: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_162: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_163: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_164: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_165: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_166: "f32[1536, 512]" = torch.ops.aten.full.default([1536, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_167: "f32[1536]" = torch.ops.aten.full.default([1536], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_168: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_169: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_170: "f32[2048, 512]" = torch.ops.aten.full.default([2048, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_171: "f32[2048]" = torch.ops.aten.full.default([2048], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_172: "f32[512, 2048]" = torch.ops.aten.full.default([512, 2048], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_173: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_174: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_175: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_176: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_177: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_178: "f32[1536, 512]" = torch.ops.aten.full.default([1536, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_179: "f32[1536]" = torch.ops.aten.full.default([1536], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_180: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_181: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_182: "f32[2048, 512]" = torch.ops.aten.full.default([2048, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_183: "f32[2048]" = torch.ops.aten.full.default([2048], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_184: "f32[512, 2048]" = torch.ops.aten.full.default([512, 2048], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_185: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_186: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_187: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_188: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_189: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_190: "f32[1536, 512]" = torch.ops.aten.full.default([1536, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_191: "f32[1536]" = torch.ops.aten.full.default([1536], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_192: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_193: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_194: "f32[2048, 512]" = torch.ops.aten.full.default([2048, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_195: "f32[2048]" = torch.ops.aten.full.default([2048], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_196: "f32[512, 2048]" = torch.ops.aten.full.default([512, 2048], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_197: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_198: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_199: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_200: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_201: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_202: "f32[1536, 512]" = torch.ops.aten.full.default([1536, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_203: "f32[1536]" = torch.ops.aten.full.default([1536], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_204: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_205: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_206: "f32[2048, 512]" = torch.ops.aten.full.default([2048, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_207: "f32[2048]" = torch.ops.aten.full.default([2048], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_208: "f32[512, 2048]" = torch.ops.aten.full.default([512, 2048], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_209: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_210: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_211: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_212: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_213: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_214: "f32[1536, 512]" = torch.ops.aten.full.default([1536, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_215: "f32[1536]" = torch.ops.aten.full.default([1536], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_216: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_217: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_218: "f32[2048, 512]" = torch.ops.aten.full.default([2048, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_219: "f32[2048]" = torch.ops.aten.full.default([2048], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_220: "f32[512, 2048]" = torch.ops.aten.full.default([512, 2048], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_221: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_222: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_223: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_224: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_225: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_226: "f32[1536, 512]" = torch.ops.aten.full.default([1536, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_227: "f32[1536]" = torch.ops.aten.full.default([1536], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_228: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_229: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_230: "f32[2048, 512]" = torch.ops.aten.full.default([2048, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_231: "f32[2048]" = torch.ops.aten.full.default([2048], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_232: "f32[512, 2048]" = torch.ops.aten.full.default([512, 2048], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_233: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_234: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_235: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_236: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_237: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_238: "f32[1536, 512]" = torch.ops.aten.full.default([1536, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_239: "f32[1536]" = torch.ops.aten.full.default([1536], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_240: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_241: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_242: "f32[2048, 512]" = torch.ops.aten.full.default([2048, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_243: "f32[2048]" = torch.ops.aten.full.default([2048], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_244: "f32[512, 2048]" = torch.ops.aten.full.default([512, 2048], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_245: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_246: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_247: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_248: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_249: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_250: "f32[1536, 512]" = torch.ops.aten.full.default([1536, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_251: "f32[1536]" = torch.ops.aten.full.default([1536], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_252: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_253: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_254: "f32[2048, 512]" = torch.ops.aten.full.default([2048, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_255: "f32[2048]" = torch.ops.aten.full.default([2048], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_256: "f32[512, 2048]" = torch.ops.aten.full.default([512, 2048], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_257: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_258: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_259: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_260: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_261: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_262: "f32[1536, 512]" = torch.ops.aten.full.default([1536, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_263: "f32[1536]" = torch.ops.aten.full.default([1536], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_264: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_265: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_266: "f32[2048, 512]" = torch.ops.aten.full.default([2048, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_267: "f32[2048]" = torch.ops.aten.full.default([2048], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_268: "f32[512, 2048]" = torch.ops.aten.full.default([512, 2048], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_269: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_270: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_271: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_272: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_273: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_274: "f32[1536, 512]" = torch.ops.aten.full.default([1536, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_275: "f32[1536]" = torch.ops.aten.full.default([1536], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_276: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_277: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_278: "f32[2048, 512]" = torch.ops.aten.full.default([2048, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_279: "f32[2048]" = torch.ops.aten.full.default([2048], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_280: "f32[512, 2048]" = torch.ops.aten.full.default([512, 2048], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_281: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_282: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_283: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_284: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_285: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_286: "f32[1536, 512]" = torch.ops.aten.full.default([1536, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_287: "f32[1536]" = torch.ops.aten.full.default([1536], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_288: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_289: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_290: "f32[2048, 512]" = torch.ops.aten.full.default([2048, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_291: "f32[2048]" = torch.ops.aten.full.default([2048], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_292: "f32[512, 2048]" = torch.ops.aten.full.default([512, 2048], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_293: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_294: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_295: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_296: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_297: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_298: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_299: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_300: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _foreach_addcmul_scalar = torch.ops.aten._foreach_addcmul.Scalar([arg304_1, arg302_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1, arg728_1, arg729_1, arg730_1, arg731_1, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, arg737_1, arg738_1, arg739_1, arg740_1, arg741_1, arg742_1, arg743_1, arg744_1, arg745_1, arg746_1, arg747_1, arg748_1, arg749_1, arg750_1, arg751_1, arg752_1, arg753_1, arg754_1, arg755_1, arg756_1, arg757_1, arg758_1, arg759_1, arg760_1, arg761_1, arg762_1, arg763_1, arg764_1, arg765_1, arg766_1, arg767_1, arg768_1, arg769_1, arg770_1, arg771_1, arg772_1, arg773_1, arg774_1, arg775_1, arg776_1, arg777_1, arg778_1, arg779_1, arg780_1, arg781_1, arg782_1, arg783_1, arg784_1, arg785_1, arg786_1, arg787_1, arg788_1, arg789_1, arg790_1, arg791_1, arg792_1, arg793_1, arg794_1, arg795_1, arg796_1, arg797_1, arg798_1, arg799_1, arg800_1, arg801_1, arg802_1, arg803_1, arg804_1, arg805_1, arg806_1, arg807_1, arg808_1, arg809_1, arg810_1, arg811_1, arg812_1, arg813_1, arg814_1, arg815_1, arg816_1, arg817_1, arg818_1, arg819_1, arg820_1, arg821_1, arg822_1, arg823_1, arg824_1, arg825_1, arg826_1, arg827_1, arg828_1, arg829_1, arg830_1, arg831_1, arg832_1, arg833_1, arg834_1, arg835_1, arg836_1, arg837_1, arg838_1, arg839_1, arg840_1, arg841_1, arg842_1, arg843_1, arg844_1, arg845_1, arg846_1, arg847_1, arg848_1, arg849_1, arg850_1, arg851_1, arg852_1, arg853_1, arg854_1, arg855_1, arg856_1, arg857_1, arg858_1, arg859_1, arg860_1, arg861_1, arg862_1, arg863_1, arg864_1, arg865_1, arg866_1, arg867_1, arg868_1, arg869_1, arg870_1, arg871_1, arg872_1, arg873_1, arg874_1, arg875_1, arg876_1, arg877_1, arg878_1, arg879_1, arg880_1, arg881_1, arg882_1, arg883_1, arg884_1, arg885_1, arg886_1, arg887_1, arg888_1, arg889_1, arg890_1, arg891_1, arg892_1, arg893_1, arg894_1, arg895_1, arg896_1, arg897_1, arg898_1, arg899_1, arg900_1, arg901_1, arg902_1, arg903_1, arg904_1], [full_default, full_default_1, full_default_2, full_default_3, full_default_4, full_default_5, full_default_6, full_default_7, full_default_8, full_default_9, full_default_10, full_default_11, full_default_12, full_default_13, full_default_14, full_default_15, full_default_16, full_default_17, full_default_18, full_default_19, full_default_20, full_default_21, full_default_22, full_default_23, full_default_24, full_default_25, full_default_26, full_default_27, full_default_28, full_default_29, full_default_30, full_default_31, full_default_32, full_default_33, full_default_34, full_default_35, full_default_36, full_default_37, full_default_38, full_default_39, full_default_40, full_default_41, full_default_42, full_default_43, full_default_44, full_default_45, full_default_46, full_default_47, full_default_48, full_default_49, full_default_50, full_default_51, full_default_52, full_default_53, full_default_54, full_default_55, full_default_56, full_default_57, full_default_58, full_default_59, full_default_60, full_default_61, full_default_62, full_default_63, full_default_64, full_default_65, full_default_66, full_default_67, full_default_68, full_default_69, full_default_70, full_default_71, full_default_72, full_default_73, full_default_74, full_default_75, full_default_76, full_default_77, full_default_78, full_default_79, full_default_80, full_default_81, full_default_82, full_default_83, full_default_84, full_default_85, full_default_86, full_default_87, full_default_88, full_default_89, full_default_90, full_default_91, full_default_92, full_default_93, full_default_94, full_default_95, full_default_96, full_default_97, full_default_98, full_default_99, full_default_100, full_default_101, full_default_102, full_default_103, full_default_104, full_default_105, full_default_106, full_default_107, full_default_108, full_default_109, full_default_110, full_default_111, full_default_112, full_default_113, full_default_114, full_default_115, full_default_116, full_default_117, full_default_118, full_default_119, full_default_120, full_default_121, full_default_122, full_default_123, full_default_124, full_default_125, full_default_126, full_default_127, full_default_128, full_default_129, full_default_130, full_default_131, full_default_132, full_default_133, full_default_134, full_default_135, full_default_136, full_default_137, full_default_138, full_default_139, full_default_140, full_default_141, full_default_142, full_default_143, full_default_144, full_default_145, full_default_146, full_default_147, full_default_148, full_default_149, full_default_150, full_default_151, full_default_152, full_default_153, full_default_154, full_default_155, full_default_156, full_default_157, full_default_158, full_default_159, full_default_160, full_default_161, full_default_162, full_default_163, full_default_164, full_default_165, full_default_166, full_default_167, full_default_168, full_default_169, full_default_170, full_default_171, full_default_172, full_default_173, full_default_174, full_default_175, full_default_176, full_default_177, full_default_178, full_default_179, full_default_180, full_default_181, full_default_182, full_default_183, full_default_184, full_default_185, full_default_186, full_default_187, full_default_188, full_default_189, full_default_190, full_default_191, full_default_192, full_default_193, full_default_194, full_default_195, full_default_196, full_default_197, full_default_198, full_default_199, full_default_200, full_default_201, full_default_202, full_default_203, full_default_204, full_default_205, full_default_206, full_default_207, full_default_208, full_default_209, full_default_210, full_default_211, full_default_212, full_default_213, full_default_214, full_default_215, full_default_216, full_default_217, full_default_218, full_default_219, full_default_220, full_default_221, full_default_222, full_default_223, full_default_224, full_default_225, full_default_226, full_default_227, full_default_228, full_default_229, full_default_230, full_default_231, full_default_232, full_default_233, full_default_234, full_default_235, full_default_236, full_default_237, full_default_238, full_default_239, full_default_240, full_default_241, full_default_242, full_default_243, full_default_244, full_default_245, full_default_246, full_default_247, full_default_248, full_default_249, full_default_250, full_default_251, full_default_252, full_default_253, full_default_254, full_default_255, full_default_256, full_default_257, full_default_258, full_default_259, full_default_260, full_default_261, full_default_262, full_default_263, full_default_264, full_default_265, full_default_266, full_default_267, full_default_268, full_default_269, full_default_270, full_default_271, full_default_272, full_default_273, full_default_274, full_default_275, full_default_276, full_default_277, full_default_278, full_default_279, full_default_280, full_default_281, full_default_282, full_default_283, full_default_284, full_default_285, full_default_286, full_default_287, full_default_288, full_default_289, full_default_290, full_default_291, full_default_292, full_default_293, full_default_294, full_default_295, full_default_296, full_default_297, full_default_298, full_default_299, full_default_300], [getitem_301, getitem_302, getitem_303, getitem_304, getitem_305, getitem_306, getitem_307, getitem_308, getitem_309, getitem_310, getitem_311, getitem_312, getitem_313, getitem_314, getitem_315, getitem_316, getitem_317, getitem_318, getitem_319, getitem_320, getitem_321, getitem_322, getitem_323, getitem_324, getitem_325, getitem_326, getitem_327, getitem_328, getitem_329, getitem_330, getitem_331, getitem_332, getitem_333, getitem_334, getitem_335, getitem_336, getitem_337, getitem_338, getitem_339, getitem_340, getitem_341, getitem_342, getitem_343, getitem_344, getitem_345, getitem_346, getitem_347, getitem_348, getitem_349, getitem_350, getitem_351, getitem_352, getitem_353, getitem_354, getitem_355, getitem_356, getitem_357, getitem_358, getitem_359, getitem_360, getitem_361, getitem_362, getitem_363, getitem_364, getitem_365, getitem_366, getitem_367, getitem_368, getitem_369, getitem_370, getitem_371, getitem_372, getitem_373, getitem_374, getitem_375, getitem_376, getitem_377, getitem_378, getitem_379, getitem_380, getitem_381, getitem_382, getitem_383, getitem_384, getitem_385, getitem_386, getitem_387, getitem_388, getitem_389, getitem_390, getitem_391, getitem_392, getitem_393, getitem_394, getitem_395, getitem_396, getitem_397, getitem_398, getitem_399, getitem_400, getitem_401, getitem_402, getitem_403, getitem_404, getitem_405, getitem_406, getitem_407, getitem_408, getitem_409, getitem_410, getitem_411, getitem_412, getitem_413, getitem_414, getitem_415, getitem_416, getitem_417, getitem_418, getitem_419, getitem_420, getitem_421, getitem_422, getitem_423, getitem_424, getitem_425, getitem_426, getitem_427, getitem_428, getitem_429, getitem_430, getitem_431, getitem_432, getitem_433, getitem_434, getitem_435, getitem_436, getitem_437, getitem_438, getitem_439, getitem_440, getitem_441, getitem_442, getitem_443, getitem_444, getitem_445, getitem_446, getitem_447, getitem_448, getitem_449, getitem_450, getitem_451, getitem_452, getitem_453, getitem_454, getitem_455, getitem_456, getitem_457, getitem_458, getitem_459, getitem_460, getitem_461, getitem_462, getitem_463, getitem_464, getitem_465, getitem_466, getitem_467, getitem_468, getitem_469, getitem_470, getitem_471, getitem_472, getitem_473, getitem_474, getitem_475, getitem_476, getitem_477, getitem_478, getitem_479, getitem_480, getitem_481, getitem_482, getitem_483, getitem_484, getitem_485, getitem_486, getitem_487, getitem_488, getitem_489, getitem_490, getitem_491, getitem_492, getitem_493, getitem_494, getitem_495, getitem_496, getitem_497, getitem_498, getitem_499, getitem_500, getitem_501, getitem_502, getitem_503, getitem_504, getitem_505, getitem_506, getitem_507, getitem_508, getitem_509, getitem_510, getitem_511, getitem_512, getitem_513, getitem_514, getitem_515, getitem_516, getitem_517, getitem_518, getitem_519, getitem_520, getitem_521, getitem_522, getitem_523, getitem_524, getitem_525, getitem_526, getitem_527, getitem_528, getitem_529, getitem_530, getitem_531, getitem_532, getitem_533, getitem_534, getitem_535, getitem_536, getitem_537, getitem_538, getitem_539, getitem_540, getitem_541, getitem_542, getitem_543, getitem_544, getitem_545, getitem_546, getitem_547, getitem_548, getitem_549, getitem_550, getitem_551, getitem_552, getitem_553, getitem_554, getitem_555, getitem_556, getitem_557, getitem_558, getitem_559, getitem_560, getitem_561, getitem_562, getitem_563, getitem_564, getitem_565, getitem_566, getitem_567, getitem_568, getitem_569, getitem_570, getitem_571, getitem_572, getitem_573, getitem_574, getitem_575, getitem_576, getitem_577, getitem_578, getitem_579, getitem_580, getitem_581, getitem_582, getitem_583, getitem_584, getitem_585, getitem_586, getitem_587, getitem_588, getitem_589, getitem_590, getitem_591, getitem_592, getitem_593, getitem_594, getitem_595, getitem_596, getitem_597, getitem_598, getitem_599, getitem_600, getitem_601]);  arg304_1 = arg302_1 = arg606_1 = arg607_1 = arg608_1 = arg609_1 = arg610_1 = arg611_1 = arg612_1 = arg613_1 = arg614_1 = arg615_1 = arg616_1 = arg617_1 = arg618_1 = arg619_1 = arg620_1 = arg621_1 = arg622_1 = arg623_1 = arg624_1 = arg625_1 = arg626_1 = arg627_1 = arg628_1 = arg629_1 = arg630_1 = arg631_1 = arg632_1 = arg633_1 = arg634_1 = arg635_1 = arg636_1 = arg637_1 = arg638_1 = arg639_1 = arg640_1 = arg641_1 = arg642_1 = arg643_1 = arg644_1 = arg645_1 = arg646_1 = arg647_1 = arg648_1 = arg649_1 = arg650_1 = arg651_1 = arg652_1 = arg653_1 = arg654_1 = arg655_1 = arg656_1 = arg657_1 = arg658_1 = arg659_1 = arg660_1 = arg661_1 = arg662_1 = arg663_1 = arg664_1 = arg665_1 = arg666_1 = arg667_1 = arg668_1 = arg669_1 = arg670_1 = arg671_1 = arg672_1 = arg673_1 = arg674_1 = arg675_1 = arg676_1 = arg677_1 = arg678_1 = arg679_1 = arg680_1 = arg681_1 = arg682_1 = arg683_1 = arg684_1 = arg685_1 = arg686_1 = arg687_1 = arg688_1 = arg689_1 = arg690_1 = arg691_1 = arg692_1 = arg693_1 = arg694_1 = arg695_1 = arg696_1 = arg697_1 = arg698_1 = arg699_1 = arg700_1 = arg701_1 = arg702_1 = arg703_1 = arg704_1 = arg705_1 = arg706_1 = arg707_1 = arg708_1 = arg709_1 = arg710_1 = arg711_1 = arg712_1 = arg713_1 = arg714_1 = arg715_1 = arg716_1 = arg717_1 = arg718_1 = arg719_1 = arg720_1 = arg721_1 = arg722_1 = arg723_1 = arg724_1 = arg725_1 = arg726_1 = arg727_1 = arg728_1 = arg729_1 = arg730_1 = arg731_1 = arg732_1 = arg733_1 = arg734_1 = arg735_1 = arg736_1 = arg737_1 = arg738_1 = arg739_1 = arg740_1 = arg741_1 = arg742_1 = arg743_1 = arg744_1 = arg745_1 = arg746_1 = arg747_1 = arg748_1 = arg749_1 = arg750_1 = arg751_1 = arg752_1 = arg753_1 = arg754_1 = arg755_1 = arg756_1 = arg757_1 = arg758_1 = arg759_1 = arg760_1 = arg761_1 = arg762_1 = arg763_1 = arg764_1 = arg765_1 = arg766_1 = arg767_1 = arg768_1 = arg769_1 = arg770_1 = arg771_1 = arg772_1 = arg773_1 = arg774_1 = arg775_1 = arg776_1 = arg777_1 = arg778_1 = arg779_1 = arg780_1 = arg781_1 = arg782_1 = arg783_1 = arg784_1 = arg785_1 = arg786_1 = arg787_1 = arg788_1 = arg789_1 = arg790_1 = arg791_1 = arg792_1 = arg793_1 = arg794_1 = arg795_1 = arg796_1 = arg797_1 = arg798_1 = arg799_1 = arg800_1 = arg801_1 = arg802_1 = arg803_1 = arg804_1 = arg805_1 = arg806_1 = arg807_1 = arg808_1 = arg809_1 = arg810_1 = arg811_1 = arg812_1 = arg813_1 = arg814_1 = arg815_1 = arg816_1 = arg817_1 = arg818_1 = arg819_1 = arg820_1 = arg821_1 = arg822_1 = arg823_1 = arg824_1 = arg825_1 = arg826_1 = arg827_1 = arg828_1 = arg829_1 = arg830_1 = arg831_1 = arg832_1 = arg833_1 = arg834_1 = arg835_1 = arg836_1 = arg837_1 = arg838_1 = arg839_1 = arg840_1 = arg841_1 = arg842_1 = arg843_1 = arg844_1 = arg845_1 = arg846_1 = arg847_1 = arg848_1 = arg849_1 = arg850_1 = arg851_1 = arg852_1 = arg853_1 = arg854_1 = arg855_1 = arg856_1 = arg857_1 = arg858_1 = arg859_1 = arg860_1 = arg861_1 = arg862_1 = arg863_1 = arg864_1 = arg865_1 = arg866_1 = arg867_1 = arg868_1 = arg869_1 = arg870_1 = arg871_1 = arg872_1 = arg873_1 = arg874_1 = arg875_1 = arg876_1 = arg877_1 = arg878_1 = arg879_1 = arg880_1 = arg881_1 = arg882_1 = arg883_1 = arg884_1 = arg885_1 = arg886_1 = arg887_1 = arg888_1 = arg889_1 = arg890_1 = arg891_1 = arg892_1 = arg893_1 = arg894_1 = arg895_1 = arg896_1 = arg897_1 = arg898_1 = arg899_1 = arg900_1 = arg901_1 = arg902_1 = arg903_1 = arg904_1 = full_default = full_default_1 = full_default_2 = full_default_3 = full_default_4 = full_default_5 = full_default_6 = full_default_7 = full_default_8 = full_default_9 = full_default_10 = full_default_11 = full_default_12 = full_default_13 = full_default_14 = full_default_15 = full_default_16 = full_default_17 = full_default_18 = full_default_19 = full_default_20 = full_default_21 = full_default_22 = full_default_23 = full_default_24 = full_default_25 = full_default_26 = full_default_27 = full_default_28 = full_default_29 = full_default_30 = full_default_31 = full_default_32 = full_default_33 = full_default_34 = full_default_35 = full_default_36 = full_default_37 = full_default_38 = full_default_39 = full_default_40 = full_default_41 = full_default_42 = full_default_43 = full_default_44 = full_default_45 = full_default_46 = full_default_47 = full_default_48 = full_default_49 = full_default_50 = full_default_51 = full_default_52 = full_default_53 = full_default_54 = full_default_55 = full_default_56 = full_default_57 = full_default_58 = full_default_59 = full_default_60 = full_default_61 = full_default_62 = full_default_63 = full_default_64 = full_default_65 = full_default_66 = full_default_67 = full_default_68 = full_default_69 = full_default_70 = full_default_71 = full_default_72 = full_default_73 = full_default_74 = full_default_75 = full_default_76 = full_default_77 = full_default_78 = full_default_79 = full_default_80 = full_default_81 = full_default_82 = full_default_83 = full_default_84 = full_default_85 = full_default_86 = full_default_87 = full_default_88 = full_default_89 = full_default_90 = full_default_91 = full_default_92 = full_default_93 = full_default_94 = full_default_95 = full_default_96 = full_default_97 = full_default_98 = full_default_99 = full_default_100 = full_default_101 = full_default_102 = full_default_103 = full_default_104 = full_default_105 = full_default_106 = full_default_107 = full_default_108 = full_default_109 = full_default_110 = full_default_111 = full_default_112 = full_default_113 = full_default_114 = full_default_115 = full_default_116 = full_default_117 = full_default_118 = full_default_119 = full_default_120 = full_default_121 = full_default_122 = full_default_123 = full_default_124 = full_default_125 = full_default_126 = full_default_127 = full_default_128 = full_default_129 = full_default_130 = full_default_131 = full_default_132 = full_default_133 = full_default_134 = full_default_135 = full_default_136 = full_default_137 = full_default_138 = full_default_139 = full_default_140 = full_default_141 = full_default_142 = full_default_143 = full_default_144 = full_default_145 = full_default_146 = full_default_147 = full_default_148 = full_default_149 = full_default_150 = full_default_151 = full_default_152 = full_default_153 = full_default_154 = full_default_155 = full_default_156 = full_default_157 = full_default_158 = full_default_159 = full_default_160 = full_default_161 = full_default_162 = full_default_163 = full_default_164 = full_default_165 = full_default_166 = full_default_167 = full_default_168 = full_default_169 = full_default_170 = full_default_171 = full_default_172 = full_default_173 = full_default_174 = full_default_175 = full_default_176 = full_default_177 = full_default_178 = full_default_179 = full_default_180 = full_default_181 = full_default_182 = full_default_183 = full_default_184 = full_default_185 = full_default_186 = full_default_187 = full_default_188 = full_default_189 = full_default_190 = full_default_191 = full_default_192 = full_default_193 = full_default_194 = full_default_195 = full_default_196 = full_default_197 = full_default_198 = full_default_199 = full_default_200 = full_default_201 = full_default_202 = full_default_203 = full_default_204 = full_default_205 = full_default_206 = full_default_207 = full_default_208 = full_default_209 = full_default_210 = full_default_211 = full_default_212 = full_default_213 = full_default_214 = full_default_215 = full_default_216 = full_default_217 = full_default_218 = full_default_219 = full_default_220 = full_default_221 = full_default_222 = full_default_223 = full_default_224 = full_default_225 = full_default_226 = full_default_227 = full_default_228 = full_default_229 = full_default_230 = full_default_231 = full_default_232 = full_default_233 = full_default_234 = full_default_235 = full_default_236 = full_default_237 = full_default_238 = full_default_239 = full_default_240 = full_default_241 = full_default_242 = full_default_243 = full_default_244 = full_default_245 = full_default_246 = full_default_247 = full_default_248 = full_default_249 = full_default_250 = full_default_251 = full_default_252 = full_default_253 = full_default_254 = full_default_255 = full_default_256 = full_default_257 = full_default_258 = full_default_259 = full_default_260 = full_default_261 = full_default_262 = full_default_263 = full_default_264 = full_default_265 = full_default_266 = full_default_267 = full_default_268 = full_default_269 = full_default_270 = full_default_271 = full_default_272 = full_default_273 = full_default_274 = full_default_275 = full_default_276 = full_default_277 = full_default_278 = full_default_279 = full_default_280 = full_default_281 = full_default_282 = full_default_283 = full_default_284 = full_default_285 = full_default_286 = full_default_287 = full_default_288 = full_default_289 = full_default_290 = full_default_291 = full_default_292 = full_default_293 = full_default_294 = full_default_295 = full_default_296 = full_default_297 = full_default_298 = full_default_299 = full_default_300 = getitem_301 = getitem_302 = getitem_303 = getitem_304 = getitem_305 = getitem_306 = getitem_307 = getitem_308 = getitem_309 = getitem_310 = getitem_311 = getitem_312 = getitem_313 = getitem_314 = getitem_315 = getitem_316 = getitem_317 = getitem_318 = getitem_319 = getitem_320 = getitem_321 = getitem_322 = getitem_323 = getitem_324 = getitem_325 = getitem_326 = getitem_327 = getitem_328 = getitem_329 = getitem_330 = getitem_331 = getitem_332 = getitem_333 = getitem_334 = getitem_335 = getitem_336 = getitem_337 = getitem_338 = getitem_339 = getitem_340 = getitem_341 = getitem_342 = getitem_343 = getitem_344 = getitem_345 = getitem_346 = getitem_347 = getitem_348 = getitem_349 = getitem_350 = getitem_351 = getitem_352 = getitem_353 = getitem_354 = getitem_355 = getitem_356 = getitem_357 = getitem_358 = getitem_359 = getitem_360 = getitem_361 = getitem_362 = getitem_363 = getitem_364 = getitem_365 = getitem_366 = getitem_367 = getitem_368 = getitem_369 = getitem_370 = getitem_371 = getitem_372 = getitem_373 = getitem_374 = getitem_375 = getitem_376 = getitem_377 = getitem_378 = getitem_379 = getitem_380 = getitem_381 = getitem_382 = getitem_383 = getitem_384 = getitem_385 = getitem_386 = getitem_387 = getitem_388 = getitem_389 = getitem_390 = getitem_391 = getitem_392 = getitem_393 = getitem_394 = getitem_395 = getitem_396 = getitem_397 = getitem_398 = getitem_399 = getitem_400 = getitem_401 = getitem_402 = getitem_403 = getitem_404 = getitem_405 = getitem_406 = getitem_407 = getitem_408 = getitem_409 = getitem_410 = getitem_411 = getitem_412 = getitem_413 = getitem_414 = getitem_415 = getitem_416 = getitem_417 = getitem_418 = getitem_419 = getitem_420 = getitem_421 = getitem_422 = getitem_423 = getitem_424 = getitem_425 = getitem_426 = getitem_427 = getitem_428 = getitem_429 = getitem_430 = getitem_431 = getitem_432 = getitem_433 = getitem_434 = getitem_435 = getitem_436 = getitem_437 = getitem_438 = getitem_439 = getitem_440 = getitem_441 = getitem_442 = getitem_443 = getitem_444 = getitem_445 = getitem_446 = getitem_447 = getitem_448 = getitem_449 = getitem_450 = getitem_451 = getitem_452 = getitem_453 = getitem_454 = getitem_455 = getitem_456 = getitem_457 = getitem_458 = getitem_459 = getitem_460 = getitem_461 = getitem_462 = getitem_463 = getitem_464 = getitem_465 = getitem_466 = getitem_467 = getitem_468 = getitem_469 = getitem_470 = getitem_471 = getitem_472 = getitem_473 = getitem_474 = getitem_475 = getitem_476 = getitem_477 = getitem_478 = getitem_479 = getitem_480 = getitem_481 = getitem_482 = getitem_483 = getitem_484 = getitem_485 = getitem_486 = getitem_487 = getitem_488 = getitem_489 = getitem_490 = getitem_491 = getitem_492 = getitem_493 = getitem_494 = getitem_495 = getitem_496 = getitem_497 = getitem_498 = getitem_499 = getitem_500 = getitem_501 = getitem_502 = getitem_503 = getitem_504 = getitem_505 = getitem_506 = getitem_507 = getitem_508 = getitem_509 = getitem_510 = getitem_511 = getitem_512 = getitem_513 = getitem_514 = getitem_515 = getitem_516 = getitem_517 = getitem_518 = getitem_519 = getitem_520 = getitem_521 = getitem_522 = getitem_523 = getitem_524 = getitem_525 = getitem_526 = getitem_527 = getitem_528 = getitem_529 = getitem_530 = getitem_531 = getitem_532 = getitem_533 = getitem_534 = getitem_535 = getitem_536 = getitem_537 = getitem_538 = getitem_539 = getitem_540 = getitem_541 = getitem_542 = getitem_543 = getitem_544 = getitem_545 = getitem_546 = getitem_547 = getitem_548 = getitem_549 = getitem_550 = getitem_551 = getitem_552 = getitem_553 = getitem_554 = getitem_555 = getitem_556 = getitem_557 = getitem_558 = getitem_559 = getitem_560 = getitem_561 = getitem_562 = getitem_563 = getitem_564 = getitem_565 = getitem_566 = getitem_567 = getitem_568 = getitem_569 = getitem_570 = getitem_571 = getitem_572 = getitem_573 = getitem_574 = getitem_575 = getitem_576 = getitem_577 = getitem_578 = getitem_579 = getitem_580 = getitem_581 = getitem_582 = getitem_583 = getitem_584 = getitem_585 = getitem_586 = getitem_587 = getitem_588 = getitem_589 = getitem_590 = getitem_591 = getitem_592 = getitem_593 = getitem_594 = getitem_595 = getitem_596 = getitem_597 = getitem_598 = getitem_599 = getitem_600 = getitem_601 = None
        getitem: "f32[768]" = _foreach_addcmul_scalar[0]
        getitem_602: "f32[50, 768]" = _foreach_addcmul_scalar[1]
        getitem_603: "f32[768, 512]" = _foreach_addcmul_scalar[2]
        getitem_604: "f32[768, 3, 32, 32]" = _foreach_addcmul_scalar[3]
        getitem_605: "f32[768]" = _foreach_addcmul_scalar[4]
        getitem_606: "f32[768]" = _foreach_addcmul_scalar[5]
        getitem_607: "f32[2304, 768]" = _foreach_addcmul_scalar[6]
        getitem_608: "f32[2304]" = _foreach_addcmul_scalar[7]
        getitem_609: "f32[768, 768]" = _foreach_addcmul_scalar[8]
        getitem_610: "f32[768]" = _foreach_addcmul_scalar[9]
        getitem_611: "f32[3072, 768]" = _foreach_addcmul_scalar[10]
        getitem_612: "f32[3072]" = _foreach_addcmul_scalar[11]
        getitem_613: "f32[768, 3072]" = _foreach_addcmul_scalar[12]
        getitem_614: "f32[768]" = _foreach_addcmul_scalar[13]
        getitem_615: "f32[768]" = _foreach_addcmul_scalar[14]
        getitem_616: "f32[768]" = _foreach_addcmul_scalar[15]
        getitem_617: "f32[768]" = _foreach_addcmul_scalar[16]
        getitem_618: "f32[768]" = _foreach_addcmul_scalar[17]
        getitem_619: "f32[2304, 768]" = _foreach_addcmul_scalar[18]
        getitem_620: "f32[2304]" = _foreach_addcmul_scalar[19]
        getitem_621: "f32[768, 768]" = _foreach_addcmul_scalar[20]
        getitem_622: "f32[768]" = _foreach_addcmul_scalar[21]
        getitem_623: "f32[3072, 768]" = _foreach_addcmul_scalar[22]
        getitem_624: "f32[3072]" = _foreach_addcmul_scalar[23]
        getitem_625: "f32[768, 3072]" = _foreach_addcmul_scalar[24]
        getitem_626: "f32[768]" = _foreach_addcmul_scalar[25]
        getitem_627: "f32[768]" = _foreach_addcmul_scalar[26]
        getitem_628: "f32[768]" = _foreach_addcmul_scalar[27]
        getitem_629: "f32[768]" = _foreach_addcmul_scalar[28]
        getitem_630: "f32[768]" = _foreach_addcmul_scalar[29]
        getitem_631: "f32[2304, 768]" = _foreach_addcmul_scalar[30]
        getitem_632: "f32[2304]" = _foreach_addcmul_scalar[31]
        getitem_633: "f32[768, 768]" = _foreach_addcmul_scalar[32]
        getitem_634: "f32[768]" = _foreach_addcmul_scalar[33]
        getitem_635: "f32[3072, 768]" = _foreach_addcmul_scalar[34]
        getitem_636: "f32[3072]" = _foreach_addcmul_scalar[35]
        getitem_637: "f32[768, 3072]" = _foreach_addcmul_scalar[36]
        getitem_638: "f32[768]" = _foreach_addcmul_scalar[37]
        getitem_639: "f32[768]" = _foreach_addcmul_scalar[38]
        getitem_640: "f32[768]" = _foreach_addcmul_scalar[39]
        getitem_641: "f32[768]" = _foreach_addcmul_scalar[40]
        getitem_642: "f32[768]" = _foreach_addcmul_scalar[41]
        getitem_643: "f32[2304, 768]" = _foreach_addcmul_scalar[42]
        getitem_644: "f32[2304]" = _foreach_addcmul_scalar[43]
        getitem_645: "f32[768, 768]" = _foreach_addcmul_scalar[44]
        getitem_646: "f32[768]" = _foreach_addcmul_scalar[45]
        getitem_647: "f32[3072, 768]" = _foreach_addcmul_scalar[46]
        getitem_648: "f32[3072]" = _foreach_addcmul_scalar[47]
        getitem_649: "f32[768, 3072]" = _foreach_addcmul_scalar[48]
        getitem_650: "f32[768]" = _foreach_addcmul_scalar[49]
        getitem_651: "f32[768]" = _foreach_addcmul_scalar[50]
        getitem_652: "f32[768]" = _foreach_addcmul_scalar[51]
        getitem_653: "f32[768]" = _foreach_addcmul_scalar[52]
        getitem_654: "f32[768]" = _foreach_addcmul_scalar[53]
        getitem_655: "f32[2304, 768]" = _foreach_addcmul_scalar[54]
        getitem_656: "f32[2304]" = _foreach_addcmul_scalar[55]
        getitem_657: "f32[768, 768]" = _foreach_addcmul_scalar[56]
        getitem_658: "f32[768]" = _foreach_addcmul_scalar[57]
        getitem_659: "f32[3072, 768]" = _foreach_addcmul_scalar[58]
        getitem_660: "f32[3072]" = _foreach_addcmul_scalar[59]
        getitem_661: "f32[768, 3072]" = _foreach_addcmul_scalar[60]
        getitem_662: "f32[768]" = _foreach_addcmul_scalar[61]
        getitem_663: "f32[768]" = _foreach_addcmul_scalar[62]
        getitem_664: "f32[768]" = _foreach_addcmul_scalar[63]
        getitem_665: "f32[768]" = _foreach_addcmul_scalar[64]
        getitem_666: "f32[768]" = _foreach_addcmul_scalar[65]
        getitem_667: "f32[2304, 768]" = _foreach_addcmul_scalar[66]
        getitem_668: "f32[2304]" = _foreach_addcmul_scalar[67]
        getitem_669: "f32[768, 768]" = _foreach_addcmul_scalar[68]
        getitem_670: "f32[768]" = _foreach_addcmul_scalar[69]
        getitem_671: "f32[3072, 768]" = _foreach_addcmul_scalar[70]
        getitem_672: "f32[3072]" = _foreach_addcmul_scalar[71]
        getitem_673: "f32[768, 3072]" = _foreach_addcmul_scalar[72]
        getitem_674: "f32[768]" = _foreach_addcmul_scalar[73]
        getitem_675: "f32[768]" = _foreach_addcmul_scalar[74]
        getitem_676: "f32[768]" = _foreach_addcmul_scalar[75]
        getitem_677: "f32[768]" = _foreach_addcmul_scalar[76]
        getitem_678: "f32[768]" = _foreach_addcmul_scalar[77]
        getitem_679: "f32[2304, 768]" = _foreach_addcmul_scalar[78]
        getitem_680: "f32[2304]" = _foreach_addcmul_scalar[79]
        getitem_681: "f32[768, 768]" = _foreach_addcmul_scalar[80]
        getitem_682: "f32[768]" = _foreach_addcmul_scalar[81]
        getitem_683: "f32[3072, 768]" = _foreach_addcmul_scalar[82]
        getitem_684: "f32[3072]" = _foreach_addcmul_scalar[83]
        getitem_685: "f32[768, 3072]" = _foreach_addcmul_scalar[84]
        getitem_686: "f32[768]" = _foreach_addcmul_scalar[85]
        getitem_687: "f32[768]" = _foreach_addcmul_scalar[86]
        getitem_688: "f32[768]" = _foreach_addcmul_scalar[87]
        getitem_689: "f32[768]" = _foreach_addcmul_scalar[88]
        getitem_690: "f32[768]" = _foreach_addcmul_scalar[89]
        getitem_691: "f32[2304, 768]" = _foreach_addcmul_scalar[90]
        getitem_692: "f32[2304]" = _foreach_addcmul_scalar[91]
        getitem_693: "f32[768, 768]" = _foreach_addcmul_scalar[92]
        getitem_694: "f32[768]" = _foreach_addcmul_scalar[93]
        getitem_695: "f32[3072, 768]" = _foreach_addcmul_scalar[94]
        getitem_696: "f32[3072]" = _foreach_addcmul_scalar[95]
        getitem_697: "f32[768, 3072]" = _foreach_addcmul_scalar[96]
        getitem_698: "f32[768]" = _foreach_addcmul_scalar[97]
        getitem_699: "f32[768]" = _foreach_addcmul_scalar[98]
        getitem_700: "f32[768]" = _foreach_addcmul_scalar[99]
        getitem_701: "f32[768]" = _foreach_addcmul_scalar[100]
        getitem_702: "f32[768]" = _foreach_addcmul_scalar[101]
        getitem_703: "f32[2304, 768]" = _foreach_addcmul_scalar[102]
        getitem_704: "f32[2304]" = _foreach_addcmul_scalar[103]
        getitem_705: "f32[768, 768]" = _foreach_addcmul_scalar[104]
        getitem_706: "f32[768]" = _foreach_addcmul_scalar[105]
        getitem_707: "f32[3072, 768]" = _foreach_addcmul_scalar[106]
        getitem_708: "f32[3072]" = _foreach_addcmul_scalar[107]
        getitem_709: "f32[768, 3072]" = _foreach_addcmul_scalar[108]
        getitem_710: "f32[768]" = _foreach_addcmul_scalar[109]
        getitem_711: "f32[768]" = _foreach_addcmul_scalar[110]
        getitem_712: "f32[768]" = _foreach_addcmul_scalar[111]
        getitem_713: "f32[768]" = _foreach_addcmul_scalar[112]
        getitem_714: "f32[768]" = _foreach_addcmul_scalar[113]
        getitem_715: "f32[2304, 768]" = _foreach_addcmul_scalar[114]
        getitem_716: "f32[2304]" = _foreach_addcmul_scalar[115]
        getitem_717: "f32[768, 768]" = _foreach_addcmul_scalar[116]
        getitem_718: "f32[768]" = _foreach_addcmul_scalar[117]
        getitem_719: "f32[3072, 768]" = _foreach_addcmul_scalar[118]
        getitem_720: "f32[3072]" = _foreach_addcmul_scalar[119]
        getitem_721: "f32[768, 3072]" = _foreach_addcmul_scalar[120]
        getitem_722: "f32[768]" = _foreach_addcmul_scalar[121]
        getitem_723: "f32[768]" = _foreach_addcmul_scalar[122]
        getitem_724: "f32[768]" = _foreach_addcmul_scalar[123]
        getitem_725: "f32[768]" = _foreach_addcmul_scalar[124]
        getitem_726: "f32[768]" = _foreach_addcmul_scalar[125]
        getitem_727: "f32[2304, 768]" = _foreach_addcmul_scalar[126]
        getitem_728: "f32[2304]" = _foreach_addcmul_scalar[127]
        getitem_729: "f32[768, 768]" = _foreach_addcmul_scalar[128]
        getitem_730: "f32[768]" = _foreach_addcmul_scalar[129]
        getitem_731: "f32[3072, 768]" = _foreach_addcmul_scalar[130]
        getitem_732: "f32[3072]" = _foreach_addcmul_scalar[131]
        getitem_733: "f32[768, 3072]" = _foreach_addcmul_scalar[132]
        getitem_734: "f32[768]" = _foreach_addcmul_scalar[133]
        getitem_735: "f32[768]" = _foreach_addcmul_scalar[134]
        getitem_736: "f32[768]" = _foreach_addcmul_scalar[135]
        getitem_737: "f32[768]" = _foreach_addcmul_scalar[136]
        getitem_738: "f32[768]" = _foreach_addcmul_scalar[137]
        getitem_739: "f32[2304, 768]" = _foreach_addcmul_scalar[138]
        getitem_740: "f32[2304]" = _foreach_addcmul_scalar[139]
        getitem_741: "f32[768, 768]" = _foreach_addcmul_scalar[140]
        getitem_742: "f32[768]" = _foreach_addcmul_scalar[141]
        getitem_743: "f32[3072, 768]" = _foreach_addcmul_scalar[142]
        getitem_744: "f32[3072]" = _foreach_addcmul_scalar[143]
        getitem_745: "f32[768, 3072]" = _foreach_addcmul_scalar[144]
        getitem_746: "f32[768]" = _foreach_addcmul_scalar[145]
        getitem_747: "f32[768]" = _foreach_addcmul_scalar[146]
        getitem_748: "f32[768]" = _foreach_addcmul_scalar[147]
        getitem_749: "f32[768]" = _foreach_addcmul_scalar[148]
        getitem_750: "f32[768]" = _foreach_addcmul_scalar[149]
        getitem_751: "f32[768]" = _foreach_addcmul_scalar[150]
        getitem_752: "f32[768]" = _foreach_addcmul_scalar[151]
        getitem_753: "f32[77, 512]" = _foreach_addcmul_scalar[152]
        getitem_754: "f32[49408, 512]" = _foreach_addcmul_scalar[153]
        getitem_755: "f32[1536, 512]" = _foreach_addcmul_scalar[154]
        getitem_756: "f32[1536]" = _foreach_addcmul_scalar[155]
        getitem_757: "f32[512, 512]" = _foreach_addcmul_scalar[156]
        getitem_758: "f32[512]" = _foreach_addcmul_scalar[157]
        getitem_759: "f32[2048, 512]" = _foreach_addcmul_scalar[158]
        getitem_760: "f32[2048]" = _foreach_addcmul_scalar[159]
        getitem_761: "f32[512, 2048]" = _foreach_addcmul_scalar[160]
        getitem_762: "f32[512]" = _foreach_addcmul_scalar[161]
        getitem_763: "f32[512]" = _foreach_addcmul_scalar[162]
        getitem_764: "f32[512]" = _foreach_addcmul_scalar[163]
        getitem_765: "f32[512]" = _foreach_addcmul_scalar[164]
        getitem_766: "f32[512]" = _foreach_addcmul_scalar[165]
        getitem_767: "f32[1536, 512]" = _foreach_addcmul_scalar[166]
        getitem_768: "f32[1536]" = _foreach_addcmul_scalar[167]
        getitem_769: "f32[512, 512]" = _foreach_addcmul_scalar[168]
        getitem_770: "f32[512]" = _foreach_addcmul_scalar[169]
        getitem_771: "f32[2048, 512]" = _foreach_addcmul_scalar[170]
        getitem_772: "f32[2048]" = _foreach_addcmul_scalar[171]
        getitem_773: "f32[512, 2048]" = _foreach_addcmul_scalar[172]
        getitem_774: "f32[512]" = _foreach_addcmul_scalar[173]
        getitem_775: "f32[512]" = _foreach_addcmul_scalar[174]
        getitem_776: "f32[512]" = _foreach_addcmul_scalar[175]
        getitem_777: "f32[512]" = _foreach_addcmul_scalar[176]
        getitem_778: "f32[512]" = _foreach_addcmul_scalar[177]
        getitem_779: "f32[1536, 512]" = _foreach_addcmul_scalar[178]
        getitem_780: "f32[1536]" = _foreach_addcmul_scalar[179]
        getitem_781: "f32[512, 512]" = _foreach_addcmul_scalar[180]
        getitem_782: "f32[512]" = _foreach_addcmul_scalar[181]
        getitem_783: "f32[2048, 512]" = _foreach_addcmul_scalar[182]
        getitem_784: "f32[2048]" = _foreach_addcmul_scalar[183]
        getitem_785: "f32[512, 2048]" = _foreach_addcmul_scalar[184]
        getitem_786: "f32[512]" = _foreach_addcmul_scalar[185]
        getitem_787: "f32[512]" = _foreach_addcmul_scalar[186]
        getitem_788: "f32[512]" = _foreach_addcmul_scalar[187]
        getitem_789: "f32[512]" = _foreach_addcmul_scalar[188]
        getitem_790: "f32[512]" = _foreach_addcmul_scalar[189]
        getitem_791: "f32[1536, 512]" = _foreach_addcmul_scalar[190]
        getitem_792: "f32[1536]" = _foreach_addcmul_scalar[191]
        getitem_793: "f32[512, 512]" = _foreach_addcmul_scalar[192]
        getitem_794: "f32[512]" = _foreach_addcmul_scalar[193]
        getitem_795: "f32[2048, 512]" = _foreach_addcmul_scalar[194]
        getitem_796: "f32[2048]" = _foreach_addcmul_scalar[195]
        getitem_797: "f32[512, 2048]" = _foreach_addcmul_scalar[196]
        getitem_798: "f32[512]" = _foreach_addcmul_scalar[197]
        getitem_799: "f32[512]" = _foreach_addcmul_scalar[198]
        getitem_800: "f32[512]" = _foreach_addcmul_scalar[199]
        getitem_801: "f32[512]" = _foreach_addcmul_scalar[200]
        getitem_802: "f32[512]" = _foreach_addcmul_scalar[201]
        getitem_803: "f32[1536, 512]" = _foreach_addcmul_scalar[202]
        getitem_804: "f32[1536]" = _foreach_addcmul_scalar[203]
        getitem_805: "f32[512, 512]" = _foreach_addcmul_scalar[204]
        getitem_806: "f32[512]" = _foreach_addcmul_scalar[205]
        getitem_807: "f32[2048, 512]" = _foreach_addcmul_scalar[206]
        getitem_808: "f32[2048]" = _foreach_addcmul_scalar[207]
        getitem_809: "f32[512, 2048]" = _foreach_addcmul_scalar[208]
        getitem_810: "f32[512]" = _foreach_addcmul_scalar[209]
        getitem_811: "f32[512]" = _foreach_addcmul_scalar[210]
        getitem_812: "f32[512]" = _foreach_addcmul_scalar[211]
        getitem_813: "f32[512]" = _foreach_addcmul_scalar[212]
        getitem_814: "f32[512]" = _foreach_addcmul_scalar[213]
        getitem_815: "f32[1536, 512]" = _foreach_addcmul_scalar[214]
        getitem_816: "f32[1536]" = _foreach_addcmul_scalar[215]
        getitem_817: "f32[512, 512]" = _foreach_addcmul_scalar[216]
        getitem_818: "f32[512]" = _foreach_addcmul_scalar[217]
        getitem_819: "f32[2048, 512]" = _foreach_addcmul_scalar[218]
        getitem_820: "f32[2048]" = _foreach_addcmul_scalar[219]
        getitem_821: "f32[512, 2048]" = _foreach_addcmul_scalar[220]
        getitem_822: "f32[512]" = _foreach_addcmul_scalar[221]
        getitem_823: "f32[512]" = _foreach_addcmul_scalar[222]
        getitem_824: "f32[512]" = _foreach_addcmul_scalar[223]
        getitem_825: "f32[512]" = _foreach_addcmul_scalar[224]
        getitem_826: "f32[512]" = _foreach_addcmul_scalar[225]
        getitem_827: "f32[1536, 512]" = _foreach_addcmul_scalar[226]
        getitem_828: "f32[1536]" = _foreach_addcmul_scalar[227]
        getitem_829: "f32[512, 512]" = _foreach_addcmul_scalar[228]
        getitem_830: "f32[512]" = _foreach_addcmul_scalar[229]
        getitem_831: "f32[2048, 512]" = _foreach_addcmul_scalar[230]
        getitem_832: "f32[2048]" = _foreach_addcmul_scalar[231]
        getitem_833: "f32[512, 2048]" = _foreach_addcmul_scalar[232]
        getitem_834: "f32[512]" = _foreach_addcmul_scalar[233]
        getitem_835: "f32[512]" = _foreach_addcmul_scalar[234]
        getitem_836: "f32[512]" = _foreach_addcmul_scalar[235]
        getitem_837: "f32[512]" = _foreach_addcmul_scalar[236]
        getitem_838: "f32[512]" = _foreach_addcmul_scalar[237]
        getitem_839: "f32[1536, 512]" = _foreach_addcmul_scalar[238]
        getitem_840: "f32[1536]" = _foreach_addcmul_scalar[239]
        getitem_841: "f32[512, 512]" = _foreach_addcmul_scalar[240]
        getitem_842: "f32[512]" = _foreach_addcmul_scalar[241]
        getitem_843: "f32[2048, 512]" = _foreach_addcmul_scalar[242]
        getitem_844: "f32[2048]" = _foreach_addcmul_scalar[243]
        getitem_845: "f32[512, 2048]" = _foreach_addcmul_scalar[244]
        getitem_846: "f32[512]" = _foreach_addcmul_scalar[245]
        getitem_847: "f32[512]" = _foreach_addcmul_scalar[246]
        getitem_848: "f32[512]" = _foreach_addcmul_scalar[247]
        getitem_849: "f32[512]" = _foreach_addcmul_scalar[248]
        getitem_850: "f32[512]" = _foreach_addcmul_scalar[249]
        getitem_851: "f32[1536, 512]" = _foreach_addcmul_scalar[250]
        getitem_852: "f32[1536]" = _foreach_addcmul_scalar[251]
        getitem_853: "f32[512, 512]" = _foreach_addcmul_scalar[252]
        getitem_854: "f32[512]" = _foreach_addcmul_scalar[253]
        getitem_855: "f32[2048, 512]" = _foreach_addcmul_scalar[254]
        getitem_856: "f32[2048]" = _foreach_addcmul_scalar[255]
        getitem_857: "f32[512, 2048]" = _foreach_addcmul_scalar[256]
        getitem_858: "f32[512]" = _foreach_addcmul_scalar[257]
        getitem_859: "f32[512]" = _foreach_addcmul_scalar[258]
        getitem_860: "f32[512]" = _foreach_addcmul_scalar[259]
        getitem_861: "f32[512]" = _foreach_addcmul_scalar[260]
        getitem_862: "f32[512]" = _foreach_addcmul_scalar[261]
        getitem_863: "f32[1536, 512]" = _foreach_addcmul_scalar[262]
        getitem_864: "f32[1536]" = _foreach_addcmul_scalar[263]
        getitem_865: "f32[512, 512]" = _foreach_addcmul_scalar[264]
        getitem_866: "f32[512]" = _foreach_addcmul_scalar[265]
        getitem_867: "f32[2048, 512]" = _foreach_addcmul_scalar[266]
        getitem_868: "f32[2048]" = _foreach_addcmul_scalar[267]
        getitem_869: "f32[512, 2048]" = _foreach_addcmul_scalar[268]
        getitem_870: "f32[512]" = _foreach_addcmul_scalar[269]
        getitem_871: "f32[512]" = _foreach_addcmul_scalar[270]
        getitem_872: "f32[512]" = _foreach_addcmul_scalar[271]
        getitem_873: "f32[512]" = _foreach_addcmul_scalar[272]
        getitem_874: "f32[512]" = _foreach_addcmul_scalar[273]
        getitem_875: "f32[1536, 512]" = _foreach_addcmul_scalar[274]
        getitem_876: "f32[1536]" = _foreach_addcmul_scalar[275]
        getitem_877: "f32[512, 512]" = _foreach_addcmul_scalar[276]
        getitem_878: "f32[512]" = _foreach_addcmul_scalar[277]
        getitem_879: "f32[2048, 512]" = _foreach_addcmul_scalar[278]
        getitem_880: "f32[2048]" = _foreach_addcmul_scalar[279]
        getitem_881: "f32[512, 2048]" = _foreach_addcmul_scalar[280]
        getitem_882: "f32[512]" = _foreach_addcmul_scalar[281]
        getitem_883: "f32[512]" = _foreach_addcmul_scalar[282]
        getitem_884: "f32[512]" = _foreach_addcmul_scalar[283]
        getitem_885: "f32[512]" = _foreach_addcmul_scalar[284]
        getitem_886: "f32[512]" = _foreach_addcmul_scalar[285]
        getitem_887: "f32[1536, 512]" = _foreach_addcmul_scalar[286]
        getitem_888: "f32[1536]" = _foreach_addcmul_scalar[287]
        getitem_889: "f32[512, 512]" = _foreach_addcmul_scalar[288]
        getitem_890: "f32[512]" = _foreach_addcmul_scalar[289]
        getitem_891: "f32[2048, 512]" = _foreach_addcmul_scalar[290]
        getitem_892: "f32[2048]" = _foreach_addcmul_scalar[291]
        getitem_893: "f32[512, 2048]" = _foreach_addcmul_scalar[292]
        getitem_894: "f32[512]" = _foreach_addcmul_scalar[293]
        getitem_895: "f32[512]" = _foreach_addcmul_scalar[294]
        getitem_896: "f32[512]" = _foreach_addcmul_scalar[295]
        getitem_897: "f32[512]" = _foreach_addcmul_scalar[296]
        getitem_898: "f32[512]" = _foreach_addcmul_scalar[297]
        getitem_899: "f32[512]" = _foreach_addcmul_scalar[298]
        getitem_900: "f32[512]" = _foreach_addcmul_scalar[299]
        getitem_901: "f32[512, 512]" = _foreach_addcmul_scalar[300];  _foreach_addcmul_scalar = None
        _foreach_div_list = torch.ops.aten._foreach_div.List([getitem_4515, getitem_4516, getitem_4517, getitem_4518, getitem_4519, getitem_4520, getitem_4521, getitem_4522, getitem_4523, getitem_4524, getitem_4525, getitem_4526, getitem_4527, getitem_4528, getitem_4529, getitem_4530, getitem_4531, getitem_4532, getitem_4533, getitem_4534, getitem_4535, getitem_4536, getitem_4537, getitem_4538, getitem_4539, getitem_4540, getitem_4541, getitem_4542, getitem_4543, getitem_4544, getitem_4545, getitem_4546, getitem_4547, getitem_4548, getitem_4549, getitem_4550, getitem_4551, getitem_4552, getitem_4553, getitem_4554, getitem_4555, getitem_4556, getitem_4557, getitem_4558, getitem_4559, getitem_4560, getitem_4561, getitem_4562, getitem_4563, getitem_4564, getitem_4565, getitem_4566, getitem_4567, getitem_4568, getitem_4569, getitem_4570, getitem_4571, getitem_4572, getitem_4573, getitem_4574, getitem_4575, getitem_4576, getitem_4577, getitem_4578, getitem_4579, getitem_4580, getitem_4581, getitem_4582, getitem_4583, getitem_4584, getitem_4585, getitem_4586, getitem_4587, getitem_4588, getitem_4589, getitem_4590, getitem_4591, getitem_4592, getitem_4593, getitem_4594, getitem_4595, getitem_4596, getitem_4597, getitem_4598, getitem_4599, getitem_4600, getitem_4601, getitem_4602, getitem_4603, getitem_4604, getitem_4605, getitem_4606, getitem_4607, getitem_4608, getitem_4609, getitem_4610, getitem_4611, getitem_4612, getitem_4613, getitem_4614, getitem_4615, getitem_4616, getitem_4617, getitem_4618, getitem_4619, getitem_4620, getitem_4621, getitem_4622, getitem_4623, getitem_4624, getitem_4625, getitem_4626, getitem_4627, getitem_4628, getitem_4629, getitem_4630, getitem_4631, getitem_4632, getitem_4633, getitem_4634, getitem_4635, getitem_4636, getitem_4637, getitem_4638, getitem_4639, getitem_4640, getitem_4641, getitem_4642, getitem_4643, getitem_4644, getitem_4645, getitem_4646, getitem_4647, getitem_4648, getitem_4649, getitem_4650, getitem_4651, getitem_4652, getitem_4653, getitem_4654, getitem_4655, getitem_4656, getitem_4657, getitem_4658, getitem_4659, getitem_4660, getitem_4661, getitem_4662, getitem_4663, getitem_4664, getitem_4665, getitem_4666, getitem_4667, getitem_4668, getitem_4669, getitem_4670, getitem_4671, getitem_4672, getitem_4673, getitem_4674, getitem_4675, getitem_4676, getitem_4677, getitem_4678, getitem_4679, getitem_4680, getitem_4681, getitem_4682, getitem_4683, getitem_4684, getitem_4685, getitem_4686, getitem_4687, getitem_4688, getitem_4689, getitem_4690, getitem_4691, getitem_4692, getitem_4693, getitem_4694, getitem_4695, getitem_4696, getitem_4697, getitem_4698, getitem_4699, getitem_4700, getitem_4701, getitem_4702, getitem_4703, getitem_4704, getitem_4705, getitem_4706, getitem_4707, getitem_4708, getitem_4709, getitem_4710, getitem_4711, getitem_4712, getitem_4713, getitem_4714, getitem_4715, getitem_4716, getitem_4717, getitem_4718, getitem_4719, getitem_4720, getitem_4721, getitem_4722, getitem_4723, getitem_4724, getitem_4725, getitem_4726, getitem_4727, getitem_4728, getitem_4729, getitem_4730, getitem_4731, getitem_4732, getitem_4733, getitem_4734, getitem_4735, getitem_4736, getitem_4737, getitem_4738, getitem_4739, getitem_4740, getitem_4741, getitem_4742, getitem_4743, getitem_4744, getitem_4745, getitem_4746, getitem_4747, getitem_4748, getitem_4749, getitem_4750, getitem_4751, getitem_4752, getitem_4753, getitem_4754, getitem_4755, getitem_4756, getitem_4757, getitem_4758, getitem_4759, getitem_4760, getitem_4761, getitem_4762, getitem_4763, getitem_4764, getitem_4765, getitem_4766, getitem_4767, getitem_4768, getitem_4769, getitem_4770, getitem_4771, getitem_4772, getitem_4773, getitem_4774, getitem_4775, getitem_4776, getitem_4777, getitem_4778, getitem_4779, getitem_4780, getitem_4781, getitem_4782, getitem_4783, getitem_4784, getitem_4785, getitem_4786, getitem_4787, getitem_4788, getitem_4789, getitem_4790, getitem_4791, getitem_4792, getitem_4793, getitem_4794, getitem_4795, getitem_4796, getitem_4797, getitem_4798, getitem_4799, getitem_4800, getitem_4801, getitem_4802, getitem_4803, getitem_4804, getitem_4805, getitem_4806, getitem_4807, getitem_4808, getitem_4809, getitem_4810, getitem_4811, getitem_4812, getitem_4813, getitem_4814, getitem_4815], [getitem_3311, getitem_3312, getitem_3313, getitem_3314, getitem_3315, getitem_3316, getitem_3317, getitem_3318, getitem_3319, getitem_3320, getitem_3321, getitem_3322, getitem_3323, getitem_3324, getitem_3325, getitem_3326, getitem_3327, getitem_3328, getitem_3329, getitem_3330, getitem_3331, getitem_3332, getitem_3333, getitem_3334, getitem_3335, getitem_3336, getitem_3337, getitem_3338, getitem_3339, getitem_3340, getitem_3341, getitem_3342, getitem_3343, getitem_3344, getitem_3345, getitem_3346, getitem_3347, getitem_3348, getitem_3349, getitem_3350, getitem_3351, getitem_3352, getitem_3353, getitem_3354, getitem_3355, getitem_3356, getitem_3357, getitem_3358, getitem_3359, getitem_3360, getitem_3361, getitem_3362, getitem_3363, getitem_3364, getitem_3365, getitem_3366, getitem_3367, getitem_3368, getitem_3369, getitem_3370, getitem_3371, getitem_3372, getitem_3373, getitem_3374, getitem_3375, getitem_3376, getitem_3377, getitem_3378, getitem_3379, getitem_3380, getitem_3381, getitem_3382, getitem_3383, getitem_3384, getitem_3385, getitem_3386, getitem_3387, getitem_3388, getitem_3389, getitem_3390, getitem_3391, getitem_3392, getitem_3393, getitem_3394, getitem_3395, getitem_3396, getitem_3397, getitem_3398, getitem_3399, getitem_3400, getitem_3401, getitem_3402, getitem_3403, getitem_3404, getitem_3405, getitem_3406, getitem_3407, getitem_3408, getitem_3409, getitem_3410, getitem_3411, getitem_3412, getitem_3413, getitem_3414, getitem_3415, getitem_3416, getitem_3417, getitem_3418, getitem_3419, getitem_3420, getitem_3421, getitem_3422, getitem_3423, getitem_3424, getitem_3425, getitem_3426, getitem_3427, getitem_3428, getitem_3429, getitem_3430, getitem_3431, getitem_3432, getitem_3433, getitem_3434, getitem_3435, getitem_3436, getitem_3437, getitem_3438, getitem_3439, getitem_3440, getitem_3441, getitem_3442, getitem_3443, getitem_3444, getitem_3445, getitem_3446, getitem_3447, getitem_3448, getitem_3449, getitem_3450, getitem_3451, getitem_3452, getitem_3453, getitem_3454, getitem_3455, getitem_3456, getitem_3457, getitem_3458, getitem_3459, getitem_3460, getitem_3461, getitem_3462, getitem_3463, getitem_3464, getitem_3465, getitem_3466, getitem_3467, getitem_3468, getitem_3469, getitem_3470, getitem_3471, getitem_3472, getitem_3473, getitem_3474, getitem_3475, getitem_3476, getitem_3477, getitem_3478, getitem_3479, getitem_3480, getitem_3481, getitem_3482, getitem_3483, getitem_3484, getitem_3485, getitem_3486, getitem_3487, getitem_3488, getitem_3489, getitem_3490, getitem_3491, getitem_3492, getitem_3493, getitem_3494, getitem_3495, getitem_3496, getitem_3497, getitem_3498, getitem_3499, getitem_3500, getitem_3501, getitem_3502, getitem_3503, getitem_3504, getitem_3505, getitem_3506, getitem_3507, getitem_3508, getitem_3509, getitem_3510, getitem_3511, getitem_3512, getitem_3513, getitem_3514, getitem_3515, getitem_3516, getitem_3517, getitem_3518, getitem_3519, getitem_3520, getitem_3521, getitem_3522, getitem_3523, getitem_3524, getitem_3525, getitem_3526, getitem_3527, getitem_3528, getitem_3529, getitem_3530, getitem_3531, getitem_3532, getitem_3533, getitem_3534, getitem_3535, getitem_3536, getitem_3537, getitem_3538, getitem_3539, getitem_3540, getitem_3541, getitem_3542, getitem_3543, getitem_3544, getitem_3545, getitem_3546, getitem_3547, getitem_3548, getitem_3549, getitem_3550, getitem_3551, getitem_3552, getitem_3553, getitem_3554, getitem_3555, getitem_3556, getitem_3557, getitem_3558, getitem_3559, getitem_3560, getitem_3561, getitem_3562, getitem_3563, getitem_3564, getitem_3565, getitem_3566, getitem_3567, getitem_3568, getitem_3569, getitem_3570, getitem_3571, getitem_3572, getitem_3573, getitem_3574, getitem_3575, getitem_3576, getitem_3577, getitem_3578, getitem_3579, getitem_3580, getitem_3581, getitem_3582, getitem_3583, getitem_3584, getitem_3585, getitem_3586, getitem_3587, getitem_3588, getitem_3589, getitem_3590, getitem_3591, getitem_3592, getitem_3593, getitem_3594, getitem_3595, getitem_3596, getitem_3597, getitem_3598, getitem_3599, getitem_3600, getitem_3601, getitem_3602, getitem_3603, getitem_3604, getitem_3605, getitem_3606, getitem_3607, getitem_3608, getitem_3609, getitem_3610, getitem_3611]);  getitem_4515 = getitem_4516 = getitem_4517 = getitem_4518 = getitem_4519 = getitem_4520 = getitem_4521 = getitem_4522 = getitem_4523 = getitem_4524 = getitem_4525 = getitem_4526 = getitem_4527 = getitem_4528 = getitem_4529 = getitem_4530 = getitem_4531 = getitem_4532 = getitem_4533 = getitem_4534 = getitem_4535 = getitem_4536 = getitem_4537 = getitem_4538 = getitem_4539 = getitem_4540 = getitem_4541 = getitem_4542 = getitem_4543 = getitem_4544 = getitem_4545 = getitem_4546 = getitem_4547 = getitem_4548 = getitem_4549 = getitem_4550 = getitem_4551 = getitem_4552 = getitem_4553 = getitem_4554 = getitem_4555 = getitem_4556 = getitem_4557 = getitem_4558 = getitem_4559 = getitem_4560 = getitem_4561 = getitem_4562 = getitem_4563 = getitem_4564 = getitem_4565 = getitem_4566 = getitem_4567 = getitem_4568 = getitem_4569 = getitem_4570 = getitem_4571 = getitem_4572 = getitem_4573 = getitem_4574 = getitem_4575 = getitem_4576 = getitem_4577 = getitem_4578 = getitem_4579 = getitem_4580 = getitem_4581 = getitem_4582 = getitem_4583 = getitem_4584 = getitem_4585 = getitem_4586 = getitem_4587 = getitem_4588 = getitem_4589 = getitem_4590 = getitem_4591 = getitem_4592 = getitem_4593 = getitem_4594 = getitem_4595 = getitem_4596 = getitem_4597 = getitem_4598 = getitem_4599 = getitem_4600 = getitem_4601 = getitem_4602 = getitem_4603 = getitem_4604 = getitem_4605 = getitem_4606 = getitem_4607 = getitem_4608 = getitem_4609 = getitem_4610 = getitem_4611 = getitem_4612 = getitem_4613 = getitem_4614 = getitem_4615 = getitem_4616 = getitem_4617 = getitem_4618 = getitem_4619 = getitem_4620 = getitem_4621 = getitem_4622 = getitem_4623 = getitem_4624 = getitem_4625 = getitem_4626 = getitem_4627 = getitem_4628 = getitem_4629 = getitem_4630 = getitem_4631 = getitem_4632 = getitem_4633 = getitem_4634 = getitem_4635 = getitem_4636 = getitem_4637 = getitem_4638 = getitem_4639 = getitem_4640 = getitem_4641 = getitem_4642 = getitem_4643 = getitem_4644 = getitem_4645 = getitem_4646 = getitem_4647 = getitem_4648 = getitem_4649 = getitem_4650 = getitem_4651 = getitem_4652 = getitem_4653 = getitem_4654 = getitem_4655 = getitem_4656 = getitem_4657 = getitem_4658 = getitem_4659 = getitem_4660 = getitem_4661 = getitem_4662 = getitem_4663 = getitem_4664 = getitem_4665 = getitem_4666 = getitem_4667 = getitem_4668 = getitem_4669 = getitem_4670 = getitem_4671 = getitem_4672 = getitem_4673 = getitem_4674 = getitem_4675 = getitem_4676 = getitem_4677 = getitem_4678 = getitem_4679 = getitem_4680 = getitem_4681 = getitem_4682 = getitem_4683 = getitem_4684 = getitem_4685 = getitem_4686 = getitem_4687 = getitem_4688 = getitem_4689 = getitem_4690 = getitem_4691 = getitem_4692 = getitem_4693 = getitem_4694 = getitem_4695 = getitem_4696 = getitem_4697 = getitem_4698 = getitem_4699 = getitem_4700 = getitem_4701 = getitem_4702 = getitem_4703 = getitem_4704 = getitem_4705 = getitem_4706 = getitem_4707 = getitem_4708 = getitem_4709 = getitem_4710 = getitem_4711 = getitem_4712 = getitem_4713 = getitem_4714 = getitem_4715 = getitem_4716 = getitem_4717 = getitem_4718 = getitem_4719 = getitem_4720 = getitem_4721 = getitem_4722 = getitem_4723 = getitem_4724 = getitem_4725 = getitem_4726 = getitem_4727 = getitem_4728 = getitem_4729 = getitem_4730 = getitem_4731 = getitem_4732 = getitem_4733 = getitem_4734 = getitem_4735 = getitem_4736 = getitem_4737 = getitem_4738 = getitem_4739 = getitem_4740 = getitem_4741 = getitem_4742 = getitem_4743 = getitem_4744 = getitem_4745 = getitem_4746 = getitem_4747 = getitem_4748 = getitem_4749 = getitem_4750 = getitem_4751 = getitem_4752 = getitem_4753 = getitem_4754 = getitem_4755 = getitem_4756 = getitem_4757 = getitem_4758 = getitem_4759 = getitem_4760 = getitem_4761 = getitem_4762 = getitem_4763 = getitem_4764 = getitem_4765 = getitem_4766 = getitem_4767 = getitem_4768 = getitem_4769 = getitem_4770 = getitem_4771 = getitem_4772 = getitem_4773 = getitem_4774 = getitem_4775 = getitem_4776 = getitem_4777 = getitem_4778 = getitem_4779 = getitem_4780 = getitem_4781 = getitem_4782 = getitem_4783 = getitem_4784 = getitem_4785 = getitem_4786 = getitem_4787 = getitem_4788 = getitem_4789 = getitem_4790 = getitem_4791 = getitem_4792 = getitem_4793 = getitem_4794 = getitem_4795 = getitem_4796 = getitem_4797 = getitem_4798 = getitem_4799 = getitem_4800 = getitem_4801 = getitem_4802 = getitem_4803 = getitem_4804 = getitem_4805 = getitem_4806 = getitem_4807 = getitem_4808 = getitem_4809 = getitem_4810 = getitem_4811 = getitem_4812 = getitem_4813 = getitem_4814 = getitem_4815 = getitem_3311 = getitem_3312 = getitem_3313 = getitem_3314 = getitem_3315 = getitem_3316 = getitem_3317 = getitem_3318 = getitem_3319 = getitem_3320 = getitem_3321 = getitem_3322 = getitem_3323 = getitem_3324 = getitem_3325 = getitem_3326 = getitem_3327 = getitem_3328 = getitem_3329 = getitem_3330 = getitem_3331 = getitem_3332 = getitem_3333 = getitem_3334 = getitem_3335 = getitem_3336 = getitem_3337 = getitem_3338 = getitem_3339 = getitem_3340 = getitem_3341 = getitem_3342 = getitem_3343 = getitem_3344 = getitem_3345 = getitem_3346 = getitem_3347 = getitem_3348 = getitem_3349 = getitem_3350 = getitem_3351 = getitem_3352 = getitem_3353 = getitem_3354 = getitem_3355 = getitem_3356 = getitem_3357 = getitem_3358 = getitem_3359 = getitem_3360 = getitem_3361 = getitem_3362 = getitem_3363 = getitem_3364 = getitem_3365 = getitem_3366 = getitem_3367 = getitem_3368 = getitem_3369 = getitem_3370 = getitem_3371 = getitem_3372 = getitem_3373 = getitem_3374 = getitem_3375 = getitem_3376 = getitem_3377 = getitem_3378 = getitem_3379 = getitem_3380 = getitem_3381 = getitem_3382 = getitem_3383 = getitem_3384 = getitem_3385 = getitem_3386 = getitem_3387 = getitem_3388 = getitem_3389 = getitem_3390 = getitem_3391 = getitem_3392 = getitem_3393 = getitem_3394 = getitem_3395 = getitem_3396 = getitem_3397 = getitem_3398 = getitem_3399 = getitem_3400 = getitem_3401 = getitem_3402 = getitem_3403 = getitem_3404 = getitem_3405 = getitem_3406 = getitem_3407 = getitem_3408 = getitem_3409 = getitem_3410 = getitem_3411 = getitem_3412 = getitem_3413 = getitem_3414 = getitem_3415 = getitem_3416 = getitem_3417 = getitem_3418 = getitem_3419 = getitem_3420 = getitem_3421 = getitem_3422 = getitem_3423 = getitem_3424 = getitem_3425 = getitem_3426 = getitem_3427 = getitem_3428 = getitem_3429 = getitem_3430 = getitem_3431 = getitem_3432 = getitem_3433 = getitem_3434 = getitem_3435 = getitem_3436 = getitem_3437 = getitem_3438 = getitem_3439 = getitem_3440 = getitem_3441 = getitem_3442 = getitem_3443 = getitem_3444 = getitem_3445 = getitem_3446 = getitem_3447 = getitem_3448 = getitem_3449 = getitem_3450 = getitem_3451 = getitem_3452 = getitem_3453 = getitem_3454 = getitem_3455 = getitem_3456 = getitem_3457 = getitem_3458 = getitem_3459 = getitem_3460 = getitem_3461 = getitem_3462 = getitem_3463 = getitem_3464 = getitem_3465 = getitem_3466 = getitem_3467 = getitem_3468 = getitem_3469 = getitem_3470 = getitem_3471 = getitem_3472 = getitem_3473 = getitem_3474 = getitem_3475 = getitem_3476 = getitem_3477 = getitem_3478 = getitem_3479 = getitem_3480 = getitem_3481 = getitem_3482 = getitem_3483 = getitem_3484 = getitem_3485 = getitem_3486 = getitem_3487 = getitem_3488 = getitem_3489 = getitem_3490 = getitem_3491 = getitem_3492 = getitem_3493 = getitem_3494 = getitem_3495 = getitem_3496 = getitem_3497 = getitem_3498 = getitem_3499 = getitem_3500 = getitem_3501 = getitem_3502 = getitem_3503 = getitem_3504 = getitem_3505 = getitem_3506 = getitem_3507 = getitem_3508 = getitem_3509 = getitem_3510 = getitem_3511 = getitem_3512 = getitem_3513 = getitem_3514 = getitem_3515 = getitem_3516 = getitem_3517 = getitem_3518 = getitem_3519 = getitem_3520 = getitem_3521 = getitem_3522 = getitem_3523 = getitem_3524 = getitem_3525 = getitem_3526 = getitem_3527 = getitem_3528 = getitem_3529 = getitem_3530 = getitem_3531 = getitem_3532 = getitem_3533 = getitem_3534 = getitem_3535 = getitem_3536 = getitem_3537 = getitem_3538 = getitem_3539 = getitem_3540 = getitem_3541 = getitem_3542 = getitem_3543 = getitem_3544 = getitem_3545 = getitem_3546 = getitem_3547 = getitem_3548 = getitem_3549 = getitem_3550 = getitem_3551 = getitem_3552 = getitem_3553 = getitem_3554 = getitem_3555 = getitem_3556 = getitem_3557 = getitem_3558 = getitem_3559 = getitem_3560 = getitem_3561 = getitem_3562 = getitem_3563 = getitem_3564 = getitem_3565 = getitem_3566 = getitem_3567 = getitem_3568 = getitem_3569 = getitem_3570 = getitem_3571 = getitem_3572 = getitem_3573 = getitem_3574 = getitem_3575 = getitem_3576 = getitem_3577 = getitem_3578 = getitem_3579 = getitem_3580 = getitem_3581 = getitem_3582 = getitem_3583 = getitem_3584 = getitem_3585 = getitem_3586 = getitem_3587 = getitem_3588 = getitem_3589 = getitem_3590 = getitem_3591 = getitem_3592 = getitem_3593 = getitem_3594 = getitem_3595 = getitem_3596 = getitem_3597 = getitem_3598 = getitem_3599 = getitem_3600 = getitem_3601 = getitem_3602 = getitem_3603 = getitem_3604 = getitem_3605 = getitem_3606 = getitem_3607 = getitem_3608 = getitem_3609 = getitem_3610 = getitem_3611 = None
        getitem_3612: "f32[768]" = _foreach_div_list[0]
        getitem_3613: "f32[50, 768]" = _foreach_div_list[1]
        getitem_3614: "f32[768, 512]" = _foreach_div_list[2]
        getitem_3615: "f32[768, 3, 32, 32]" = _foreach_div_list[3]
        getitem_3616: "f32[768]" = _foreach_div_list[4]
        getitem_3617: "f32[768]" = _foreach_div_list[5]
        getitem_3618: "f32[2304, 768]" = _foreach_div_list[6]
        getitem_3619: "f32[2304]" = _foreach_div_list[7]
        getitem_3620: "f32[768, 768]" = _foreach_div_list[8]
        getitem_3621: "f32[768]" = _foreach_div_list[9]
        getitem_3622: "f32[3072, 768]" = _foreach_div_list[10]
        getitem_3623: "f32[3072]" = _foreach_div_list[11]
        getitem_3624: "f32[768, 3072]" = _foreach_div_list[12]
        getitem_3625: "f32[768]" = _foreach_div_list[13]
        getitem_3626: "f32[768]" = _foreach_div_list[14]
        getitem_3627: "f32[768]" = _foreach_div_list[15]
        getitem_3628: "f32[768]" = _foreach_div_list[16]
        getitem_3629: "f32[768]" = _foreach_div_list[17]
        getitem_3630: "f32[2304, 768]" = _foreach_div_list[18]
        getitem_3631: "f32[2304]" = _foreach_div_list[19]
        getitem_3632: "f32[768, 768]" = _foreach_div_list[20]
        getitem_3633: "f32[768]" = _foreach_div_list[21]
        getitem_3634: "f32[3072, 768]" = _foreach_div_list[22]
        getitem_3635: "f32[3072]" = _foreach_div_list[23]
        getitem_3636: "f32[768, 3072]" = _foreach_div_list[24]
        getitem_3637: "f32[768]" = _foreach_div_list[25]
        getitem_3638: "f32[768]" = _foreach_div_list[26]
        getitem_3639: "f32[768]" = _foreach_div_list[27]
        getitem_3640: "f32[768]" = _foreach_div_list[28]
        getitem_3641: "f32[768]" = _foreach_div_list[29]
        getitem_3642: "f32[2304, 768]" = _foreach_div_list[30]
        getitem_3643: "f32[2304]" = _foreach_div_list[31]
        getitem_3644: "f32[768, 768]" = _foreach_div_list[32]
        getitem_3645: "f32[768]" = _foreach_div_list[33]
        getitem_3646: "f32[3072, 768]" = _foreach_div_list[34]
        getitem_3647: "f32[3072]" = _foreach_div_list[35]
        getitem_3648: "f32[768, 3072]" = _foreach_div_list[36]
        getitem_3649: "f32[768]" = _foreach_div_list[37]
        getitem_3650: "f32[768]" = _foreach_div_list[38]
        getitem_3651: "f32[768]" = _foreach_div_list[39]
        getitem_3652: "f32[768]" = _foreach_div_list[40]
        getitem_3653: "f32[768]" = _foreach_div_list[41]
        getitem_3654: "f32[2304, 768]" = _foreach_div_list[42]
        getitem_3655: "f32[2304]" = _foreach_div_list[43]
        getitem_3656: "f32[768, 768]" = _foreach_div_list[44]
        getitem_3657: "f32[768]" = _foreach_div_list[45]
        getitem_3658: "f32[3072, 768]" = _foreach_div_list[46]
        getitem_3659: "f32[3072]" = _foreach_div_list[47]
        getitem_3660: "f32[768, 3072]" = _foreach_div_list[48]
        getitem_3661: "f32[768]" = _foreach_div_list[49]
        getitem_3662: "f32[768]" = _foreach_div_list[50]
        getitem_3663: "f32[768]" = _foreach_div_list[51]
        getitem_3664: "f32[768]" = _foreach_div_list[52]
        getitem_3665: "f32[768]" = _foreach_div_list[53]
        getitem_3666: "f32[2304, 768]" = _foreach_div_list[54]
        getitem_3667: "f32[2304]" = _foreach_div_list[55]
        getitem_3668: "f32[768, 768]" = _foreach_div_list[56]
        getitem_3669: "f32[768]" = _foreach_div_list[57]
        getitem_3670: "f32[3072, 768]" = _foreach_div_list[58]
        getitem_3671: "f32[3072]" = _foreach_div_list[59]
        getitem_3672: "f32[768, 3072]" = _foreach_div_list[60]
        getitem_3673: "f32[768]" = _foreach_div_list[61]
        getitem_3674: "f32[768]" = _foreach_div_list[62]
        getitem_3675: "f32[768]" = _foreach_div_list[63]
        getitem_3676: "f32[768]" = _foreach_div_list[64]
        getitem_3677: "f32[768]" = _foreach_div_list[65]
        getitem_3678: "f32[2304, 768]" = _foreach_div_list[66]
        getitem_3679: "f32[2304]" = _foreach_div_list[67]
        getitem_3680: "f32[768, 768]" = _foreach_div_list[68]
        getitem_3681: "f32[768]" = _foreach_div_list[69]
        getitem_3682: "f32[3072, 768]" = _foreach_div_list[70]
        getitem_3683: "f32[3072]" = _foreach_div_list[71]
        getitem_3684: "f32[768, 3072]" = _foreach_div_list[72]
        getitem_3685: "f32[768]" = _foreach_div_list[73]
        getitem_3686: "f32[768]" = _foreach_div_list[74]
        getitem_3687: "f32[768]" = _foreach_div_list[75]
        getitem_3688: "f32[768]" = _foreach_div_list[76]
        getitem_3689: "f32[768]" = _foreach_div_list[77]
        getitem_3690: "f32[2304, 768]" = _foreach_div_list[78]
        getitem_3691: "f32[2304]" = _foreach_div_list[79]
        getitem_3692: "f32[768, 768]" = _foreach_div_list[80]
        getitem_3693: "f32[768]" = _foreach_div_list[81]
        getitem_3694: "f32[3072, 768]" = _foreach_div_list[82]
        getitem_3695: "f32[3072]" = _foreach_div_list[83]
        getitem_3696: "f32[768, 3072]" = _foreach_div_list[84]
        getitem_3697: "f32[768]" = _foreach_div_list[85]
        getitem_3698: "f32[768]" = _foreach_div_list[86]
        getitem_3699: "f32[768]" = _foreach_div_list[87]
        getitem_3700: "f32[768]" = _foreach_div_list[88]
        getitem_3701: "f32[768]" = _foreach_div_list[89]
        getitem_3702: "f32[2304, 768]" = _foreach_div_list[90]
        getitem_3703: "f32[2304]" = _foreach_div_list[91]
        getitem_3704: "f32[768, 768]" = _foreach_div_list[92]
        getitem_3705: "f32[768]" = _foreach_div_list[93]
        getitem_3706: "f32[3072, 768]" = _foreach_div_list[94]
        getitem_3707: "f32[3072]" = _foreach_div_list[95]
        getitem_3708: "f32[768, 3072]" = _foreach_div_list[96]
        getitem_3709: "f32[768]" = _foreach_div_list[97]
        getitem_3710: "f32[768]" = _foreach_div_list[98]
        getitem_3711: "f32[768]" = _foreach_div_list[99]
        getitem_3712: "f32[768]" = _foreach_div_list[100]
        getitem_3713: "f32[768]" = _foreach_div_list[101]
        getitem_3714: "f32[2304, 768]" = _foreach_div_list[102]
        getitem_3715: "f32[2304]" = _foreach_div_list[103]
        getitem_3716: "f32[768, 768]" = _foreach_div_list[104]
        getitem_3717: "f32[768]" = _foreach_div_list[105]
        getitem_3718: "f32[3072, 768]" = _foreach_div_list[106]
        getitem_3719: "f32[3072]" = _foreach_div_list[107]
        getitem_3720: "f32[768, 3072]" = _foreach_div_list[108]
        getitem_3721: "f32[768]" = _foreach_div_list[109]
        getitem_3722: "f32[768]" = _foreach_div_list[110]
        getitem_3723: "f32[768]" = _foreach_div_list[111]
        getitem_3724: "f32[768]" = _foreach_div_list[112]
        getitem_3725: "f32[768]" = _foreach_div_list[113]
        getitem_3726: "f32[2304, 768]" = _foreach_div_list[114]
        getitem_3727: "f32[2304]" = _foreach_div_list[115]
        getitem_3728: "f32[768, 768]" = _foreach_div_list[116]
        getitem_3729: "f32[768]" = _foreach_div_list[117]
        getitem_3730: "f32[3072, 768]" = _foreach_div_list[118]
        getitem_3731: "f32[3072]" = _foreach_div_list[119]
        getitem_3732: "f32[768, 3072]" = _foreach_div_list[120]
        getitem_3733: "f32[768]" = _foreach_div_list[121]
        getitem_3734: "f32[768]" = _foreach_div_list[122]
        getitem_3735: "f32[768]" = _foreach_div_list[123]
        getitem_3736: "f32[768]" = _foreach_div_list[124]
        getitem_3737: "f32[768]" = _foreach_div_list[125]
        getitem_3738: "f32[2304, 768]" = _foreach_div_list[126]
        getitem_3739: "f32[2304]" = _foreach_div_list[127]
        getitem_3740: "f32[768, 768]" = _foreach_div_list[128]
        getitem_3741: "f32[768]" = _foreach_div_list[129]
        getitem_3742: "f32[3072, 768]" = _foreach_div_list[130]
        getitem_3743: "f32[3072]" = _foreach_div_list[131]
        getitem_3744: "f32[768, 3072]" = _foreach_div_list[132]
        getitem_3745: "f32[768]" = _foreach_div_list[133]
        getitem_3746: "f32[768]" = _foreach_div_list[134]
        getitem_3747: "f32[768]" = _foreach_div_list[135]
        getitem_3748: "f32[768]" = _foreach_div_list[136]
        getitem_3749: "f32[768]" = _foreach_div_list[137]
        getitem_3750: "f32[2304, 768]" = _foreach_div_list[138]
        getitem_3751: "f32[2304]" = _foreach_div_list[139]
        getitem_3752: "f32[768, 768]" = _foreach_div_list[140]
        getitem_3753: "f32[768]" = _foreach_div_list[141]
        getitem_3754: "f32[3072, 768]" = _foreach_div_list[142]
        getitem_3755: "f32[3072]" = _foreach_div_list[143]
        getitem_3756: "f32[768, 3072]" = _foreach_div_list[144]
        getitem_3757: "f32[768]" = _foreach_div_list[145]
        getitem_3758: "f32[768]" = _foreach_div_list[146]
        getitem_3759: "f32[768]" = _foreach_div_list[147]
        getitem_3760: "f32[768]" = _foreach_div_list[148]
        getitem_3761: "f32[768]" = _foreach_div_list[149]
        getitem_3762: "f32[768]" = _foreach_div_list[150]
        getitem_3763: "f32[768]" = _foreach_div_list[151]
        getitem_3764: "f32[77, 512]" = _foreach_div_list[152]
        getitem_3765: "f32[49408, 512]" = _foreach_div_list[153]
        getitem_3766: "f32[1536, 512]" = _foreach_div_list[154]
        getitem_3767: "f32[1536]" = _foreach_div_list[155]
        getitem_3768: "f32[512, 512]" = _foreach_div_list[156]
        getitem_3769: "f32[512]" = _foreach_div_list[157]
        getitem_3770: "f32[2048, 512]" = _foreach_div_list[158]
        getitem_3771: "f32[2048]" = _foreach_div_list[159]
        getitem_3772: "f32[512, 2048]" = _foreach_div_list[160]
        getitem_3773: "f32[512]" = _foreach_div_list[161]
        getitem_3774: "f32[512]" = _foreach_div_list[162]
        getitem_3775: "f32[512]" = _foreach_div_list[163]
        getitem_3776: "f32[512]" = _foreach_div_list[164]
        getitem_3777: "f32[512]" = _foreach_div_list[165]
        getitem_3778: "f32[1536, 512]" = _foreach_div_list[166]
        getitem_3779: "f32[1536]" = _foreach_div_list[167]
        getitem_3780: "f32[512, 512]" = _foreach_div_list[168]
        getitem_3781: "f32[512]" = _foreach_div_list[169]
        getitem_3782: "f32[2048, 512]" = _foreach_div_list[170]
        getitem_3783: "f32[2048]" = _foreach_div_list[171]
        getitem_3784: "f32[512, 2048]" = _foreach_div_list[172]
        getitem_3785: "f32[512]" = _foreach_div_list[173]
        getitem_3786: "f32[512]" = _foreach_div_list[174]
        getitem_3787: "f32[512]" = _foreach_div_list[175]
        getitem_3788: "f32[512]" = _foreach_div_list[176]
        getitem_3789: "f32[512]" = _foreach_div_list[177]
        getitem_3790: "f32[1536, 512]" = _foreach_div_list[178]
        getitem_3791: "f32[1536]" = _foreach_div_list[179]
        getitem_3792: "f32[512, 512]" = _foreach_div_list[180]
        getitem_3793: "f32[512]" = _foreach_div_list[181]
        getitem_3794: "f32[2048, 512]" = _foreach_div_list[182]
        getitem_3795: "f32[2048]" = _foreach_div_list[183]
        getitem_3796: "f32[512, 2048]" = _foreach_div_list[184]
        getitem_3797: "f32[512]" = _foreach_div_list[185]
        getitem_3798: "f32[512]" = _foreach_div_list[186]
        getitem_3799: "f32[512]" = _foreach_div_list[187]
        getitem_3800: "f32[512]" = _foreach_div_list[188]
        getitem_3801: "f32[512]" = _foreach_div_list[189]
        getitem_3802: "f32[1536, 512]" = _foreach_div_list[190]
        getitem_3803: "f32[1536]" = _foreach_div_list[191]
        getitem_3804: "f32[512, 512]" = _foreach_div_list[192]
        getitem_3805: "f32[512]" = _foreach_div_list[193]
        getitem_3806: "f32[2048, 512]" = _foreach_div_list[194]
        getitem_3807: "f32[2048]" = _foreach_div_list[195]
        getitem_3808: "f32[512, 2048]" = _foreach_div_list[196]
        getitem_3809: "f32[512]" = _foreach_div_list[197]
        getitem_3810: "f32[512]" = _foreach_div_list[198]
        getitem_3811: "f32[512]" = _foreach_div_list[199]
        getitem_3812: "f32[512]" = _foreach_div_list[200]
        getitem_3813: "f32[512]" = _foreach_div_list[201]
        getitem_3814: "f32[1536, 512]" = _foreach_div_list[202]
        getitem_3815: "f32[1536]" = _foreach_div_list[203]
        getitem_3816: "f32[512, 512]" = _foreach_div_list[204]
        getitem_3817: "f32[512]" = _foreach_div_list[205]
        getitem_3818: "f32[2048, 512]" = _foreach_div_list[206]
        getitem_3819: "f32[2048]" = _foreach_div_list[207]
        getitem_3820: "f32[512, 2048]" = _foreach_div_list[208]
        getitem_3821: "f32[512]" = _foreach_div_list[209]
        getitem_3822: "f32[512]" = _foreach_div_list[210]
        getitem_3823: "f32[512]" = _foreach_div_list[211]
        getitem_3824: "f32[512]" = _foreach_div_list[212]
        getitem_3825: "f32[512]" = _foreach_div_list[213]
        getitem_3826: "f32[1536, 512]" = _foreach_div_list[214]
        getitem_3827: "f32[1536]" = _foreach_div_list[215]
        getitem_3828: "f32[512, 512]" = _foreach_div_list[216]
        getitem_3829: "f32[512]" = _foreach_div_list[217]
        getitem_3830: "f32[2048, 512]" = _foreach_div_list[218]
        getitem_3831: "f32[2048]" = _foreach_div_list[219]
        getitem_3832: "f32[512, 2048]" = _foreach_div_list[220]
        getitem_3833: "f32[512]" = _foreach_div_list[221]
        getitem_3834: "f32[512]" = _foreach_div_list[222]
        getitem_3835: "f32[512]" = _foreach_div_list[223]
        getitem_3836: "f32[512]" = _foreach_div_list[224]
        getitem_3837: "f32[512]" = _foreach_div_list[225]
        getitem_3838: "f32[1536, 512]" = _foreach_div_list[226]
        getitem_3839: "f32[1536]" = _foreach_div_list[227]
        getitem_3840: "f32[512, 512]" = _foreach_div_list[228]
        getitem_3841: "f32[512]" = _foreach_div_list[229]
        getitem_3842: "f32[2048, 512]" = _foreach_div_list[230]
        getitem_3843: "f32[2048]" = _foreach_div_list[231]
        getitem_3844: "f32[512, 2048]" = _foreach_div_list[232]
        getitem_3845: "f32[512]" = _foreach_div_list[233]
        getitem_3846: "f32[512]" = _foreach_div_list[234]
        getitem_3847: "f32[512]" = _foreach_div_list[235]
        getitem_3848: "f32[512]" = _foreach_div_list[236]
        getitem_3849: "f32[512]" = _foreach_div_list[237]
        getitem_3850: "f32[1536, 512]" = _foreach_div_list[238]
        getitem_3851: "f32[1536]" = _foreach_div_list[239]
        getitem_3852: "f32[512, 512]" = _foreach_div_list[240]
        getitem_3853: "f32[512]" = _foreach_div_list[241]
        getitem_3854: "f32[2048, 512]" = _foreach_div_list[242]
        getitem_3855: "f32[2048]" = _foreach_div_list[243]
        getitem_3856: "f32[512, 2048]" = _foreach_div_list[244]
        getitem_3857: "f32[512]" = _foreach_div_list[245]
        getitem_3858: "f32[512]" = _foreach_div_list[246]
        getitem_3859: "f32[512]" = _foreach_div_list[247]
        getitem_3860: "f32[512]" = _foreach_div_list[248]
        getitem_3861: "f32[512]" = _foreach_div_list[249]
        getitem_3862: "f32[1536, 512]" = _foreach_div_list[250]
        getitem_3863: "f32[1536]" = _foreach_div_list[251]
        getitem_3864: "f32[512, 512]" = _foreach_div_list[252]
        getitem_3865: "f32[512]" = _foreach_div_list[253]
        getitem_3866: "f32[2048, 512]" = _foreach_div_list[254]
        getitem_3867: "f32[2048]" = _foreach_div_list[255]
        getitem_3868: "f32[512, 2048]" = _foreach_div_list[256]
        getitem_3869: "f32[512]" = _foreach_div_list[257]
        getitem_3870: "f32[512]" = _foreach_div_list[258]
        getitem_3871: "f32[512]" = _foreach_div_list[259]
        getitem_3872: "f32[512]" = _foreach_div_list[260]
        getitem_3873: "f32[512]" = _foreach_div_list[261]
        getitem_3874: "f32[1536, 512]" = _foreach_div_list[262]
        getitem_3875: "f32[1536]" = _foreach_div_list[263]
        getitem_3876: "f32[512, 512]" = _foreach_div_list[264]
        getitem_3877: "f32[512]" = _foreach_div_list[265]
        getitem_3878: "f32[2048, 512]" = _foreach_div_list[266]
        getitem_3879: "f32[2048]" = _foreach_div_list[267]
        getitem_3880: "f32[512, 2048]" = _foreach_div_list[268]
        getitem_3881: "f32[512]" = _foreach_div_list[269]
        getitem_3882: "f32[512]" = _foreach_div_list[270]
        getitem_3883: "f32[512]" = _foreach_div_list[271]
        getitem_3884: "f32[512]" = _foreach_div_list[272]
        getitem_3885: "f32[512]" = _foreach_div_list[273]
        getitem_3886: "f32[1536, 512]" = _foreach_div_list[274]
        getitem_3887: "f32[1536]" = _foreach_div_list[275]
        getitem_3888: "f32[512, 512]" = _foreach_div_list[276]
        getitem_3889: "f32[512]" = _foreach_div_list[277]
        getitem_3890: "f32[2048, 512]" = _foreach_div_list[278]
        getitem_3891: "f32[2048]" = _foreach_div_list[279]
        getitem_3892: "f32[512, 2048]" = _foreach_div_list[280]
        getitem_3893: "f32[512]" = _foreach_div_list[281]
        getitem_3894: "f32[512]" = _foreach_div_list[282]
        getitem_3895: "f32[512]" = _foreach_div_list[283]
        getitem_3896: "f32[512]" = _foreach_div_list[284]
        getitem_3897: "f32[512]" = _foreach_div_list[285]
        getitem_3898: "f32[1536, 512]" = _foreach_div_list[286]
        getitem_3899: "f32[1536]" = _foreach_div_list[287]
        getitem_3900: "f32[512, 512]" = _foreach_div_list[288]
        getitem_3901: "f32[512]" = _foreach_div_list[289]
        getitem_3902: "f32[2048, 512]" = _foreach_div_list[290]
        getitem_3903: "f32[2048]" = _foreach_div_list[291]
        getitem_3904: "f32[512, 2048]" = _foreach_div_list[292]
        getitem_3905: "f32[512]" = _foreach_div_list[293]
        getitem_3906: "f32[512]" = _foreach_div_list[294]
        getitem_3907: "f32[512]" = _foreach_div_list[295]
        getitem_3908: "f32[512]" = _foreach_div_list[296]
        getitem_3909: "f32[512]" = _foreach_div_list[297]
        getitem_3910: "f32[512]" = _foreach_div_list[298]
        getitem_3911: "f32[512]" = _foreach_div_list[299]
        getitem_3912: "f32[512, 512]" = _foreach_div_list[300];  _foreach_div_list = None
        return (getitem, getitem_602, getitem_603, getitem_604, getitem_605, getitem_606, getitem_607, getitem_608, getitem_609, getitem_610, getitem_611, getitem_612, getitem_613, getitem_614, getitem_615, getitem_616, getitem_617, getitem_618, getitem_619, getitem_620, getitem_621, getitem_622, getitem_623, getitem_624, getitem_625, getitem_626, getitem_627, getitem_628, getitem_629, getitem_630, getitem_631, getitem_632, getitem_633, getitem_634, getitem_635, getitem_636, getitem_637, getitem_638, getitem_639, getitem_640, getitem_641, getitem_642, getitem_643, getitem_644, getitem_645, getitem_646, getitem_647, getitem_648, getitem_649, getitem_650, getitem_651, getitem_652, getitem_653, getitem_654, getitem_655, getitem_656, getitem_657, getitem_658, getitem_659, getitem_660, getitem_661, getitem_662, getitem_663, getitem_664, getitem_665, getitem_666, getitem_667, getitem_668, getitem_669, getitem_670, getitem_671, getitem_672, getitem_673, getitem_674, getitem_675, getitem_676, getitem_677, getitem_678, getitem_679, getitem_680, getitem_681, getitem_682, getitem_683, getitem_684, getitem_685, getitem_686, getitem_687, getitem_688, getitem_689, getitem_690, getitem_691, getitem_692, getitem_693, getitem_694, getitem_695, getitem_696, getitem_697, getitem_698, getitem_699, getitem_700, getitem_701, getitem_702, getitem_703, getitem_704, getitem_705, getitem_706, getitem_707, getitem_708, getitem_709, getitem_710, getitem_711, getitem_712, getitem_713, getitem_714, getitem_715, getitem_716, getitem_717, getitem_718, getitem_719, getitem_720, getitem_721, getitem_722, getitem_723, getitem_724, getitem_725, getitem_726, getitem_727, getitem_728, getitem_729, getitem_730, getitem_731, getitem_732, getitem_733, getitem_734, getitem_735, getitem_736, getitem_737, getitem_738, getitem_739, getitem_740, getitem_741, getitem_742, getitem_743, getitem_744, getitem_745, getitem_746, getitem_747, getitem_748, getitem_749, getitem_750, getitem_751, getitem_752, getitem_753, getitem_754, getitem_755, getitem_756, getitem_757, getitem_758, getitem_759, getitem_760, getitem_761, getitem_762, getitem_763, getitem_764, getitem_765, getitem_766, getitem_767, getitem_768, getitem_769, getitem_770, getitem_771, getitem_772, getitem_773, getitem_774, getitem_775, getitem_776, getitem_777, getitem_778, getitem_779, getitem_780, getitem_781, getitem_782, getitem_783, getitem_784, getitem_785, getitem_786, getitem_787, getitem_788, getitem_789, getitem_790, getitem_791, getitem_792, getitem_793, getitem_794, getitem_795, getitem_796, getitem_797, getitem_798, getitem_799, getitem_800, getitem_801, getitem_802, getitem_803, getitem_804, getitem_805, getitem_806, getitem_807, getitem_808, getitem_809, getitem_810, getitem_811, getitem_812, getitem_813, getitem_814, getitem_815, getitem_816, getitem_817, getitem_818, getitem_819, getitem_820, getitem_821, getitem_822, getitem_823, getitem_824, getitem_825, getitem_826, getitem_827, getitem_828, getitem_829, getitem_830, getitem_831, getitem_832, getitem_833, getitem_834, getitem_835, getitem_836, getitem_837, getitem_838, getitem_839, getitem_840, getitem_841, getitem_842, getitem_843, getitem_844, getitem_845, getitem_846, getitem_847, getitem_848, getitem_849, getitem_850, getitem_851, getitem_852, getitem_853, getitem_854, getitem_855, getitem_856, getitem_857, getitem_858, getitem_859, getitem_860, getitem_861, getitem_862, getitem_863, getitem_864, getitem_865, getitem_866, getitem_867, getitem_868, getitem_869, getitem_870, getitem_871, getitem_872, getitem_873, getitem_874, getitem_875, getitem_876, getitem_877, getitem_878, getitem_879, getitem_880, getitem_881, getitem_882, getitem_883, getitem_884, getitem_885, getitem_886, getitem_887, getitem_888, getitem_889, getitem_890, getitem_891, getitem_892, getitem_893, getitem_894, getitem_895, getitem_896, getitem_897, getitem_898, getitem_899, getitem_900, getitem_901, getitem_3612, getitem_3613, getitem_3614, getitem_3615, getitem_3616, getitem_3617, getitem_3618, getitem_3619, getitem_3620, getitem_3621, getitem_3622, getitem_3623, getitem_3624, getitem_3625, getitem_3626, getitem_3627, getitem_3628, getitem_3629, getitem_3630, getitem_3631, getitem_3632, getitem_3633, getitem_3634, getitem_3635, getitem_3636, getitem_3637, getitem_3638, getitem_3639, getitem_3640, getitem_3641, getitem_3642, getitem_3643, getitem_3644, getitem_3645, getitem_3646, getitem_3647, getitem_3648, getitem_3649, getitem_3650, getitem_3651, getitem_3652, getitem_3653, getitem_3654, getitem_3655, getitem_3656, getitem_3657, getitem_3658, getitem_3659, getitem_3660, getitem_3661, getitem_3662, getitem_3663, getitem_3664, getitem_3665, getitem_3666, getitem_3667, getitem_3668, getitem_3669, getitem_3670, getitem_3671, getitem_3672, getitem_3673, getitem_3674, getitem_3675, getitem_3676, getitem_3677, getitem_3678, getitem_3679, getitem_3680, getitem_3681, getitem_3682, getitem_3683, getitem_3684, getitem_3685, getitem_3686, getitem_3687, getitem_3688, getitem_3689, getitem_3690, getitem_3691, getitem_3692, getitem_3693, getitem_3694, getitem_3695, getitem_3696, getitem_3697, getitem_3698, getitem_3699, getitem_3700, getitem_3701, getitem_3702, getitem_3703, getitem_3704, getitem_3705, getitem_3706, getitem_3707, getitem_3708, getitem_3709, getitem_3710, getitem_3711, getitem_3712, getitem_3713, getitem_3714, getitem_3715, getitem_3716, getitem_3717, getitem_3718, getitem_3719, getitem_3720, getitem_3721, getitem_3722, getitem_3723, getitem_3724, getitem_3725, getitem_3726, getitem_3727, getitem_3728, getitem_3729, getitem_3730, getitem_3731, getitem_3732, getitem_3733, getitem_3734, getitem_3735, getitem_3736, getitem_3737, getitem_3738, getitem_3739, getitem_3740, getitem_3741, getitem_3742, getitem_3743, getitem_3744, getitem_3745, getitem_3746, getitem_3747, getitem_3748, getitem_3749, getitem_3750, getitem_3751, getitem_3752, getitem_3753, getitem_3754, getitem_3755, getitem_3756, getitem_3757, getitem_3758, getitem_3759, getitem_3760, getitem_3761, getitem_3762, getitem_3763, getitem_3764, getitem_3765, getitem_3766, getitem_3767, getitem_3768, getitem_3769, getitem_3770, getitem_3771, getitem_3772, getitem_3773, getitem_3774, getitem_3775, getitem_3776, getitem_3777, getitem_3778, getitem_3779, getitem_3780, getitem_3781, getitem_3782, getitem_3783, getitem_3784, getitem_3785, getitem_3786, getitem_3787, getitem_3788, getitem_3789, getitem_3790, getitem_3791, getitem_3792, getitem_3793, getitem_3794, getitem_3795, getitem_3796, getitem_3797, getitem_3798, getitem_3799, getitem_3800, getitem_3801, getitem_3802, getitem_3803, getitem_3804, getitem_3805, getitem_3806, getitem_3807, getitem_3808, getitem_3809, getitem_3810, getitem_3811, getitem_3812, getitem_3813, getitem_3814, getitem_3815, getitem_3816, getitem_3817, getitem_3818, getitem_3819, getitem_3820, getitem_3821, getitem_3822, getitem_3823, getitem_3824, getitem_3825, getitem_3826, getitem_3827, getitem_3828, getitem_3829, getitem_3830, getitem_3831, getitem_3832, getitem_3833, getitem_3834, getitem_3835, getitem_3836, getitem_3837, getitem_3838, getitem_3839, getitem_3840, getitem_3841, getitem_3842, getitem_3843, getitem_3844, getitem_3845, getitem_3846, getitem_3847, getitem_3848, getitem_3849, getitem_3850, getitem_3851, getitem_3852, getitem_3853, getitem_3854, getitem_3855, getitem_3856, getitem_3857, getitem_3858, getitem_3859, getitem_3860, getitem_3861, getitem_3862, getitem_3863, getitem_3864, getitem_3865, getitem_3866, getitem_3867, getitem_3868, getitem_3869, getitem_3870, getitem_3871, getitem_3872, getitem_3873, getitem_3874, getitem_3875, getitem_3876, getitem_3877, getitem_3878, getitem_3879, getitem_3880, getitem_3881, getitem_3882, getitem_3883, getitem_3884, getitem_3885, getitem_3886, getitem_3887, getitem_3888, getitem_3889, getitem_3890, getitem_3891, getitem_3892, getitem_3893, getitem_3894, getitem_3895, getitem_3896, getitem_3897, getitem_3898, getitem_3899, getitem_3900, getitem_3901, getitem_3902, getitem_3903, getitem_3904, getitem_3905, getitem_3906, getitem_3907, getitem_3908, getitem_3909, getitem_3910, getitem_3911, getitem_3912)


def _default_make_inputs():
    return [
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 512], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3, 32, 32], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([77, 512], dtype=torch.float32, device='cuda'),
    torch.randn([49408, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 512], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3, 32, 32], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([77, 512], dtype=torch.float32, device='cuda'),
    torch.randn([49408, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 512], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3, 32, 32], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([77, 512], dtype=torch.float32, device='cuda'),
    torch.randn([49408, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
