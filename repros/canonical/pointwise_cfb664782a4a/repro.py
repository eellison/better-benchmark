"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s4_g59
Pattern hash: cfb664782a4a
Shape hash: 372209a9
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg134_1: "f32[32128, 512]", arg132_1: "f32[512, 512]", arg266_1: "f32[512, 512]", arg267_1: "f32[512, 512]", arg268_1: "f32[512, 512]", arg269_1: "f32[32, 8]", arg270_1: "f32[512]", arg271_1: "f32[2048, 512]", arg272_1: "f32[512, 2048]", arg273_1: "f32[512]", arg274_1: "f32[512, 512]", arg275_1: "f32[512, 512]", arg276_1: "f32[512, 512]", arg277_1: "f32[512, 512]", arg278_1: "f32[512]", arg279_1: "f32[2048, 512]", arg280_1: "f32[512, 2048]", arg281_1: "f32[512]", arg282_1: "f32[512, 512]", arg283_1: "f32[512, 512]", arg284_1: "f32[512, 512]", arg285_1: "f32[512, 512]", arg286_1: "f32[512]", arg287_1: "f32[2048, 512]", arg288_1: "f32[512, 2048]", arg289_1: "f32[512]", arg290_1: "f32[512, 512]", arg291_1: "f32[512, 512]", arg292_1: "f32[512, 512]", arg293_1: "f32[512, 512]", arg294_1: "f32[512]", arg295_1: "f32[2048, 512]", arg296_1: "f32[512, 2048]", arg297_1: "f32[512]", arg298_1: "f32[512, 512]", arg299_1: "f32[512, 512]", arg300_1: "f32[512, 512]", arg301_1: "f32[512, 512]", arg302_1: "f32[512]", arg303_1: "f32[2048, 512]", arg304_1: "f32[512, 2048]", arg305_1: "f32[512]", arg306_1: "f32[512, 512]", arg307_1: "f32[512, 512]", arg308_1: "f32[512, 512]", arg309_1: "f32[512, 512]", arg310_1: "f32[512]", arg311_1: "f32[2048, 512]", arg312_1: "f32[512, 2048]", arg313_1: "f32[512]", arg314_1: "f32[512]", arg315_1: "f32[512, 512]", arg316_1: "f32[512, 512]", arg317_1: "f32[512, 512]", arg318_1: "f32[512, 512]", arg319_1: "f32[32, 8]", arg320_1: "f32[512]", arg321_1: "f32[512, 512]", arg322_1: "f32[512, 512]", arg323_1: "f32[512, 512]", arg324_1: "f32[512, 512]", arg325_1: "f32[512]", arg326_1: "f32[2048, 512]", arg327_1: "f32[512, 2048]", arg328_1: "f32[512]", arg329_1: "f32[512, 512]", arg330_1: "f32[512, 512]", arg331_1: "f32[512, 512]", arg332_1: "f32[512, 512]", arg333_1: "f32[512]", arg334_1: "f32[512, 512]", arg335_1: "f32[512, 512]", arg336_1: "f32[512, 512]", arg337_1: "f32[512, 512]", arg338_1: "f32[512]", arg339_1: "f32[2048, 512]", arg340_1: "f32[512, 2048]", arg341_1: "f32[512]", arg342_1: "f32[512, 512]", arg343_1: "f32[512, 512]", arg344_1: "f32[512, 512]", arg345_1: "f32[512, 512]", arg346_1: "f32[512]", arg347_1: "f32[512, 512]", arg348_1: "f32[512, 512]", arg349_1: "f32[512, 512]", arg350_1: "f32[512, 512]", arg351_1: "f32[512]", arg352_1: "f32[2048, 512]", arg353_1: "f32[512, 2048]", arg354_1: "f32[512]", arg355_1: "f32[512, 512]", arg356_1: "f32[512, 512]", arg357_1: "f32[512, 512]", arg358_1: "f32[512, 512]", arg359_1: "f32[512]", arg360_1: "f32[512, 512]", arg361_1: "f32[512, 512]", arg362_1: "f32[512, 512]", arg363_1: "f32[512, 512]", arg364_1: "f32[512]", arg365_1: "f32[2048, 512]", arg366_1: "f32[512, 2048]", arg367_1: "f32[512]", arg368_1: "f32[512, 512]", arg369_1: "f32[512, 512]", arg370_1: "f32[512, 512]", arg371_1: "f32[512, 512]", arg372_1: "f32[512]", arg373_1: "f32[512, 512]", arg374_1: "f32[512, 512]", arg375_1: "f32[512, 512]", arg376_1: "f32[512, 512]", arg377_1: "f32[512]", arg378_1: "f32[2048, 512]", arg379_1: "f32[512, 2048]", arg380_1: "f32[512]", arg381_1: "f32[512, 512]", arg382_1: "f32[512, 512]", arg383_1: "f32[512, 512]", arg384_1: "f32[512, 512]", arg385_1: "f32[512]", arg386_1: "f32[512, 512]", arg387_1: "f32[512, 512]", arg388_1: "f32[512, 512]", arg389_1: "f32[512, 512]", arg390_1: "f32[512]", arg391_1: "f32[2048, 512]", arg392_1: "f32[512, 2048]", arg393_1: "f32[512]", arg394_1: "f32[512]", getitem_131: "f32[32128, 512]", getitem_132: "f32[512, 512]", getitem_133: "f32[512, 512]", getitem_134: "f32[512, 512]", getitem_135: "f32[512, 512]", getitem_136: "f32[32, 8]", getitem_137: "f32[512]", getitem_138: "f32[2048, 512]", getitem_139: "f32[512, 2048]", getitem_140: "f32[512]", getitem_141: "f32[512, 512]", getitem_142: "f32[512, 512]", getitem_143: "f32[512, 512]", getitem_144: "f32[512, 512]", getitem_145: "f32[512]", getitem_146: "f32[2048, 512]", getitem_147: "f32[512, 2048]", getitem_148: "f32[512]", getitem_149: "f32[512, 512]", getitem_150: "f32[512, 512]", getitem_151: "f32[512, 512]", getitem_152: "f32[512, 512]", getitem_153: "f32[512]", getitem_154: "f32[2048, 512]", getitem_155: "f32[512, 2048]", getitem_156: "f32[512]", getitem_157: "f32[512, 512]", getitem_158: "f32[512, 512]", getitem_159: "f32[512, 512]", getitem_160: "f32[512, 512]", getitem_161: "f32[512]", getitem_162: "f32[2048, 512]", getitem_163: "f32[512, 2048]", getitem_164: "f32[512]", getitem_165: "f32[512, 512]", getitem_166: "f32[512, 512]", getitem_167: "f32[512, 512]", getitem_168: "f32[512, 512]", getitem_169: "f32[512]", getitem_170: "f32[2048, 512]", getitem_171: "f32[512, 2048]", getitem_172: "f32[512]", getitem_173: "f32[512, 512]", getitem_174: "f32[512, 512]", getitem_175: "f32[512, 512]", getitem_176: "f32[512, 512]", getitem_177: "f32[512]", getitem_178: "f32[2048, 512]", getitem_179: "f32[512, 2048]", getitem_180: "f32[512]", getitem_181: "f32[512]", getitem_182: "f32[512, 512]", getitem_183: "f32[512, 512]", getitem_184: "f32[512, 512]", getitem_185: "f32[512, 512]", getitem_186: "f32[32, 8]", getitem_187: "f32[512]", getitem_188: "f32[512, 512]", getitem_189: "f32[512, 512]", getitem_190: "f32[512, 512]", getitem_191: "f32[512, 512]", getitem_192: "f32[512]", getitem_193: "f32[2048, 512]", getitem_194: "f32[512, 2048]", getitem_195: "f32[512]", getitem_196: "f32[512, 512]", getitem_197: "f32[512, 512]", getitem_198: "f32[512, 512]", getitem_199: "f32[512, 512]", getitem_200: "f32[512]", getitem_201: "f32[512, 512]", getitem_202: "f32[512, 512]", getitem_203: "f32[512, 512]", getitem_204: "f32[512, 512]", getitem_205: "f32[512]", getitem_206: "f32[2048, 512]", getitem_207: "f32[512, 2048]", getitem_208: "f32[512]", getitem_209: "f32[512, 512]", getitem_210: "f32[512, 512]", getitem_211: "f32[512, 512]", getitem_212: "f32[512, 512]", getitem_213: "f32[512]", getitem_214: "f32[512, 512]", getitem_215: "f32[512, 512]", getitem_216: "f32[512, 512]", getitem_217: "f32[512, 512]", getitem_218: "f32[512]", getitem_219: "f32[2048, 512]", getitem_220: "f32[512, 2048]", getitem_221: "f32[512]", getitem_222: "f32[512, 512]", getitem_223: "f32[512, 512]", getitem_224: "f32[512, 512]", getitem_225: "f32[512, 512]", getitem_226: "f32[512]", getitem_227: "f32[512, 512]", getitem_228: "f32[512, 512]", getitem_229: "f32[512, 512]", getitem_230: "f32[512, 512]", getitem_231: "f32[512]", getitem_232: "f32[2048, 512]", getitem_233: "f32[512, 2048]", getitem_234: "f32[512]", getitem_235: "f32[512, 512]", getitem_236: "f32[512, 512]", getitem_237: "f32[512, 512]", getitem_238: "f32[512, 512]", getitem_239: "f32[512]", getitem_240: "f32[512, 512]", getitem_241: "f32[512, 512]", getitem_242: "f32[512, 512]", getitem_243: "f32[512, 512]", getitem_244: "f32[512]", getitem_245: "f32[2048, 512]", getitem_246: "f32[512, 2048]", getitem_247: "f32[512]", getitem_248: "f32[512, 512]", getitem_249: "f32[512, 512]", getitem_250: "f32[512, 512]", getitem_251: "f32[512, 512]", getitem_252: "f32[512]", getitem_253: "f32[512, 512]", getitem_254: "f32[512, 512]", getitem_255: "f32[512, 512]", getitem_256: "f32[512, 512]", getitem_257: "f32[512]", getitem_258: "f32[2048, 512]", getitem_259: "f32[512, 2048]", getitem_260: "f32[512]", getitem_261: "f32[512]", getitem_1965: "f32[32128, 512]", getitem_1966: "f32[512, 512]", getitem_1967: "f32[512, 512]", getitem_1968: "f32[512, 512]", getitem_1969: "f32[512, 512]", getitem_1970: "f32[32, 8]", getitem_1971: "f32[512]", getitem_1972: "f32[2048, 512]", getitem_1973: "f32[512, 2048]", getitem_1974: "f32[512]", getitem_1975: "f32[512, 512]", getitem_1976: "f32[512, 512]", getitem_1977: "f32[512, 512]", getitem_1978: "f32[512, 512]", getitem_1979: "f32[512]", getitem_1980: "f32[2048, 512]", getitem_1981: "f32[512, 2048]", getitem_1982: "f32[512]", getitem_1983: "f32[512, 512]", getitem_1984: "f32[512, 512]", getitem_1985: "f32[512, 512]", getitem_1986: "f32[512, 512]", getitem_1987: "f32[512]", getitem_1988: "f32[2048, 512]", getitem_1989: "f32[512, 2048]", getitem_1990: "f32[512]", getitem_1991: "f32[512, 512]", getitem_1992: "f32[512, 512]", getitem_1993: "f32[512, 512]", getitem_1994: "f32[512, 512]", getitem_1995: "f32[512]", getitem_1996: "f32[2048, 512]", getitem_1997: "f32[512, 2048]", getitem_1998: "f32[512]", getitem_1999: "f32[512, 512]", getitem_2000: "f32[512, 512]", getitem_2001: "f32[512, 512]", getitem_2002: "f32[512, 512]", getitem_2003: "f32[512]", getitem_2004: "f32[2048, 512]", getitem_2005: "f32[512, 2048]", getitem_2006: "f32[512]", getitem_2007: "f32[512, 512]", getitem_2008: "f32[512, 512]", getitem_2009: "f32[512, 512]", getitem_2010: "f32[512, 512]", getitem_2011: "f32[512]", getitem_2012: "f32[2048, 512]", getitem_2013: "f32[512, 2048]", getitem_2014: "f32[512]", getitem_2015: "f32[512]", getitem_2016: "f32[512, 512]", getitem_2017: "f32[512, 512]", getitem_2018: "f32[512, 512]", getitem_2019: "f32[512, 512]", getitem_2020: "f32[32, 8]", getitem_2021: "f32[512]", getitem_2022: "f32[512, 512]", getitem_2023: "f32[512, 512]", getitem_2024: "f32[512, 512]", getitem_2025: "f32[512, 512]", getitem_2026: "f32[512]", getitem_2027: "f32[2048, 512]", getitem_2028: "f32[512, 2048]", getitem_2029: "f32[512]", getitem_2030: "f32[512, 512]", getitem_2031: "f32[512, 512]", getitem_2032: "f32[512, 512]", getitem_2033: "f32[512, 512]", getitem_2034: "f32[512]", getitem_2035: "f32[512, 512]", getitem_2036: "f32[512, 512]", getitem_2037: "f32[512, 512]", getitem_2038: "f32[512, 512]", getitem_2039: "f32[512]", getitem_2040: "f32[2048, 512]", getitem_2041: "f32[512, 2048]", getitem_2042: "f32[512]", getitem_2043: "f32[512, 512]", getitem_2044: "f32[512, 512]", getitem_2045: "f32[512, 512]", getitem_2046: "f32[512, 512]", getitem_2047: "f32[512]", getitem_2048: "f32[512, 512]", getitem_2049: "f32[512, 512]", getitem_2050: "f32[512, 512]", getitem_2051: "f32[512, 512]", getitem_2052: "f32[512]", getitem_2053: "f32[2048, 512]", getitem_2054: "f32[512, 2048]", getitem_2055: "f32[512]", getitem_2056: "f32[512, 512]", getitem_2057: "f32[512, 512]", getitem_2058: "f32[512, 512]", getitem_2059: "f32[512, 512]", getitem_2060: "f32[512]", getitem_2061: "f32[512, 512]", getitem_2062: "f32[512, 512]", getitem_2063: "f32[512, 512]", getitem_2064: "f32[512, 512]", getitem_2065: "f32[512]", getitem_2066: "f32[2048, 512]", getitem_2067: "f32[512, 2048]", getitem_2068: "f32[512]", getitem_2069: "f32[512, 512]", getitem_2070: "f32[512, 512]", getitem_2071: "f32[512, 512]", getitem_2072: "f32[512, 512]", getitem_2073: "f32[512]", getitem_2074: "f32[512, 512]", getitem_2075: "f32[512, 512]", getitem_2076: "f32[512, 512]", getitem_2077: "f32[512, 512]", getitem_2078: "f32[512]", getitem_2079: "f32[2048, 512]", getitem_2080: "f32[512, 2048]", getitem_2081: "f32[512]", getitem_2082: "f32[512, 512]", getitem_2083: "f32[512, 512]", getitem_2084: "f32[512, 512]", getitem_2085: "f32[512, 512]", getitem_2086: "f32[512]", getitem_2087: "f32[512, 512]", getitem_2088: "f32[512, 512]", getitem_2089: "f32[512, 512]", getitem_2090: "f32[512, 512]", getitem_2091: "f32[512]", getitem_2092: "f32[2048, 512]", getitem_2093: "f32[512, 2048]", getitem_2094: "f32[512]", getitem_2095: "f32[512]", getitem_1441: "f32[]", getitem_1442: "f32[]", getitem_1443: "f32[]", getitem_1444: "f32[]", getitem_1445: "f32[]", getitem_1446: "f32[]", getitem_1447: "f32[]", getitem_1448: "f32[]", getitem_1449: "f32[]", getitem_1450: "f32[]", getitem_1451: "f32[]", getitem_1452: "f32[]", getitem_1453: "f32[]", getitem_1454: "f32[]", getitem_1455: "f32[]", getitem_1456: "f32[]", getitem_1457: "f32[]", getitem_1458: "f32[]", getitem_1459: "f32[]", getitem_1460: "f32[]", getitem_1461: "f32[]", getitem_1462: "f32[]", getitem_1463: "f32[]", getitem_1464: "f32[]", getitem_1465: "f32[]", getitem_1466: "f32[]", getitem_1467: "f32[]", getitem_1468: "f32[]", getitem_1469: "f32[]", getitem_1470: "f32[]", getitem_1471: "f32[]", getitem_1472: "f32[]", getitem_1473: "f32[]", getitem_1474: "f32[]", getitem_1475: "f32[]", getitem_1476: "f32[]", getitem_1477: "f32[]", getitem_1478: "f32[]", getitem_1479: "f32[]", getitem_1480: "f32[]", getitem_1481: "f32[]", getitem_1482: "f32[]", getitem_1483: "f32[]", getitem_1484: "f32[]", getitem_1485: "f32[]", getitem_1486: "f32[]", getitem_1487: "f32[]", getitem_1488: "f32[]", getitem_1489: "f32[]", getitem_1490: "f32[]", getitem_1491: "f32[]", getitem_1492: "f32[]", getitem_1493: "f32[]", getitem_1494: "f32[]", getitem_1495: "f32[]", getitem_1496: "f32[]", getitem_1497: "f32[]", getitem_1498: "f32[]", getitem_1499: "f32[]", getitem_1500: "f32[]", getitem_1501: "f32[]", getitem_1502: "f32[]", getitem_1503: "f32[]", getitem_1504: "f32[]", getitem_1505: "f32[]", getitem_1506: "f32[]", getitem_1507: "f32[]", getitem_1508: "f32[]", getitem_1509: "f32[]", getitem_1510: "f32[]", getitem_1511: "f32[]", getitem_1512: "f32[]", getitem_1513: "f32[]", getitem_1514: "f32[]", getitem_1515: "f32[]", getitem_1516: "f32[]", getitem_1517: "f32[]", getitem_1518: "f32[]", getitem_1519: "f32[]", getitem_1520: "f32[]", getitem_1521: "f32[]", getitem_1522: "f32[]", getitem_1523: "f32[]", getitem_1524: "f32[]", getitem_1525: "f32[]", getitem_1526: "f32[]", getitem_1527: "f32[]", getitem_1528: "f32[]", getitem_1529: "f32[]", getitem_1530: "f32[]", getitem_1531: "f32[]", getitem_1532: "f32[]", getitem_1533: "f32[]", getitem_1534: "f32[]", getitem_1535: "f32[]", getitem_1536: "f32[]", getitem_1537: "f32[]", getitem_1538: "f32[]", getitem_1539: "f32[]", getitem_1540: "f32[]", getitem_1541: "f32[]", getitem_1542: "f32[]", getitem_1543: "f32[]", getitem_1544: "f32[]", getitem_1545: "f32[]", getitem_1546: "f32[]", getitem_1547: "f32[]", getitem_1548: "f32[]", getitem_1549: "f32[]", getitem_1550: "f32[]", getitem_1551: "f32[]", getitem_1552: "f32[]", getitem_1553: "f32[]", getitem_1554: "f32[]", getitem_1555: "f32[]", getitem_1556: "f32[]", getitem_1557: "f32[]", getitem_1558: "f32[]", getitem_1559: "f32[]", getitem_1560: "f32[]", getitem_1561: "f32[]", getitem_1562: "f32[]", getitem_1563: "f32[]", getitem_1564: "f32[]", getitem_1565: "f32[]", getitem_1566: "f32[]", getitem_1567: "f32[]", getitem_1568: "f32[]", getitem_1569: "f32[]", getitem_1570: "f32[]", getitem_1571: "f32[]"):
        # No stacktrace found for following nodes
        full_default: "f32[32128, 512]" = torch.ops.aten.full.default([32128, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_3: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_4: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_5: "f32[32, 8]" = torch.ops.aten.full.default([32, 8], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_6: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_7: "f32[2048, 512]" = torch.ops.aten.full.default([2048, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_8: "f32[512, 2048]" = torch.ops.aten.full.default([512, 2048], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_9: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_10: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_11: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_12: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_13: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_14: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_15: "f32[2048, 512]" = torch.ops.aten.full.default([2048, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_16: "f32[512, 2048]" = torch.ops.aten.full.default([512, 2048], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_17: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_18: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_19: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_20: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_21: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_22: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_23: "f32[2048, 512]" = torch.ops.aten.full.default([2048, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_24: "f32[512, 2048]" = torch.ops.aten.full.default([512, 2048], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_25: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_26: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_27: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_28: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_29: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_30: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_31: "f32[2048, 512]" = torch.ops.aten.full.default([2048, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_32: "f32[512, 2048]" = torch.ops.aten.full.default([512, 2048], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_33: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_34: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_35: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_36: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_37: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_38: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_39: "f32[2048, 512]" = torch.ops.aten.full.default([2048, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_40: "f32[512, 2048]" = torch.ops.aten.full.default([512, 2048], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_41: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_42: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_43: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_44: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_45: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_46: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_47: "f32[2048, 512]" = torch.ops.aten.full.default([2048, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_48: "f32[512, 2048]" = torch.ops.aten.full.default([512, 2048], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_49: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_50: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_51: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_52: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_53: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_54: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_55: "f32[32, 8]" = torch.ops.aten.full.default([32, 8], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_56: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_57: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_58: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_59: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_60: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_61: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_62: "f32[2048, 512]" = torch.ops.aten.full.default([2048, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_63: "f32[512, 2048]" = torch.ops.aten.full.default([512, 2048], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_64: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_65: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_66: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_67: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_68: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_69: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_70: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_71: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_72: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_73: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_74: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_75: "f32[2048, 512]" = torch.ops.aten.full.default([2048, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_76: "f32[512, 2048]" = torch.ops.aten.full.default([512, 2048], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_77: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_78: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_79: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_80: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_81: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_82: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_83: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_84: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_85: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_86: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_87: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_88: "f32[2048, 512]" = torch.ops.aten.full.default([2048, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_89: "f32[512, 2048]" = torch.ops.aten.full.default([512, 2048], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_90: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_91: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_92: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_93: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_94: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_95: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_96: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_97: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_98: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_99: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_100: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_101: "f32[2048, 512]" = torch.ops.aten.full.default([2048, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_102: "f32[512, 2048]" = torch.ops.aten.full.default([512, 2048], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_103: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_104: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_105: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_106: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_107: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_108: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_109: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_110: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_111: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_112: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_113: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_114: "f32[2048, 512]" = torch.ops.aten.full.default([2048, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_115: "f32[512, 2048]" = torch.ops.aten.full.default([512, 2048], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_116: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_117: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_118: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_119: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_120: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_121: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_122: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_123: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_124: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_125: "f32[512, 512]" = torch.ops.aten.full.default([512, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_126: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_127: "f32[2048, 512]" = torch.ops.aten.full.default([2048, 512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_128: "f32[512, 2048]" = torch.ops.aten.full.default([512, 2048], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_129: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_130: "f32[512]" = torch.ops.aten.full.default([512], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _foreach_addcmul_scalar = torch.ops.aten._foreach_addcmul.Scalar([arg134_1, arg132_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1], [full_default, full_default_1, full_default_2, full_default_3, full_default_4, full_default_5, full_default_6, full_default_7, full_default_8, full_default_9, full_default_10, full_default_11, full_default_12, full_default_13, full_default_14, full_default_15, full_default_16, full_default_17, full_default_18, full_default_19, full_default_20, full_default_21, full_default_22, full_default_23, full_default_24, full_default_25, full_default_26, full_default_27, full_default_28, full_default_29, full_default_30, full_default_31, full_default_32, full_default_33, full_default_34, full_default_35, full_default_36, full_default_37, full_default_38, full_default_39, full_default_40, full_default_41, full_default_42, full_default_43, full_default_44, full_default_45, full_default_46, full_default_47, full_default_48, full_default_49, full_default_50, full_default_51, full_default_52, full_default_53, full_default_54, full_default_55, full_default_56, full_default_57, full_default_58, full_default_59, full_default_60, full_default_61, full_default_62, full_default_63, full_default_64, full_default_65, full_default_66, full_default_67, full_default_68, full_default_69, full_default_70, full_default_71, full_default_72, full_default_73, full_default_74, full_default_75, full_default_76, full_default_77, full_default_78, full_default_79, full_default_80, full_default_81, full_default_82, full_default_83, full_default_84, full_default_85, full_default_86, full_default_87, full_default_88, full_default_89, full_default_90, full_default_91, full_default_92, full_default_93, full_default_94, full_default_95, full_default_96, full_default_97, full_default_98, full_default_99, full_default_100, full_default_101, full_default_102, full_default_103, full_default_104, full_default_105, full_default_106, full_default_107, full_default_108, full_default_109, full_default_110, full_default_111, full_default_112, full_default_113, full_default_114, full_default_115, full_default_116, full_default_117, full_default_118, full_default_119, full_default_120, full_default_121, full_default_122, full_default_123, full_default_124, full_default_125, full_default_126, full_default_127, full_default_128, full_default_129, full_default_130], [getitem_131, getitem_132, getitem_133, getitem_134, getitem_135, getitem_136, getitem_137, getitem_138, getitem_139, getitem_140, getitem_141, getitem_142, getitem_143, getitem_144, getitem_145, getitem_146, getitem_147, getitem_148, getitem_149, getitem_150, getitem_151, getitem_152, getitem_153, getitem_154, getitem_155, getitem_156, getitem_157, getitem_158, getitem_159, getitem_160, getitem_161, getitem_162, getitem_163, getitem_164, getitem_165, getitem_166, getitem_167, getitem_168, getitem_169, getitem_170, getitem_171, getitem_172, getitem_173, getitem_174, getitem_175, getitem_176, getitem_177, getitem_178, getitem_179, getitem_180, getitem_181, getitem_182, getitem_183, getitem_184, getitem_185, getitem_186, getitem_187, getitem_188, getitem_189, getitem_190, getitem_191, getitem_192, getitem_193, getitem_194, getitem_195, getitem_196, getitem_197, getitem_198, getitem_199, getitem_200, getitem_201, getitem_202, getitem_203, getitem_204, getitem_205, getitem_206, getitem_207, getitem_208, getitem_209, getitem_210, getitem_211, getitem_212, getitem_213, getitem_214, getitem_215, getitem_216, getitem_217, getitem_218, getitem_219, getitem_220, getitem_221, getitem_222, getitem_223, getitem_224, getitem_225, getitem_226, getitem_227, getitem_228, getitem_229, getitem_230, getitem_231, getitem_232, getitem_233, getitem_234, getitem_235, getitem_236, getitem_237, getitem_238, getitem_239, getitem_240, getitem_241, getitem_242, getitem_243, getitem_244, getitem_245, getitem_246, getitem_247, getitem_248, getitem_249, getitem_250, getitem_251, getitem_252, getitem_253, getitem_254, getitem_255, getitem_256, getitem_257, getitem_258, getitem_259, getitem_260, getitem_261]);  arg134_1 = arg132_1 = arg266_1 = arg267_1 = arg268_1 = arg269_1 = arg270_1 = arg271_1 = arg272_1 = arg273_1 = arg274_1 = arg275_1 = arg276_1 = arg277_1 = arg278_1 = arg279_1 = arg280_1 = arg281_1 = arg282_1 = arg283_1 = arg284_1 = arg285_1 = arg286_1 = arg287_1 = arg288_1 = arg289_1 = arg290_1 = arg291_1 = arg292_1 = arg293_1 = arg294_1 = arg295_1 = arg296_1 = arg297_1 = arg298_1 = arg299_1 = arg300_1 = arg301_1 = arg302_1 = arg303_1 = arg304_1 = arg305_1 = arg306_1 = arg307_1 = arg308_1 = arg309_1 = arg310_1 = arg311_1 = arg312_1 = arg313_1 = arg314_1 = arg315_1 = arg316_1 = arg317_1 = arg318_1 = arg319_1 = arg320_1 = arg321_1 = arg322_1 = arg323_1 = arg324_1 = arg325_1 = arg326_1 = arg327_1 = arg328_1 = arg329_1 = arg330_1 = arg331_1 = arg332_1 = arg333_1 = arg334_1 = arg335_1 = arg336_1 = arg337_1 = arg338_1 = arg339_1 = arg340_1 = arg341_1 = arg342_1 = arg343_1 = arg344_1 = arg345_1 = arg346_1 = arg347_1 = arg348_1 = arg349_1 = arg350_1 = arg351_1 = arg352_1 = arg353_1 = arg354_1 = arg355_1 = arg356_1 = arg357_1 = arg358_1 = arg359_1 = arg360_1 = arg361_1 = arg362_1 = arg363_1 = arg364_1 = arg365_1 = arg366_1 = arg367_1 = arg368_1 = arg369_1 = arg370_1 = arg371_1 = arg372_1 = arg373_1 = arg374_1 = arg375_1 = arg376_1 = arg377_1 = arg378_1 = arg379_1 = arg380_1 = arg381_1 = arg382_1 = arg383_1 = arg384_1 = arg385_1 = arg386_1 = arg387_1 = arg388_1 = arg389_1 = arg390_1 = arg391_1 = arg392_1 = arg393_1 = arg394_1 = full_default = full_default_1 = full_default_2 = full_default_3 = full_default_4 = full_default_5 = full_default_6 = full_default_7 = full_default_8 = full_default_9 = full_default_10 = full_default_11 = full_default_12 = full_default_13 = full_default_14 = full_default_15 = full_default_16 = full_default_17 = full_default_18 = full_default_19 = full_default_20 = full_default_21 = full_default_22 = full_default_23 = full_default_24 = full_default_25 = full_default_26 = full_default_27 = full_default_28 = full_default_29 = full_default_30 = full_default_31 = full_default_32 = full_default_33 = full_default_34 = full_default_35 = full_default_36 = full_default_37 = full_default_38 = full_default_39 = full_default_40 = full_default_41 = full_default_42 = full_default_43 = full_default_44 = full_default_45 = full_default_46 = full_default_47 = full_default_48 = full_default_49 = full_default_50 = full_default_51 = full_default_52 = full_default_53 = full_default_54 = full_default_55 = full_default_56 = full_default_57 = full_default_58 = full_default_59 = full_default_60 = full_default_61 = full_default_62 = full_default_63 = full_default_64 = full_default_65 = full_default_66 = full_default_67 = full_default_68 = full_default_69 = full_default_70 = full_default_71 = full_default_72 = full_default_73 = full_default_74 = full_default_75 = full_default_76 = full_default_77 = full_default_78 = full_default_79 = full_default_80 = full_default_81 = full_default_82 = full_default_83 = full_default_84 = full_default_85 = full_default_86 = full_default_87 = full_default_88 = full_default_89 = full_default_90 = full_default_91 = full_default_92 = full_default_93 = full_default_94 = full_default_95 = full_default_96 = full_default_97 = full_default_98 = full_default_99 = full_default_100 = full_default_101 = full_default_102 = full_default_103 = full_default_104 = full_default_105 = full_default_106 = full_default_107 = full_default_108 = full_default_109 = full_default_110 = full_default_111 = full_default_112 = full_default_113 = full_default_114 = full_default_115 = full_default_116 = full_default_117 = full_default_118 = full_default_119 = full_default_120 = full_default_121 = full_default_122 = full_default_123 = full_default_124 = full_default_125 = full_default_126 = full_default_127 = full_default_128 = full_default_129 = full_default_130 = getitem_131 = getitem_132 = getitem_133 = getitem_134 = getitem_135 = getitem_136 = getitem_137 = getitem_138 = getitem_139 = getitem_140 = getitem_141 = getitem_142 = getitem_143 = getitem_144 = getitem_145 = getitem_146 = getitem_147 = getitem_148 = getitem_149 = getitem_150 = getitem_151 = getitem_152 = getitem_153 = getitem_154 = getitem_155 = getitem_156 = getitem_157 = getitem_158 = getitem_159 = getitem_160 = getitem_161 = getitem_162 = getitem_163 = getitem_164 = getitem_165 = getitem_166 = getitem_167 = getitem_168 = getitem_169 = getitem_170 = getitem_171 = getitem_172 = getitem_173 = getitem_174 = getitem_175 = getitem_176 = getitem_177 = getitem_178 = getitem_179 = getitem_180 = getitem_181 = getitem_182 = getitem_183 = getitem_184 = getitem_185 = getitem_186 = getitem_187 = getitem_188 = getitem_189 = getitem_190 = getitem_191 = getitem_192 = getitem_193 = getitem_194 = getitem_195 = getitem_196 = getitem_197 = getitem_198 = getitem_199 = getitem_200 = getitem_201 = getitem_202 = getitem_203 = getitem_204 = getitem_205 = getitem_206 = getitem_207 = getitem_208 = getitem_209 = getitem_210 = getitem_211 = getitem_212 = getitem_213 = getitem_214 = getitem_215 = getitem_216 = getitem_217 = getitem_218 = getitem_219 = getitem_220 = getitem_221 = getitem_222 = getitem_223 = getitem_224 = getitem_225 = getitem_226 = getitem_227 = getitem_228 = getitem_229 = getitem_230 = getitem_231 = getitem_232 = getitem_233 = getitem_234 = getitem_235 = getitem_236 = getitem_237 = getitem_238 = getitem_239 = getitem_240 = getitem_241 = getitem_242 = getitem_243 = getitem_244 = getitem_245 = getitem_246 = getitem_247 = getitem_248 = getitem_249 = getitem_250 = getitem_251 = getitem_252 = getitem_253 = getitem_254 = getitem_255 = getitem_256 = getitem_257 = getitem_258 = getitem_259 = getitem_260 = getitem_261 = None
        getitem: "f32[32128, 512]" = _foreach_addcmul_scalar[0]
        getitem_262: "f32[512, 512]" = _foreach_addcmul_scalar[1]
        getitem_263: "f32[512, 512]" = _foreach_addcmul_scalar[2]
        getitem_264: "f32[512, 512]" = _foreach_addcmul_scalar[3]
        getitem_265: "f32[512, 512]" = _foreach_addcmul_scalar[4]
        getitem_266: "f32[32, 8]" = _foreach_addcmul_scalar[5]
        getitem_267: "f32[512]" = _foreach_addcmul_scalar[6]
        getitem_268: "f32[2048, 512]" = _foreach_addcmul_scalar[7]
        getitem_269: "f32[512, 2048]" = _foreach_addcmul_scalar[8]
        getitem_270: "f32[512]" = _foreach_addcmul_scalar[9]
        getitem_271: "f32[512, 512]" = _foreach_addcmul_scalar[10]
        getitem_272: "f32[512, 512]" = _foreach_addcmul_scalar[11]
        getitem_273: "f32[512, 512]" = _foreach_addcmul_scalar[12]
        getitem_274: "f32[512, 512]" = _foreach_addcmul_scalar[13]
        getitem_275: "f32[512]" = _foreach_addcmul_scalar[14]
        getitem_276: "f32[2048, 512]" = _foreach_addcmul_scalar[15]
        getitem_277: "f32[512, 2048]" = _foreach_addcmul_scalar[16]
        getitem_278: "f32[512]" = _foreach_addcmul_scalar[17]
        getitem_279: "f32[512, 512]" = _foreach_addcmul_scalar[18]
        getitem_280: "f32[512, 512]" = _foreach_addcmul_scalar[19]
        getitem_281: "f32[512, 512]" = _foreach_addcmul_scalar[20]
        getitem_282: "f32[512, 512]" = _foreach_addcmul_scalar[21]
        getitem_283: "f32[512]" = _foreach_addcmul_scalar[22]
        getitem_284: "f32[2048, 512]" = _foreach_addcmul_scalar[23]
        getitem_285: "f32[512, 2048]" = _foreach_addcmul_scalar[24]
        getitem_286: "f32[512]" = _foreach_addcmul_scalar[25]
        getitem_287: "f32[512, 512]" = _foreach_addcmul_scalar[26]
        getitem_288: "f32[512, 512]" = _foreach_addcmul_scalar[27]
        getitem_289: "f32[512, 512]" = _foreach_addcmul_scalar[28]
        getitem_290: "f32[512, 512]" = _foreach_addcmul_scalar[29]
        getitem_291: "f32[512]" = _foreach_addcmul_scalar[30]
        getitem_292: "f32[2048, 512]" = _foreach_addcmul_scalar[31]
        getitem_293: "f32[512, 2048]" = _foreach_addcmul_scalar[32]
        getitem_294: "f32[512]" = _foreach_addcmul_scalar[33]
        getitem_295: "f32[512, 512]" = _foreach_addcmul_scalar[34]
        getitem_296: "f32[512, 512]" = _foreach_addcmul_scalar[35]
        getitem_297: "f32[512, 512]" = _foreach_addcmul_scalar[36]
        getitem_298: "f32[512, 512]" = _foreach_addcmul_scalar[37]
        getitem_299: "f32[512]" = _foreach_addcmul_scalar[38]
        getitem_300: "f32[2048, 512]" = _foreach_addcmul_scalar[39]
        getitem_301: "f32[512, 2048]" = _foreach_addcmul_scalar[40]
        getitem_302: "f32[512]" = _foreach_addcmul_scalar[41]
        getitem_303: "f32[512, 512]" = _foreach_addcmul_scalar[42]
        getitem_304: "f32[512, 512]" = _foreach_addcmul_scalar[43]
        getitem_305: "f32[512, 512]" = _foreach_addcmul_scalar[44]
        getitem_306: "f32[512, 512]" = _foreach_addcmul_scalar[45]
        getitem_307: "f32[512]" = _foreach_addcmul_scalar[46]
        getitem_308: "f32[2048, 512]" = _foreach_addcmul_scalar[47]
        getitem_309: "f32[512, 2048]" = _foreach_addcmul_scalar[48]
        getitem_310: "f32[512]" = _foreach_addcmul_scalar[49]
        getitem_311: "f32[512]" = _foreach_addcmul_scalar[50]
        getitem_312: "f32[512, 512]" = _foreach_addcmul_scalar[51]
        getitem_313: "f32[512, 512]" = _foreach_addcmul_scalar[52]
        getitem_314: "f32[512, 512]" = _foreach_addcmul_scalar[53]
        getitem_315: "f32[512, 512]" = _foreach_addcmul_scalar[54]
        getitem_316: "f32[32, 8]" = _foreach_addcmul_scalar[55]
        getitem_317: "f32[512]" = _foreach_addcmul_scalar[56]
        getitem_318: "f32[512, 512]" = _foreach_addcmul_scalar[57]
        getitem_319: "f32[512, 512]" = _foreach_addcmul_scalar[58]
        getitem_320: "f32[512, 512]" = _foreach_addcmul_scalar[59]
        getitem_321: "f32[512, 512]" = _foreach_addcmul_scalar[60]
        getitem_322: "f32[512]" = _foreach_addcmul_scalar[61]
        getitem_323: "f32[2048, 512]" = _foreach_addcmul_scalar[62]
        getitem_324: "f32[512, 2048]" = _foreach_addcmul_scalar[63]
        getitem_325: "f32[512]" = _foreach_addcmul_scalar[64]
        getitem_326: "f32[512, 512]" = _foreach_addcmul_scalar[65]
        getitem_327: "f32[512, 512]" = _foreach_addcmul_scalar[66]
        getitem_328: "f32[512, 512]" = _foreach_addcmul_scalar[67]
        getitem_329: "f32[512, 512]" = _foreach_addcmul_scalar[68]
        getitem_330: "f32[512]" = _foreach_addcmul_scalar[69]
        getitem_331: "f32[512, 512]" = _foreach_addcmul_scalar[70]
        getitem_332: "f32[512, 512]" = _foreach_addcmul_scalar[71]
        getitem_333: "f32[512, 512]" = _foreach_addcmul_scalar[72]
        getitem_334: "f32[512, 512]" = _foreach_addcmul_scalar[73]
        getitem_335: "f32[512]" = _foreach_addcmul_scalar[74]
        getitem_336: "f32[2048, 512]" = _foreach_addcmul_scalar[75]
        getitem_337: "f32[512, 2048]" = _foreach_addcmul_scalar[76]
        getitem_338: "f32[512]" = _foreach_addcmul_scalar[77]
        getitem_339: "f32[512, 512]" = _foreach_addcmul_scalar[78]
        getitem_340: "f32[512, 512]" = _foreach_addcmul_scalar[79]
        getitem_341: "f32[512, 512]" = _foreach_addcmul_scalar[80]
        getitem_342: "f32[512, 512]" = _foreach_addcmul_scalar[81]
        getitem_343: "f32[512]" = _foreach_addcmul_scalar[82]
        getitem_344: "f32[512, 512]" = _foreach_addcmul_scalar[83]
        getitem_345: "f32[512, 512]" = _foreach_addcmul_scalar[84]
        getitem_346: "f32[512, 512]" = _foreach_addcmul_scalar[85]
        getitem_347: "f32[512, 512]" = _foreach_addcmul_scalar[86]
        getitem_348: "f32[512]" = _foreach_addcmul_scalar[87]
        getitem_349: "f32[2048, 512]" = _foreach_addcmul_scalar[88]
        getitem_350: "f32[512, 2048]" = _foreach_addcmul_scalar[89]
        getitem_351: "f32[512]" = _foreach_addcmul_scalar[90]
        getitem_352: "f32[512, 512]" = _foreach_addcmul_scalar[91]
        getitem_353: "f32[512, 512]" = _foreach_addcmul_scalar[92]
        getitem_354: "f32[512, 512]" = _foreach_addcmul_scalar[93]
        getitem_355: "f32[512, 512]" = _foreach_addcmul_scalar[94]
        getitem_356: "f32[512]" = _foreach_addcmul_scalar[95]
        getitem_357: "f32[512, 512]" = _foreach_addcmul_scalar[96]
        getitem_358: "f32[512, 512]" = _foreach_addcmul_scalar[97]
        getitem_359: "f32[512, 512]" = _foreach_addcmul_scalar[98]
        getitem_360: "f32[512, 512]" = _foreach_addcmul_scalar[99]
        getitem_361: "f32[512]" = _foreach_addcmul_scalar[100]
        getitem_362: "f32[2048, 512]" = _foreach_addcmul_scalar[101]
        getitem_363: "f32[512, 2048]" = _foreach_addcmul_scalar[102]
        getitem_364: "f32[512]" = _foreach_addcmul_scalar[103]
        getitem_365: "f32[512, 512]" = _foreach_addcmul_scalar[104]
        getitem_366: "f32[512, 512]" = _foreach_addcmul_scalar[105]
        getitem_367: "f32[512, 512]" = _foreach_addcmul_scalar[106]
        getitem_368: "f32[512, 512]" = _foreach_addcmul_scalar[107]
        getitem_369: "f32[512]" = _foreach_addcmul_scalar[108]
        getitem_370: "f32[512, 512]" = _foreach_addcmul_scalar[109]
        getitem_371: "f32[512, 512]" = _foreach_addcmul_scalar[110]
        getitem_372: "f32[512, 512]" = _foreach_addcmul_scalar[111]
        getitem_373: "f32[512, 512]" = _foreach_addcmul_scalar[112]
        getitem_374: "f32[512]" = _foreach_addcmul_scalar[113]
        getitem_375: "f32[2048, 512]" = _foreach_addcmul_scalar[114]
        getitem_376: "f32[512, 2048]" = _foreach_addcmul_scalar[115]
        getitem_377: "f32[512]" = _foreach_addcmul_scalar[116]
        getitem_378: "f32[512, 512]" = _foreach_addcmul_scalar[117]
        getitem_379: "f32[512, 512]" = _foreach_addcmul_scalar[118]
        getitem_380: "f32[512, 512]" = _foreach_addcmul_scalar[119]
        getitem_381: "f32[512, 512]" = _foreach_addcmul_scalar[120]
        getitem_382: "f32[512]" = _foreach_addcmul_scalar[121]
        getitem_383: "f32[512, 512]" = _foreach_addcmul_scalar[122]
        getitem_384: "f32[512, 512]" = _foreach_addcmul_scalar[123]
        getitem_385: "f32[512, 512]" = _foreach_addcmul_scalar[124]
        getitem_386: "f32[512, 512]" = _foreach_addcmul_scalar[125]
        getitem_387: "f32[512]" = _foreach_addcmul_scalar[126]
        getitem_388: "f32[2048, 512]" = _foreach_addcmul_scalar[127]
        getitem_389: "f32[512, 2048]" = _foreach_addcmul_scalar[128]
        getitem_390: "f32[512]" = _foreach_addcmul_scalar[129]
        getitem_391: "f32[512]" = _foreach_addcmul_scalar[130];  _foreach_addcmul_scalar = None
        _foreach_div_list = torch.ops.aten._foreach_div.List([getitem_1965, getitem_1966, getitem_1967, getitem_1968, getitem_1969, getitem_1970, getitem_1971, getitem_1972, getitem_1973, getitem_1974, getitem_1975, getitem_1976, getitem_1977, getitem_1978, getitem_1979, getitem_1980, getitem_1981, getitem_1982, getitem_1983, getitem_1984, getitem_1985, getitem_1986, getitem_1987, getitem_1988, getitem_1989, getitem_1990, getitem_1991, getitem_1992, getitem_1993, getitem_1994, getitem_1995, getitem_1996, getitem_1997, getitem_1998, getitem_1999, getitem_2000, getitem_2001, getitem_2002, getitem_2003, getitem_2004, getitem_2005, getitem_2006, getitem_2007, getitem_2008, getitem_2009, getitem_2010, getitem_2011, getitem_2012, getitem_2013, getitem_2014, getitem_2015, getitem_2016, getitem_2017, getitem_2018, getitem_2019, getitem_2020, getitem_2021, getitem_2022, getitem_2023, getitem_2024, getitem_2025, getitem_2026, getitem_2027, getitem_2028, getitem_2029, getitem_2030, getitem_2031, getitem_2032, getitem_2033, getitem_2034, getitem_2035, getitem_2036, getitem_2037, getitem_2038, getitem_2039, getitem_2040, getitem_2041, getitem_2042, getitem_2043, getitem_2044, getitem_2045, getitem_2046, getitem_2047, getitem_2048, getitem_2049, getitem_2050, getitem_2051, getitem_2052, getitem_2053, getitem_2054, getitem_2055, getitem_2056, getitem_2057, getitem_2058, getitem_2059, getitem_2060, getitem_2061, getitem_2062, getitem_2063, getitem_2064, getitem_2065, getitem_2066, getitem_2067, getitem_2068, getitem_2069, getitem_2070, getitem_2071, getitem_2072, getitem_2073, getitem_2074, getitem_2075, getitem_2076, getitem_2077, getitem_2078, getitem_2079, getitem_2080, getitem_2081, getitem_2082, getitem_2083, getitem_2084, getitem_2085, getitem_2086, getitem_2087, getitem_2088, getitem_2089, getitem_2090, getitem_2091, getitem_2092, getitem_2093, getitem_2094, getitem_2095], [getitem_1441, getitem_1442, getitem_1443, getitem_1444, getitem_1445, getitem_1446, getitem_1447, getitem_1448, getitem_1449, getitem_1450, getitem_1451, getitem_1452, getitem_1453, getitem_1454, getitem_1455, getitem_1456, getitem_1457, getitem_1458, getitem_1459, getitem_1460, getitem_1461, getitem_1462, getitem_1463, getitem_1464, getitem_1465, getitem_1466, getitem_1467, getitem_1468, getitem_1469, getitem_1470, getitem_1471, getitem_1472, getitem_1473, getitem_1474, getitem_1475, getitem_1476, getitem_1477, getitem_1478, getitem_1479, getitem_1480, getitem_1481, getitem_1482, getitem_1483, getitem_1484, getitem_1485, getitem_1486, getitem_1487, getitem_1488, getitem_1489, getitem_1490, getitem_1491, getitem_1492, getitem_1493, getitem_1494, getitem_1495, getitem_1496, getitem_1497, getitem_1498, getitem_1499, getitem_1500, getitem_1501, getitem_1502, getitem_1503, getitem_1504, getitem_1505, getitem_1506, getitem_1507, getitem_1508, getitem_1509, getitem_1510, getitem_1511, getitem_1512, getitem_1513, getitem_1514, getitem_1515, getitem_1516, getitem_1517, getitem_1518, getitem_1519, getitem_1520, getitem_1521, getitem_1522, getitem_1523, getitem_1524, getitem_1525, getitem_1526, getitem_1527, getitem_1528, getitem_1529, getitem_1530, getitem_1531, getitem_1532, getitem_1533, getitem_1534, getitem_1535, getitem_1536, getitem_1537, getitem_1538, getitem_1539, getitem_1540, getitem_1541, getitem_1542, getitem_1543, getitem_1544, getitem_1545, getitem_1546, getitem_1547, getitem_1548, getitem_1549, getitem_1550, getitem_1551, getitem_1552, getitem_1553, getitem_1554, getitem_1555, getitem_1556, getitem_1557, getitem_1558, getitem_1559, getitem_1560, getitem_1561, getitem_1562, getitem_1563, getitem_1564, getitem_1565, getitem_1566, getitem_1567, getitem_1568, getitem_1569, getitem_1570, getitem_1571]);  getitem_1965 = getitem_1966 = getitem_1967 = getitem_1968 = getitem_1969 = getitem_1970 = getitem_1971 = getitem_1972 = getitem_1973 = getitem_1974 = getitem_1975 = getitem_1976 = getitem_1977 = getitem_1978 = getitem_1979 = getitem_1980 = getitem_1981 = getitem_1982 = getitem_1983 = getitem_1984 = getitem_1985 = getitem_1986 = getitem_1987 = getitem_1988 = getitem_1989 = getitem_1990 = getitem_1991 = getitem_1992 = getitem_1993 = getitem_1994 = getitem_1995 = getitem_1996 = getitem_1997 = getitem_1998 = getitem_1999 = getitem_2000 = getitem_2001 = getitem_2002 = getitem_2003 = getitem_2004 = getitem_2005 = getitem_2006 = getitem_2007 = getitem_2008 = getitem_2009 = getitem_2010 = getitem_2011 = getitem_2012 = getitem_2013 = getitem_2014 = getitem_2015 = getitem_2016 = getitem_2017 = getitem_2018 = getitem_2019 = getitem_2020 = getitem_2021 = getitem_2022 = getitem_2023 = getitem_2024 = getitem_2025 = getitem_2026 = getitem_2027 = getitem_2028 = getitem_2029 = getitem_2030 = getitem_2031 = getitem_2032 = getitem_2033 = getitem_2034 = getitem_2035 = getitem_2036 = getitem_2037 = getitem_2038 = getitem_2039 = getitem_2040 = getitem_2041 = getitem_2042 = getitem_2043 = getitem_2044 = getitem_2045 = getitem_2046 = getitem_2047 = getitem_2048 = getitem_2049 = getitem_2050 = getitem_2051 = getitem_2052 = getitem_2053 = getitem_2054 = getitem_2055 = getitem_2056 = getitem_2057 = getitem_2058 = getitem_2059 = getitem_2060 = getitem_2061 = getitem_2062 = getitem_2063 = getitem_2064 = getitem_2065 = getitem_2066 = getitem_2067 = getitem_2068 = getitem_2069 = getitem_2070 = getitem_2071 = getitem_2072 = getitem_2073 = getitem_2074 = getitem_2075 = getitem_2076 = getitem_2077 = getitem_2078 = getitem_2079 = getitem_2080 = getitem_2081 = getitem_2082 = getitem_2083 = getitem_2084 = getitem_2085 = getitem_2086 = getitem_2087 = getitem_2088 = getitem_2089 = getitem_2090 = getitem_2091 = getitem_2092 = getitem_2093 = getitem_2094 = getitem_2095 = getitem_1441 = getitem_1442 = getitem_1443 = getitem_1444 = getitem_1445 = getitem_1446 = getitem_1447 = getitem_1448 = getitem_1449 = getitem_1450 = getitem_1451 = getitem_1452 = getitem_1453 = getitem_1454 = getitem_1455 = getitem_1456 = getitem_1457 = getitem_1458 = getitem_1459 = getitem_1460 = getitem_1461 = getitem_1462 = getitem_1463 = getitem_1464 = getitem_1465 = getitem_1466 = getitem_1467 = getitem_1468 = getitem_1469 = getitem_1470 = getitem_1471 = getitem_1472 = getitem_1473 = getitem_1474 = getitem_1475 = getitem_1476 = getitem_1477 = getitem_1478 = getitem_1479 = getitem_1480 = getitem_1481 = getitem_1482 = getitem_1483 = getitem_1484 = getitem_1485 = getitem_1486 = getitem_1487 = getitem_1488 = getitem_1489 = getitem_1490 = getitem_1491 = getitem_1492 = getitem_1493 = getitem_1494 = getitem_1495 = getitem_1496 = getitem_1497 = getitem_1498 = getitem_1499 = getitem_1500 = getitem_1501 = getitem_1502 = getitem_1503 = getitem_1504 = getitem_1505 = getitem_1506 = getitem_1507 = getitem_1508 = getitem_1509 = getitem_1510 = getitem_1511 = getitem_1512 = getitem_1513 = getitem_1514 = getitem_1515 = getitem_1516 = getitem_1517 = getitem_1518 = getitem_1519 = getitem_1520 = getitem_1521 = getitem_1522 = getitem_1523 = getitem_1524 = getitem_1525 = getitem_1526 = getitem_1527 = getitem_1528 = getitem_1529 = getitem_1530 = getitem_1531 = getitem_1532 = getitem_1533 = getitem_1534 = getitem_1535 = getitem_1536 = getitem_1537 = getitem_1538 = getitem_1539 = getitem_1540 = getitem_1541 = getitem_1542 = getitem_1543 = getitem_1544 = getitem_1545 = getitem_1546 = getitem_1547 = getitem_1548 = getitem_1549 = getitem_1550 = getitem_1551 = getitem_1552 = getitem_1553 = getitem_1554 = getitem_1555 = getitem_1556 = getitem_1557 = getitem_1558 = getitem_1559 = getitem_1560 = getitem_1561 = getitem_1562 = getitem_1563 = getitem_1564 = getitem_1565 = getitem_1566 = getitem_1567 = getitem_1568 = getitem_1569 = getitem_1570 = getitem_1571 = None
        getitem_1572: "f32[32128, 512]" = _foreach_div_list[0]
        getitem_1573: "f32[512, 512]" = _foreach_div_list[1]
        getitem_1574: "f32[512, 512]" = _foreach_div_list[2]
        getitem_1575: "f32[512, 512]" = _foreach_div_list[3]
        getitem_1576: "f32[512, 512]" = _foreach_div_list[4]
        getitem_1577: "f32[32, 8]" = _foreach_div_list[5]
        getitem_1578: "f32[512]" = _foreach_div_list[6]
        getitem_1579: "f32[2048, 512]" = _foreach_div_list[7]
        getitem_1580: "f32[512, 2048]" = _foreach_div_list[8]
        getitem_1581: "f32[512]" = _foreach_div_list[9]
        getitem_1582: "f32[512, 512]" = _foreach_div_list[10]
        getitem_1583: "f32[512, 512]" = _foreach_div_list[11]
        getitem_1584: "f32[512, 512]" = _foreach_div_list[12]
        getitem_1585: "f32[512, 512]" = _foreach_div_list[13]
        getitem_1586: "f32[512]" = _foreach_div_list[14]
        getitem_1587: "f32[2048, 512]" = _foreach_div_list[15]
        getitem_1588: "f32[512, 2048]" = _foreach_div_list[16]
        getitem_1589: "f32[512]" = _foreach_div_list[17]
        getitem_1590: "f32[512, 512]" = _foreach_div_list[18]
        getitem_1591: "f32[512, 512]" = _foreach_div_list[19]
        getitem_1592: "f32[512, 512]" = _foreach_div_list[20]
        getitem_1593: "f32[512, 512]" = _foreach_div_list[21]
        getitem_1594: "f32[512]" = _foreach_div_list[22]
        getitem_1595: "f32[2048, 512]" = _foreach_div_list[23]
        getitem_1596: "f32[512, 2048]" = _foreach_div_list[24]
        getitem_1597: "f32[512]" = _foreach_div_list[25]
        getitem_1598: "f32[512, 512]" = _foreach_div_list[26]
        getitem_1599: "f32[512, 512]" = _foreach_div_list[27]
        getitem_1600: "f32[512, 512]" = _foreach_div_list[28]
        getitem_1601: "f32[512, 512]" = _foreach_div_list[29]
        getitem_1602: "f32[512]" = _foreach_div_list[30]
        getitem_1603: "f32[2048, 512]" = _foreach_div_list[31]
        getitem_1604: "f32[512, 2048]" = _foreach_div_list[32]
        getitem_1605: "f32[512]" = _foreach_div_list[33]
        getitem_1606: "f32[512, 512]" = _foreach_div_list[34]
        getitem_1607: "f32[512, 512]" = _foreach_div_list[35]
        getitem_1608: "f32[512, 512]" = _foreach_div_list[36]
        getitem_1609: "f32[512, 512]" = _foreach_div_list[37]
        getitem_1610: "f32[512]" = _foreach_div_list[38]
        getitem_1611: "f32[2048, 512]" = _foreach_div_list[39]
        getitem_1612: "f32[512, 2048]" = _foreach_div_list[40]
        getitem_1613: "f32[512]" = _foreach_div_list[41]
        getitem_1614: "f32[512, 512]" = _foreach_div_list[42]
        getitem_1615: "f32[512, 512]" = _foreach_div_list[43]
        getitem_1616: "f32[512, 512]" = _foreach_div_list[44]
        getitem_1617: "f32[512, 512]" = _foreach_div_list[45]
        getitem_1618: "f32[512]" = _foreach_div_list[46]
        getitem_1619: "f32[2048, 512]" = _foreach_div_list[47]
        getitem_1620: "f32[512, 2048]" = _foreach_div_list[48]
        getitem_1621: "f32[512]" = _foreach_div_list[49]
        getitem_1622: "f32[512]" = _foreach_div_list[50]
        getitem_1623: "f32[512, 512]" = _foreach_div_list[51]
        getitem_1624: "f32[512, 512]" = _foreach_div_list[52]
        getitem_1625: "f32[512, 512]" = _foreach_div_list[53]
        getitem_1626: "f32[512, 512]" = _foreach_div_list[54]
        getitem_1627: "f32[32, 8]" = _foreach_div_list[55]
        getitem_1628: "f32[512]" = _foreach_div_list[56]
        getitem_1629: "f32[512, 512]" = _foreach_div_list[57]
        getitem_1630: "f32[512, 512]" = _foreach_div_list[58]
        getitem_1631: "f32[512, 512]" = _foreach_div_list[59]
        getitem_1632: "f32[512, 512]" = _foreach_div_list[60]
        getitem_1633: "f32[512]" = _foreach_div_list[61]
        getitem_1634: "f32[2048, 512]" = _foreach_div_list[62]
        getitem_1635: "f32[512, 2048]" = _foreach_div_list[63]
        getitem_1636: "f32[512]" = _foreach_div_list[64]
        getitem_1637: "f32[512, 512]" = _foreach_div_list[65]
        getitem_1638: "f32[512, 512]" = _foreach_div_list[66]
        getitem_1639: "f32[512, 512]" = _foreach_div_list[67]
        getitem_1640: "f32[512, 512]" = _foreach_div_list[68]
        getitem_1641: "f32[512]" = _foreach_div_list[69]
        getitem_1642: "f32[512, 512]" = _foreach_div_list[70]
        getitem_1643: "f32[512, 512]" = _foreach_div_list[71]
        getitem_1644: "f32[512, 512]" = _foreach_div_list[72]
        getitem_1645: "f32[512, 512]" = _foreach_div_list[73]
        getitem_1646: "f32[512]" = _foreach_div_list[74]
        getitem_1647: "f32[2048, 512]" = _foreach_div_list[75]
        getitem_1648: "f32[512, 2048]" = _foreach_div_list[76]
        getitem_1649: "f32[512]" = _foreach_div_list[77]
        getitem_1650: "f32[512, 512]" = _foreach_div_list[78]
        getitem_1651: "f32[512, 512]" = _foreach_div_list[79]
        getitem_1652: "f32[512, 512]" = _foreach_div_list[80]
        getitem_1653: "f32[512, 512]" = _foreach_div_list[81]
        getitem_1654: "f32[512]" = _foreach_div_list[82]
        getitem_1655: "f32[512, 512]" = _foreach_div_list[83]
        getitem_1656: "f32[512, 512]" = _foreach_div_list[84]
        getitem_1657: "f32[512, 512]" = _foreach_div_list[85]
        getitem_1658: "f32[512, 512]" = _foreach_div_list[86]
        getitem_1659: "f32[512]" = _foreach_div_list[87]
        getitem_1660: "f32[2048, 512]" = _foreach_div_list[88]
        getitem_1661: "f32[512, 2048]" = _foreach_div_list[89]
        getitem_1662: "f32[512]" = _foreach_div_list[90]
        getitem_1663: "f32[512, 512]" = _foreach_div_list[91]
        getitem_1664: "f32[512, 512]" = _foreach_div_list[92]
        getitem_1665: "f32[512, 512]" = _foreach_div_list[93]
        getitem_1666: "f32[512, 512]" = _foreach_div_list[94]
        getitem_1667: "f32[512]" = _foreach_div_list[95]
        getitem_1668: "f32[512, 512]" = _foreach_div_list[96]
        getitem_1669: "f32[512, 512]" = _foreach_div_list[97]
        getitem_1670: "f32[512, 512]" = _foreach_div_list[98]
        getitem_1671: "f32[512, 512]" = _foreach_div_list[99]
        getitem_1672: "f32[512]" = _foreach_div_list[100]
        getitem_1673: "f32[2048, 512]" = _foreach_div_list[101]
        getitem_1674: "f32[512, 2048]" = _foreach_div_list[102]
        getitem_1675: "f32[512]" = _foreach_div_list[103]
        getitem_1676: "f32[512, 512]" = _foreach_div_list[104]
        getitem_1677: "f32[512, 512]" = _foreach_div_list[105]
        getitem_1678: "f32[512, 512]" = _foreach_div_list[106]
        getitem_1679: "f32[512, 512]" = _foreach_div_list[107]
        getitem_1680: "f32[512]" = _foreach_div_list[108]
        getitem_1681: "f32[512, 512]" = _foreach_div_list[109]
        getitem_1682: "f32[512, 512]" = _foreach_div_list[110]
        getitem_1683: "f32[512, 512]" = _foreach_div_list[111]
        getitem_1684: "f32[512, 512]" = _foreach_div_list[112]
        getitem_1685: "f32[512]" = _foreach_div_list[113]
        getitem_1686: "f32[2048, 512]" = _foreach_div_list[114]
        getitem_1687: "f32[512, 2048]" = _foreach_div_list[115]
        getitem_1688: "f32[512]" = _foreach_div_list[116]
        getitem_1689: "f32[512, 512]" = _foreach_div_list[117]
        getitem_1690: "f32[512, 512]" = _foreach_div_list[118]
        getitem_1691: "f32[512, 512]" = _foreach_div_list[119]
        getitem_1692: "f32[512, 512]" = _foreach_div_list[120]
        getitem_1693: "f32[512]" = _foreach_div_list[121]
        getitem_1694: "f32[512, 512]" = _foreach_div_list[122]
        getitem_1695: "f32[512, 512]" = _foreach_div_list[123]
        getitem_1696: "f32[512, 512]" = _foreach_div_list[124]
        getitem_1697: "f32[512, 512]" = _foreach_div_list[125]
        getitem_1698: "f32[512]" = _foreach_div_list[126]
        getitem_1699: "f32[2048, 512]" = _foreach_div_list[127]
        getitem_1700: "f32[512, 2048]" = _foreach_div_list[128]
        getitem_1701: "f32[512]" = _foreach_div_list[129]
        getitem_1702: "f32[512]" = _foreach_div_list[130];  _foreach_div_list = None
        return (getitem, getitem_262, getitem_263, getitem_264, getitem_265, getitem_266, getitem_267, getitem_268, getitem_269, getitem_270, getitem_271, getitem_272, getitem_273, getitem_274, getitem_275, getitem_276, getitem_277, getitem_278, getitem_279, getitem_280, getitem_281, getitem_282, getitem_283, getitem_284, getitem_285, getitem_286, getitem_287, getitem_288, getitem_289, getitem_290, getitem_291, getitem_292, getitem_293, getitem_294, getitem_295, getitem_296, getitem_297, getitem_298, getitem_299, getitem_300, getitem_301, getitem_302, getitem_303, getitem_304, getitem_305, getitem_306, getitem_307, getitem_308, getitem_309, getitem_310, getitem_311, getitem_312, getitem_313, getitem_314, getitem_315, getitem_316, getitem_317, getitem_318, getitem_319, getitem_320, getitem_321, getitem_322, getitem_323, getitem_324, getitem_325, getitem_326, getitem_327, getitem_328, getitem_329, getitem_330, getitem_331, getitem_332, getitem_333, getitem_334, getitem_335, getitem_336, getitem_337, getitem_338, getitem_339, getitem_340, getitem_341, getitem_342, getitem_343, getitem_344, getitem_345, getitem_346, getitem_347, getitem_348, getitem_349, getitem_350, getitem_351, getitem_352, getitem_353, getitem_354, getitem_355, getitem_356, getitem_357, getitem_358, getitem_359, getitem_360, getitem_361, getitem_362, getitem_363, getitem_364, getitem_365, getitem_366, getitem_367, getitem_368, getitem_369, getitem_370, getitem_371, getitem_372, getitem_373, getitem_374, getitem_375, getitem_376, getitem_377, getitem_378, getitem_379, getitem_380, getitem_381, getitem_382, getitem_383, getitem_384, getitem_385, getitem_386, getitem_387, getitem_388, getitem_389, getitem_390, getitem_391, getitem_1572, getitem_1573, getitem_1574, getitem_1575, getitem_1576, getitem_1577, getitem_1578, getitem_1579, getitem_1580, getitem_1581, getitem_1582, getitem_1583, getitem_1584, getitem_1585, getitem_1586, getitem_1587, getitem_1588, getitem_1589, getitem_1590, getitem_1591, getitem_1592, getitem_1593, getitem_1594, getitem_1595, getitem_1596, getitem_1597, getitem_1598, getitem_1599, getitem_1600, getitem_1601, getitem_1602, getitem_1603, getitem_1604, getitem_1605, getitem_1606, getitem_1607, getitem_1608, getitem_1609, getitem_1610, getitem_1611, getitem_1612, getitem_1613, getitem_1614, getitem_1615, getitem_1616, getitem_1617, getitem_1618, getitem_1619, getitem_1620, getitem_1621, getitem_1622, getitem_1623, getitem_1624, getitem_1625, getitem_1626, getitem_1627, getitem_1628, getitem_1629, getitem_1630, getitem_1631, getitem_1632, getitem_1633, getitem_1634, getitem_1635, getitem_1636, getitem_1637, getitem_1638, getitem_1639, getitem_1640, getitem_1641, getitem_1642, getitem_1643, getitem_1644, getitem_1645, getitem_1646, getitem_1647, getitem_1648, getitem_1649, getitem_1650, getitem_1651, getitem_1652, getitem_1653, getitem_1654, getitem_1655, getitem_1656, getitem_1657, getitem_1658, getitem_1659, getitem_1660, getitem_1661, getitem_1662, getitem_1663, getitem_1664, getitem_1665, getitem_1666, getitem_1667, getitem_1668, getitem_1669, getitem_1670, getitem_1671, getitem_1672, getitem_1673, getitem_1674, getitem_1675, getitem_1676, getitem_1677, getitem_1678, getitem_1679, getitem_1680, getitem_1681, getitem_1682, getitem_1683, getitem_1684, getitem_1685, getitem_1686, getitem_1687, getitem_1688, getitem_1689, getitem_1690, getitem_1691, getitem_1692, getitem_1693, getitem_1694, getitem_1695, getitem_1696, getitem_1697, getitem_1698, getitem_1699, getitem_1700, getitem_1701, getitem_1702)


def _default_make_inputs():
    return [
    torch.randn([32128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 8], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 8], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([32128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 8], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 8], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([32128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 8], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 8], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
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
