"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s2_g53
Pattern hash: 2402427cc20e
Shape hash: fd6a4668
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg596_1: "f32[50257, 768]", arg597_1: "f32[1024, 768]", arg598_1: "f32[768]", arg599_1: "f32[768]", arg600_1: "f32[768, 2304]", arg601_1: "f32[2304]", arg602_1: "f32[768, 768]", arg603_1: "f32[768]", arg604_1: "f32[768]", arg605_1: "f32[768]", arg606_1: "f32[768, 3072]", arg607_1: "f32[3072]", arg608_1: "f32[3072, 768]", arg609_1: "f32[768]", arg610_1: "f32[768]", arg611_1: "f32[768]", arg612_1: "f32[768, 2304]", arg613_1: "f32[2304]", arg614_1: "f32[768, 768]", arg615_1: "f32[768]", arg616_1: "f32[768]", arg617_1: "f32[768]", arg618_1: "f32[768, 3072]", arg619_1: "f32[3072]", arg620_1: "f32[3072, 768]", arg621_1: "f32[768]", arg622_1: "f32[768]", arg623_1: "f32[768]", arg624_1: "f32[768, 2304]", arg625_1: "f32[2304]", arg626_1: "f32[768, 768]", arg627_1: "f32[768]", arg628_1: "f32[768]", arg629_1: "f32[768]", arg630_1: "f32[768, 3072]", arg631_1: "f32[3072]", arg632_1: "f32[3072, 768]", arg633_1: "f32[768]", arg634_1: "f32[768]", arg635_1: "f32[768]", arg636_1: "f32[768, 2304]", arg637_1: "f32[2304]", arg638_1: "f32[768, 768]", arg639_1: "f32[768]", arg640_1: "f32[768]", arg641_1: "f32[768]", arg642_1: "f32[768, 3072]", arg643_1: "f32[3072]", arg644_1: "f32[3072, 768]", arg645_1: "f32[768]", arg646_1: "f32[768]", arg647_1: "f32[768]", arg648_1: "f32[768, 2304]", arg649_1: "f32[2304]", arg650_1: "f32[768, 768]", arg651_1: "f32[768]", arg652_1: "f32[768]", arg653_1: "f32[768]", arg654_1: "f32[768, 3072]", arg655_1: "f32[3072]", arg656_1: "f32[3072, 768]", arg657_1: "f32[768]", arg658_1: "f32[768]", arg659_1: "f32[768]", arg660_1: "f32[768, 2304]", arg661_1: "f32[2304]", arg662_1: "f32[768, 768]", arg663_1: "f32[768]", arg664_1: "f32[768]", arg665_1: "f32[768]", arg666_1: "f32[768, 3072]", arg667_1: "f32[3072]", arg668_1: "f32[3072, 768]", arg669_1: "f32[768]", arg670_1: "f32[768]", arg671_1: "f32[768]", arg672_1: "f32[768, 2304]", arg673_1: "f32[2304]", arg674_1: "f32[768, 768]", arg675_1: "f32[768]", arg676_1: "f32[768]", arg677_1: "f32[768]", arg678_1: "f32[768, 3072]", arg679_1: "f32[3072]", arg680_1: "f32[3072, 768]", arg681_1: "f32[768]", arg682_1: "f32[768]", arg683_1: "f32[768]", arg684_1: "f32[768, 2304]", arg685_1: "f32[2304]", arg686_1: "f32[768, 768]", arg687_1: "f32[768]", arg688_1: "f32[768]", arg689_1: "f32[768]", arg690_1: "f32[768, 3072]", arg691_1: "f32[3072]", arg692_1: "f32[3072, 768]", arg693_1: "f32[768]", arg694_1: "f32[768]", arg695_1: "f32[768]", arg696_1: "f32[768, 2304]", arg697_1: "f32[2304]", arg698_1: "f32[768, 768]", arg699_1: "f32[768]", arg700_1: "f32[768]", arg701_1: "f32[768]", arg702_1: "f32[768, 3072]", arg703_1: "f32[3072]", arg704_1: "f32[3072, 768]", arg705_1: "f32[768]", arg706_1: "f32[768]", arg707_1: "f32[768]", arg708_1: "f32[768, 2304]", arg709_1: "f32[2304]", arg710_1: "f32[768, 768]", arg711_1: "f32[768]", arg712_1: "f32[768]", arg713_1: "f32[768]", arg714_1: "f32[768, 3072]", arg715_1: "f32[3072]", arg716_1: "f32[3072, 768]", arg717_1: "f32[768]", arg718_1: "f32[768]", arg719_1: "f32[768]", arg720_1: "f32[768, 2304]", arg721_1: "f32[2304]", arg722_1: "f32[768, 768]", arg723_1: "f32[768]", arg724_1: "f32[768]", arg725_1: "f32[768]", arg726_1: "f32[768, 3072]", arg727_1: "f32[3072]", arg728_1: "f32[3072, 768]", arg729_1: "f32[768]", arg730_1: "f32[768]", arg731_1: "f32[768]", arg732_1: "f32[768, 2304]", arg733_1: "f32[2304]", arg734_1: "f32[768, 768]", arg735_1: "f32[768]", arg736_1: "f32[768]", arg737_1: "f32[768]", arg738_1: "f32[768, 3072]", arg739_1: "f32[3072]", arg740_1: "f32[3072, 768]", arg741_1: "f32[768]", arg742_1: "f32[768]", arg743_1: "f32[768]", arg744_1: "f32[2, 768]", arg152_1: "f32[50257, 768]", arg150_1: "f32[1024, 768]", arg302_1: "f32[768]", arg303_1: "f32[768]", arg304_1: "f32[768, 2304]", arg305_1: "f32[2304]", arg306_1: "f32[768, 768]", arg307_1: "f32[768]", arg308_1: "f32[768]", arg309_1: "f32[768]", arg310_1: "f32[768, 3072]", arg311_1: "f32[3072]", arg312_1: "f32[3072, 768]", arg313_1: "f32[768]", arg314_1: "f32[768]", arg315_1: "f32[768]", arg316_1: "f32[768, 2304]", arg317_1: "f32[2304]", arg318_1: "f32[768, 768]", arg319_1: "f32[768]", arg320_1: "f32[768]", arg321_1: "f32[768]", arg322_1: "f32[768, 3072]", arg323_1: "f32[3072]", arg324_1: "f32[3072, 768]", arg325_1: "f32[768]", arg326_1: "f32[768]", arg327_1: "f32[768]", arg328_1: "f32[768, 2304]", arg329_1: "f32[2304]", arg330_1: "f32[768, 768]", arg331_1: "f32[768]", arg332_1: "f32[768]", arg333_1: "f32[768]", arg334_1: "f32[768, 3072]", arg335_1: "f32[3072]", arg336_1: "f32[3072, 768]", arg337_1: "f32[768]", arg338_1: "f32[768]", arg339_1: "f32[768]", arg340_1: "f32[768, 2304]", arg341_1: "f32[2304]", arg342_1: "f32[768, 768]", arg343_1: "f32[768]", arg344_1: "f32[768]", arg345_1: "f32[768]", arg346_1: "f32[768, 3072]", arg347_1: "f32[3072]", arg348_1: "f32[3072, 768]", arg349_1: "f32[768]", arg350_1: "f32[768]", arg351_1: "f32[768]", arg352_1: "f32[768, 2304]", arg353_1: "f32[2304]", arg354_1: "f32[768, 768]", arg355_1: "f32[768]", arg356_1: "f32[768]", arg357_1: "f32[768]", arg358_1: "f32[768, 3072]", arg359_1: "f32[3072]", arg360_1: "f32[3072, 768]", arg361_1: "f32[768]", arg362_1: "f32[768]", arg363_1: "f32[768]", arg364_1: "f32[768, 2304]", arg365_1: "f32[2304]", arg366_1: "f32[768, 768]", arg367_1: "f32[768]", arg368_1: "f32[768]", arg369_1: "f32[768]", arg370_1: "f32[768, 3072]", arg371_1: "f32[3072]", arg372_1: "f32[3072, 768]", arg373_1: "f32[768]", arg374_1: "f32[768]", arg375_1: "f32[768]", arg376_1: "f32[768, 2304]", arg377_1: "f32[2304]", arg378_1: "f32[768, 768]", arg379_1: "f32[768]", arg380_1: "f32[768]", arg381_1: "f32[768]", arg382_1: "f32[768, 3072]", arg383_1: "f32[3072]", arg384_1: "f32[3072, 768]", arg385_1: "f32[768]", arg386_1: "f32[768]", arg387_1: "f32[768]", arg388_1: "f32[768, 2304]", arg389_1: "f32[2304]", arg390_1: "f32[768, 768]", arg391_1: "f32[768]", arg392_1: "f32[768]", arg393_1: "f32[768]", arg394_1: "f32[768, 3072]", arg395_1: "f32[3072]", arg396_1: "f32[3072, 768]", arg397_1: "f32[768]", arg398_1: "f32[768]", arg399_1: "f32[768]", arg400_1: "f32[768, 2304]", arg401_1: "f32[2304]", arg402_1: "f32[768, 768]", arg403_1: "f32[768]", arg404_1: "f32[768]", arg405_1: "f32[768]", arg406_1: "f32[768, 3072]", arg407_1: "f32[3072]", arg408_1: "f32[3072, 768]", arg409_1: "f32[768]", arg410_1: "f32[768]", arg411_1: "f32[768]", arg412_1: "f32[768, 2304]", arg413_1: "f32[2304]", arg414_1: "f32[768, 768]", arg415_1: "f32[768]", arg416_1: "f32[768]", arg417_1: "f32[768]", arg418_1: "f32[768, 3072]", arg419_1: "f32[3072]", arg420_1: "f32[3072, 768]", arg421_1: "f32[768]", arg422_1: "f32[768]", arg423_1: "f32[768]", arg424_1: "f32[768, 2304]", arg425_1: "f32[2304]", arg426_1: "f32[768, 768]", arg427_1: "f32[768]", arg428_1: "f32[768]", arg429_1: "f32[768]", arg430_1: "f32[768, 3072]", arg431_1: "f32[3072]", arg432_1: "f32[3072, 768]", arg433_1: "f32[768]", arg434_1: "f32[768]", arg435_1: "f32[768]", arg436_1: "f32[768, 2304]", arg437_1: "f32[2304]", arg438_1: "f32[768, 768]", arg439_1: "f32[768]", arg440_1: "f32[768]", arg441_1: "f32[768]", arg442_1: "f32[768, 3072]", arg443_1: "f32[3072]", arg444_1: "f32[3072, 768]", arg445_1: "f32[768]", arg446_1: "f32[768]", arg447_1: "f32[768]", arg448_1: "f32[2, 768]", getitem_1490: "f32[]", getitem_1491: "f32[]", getitem_1492: "f32[]", getitem_1493: "f32[]", getitem_1494: "f32[]", getitem_1495: "f32[]", getitem_1496: "f32[]", getitem_1497: "f32[]", getitem_1498: "f32[]", getitem_1499: "f32[]", getitem_1500: "f32[]", getitem_1501: "f32[]", getitem_1502: "f32[]", getitem_1503: "f32[]", getitem_1504: "f32[]", getitem_1505: "f32[]", getitem_1506: "f32[]", getitem_1507: "f32[]", getitem_1508: "f32[]", getitem_1509: "f32[]", getitem_1510: "f32[]", getitem_1511: "f32[]", getitem_1512: "f32[]", getitem_1513: "f32[]", getitem_1514: "f32[]", getitem_1515: "f32[]", getitem_1516: "f32[]", getitem_1517: "f32[]", getitem_1518: "f32[]", getitem_1519: "f32[]", getitem_1520: "f32[]", getitem_1521: "f32[]", getitem_1522: "f32[]", getitem_1523: "f32[]", getitem_1524: "f32[]", getitem_1525: "f32[]", getitem_1526: "f32[]", getitem_1527: "f32[]", getitem_1528: "f32[]", getitem_1529: "f32[]", getitem_1530: "f32[]", getitem_1531: "f32[]", getitem_1532: "f32[]", getitem_1533: "f32[]", getitem_1534: "f32[]", getitem_1535: "f32[]", getitem_1536: "f32[]", getitem_1537: "f32[]", getitem_1538: "f32[]", getitem_1539: "f32[]", getitem_1540: "f32[]", getitem_1541: "f32[]", getitem_1542: "f32[]", getitem_1543: "f32[]", getitem_1544: "f32[]", getitem_1545: "f32[]", getitem_1546: "f32[]", getitem_1547: "f32[]", getitem_1548: "f32[]", getitem_1549: "f32[]", getitem_1550: "f32[]", getitem_1551: "f32[]", getitem_1552: "f32[]", getitem_1553: "f32[]", getitem_1554: "f32[]", getitem_1555: "f32[]", getitem_1556: "f32[]", getitem_1557: "f32[]", getitem_1558: "f32[]", getitem_1559: "f32[]", getitem_1560: "f32[]", getitem_1561: "f32[]", getitem_1562: "f32[]", getitem_1563: "f32[]", getitem_1564: "f32[]", getitem_1565: "f32[]", getitem_1566: "f32[]", getitem_1567: "f32[]", getitem_1568: "f32[]", getitem_1569: "f32[]", getitem_1570: "f32[]", getitem_1571: "f32[]", getitem_1572: "f32[]", getitem_1573: "f32[]", getitem_1574: "f32[]", getitem_1575: "f32[]", getitem_1576: "f32[]", getitem_1577: "f32[]", getitem_1578: "f32[]", getitem_1579: "f32[]", getitem_1580: "f32[]", getitem_1581: "f32[]", getitem_1582: "f32[]", getitem_1583: "f32[]", getitem_1584: "f32[]", getitem_1585: "f32[]", getitem_1586: "f32[]", getitem_1587: "f32[]", getitem_1588: "f32[]", getitem_1589: "f32[]", getitem_1590: "f32[]", getitem_1591: "f32[]", getitem_1592: "f32[]", getitem_1593: "f32[]", getitem_1594: "f32[]", getitem_1595: "f32[]", getitem_1596: "f32[]", getitem_1597: "f32[]", getitem_1598: "f32[]", getitem_1599: "f32[]", getitem_1600: "f32[]", getitem_1601: "f32[]", getitem_1602: "f32[]", getitem_1603: "f32[]", getitem_1604: "f32[]", getitem_1605: "f32[]", getitem_1606: "f32[]", getitem_1607: "f32[]", getitem_1608: "f32[]", getitem_1609: "f32[]", getitem_1610: "f32[]", getitem_1611: "f32[]", getitem_1612: "f32[]", getitem_1613: "f32[]", getitem_1614: "f32[]", getitem_1615: "f32[]", getitem_1616: "f32[]", getitem_1617: "f32[]", getitem_1618: "f32[]", getitem_1619: "f32[]", getitem_1620: "f32[]", getitem_1621: "f32[]", getitem_1622: "f32[]", getitem_1623: "f32[]", getitem_1624: "f32[]", getitem_1625: "f32[]", getitem_1626: "f32[]", getitem_1627: "f32[]", getitem_1628: "f32[]", getitem_1629: "f32[]", getitem_1630: "f32[]", getitem_1631: "f32[]", getitem_1632: "f32[]", getitem_1633: "f32[]", getitem_1634: "f32[]", getitem_1635: "f32[]", getitem_1636: "f32[]", getitem_1637: "f32[]", getitem_1638: "f32[]", getitem_2086: "f32[50257, 768]", getitem_2087: "f32[1024, 768]", getitem_2088: "f32[768]", getitem_2089: "f32[768]", getitem_2090: "f32[768, 2304]", getitem_2091: "f32[2304]", getitem_2092: "f32[768, 768]", getitem_2093: "f32[768]", getitem_2094: "f32[768]", getitem_2095: "f32[768]", getitem_2096: "f32[768, 3072]", getitem_2097: "f32[3072]", getitem_2098: "f32[3072, 768]", getitem_2099: "f32[768]", getitem_2100: "f32[768]", getitem_2101: "f32[768]", getitem_2102: "f32[768, 2304]", getitem_2103: "f32[2304]", getitem_2104: "f32[768, 768]", getitem_2105: "f32[768]", getitem_2106: "f32[768]", getitem_2107: "f32[768]", getitem_2108: "f32[768, 3072]", getitem_2109: "f32[3072]", getitem_2110: "f32[3072, 768]", getitem_2111: "f32[768]", getitem_2112: "f32[768]", getitem_2113: "f32[768]", getitem_2114: "f32[768, 2304]", getitem_2115: "f32[2304]", getitem_2116: "f32[768, 768]", getitem_2117: "f32[768]", getitem_2118: "f32[768]", getitem_2119: "f32[768]", getitem_2120: "f32[768, 3072]", getitem_2121: "f32[3072]", getitem_2122: "f32[3072, 768]", getitem_2123: "f32[768]", getitem_2124: "f32[768]", getitem_2125: "f32[768]", getitem_2126: "f32[768, 2304]", getitem_2127: "f32[2304]", getitem_2128: "f32[768, 768]", getitem_2129: "f32[768]", getitem_2130: "f32[768]", getitem_2131: "f32[768]", getitem_2132: "f32[768, 3072]", getitem_2133: "f32[3072]", getitem_2134: "f32[3072, 768]", getitem_2135: "f32[768]", getitem_2136: "f32[768]", getitem_2137: "f32[768]", getitem_2138: "f32[768, 2304]", getitem_2139: "f32[2304]", getitem_2140: "f32[768, 768]", getitem_2141: "f32[768]", getitem_2142: "f32[768]", getitem_2143: "f32[768]", getitem_2144: "f32[768, 3072]", getitem_2145: "f32[3072]", getitem_2146: "f32[3072, 768]", getitem_2147: "f32[768]", getitem_2148: "f32[768]", getitem_2149: "f32[768]", getitem_2150: "f32[768, 2304]", getitem_2151: "f32[2304]", getitem_2152: "f32[768, 768]", getitem_2153: "f32[768]", getitem_2154: "f32[768]", getitem_2155: "f32[768]", getitem_2156: "f32[768, 3072]", getitem_2157: "f32[3072]", getitem_2158: "f32[3072, 768]", getitem_2159: "f32[768]", getitem_2160: "f32[768]", getitem_2161: "f32[768]", getitem_2162: "f32[768, 2304]", getitem_2163: "f32[2304]", getitem_2164: "f32[768, 768]", getitem_2165: "f32[768]", getitem_2166: "f32[768]", getitem_2167: "f32[768]", getitem_2168: "f32[768, 3072]", getitem_2169: "f32[3072]", getitem_2170: "f32[3072, 768]", getitem_2171: "f32[768]", getitem_2172: "f32[768]", getitem_2173: "f32[768]", getitem_2174: "f32[768, 2304]", getitem_2175: "f32[2304]", getitem_2176: "f32[768, 768]", getitem_2177: "f32[768]", getitem_2178: "f32[768]", getitem_2179: "f32[768]", getitem_2180: "f32[768, 3072]", getitem_2181: "f32[3072]", getitem_2182: "f32[3072, 768]", getitem_2183: "f32[768]", getitem_2184: "f32[768]", getitem_2185: "f32[768]", getitem_2186: "f32[768, 2304]", getitem_2187: "f32[2304]", getitem_2188: "f32[768, 768]", getitem_2189: "f32[768]", getitem_2190: "f32[768]", getitem_2191: "f32[768]", getitem_2192: "f32[768, 3072]", getitem_2193: "f32[3072]", getitem_2194: "f32[3072, 768]", getitem_2195: "f32[768]", getitem_2196: "f32[768]", getitem_2197: "f32[768]", getitem_2198: "f32[768, 2304]", getitem_2199: "f32[2304]", getitem_2200: "f32[768, 768]", getitem_2201: "f32[768]", getitem_2202: "f32[768]", getitem_2203: "f32[768]", getitem_2204: "f32[768, 3072]", getitem_2205: "f32[3072]", getitem_2206: "f32[3072, 768]", getitem_2207: "f32[768]", getitem_2208: "f32[768]", getitem_2209: "f32[768]", getitem_2210: "f32[768, 2304]", getitem_2211: "f32[2304]", getitem_2212: "f32[768, 768]", getitem_2213: "f32[768]", getitem_2214: "f32[768]", getitem_2215: "f32[768]", getitem_2216: "f32[768, 3072]", getitem_2217: "f32[3072]", getitem_2218: "f32[3072, 768]", getitem_2219: "f32[768]", getitem_2220: "f32[768]", getitem_2221: "f32[768]", getitem_2222: "f32[768, 2304]", getitem_2223: "f32[2304]", getitem_2224: "f32[768, 768]", getitem_2225: "f32[768]", getitem_2226: "f32[768]", getitem_2227: "f32[768]", getitem_2228: "f32[768, 3072]", getitem_2229: "f32[3072]", getitem_2230: "f32[3072, 768]", getitem_2231: "f32[768]", getitem_2232: "f32[768]", getitem_2233: "f32[768]", getitem_2234: "f32[2, 768]"):
        # No stacktrace found for following nodes
        _foreach_sub_list = torch.ops.aten._foreach_sub.List([arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1, arg728_1, arg729_1, arg730_1, arg731_1, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, arg737_1, arg738_1, arg739_1, arg740_1, arg741_1, arg742_1, arg743_1, arg744_1], [arg152_1, arg150_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1]);  arg596_1 = arg597_1 = arg598_1 = arg599_1 = arg600_1 = arg601_1 = arg602_1 = arg603_1 = arg604_1 = arg605_1 = arg606_1 = arg607_1 = arg608_1 = arg609_1 = arg610_1 = arg611_1 = arg612_1 = arg613_1 = arg614_1 = arg615_1 = arg616_1 = arg617_1 = arg618_1 = arg619_1 = arg620_1 = arg621_1 = arg622_1 = arg623_1 = arg624_1 = arg625_1 = arg626_1 = arg627_1 = arg628_1 = arg629_1 = arg630_1 = arg631_1 = arg632_1 = arg633_1 = arg634_1 = arg635_1 = arg636_1 = arg637_1 = arg638_1 = arg639_1 = arg640_1 = arg641_1 = arg642_1 = arg643_1 = arg644_1 = arg645_1 = arg646_1 = arg647_1 = arg648_1 = arg649_1 = arg650_1 = arg651_1 = arg652_1 = arg653_1 = arg654_1 = arg655_1 = arg656_1 = arg657_1 = arg658_1 = arg659_1 = arg660_1 = arg661_1 = arg662_1 = arg663_1 = arg664_1 = arg665_1 = arg666_1 = arg667_1 = arg668_1 = arg669_1 = arg670_1 = arg671_1 = arg672_1 = arg673_1 = arg674_1 = arg675_1 = arg676_1 = arg677_1 = arg678_1 = arg679_1 = arg680_1 = arg681_1 = arg682_1 = arg683_1 = arg684_1 = arg685_1 = arg686_1 = arg687_1 = arg688_1 = arg689_1 = arg690_1 = arg691_1 = arg692_1 = arg693_1 = arg694_1 = arg695_1 = arg696_1 = arg697_1 = arg698_1 = arg699_1 = arg700_1 = arg701_1 = arg702_1 = arg703_1 = arg704_1 = arg705_1 = arg706_1 = arg707_1 = arg708_1 = arg709_1 = arg710_1 = arg711_1 = arg712_1 = arg713_1 = arg714_1 = arg715_1 = arg716_1 = arg717_1 = arg718_1 = arg719_1 = arg720_1 = arg721_1 = arg722_1 = arg723_1 = arg724_1 = arg725_1 = arg726_1 = arg727_1 = arg728_1 = arg729_1 = arg730_1 = arg731_1 = arg732_1 = arg733_1 = arg734_1 = arg735_1 = arg736_1 = arg737_1 = arg738_1 = arg739_1 = arg740_1 = arg741_1 = arg742_1 = arg743_1 = arg744_1 = arg152_1 = arg150_1 = arg302_1 = arg303_1 = arg304_1 = arg305_1 = arg306_1 = arg307_1 = arg308_1 = arg309_1 = arg310_1 = arg311_1 = arg312_1 = arg313_1 = arg314_1 = arg315_1 = arg316_1 = arg317_1 = arg318_1 = arg319_1 = arg320_1 = arg321_1 = arg322_1 = arg323_1 = arg324_1 = arg325_1 = arg326_1 = arg327_1 = arg328_1 = arg329_1 = arg330_1 = arg331_1 = arg332_1 = arg333_1 = arg334_1 = arg335_1 = arg336_1 = arg337_1 = arg338_1 = arg339_1 = arg340_1 = arg341_1 = arg342_1 = arg343_1 = arg344_1 = arg345_1 = arg346_1 = arg347_1 = arg348_1 = arg349_1 = arg350_1 = arg351_1 = arg352_1 = arg353_1 = arg354_1 = arg355_1 = arg356_1 = arg357_1 = arg358_1 = arg359_1 = arg360_1 = arg361_1 = arg362_1 = arg363_1 = arg364_1 = arg365_1 = arg366_1 = arg367_1 = arg368_1 = arg369_1 = arg370_1 = arg371_1 = arg372_1 = arg373_1 = arg374_1 = arg375_1 = arg376_1 = arg377_1 = arg378_1 = arg379_1 = arg380_1 = arg381_1 = arg382_1 = arg383_1 = arg384_1 = arg385_1 = arg386_1 = arg387_1 = arg388_1 = arg389_1 = arg390_1 = arg391_1 = arg392_1 = arg393_1 = arg394_1 = arg395_1 = arg396_1 = arg397_1 = arg398_1 = arg399_1 = arg400_1 = arg401_1 = arg402_1 = arg403_1 = arg404_1 = arg405_1 = arg406_1 = arg407_1 = arg408_1 = arg409_1 = arg410_1 = arg411_1 = arg412_1 = arg413_1 = arg414_1 = arg415_1 = arg416_1 = arg417_1 = arg418_1 = arg419_1 = arg420_1 = arg421_1 = arg422_1 = arg423_1 = arg424_1 = arg425_1 = arg426_1 = arg427_1 = arg428_1 = arg429_1 = arg430_1 = arg431_1 = arg432_1 = arg433_1 = arg434_1 = arg435_1 = arg436_1 = arg437_1 = arg438_1 = arg439_1 = arg440_1 = arg441_1 = arg442_1 = arg443_1 = arg444_1 = arg445_1 = arg446_1 = arg447_1 = arg448_1 = None
        getitem: "f32[50257, 768]" = _foreach_sub_list[0]
        getitem_1: "f32[1024, 768]" = _foreach_sub_list[1]
        getitem_2: "f32[768]" = _foreach_sub_list[2]
        getitem_3: "f32[768]" = _foreach_sub_list[3]
        getitem_4: "f32[768, 2304]" = _foreach_sub_list[4]
        getitem_5: "f32[2304]" = _foreach_sub_list[5]
        getitem_6: "f32[768, 768]" = _foreach_sub_list[6]
        getitem_7: "f32[768]" = _foreach_sub_list[7]
        getitem_8: "f32[768]" = _foreach_sub_list[8]
        getitem_9: "f32[768]" = _foreach_sub_list[9]
        getitem_10: "f32[768, 3072]" = _foreach_sub_list[10]
        getitem_11: "f32[3072]" = _foreach_sub_list[11]
        getitem_12: "f32[3072, 768]" = _foreach_sub_list[12]
        getitem_13: "f32[768]" = _foreach_sub_list[13]
        getitem_14: "f32[768]" = _foreach_sub_list[14]
        getitem_15: "f32[768]" = _foreach_sub_list[15]
        getitem_16: "f32[768, 2304]" = _foreach_sub_list[16]
        getitem_17: "f32[2304]" = _foreach_sub_list[17]
        getitem_18: "f32[768, 768]" = _foreach_sub_list[18]
        getitem_19: "f32[768]" = _foreach_sub_list[19]
        getitem_20: "f32[768]" = _foreach_sub_list[20]
        getitem_21: "f32[768]" = _foreach_sub_list[21]
        getitem_22: "f32[768, 3072]" = _foreach_sub_list[22]
        getitem_23: "f32[3072]" = _foreach_sub_list[23]
        getitem_24: "f32[3072, 768]" = _foreach_sub_list[24]
        getitem_25: "f32[768]" = _foreach_sub_list[25]
        getitem_26: "f32[768]" = _foreach_sub_list[26]
        getitem_27: "f32[768]" = _foreach_sub_list[27]
        getitem_28: "f32[768, 2304]" = _foreach_sub_list[28]
        getitem_29: "f32[2304]" = _foreach_sub_list[29]
        getitem_30: "f32[768, 768]" = _foreach_sub_list[30]
        getitem_31: "f32[768]" = _foreach_sub_list[31]
        getitem_32: "f32[768]" = _foreach_sub_list[32]
        getitem_33: "f32[768]" = _foreach_sub_list[33]
        getitem_34: "f32[768, 3072]" = _foreach_sub_list[34]
        getitem_35: "f32[3072]" = _foreach_sub_list[35]
        getitem_36: "f32[3072, 768]" = _foreach_sub_list[36]
        getitem_37: "f32[768]" = _foreach_sub_list[37]
        getitem_38: "f32[768]" = _foreach_sub_list[38]
        getitem_39: "f32[768]" = _foreach_sub_list[39]
        getitem_40: "f32[768, 2304]" = _foreach_sub_list[40]
        getitem_41: "f32[2304]" = _foreach_sub_list[41]
        getitem_42: "f32[768, 768]" = _foreach_sub_list[42]
        getitem_43: "f32[768]" = _foreach_sub_list[43]
        getitem_44: "f32[768]" = _foreach_sub_list[44]
        getitem_45: "f32[768]" = _foreach_sub_list[45]
        getitem_46: "f32[768, 3072]" = _foreach_sub_list[46]
        getitem_47: "f32[3072]" = _foreach_sub_list[47]
        getitem_48: "f32[3072, 768]" = _foreach_sub_list[48]
        getitem_49: "f32[768]" = _foreach_sub_list[49]
        getitem_50: "f32[768]" = _foreach_sub_list[50]
        getitem_51: "f32[768]" = _foreach_sub_list[51]
        getitem_52: "f32[768, 2304]" = _foreach_sub_list[52]
        getitem_53: "f32[2304]" = _foreach_sub_list[53]
        getitem_54: "f32[768, 768]" = _foreach_sub_list[54]
        getitem_55: "f32[768]" = _foreach_sub_list[55]
        getitem_56: "f32[768]" = _foreach_sub_list[56]
        getitem_57: "f32[768]" = _foreach_sub_list[57]
        getitem_58: "f32[768, 3072]" = _foreach_sub_list[58]
        getitem_59: "f32[3072]" = _foreach_sub_list[59]
        getitem_60: "f32[3072, 768]" = _foreach_sub_list[60]
        getitem_61: "f32[768]" = _foreach_sub_list[61]
        getitem_62: "f32[768]" = _foreach_sub_list[62]
        getitem_63: "f32[768]" = _foreach_sub_list[63]
        getitem_64: "f32[768, 2304]" = _foreach_sub_list[64]
        getitem_65: "f32[2304]" = _foreach_sub_list[65]
        getitem_66: "f32[768, 768]" = _foreach_sub_list[66]
        getitem_67: "f32[768]" = _foreach_sub_list[67]
        getitem_68: "f32[768]" = _foreach_sub_list[68]
        getitem_69: "f32[768]" = _foreach_sub_list[69]
        getitem_70: "f32[768, 3072]" = _foreach_sub_list[70]
        getitem_71: "f32[3072]" = _foreach_sub_list[71]
        getitem_72: "f32[3072, 768]" = _foreach_sub_list[72]
        getitem_73: "f32[768]" = _foreach_sub_list[73]
        getitem_74: "f32[768]" = _foreach_sub_list[74]
        getitem_75: "f32[768]" = _foreach_sub_list[75]
        getitem_76: "f32[768, 2304]" = _foreach_sub_list[76]
        getitem_77: "f32[2304]" = _foreach_sub_list[77]
        getitem_78: "f32[768, 768]" = _foreach_sub_list[78]
        getitem_79: "f32[768]" = _foreach_sub_list[79]
        getitem_80: "f32[768]" = _foreach_sub_list[80]
        getitem_81: "f32[768]" = _foreach_sub_list[81]
        getitem_82: "f32[768, 3072]" = _foreach_sub_list[82]
        getitem_83: "f32[3072]" = _foreach_sub_list[83]
        getitem_84: "f32[3072, 768]" = _foreach_sub_list[84]
        getitem_85: "f32[768]" = _foreach_sub_list[85]
        getitem_86: "f32[768]" = _foreach_sub_list[86]
        getitem_87: "f32[768]" = _foreach_sub_list[87]
        getitem_88: "f32[768, 2304]" = _foreach_sub_list[88]
        getitem_89: "f32[2304]" = _foreach_sub_list[89]
        getitem_90: "f32[768, 768]" = _foreach_sub_list[90]
        getitem_91: "f32[768]" = _foreach_sub_list[91]
        getitem_92: "f32[768]" = _foreach_sub_list[92]
        getitem_93: "f32[768]" = _foreach_sub_list[93]
        getitem_94: "f32[768, 3072]" = _foreach_sub_list[94]
        getitem_95: "f32[3072]" = _foreach_sub_list[95]
        getitem_96: "f32[3072, 768]" = _foreach_sub_list[96]
        getitem_97: "f32[768]" = _foreach_sub_list[97]
        getitem_98: "f32[768]" = _foreach_sub_list[98]
        getitem_99: "f32[768]" = _foreach_sub_list[99]
        getitem_100: "f32[768, 2304]" = _foreach_sub_list[100]
        getitem_101: "f32[2304]" = _foreach_sub_list[101]
        getitem_102: "f32[768, 768]" = _foreach_sub_list[102]
        getitem_103: "f32[768]" = _foreach_sub_list[103]
        getitem_104: "f32[768]" = _foreach_sub_list[104]
        getitem_105: "f32[768]" = _foreach_sub_list[105]
        getitem_106: "f32[768, 3072]" = _foreach_sub_list[106]
        getitem_107: "f32[3072]" = _foreach_sub_list[107]
        getitem_108: "f32[3072, 768]" = _foreach_sub_list[108]
        getitem_109: "f32[768]" = _foreach_sub_list[109]
        getitem_110: "f32[768]" = _foreach_sub_list[110]
        getitem_111: "f32[768]" = _foreach_sub_list[111]
        getitem_112: "f32[768, 2304]" = _foreach_sub_list[112]
        getitem_113: "f32[2304]" = _foreach_sub_list[113]
        getitem_114: "f32[768, 768]" = _foreach_sub_list[114]
        getitem_115: "f32[768]" = _foreach_sub_list[115]
        getitem_116: "f32[768]" = _foreach_sub_list[116]
        getitem_117: "f32[768]" = _foreach_sub_list[117]
        getitem_118: "f32[768, 3072]" = _foreach_sub_list[118]
        getitem_119: "f32[3072]" = _foreach_sub_list[119]
        getitem_120: "f32[3072, 768]" = _foreach_sub_list[120]
        getitem_121: "f32[768]" = _foreach_sub_list[121]
        getitem_122: "f32[768]" = _foreach_sub_list[122]
        getitem_123: "f32[768]" = _foreach_sub_list[123]
        getitem_124: "f32[768, 2304]" = _foreach_sub_list[124]
        getitem_125: "f32[2304]" = _foreach_sub_list[125]
        getitem_126: "f32[768, 768]" = _foreach_sub_list[126]
        getitem_127: "f32[768]" = _foreach_sub_list[127]
        getitem_128: "f32[768]" = _foreach_sub_list[128]
        getitem_129: "f32[768]" = _foreach_sub_list[129]
        getitem_130: "f32[768, 3072]" = _foreach_sub_list[130]
        getitem_131: "f32[3072]" = _foreach_sub_list[131]
        getitem_132: "f32[3072, 768]" = _foreach_sub_list[132]
        getitem_133: "f32[768]" = _foreach_sub_list[133]
        getitem_134: "f32[768]" = _foreach_sub_list[134]
        getitem_135: "f32[768]" = _foreach_sub_list[135]
        getitem_136: "f32[768, 2304]" = _foreach_sub_list[136]
        getitem_137: "f32[2304]" = _foreach_sub_list[137]
        getitem_138: "f32[768, 768]" = _foreach_sub_list[138]
        getitem_139: "f32[768]" = _foreach_sub_list[139]
        getitem_140: "f32[768]" = _foreach_sub_list[140]
        getitem_141: "f32[768]" = _foreach_sub_list[141]
        getitem_142: "f32[768, 3072]" = _foreach_sub_list[142]
        getitem_143: "f32[3072]" = _foreach_sub_list[143]
        getitem_144: "f32[3072, 768]" = _foreach_sub_list[144]
        getitem_145: "f32[768]" = _foreach_sub_list[145]
        getitem_146: "f32[768]" = _foreach_sub_list[146]
        getitem_147: "f32[768]" = _foreach_sub_list[147]
        getitem_148: "f32[2, 768]" = _foreach_sub_list[148];  _foreach_sub_list = None
        _foreach_reciprocal_default = torch.ops.aten._foreach_reciprocal.default([getitem_1490, getitem_1491, getitem_1492, getitem_1493, getitem_1494, getitem_1495, getitem_1496, getitem_1497, getitem_1498, getitem_1499, getitem_1500, getitem_1501, getitem_1502, getitem_1503, getitem_1504, getitem_1505, getitem_1506, getitem_1507, getitem_1508, getitem_1509, getitem_1510, getitem_1511, getitem_1512, getitem_1513, getitem_1514, getitem_1515, getitem_1516, getitem_1517, getitem_1518, getitem_1519, getitem_1520, getitem_1521, getitem_1522, getitem_1523, getitem_1524, getitem_1525, getitem_1526, getitem_1527, getitem_1528, getitem_1529, getitem_1530, getitem_1531, getitem_1532, getitem_1533, getitem_1534, getitem_1535, getitem_1536, getitem_1537, getitem_1538, getitem_1539, getitem_1540, getitem_1541, getitem_1542, getitem_1543, getitem_1544, getitem_1545, getitem_1546, getitem_1547, getitem_1548, getitem_1549, getitem_1550, getitem_1551, getitem_1552, getitem_1553, getitem_1554, getitem_1555, getitem_1556, getitem_1557, getitem_1558, getitem_1559, getitem_1560, getitem_1561, getitem_1562, getitem_1563, getitem_1564, getitem_1565, getitem_1566, getitem_1567, getitem_1568, getitem_1569, getitem_1570, getitem_1571, getitem_1572, getitem_1573, getitem_1574, getitem_1575, getitem_1576, getitem_1577, getitem_1578, getitem_1579, getitem_1580, getitem_1581, getitem_1582, getitem_1583, getitem_1584, getitem_1585, getitem_1586, getitem_1587, getitem_1588, getitem_1589, getitem_1590, getitem_1591, getitem_1592, getitem_1593, getitem_1594, getitem_1595, getitem_1596, getitem_1597, getitem_1598, getitem_1599, getitem_1600, getitem_1601, getitem_1602, getitem_1603, getitem_1604, getitem_1605, getitem_1606, getitem_1607, getitem_1608, getitem_1609, getitem_1610, getitem_1611, getitem_1612, getitem_1613, getitem_1614, getitem_1615, getitem_1616, getitem_1617, getitem_1618, getitem_1619, getitem_1620, getitem_1621, getitem_1622, getitem_1623, getitem_1624, getitem_1625, getitem_1626, getitem_1627, getitem_1628, getitem_1629, getitem_1630, getitem_1631, getitem_1632, getitem_1633, getitem_1634, getitem_1635, getitem_1636, getitem_1637, getitem_1638]);  getitem_1490 = getitem_1491 = getitem_1492 = getitem_1493 = getitem_1494 = getitem_1495 = getitem_1496 = getitem_1497 = getitem_1498 = getitem_1499 = getitem_1500 = getitem_1501 = getitem_1502 = getitem_1503 = getitem_1504 = getitem_1505 = getitem_1506 = getitem_1507 = getitem_1508 = getitem_1509 = getitem_1510 = getitem_1511 = getitem_1512 = getitem_1513 = getitem_1514 = getitem_1515 = getitem_1516 = getitem_1517 = getitem_1518 = getitem_1519 = getitem_1520 = getitem_1521 = getitem_1522 = getitem_1523 = getitem_1524 = getitem_1525 = getitem_1526 = getitem_1527 = getitem_1528 = getitem_1529 = getitem_1530 = getitem_1531 = getitem_1532 = getitem_1533 = getitem_1534 = getitem_1535 = getitem_1536 = getitem_1537 = getitem_1538 = getitem_1539 = getitem_1540 = getitem_1541 = getitem_1542 = getitem_1543 = getitem_1544 = getitem_1545 = getitem_1546 = getitem_1547 = getitem_1548 = getitem_1549 = getitem_1550 = getitem_1551 = getitem_1552 = getitem_1553 = getitem_1554 = getitem_1555 = getitem_1556 = getitem_1557 = getitem_1558 = getitem_1559 = getitem_1560 = getitem_1561 = getitem_1562 = getitem_1563 = getitem_1564 = getitem_1565 = getitem_1566 = getitem_1567 = getitem_1568 = getitem_1569 = getitem_1570 = getitem_1571 = getitem_1572 = getitem_1573 = getitem_1574 = getitem_1575 = getitem_1576 = getitem_1577 = getitem_1578 = getitem_1579 = getitem_1580 = getitem_1581 = getitem_1582 = getitem_1583 = getitem_1584 = getitem_1585 = getitem_1586 = getitem_1587 = getitem_1588 = getitem_1589 = getitem_1590 = getitem_1591 = getitem_1592 = getitem_1593 = getitem_1594 = getitem_1595 = getitem_1596 = getitem_1597 = getitem_1598 = getitem_1599 = getitem_1600 = getitem_1601 = getitem_1602 = getitem_1603 = getitem_1604 = getitem_1605 = getitem_1606 = getitem_1607 = getitem_1608 = getitem_1609 = getitem_1610 = getitem_1611 = getitem_1612 = getitem_1613 = getitem_1614 = getitem_1615 = getitem_1616 = getitem_1617 = getitem_1618 = getitem_1619 = getitem_1620 = getitem_1621 = getitem_1622 = getitem_1623 = getitem_1624 = getitem_1625 = getitem_1626 = getitem_1627 = getitem_1628 = getitem_1629 = getitem_1630 = getitem_1631 = getitem_1632 = getitem_1633 = getitem_1634 = getitem_1635 = getitem_1636 = getitem_1637 = getitem_1638 = None
        getitem_1639: "f32[]" = _foreach_reciprocal_default[0]
        getitem_1640: "f32[]" = _foreach_reciprocal_default[1]
        getitem_1641: "f32[]" = _foreach_reciprocal_default[2]
        getitem_1642: "f32[]" = _foreach_reciprocal_default[3]
        getitem_1643: "f32[]" = _foreach_reciprocal_default[4]
        getitem_1644: "f32[]" = _foreach_reciprocal_default[5]
        getitem_1645: "f32[]" = _foreach_reciprocal_default[6]
        getitem_1646: "f32[]" = _foreach_reciprocal_default[7]
        getitem_1647: "f32[]" = _foreach_reciprocal_default[8]
        getitem_1648: "f32[]" = _foreach_reciprocal_default[9]
        getitem_1649: "f32[]" = _foreach_reciprocal_default[10]
        getitem_1650: "f32[]" = _foreach_reciprocal_default[11]
        getitem_1651: "f32[]" = _foreach_reciprocal_default[12]
        getitem_1652: "f32[]" = _foreach_reciprocal_default[13]
        getitem_1653: "f32[]" = _foreach_reciprocal_default[14]
        getitem_1654: "f32[]" = _foreach_reciprocal_default[15]
        getitem_1655: "f32[]" = _foreach_reciprocal_default[16]
        getitem_1656: "f32[]" = _foreach_reciprocal_default[17]
        getitem_1657: "f32[]" = _foreach_reciprocal_default[18]
        getitem_1658: "f32[]" = _foreach_reciprocal_default[19]
        getitem_1659: "f32[]" = _foreach_reciprocal_default[20]
        getitem_1660: "f32[]" = _foreach_reciprocal_default[21]
        getitem_1661: "f32[]" = _foreach_reciprocal_default[22]
        getitem_1662: "f32[]" = _foreach_reciprocal_default[23]
        getitem_1663: "f32[]" = _foreach_reciprocal_default[24]
        getitem_1664: "f32[]" = _foreach_reciprocal_default[25]
        getitem_1665: "f32[]" = _foreach_reciprocal_default[26]
        getitem_1666: "f32[]" = _foreach_reciprocal_default[27]
        getitem_1667: "f32[]" = _foreach_reciprocal_default[28]
        getitem_1668: "f32[]" = _foreach_reciprocal_default[29]
        getitem_1669: "f32[]" = _foreach_reciprocal_default[30]
        getitem_1670: "f32[]" = _foreach_reciprocal_default[31]
        getitem_1671: "f32[]" = _foreach_reciprocal_default[32]
        getitem_1672: "f32[]" = _foreach_reciprocal_default[33]
        getitem_1673: "f32[]" = _foreach_reciprocal_default[34]
        getitem_1674: "f32[]" = _foreach_reciprocal_default[35]
        getitem_1675: "f32[]" = _foreach_reciprocal_default[36]
        getitem_1676: "f32[]" = _foreach_reciprocal_default[37]
        getitem_1677: "f32[]" = _foreach_reciprocal_default[38]
        getitem_1678: "f32[]" = _foreach_reciprocal_default[39]
        getitem_1679: "f32[]" = _foreach_reciprocal_default[40]
        getitem_1680: "f32[]" = _foreach_reciprocal_default[41]
        getitem_1681: "f32[]" = _foreach_reciprocal_default[42]
        getitem_1682: "f32[]" = _foreach_reciprocal_default[43]
        getitem_1683: "f32[]" = _foreach_reciprocal_default[44]
        getitem_1684: "f32[]" = _foreach_reciprocal_default[45]
        getitem_1685: "f32[]" = _foreach_reciprocal_default[46]
        getitem_1686: "f32[]" = _foreach_reciprocal_default[47]
        getitem_1687: "f32[]" = _foreach_reciprocal_default[48]
        getitem_1688: "f32[]" = _foreach_reciprocal_default[49]
        getitem_1689: "f32[]" = _foreach_reciprocal_default[50]
        getitem_1690: "f32[]" = _foreach_reciprocal_default[51]
        getitem_1691: "f32[]" = _foreach_reciprocal_default[52]
        getitem_1692: "f32[]" = _foreach_reciprocal_default[53]
        getitem_1693: "f32[]" = _foreach_reciprocal_default[54]
        getitem_1694: "f32[]" = _foreach_reciprocal_default[55]
        getitem_1695: "f32[]" = _foreach_reciprocal_default[56]
        getitem_1696: "f32[]" = _foreach_reciprocal_default[57]
        getitem_1697: "f32[]" = _foreach_reciprocal_default[58]
        getitem_1698: "f32[]" = _foreach_reciprocal_default[59]
        getitem_1699: "f32[]" = _foreach_reciprocal_default[60]
        getitem_1700: "f32[]" = _foreach_reciprocal_default[61]
        getitem_1701: "f32[]" = _foreach_reciprocal_default[62]
        getitem_1702: "f32[]" = _foreach_reciprocal_default[63]
        getitem_1703: "f32[]" = _foreach_reciprocal_default[64]
        getitem_1704: "f32[]" = _foreach_reciprocal_default[65]
        getitem_1705: "f32[]" = _foreach_reciprocal_default[66]
        getitem_1706: "f32[]" = _foreach_reciprocal_default[67]
        getitem_1707: "f32[]" = _foreach_reciprocal_default[68]
        getitem_1708: "f32[]" = _foreach_reciprocal_default[69]
        getitem_1709: "f32[]" = _foreach_reciprocal_default[70]
        getitem_1710: "f32[]" = _foreach_reciprocal_default[71]
        getitem_1711: "f32[]" = _foreach_reciprocal_default[72]
        getitem_1712: "f32[]" = _foreach_reciprocal_default[73]
        getitem_1713: "f32[]" = _foreach_reciprocal_default[74]
        getitem_1714: "f32[]" = _foreach_reciprocal_default[75]
        getitem_1715: "f32[]" = _foreach_reciprocal_default[76]
        getitem_1716: "f32[]" = _foreach_reciprocal_default[77]
        getitem_1717: "f32[]" = _foreach_reciprocal_default[78]
        getitem_1718: "f32[]" = _foreach_reciprocal_default[79]
        getitem_1719: "f32[]" = _foreach_reciprocal_default[80]
        getitem_1720: "f32[]" = _foreach_reciprocal_default[81]
        getitem_1721: "f32[]" = _foreach_reciprocal_default[82]
        getitem_1722: "f32[]" = _foreach_reciprocal_default[83]
        getitem_1723: "f32[]" = _foreach_reciprocal_default[84]
        getitem_1724: "f32[]" = _foreach_reciprocal_default[85]
        getitem_1725: "f32[]" = _foreach_reciprocal_default[86]
        getitem_1726: "f32[]" = _foreach_reciprocal_default[87]
        getitem_1727: "f32[]" = _foreach_reciprocal_default[88]
        getitem_1728: "f32[]" = _foreach_reciprocal_default[89]
        getitem_1729: "f32[]" = _foreach_reciprocal_default[90]
        getitem_1730: "f32[]" = _foreach_reciprocal_default[91]
        getitem_1731: "f32[]" = _foreach_reciprocal_default[92]
        getitem_1732: "f32[]" = _foreach_reciprocal_default[93]
        getitem_1733: "f32[]" = _foreach_reciprocal_default[94]
        getitem_1734: "f32[]" = _foreach_reciprocal_default[95]
        getitem_1735: "f32[]" = _foreach_reciprocal_default[96]
        getitem_1736: "f32[]" = _foreach_reciprocal_default[97]
        getitem_1737: "f32[]" = _foreach_reciprocal_default[98]
        getitem_1738: "f32[]" = _foreach_reciprocal_default[99]
        getitem_1739: "f32[]" = _foreach_reciprocal_default[100]
        getitem_1740: "f32[]" = _foreach_reciprocal_default[101]
        getitem_1741: "f32[]" = _foreach_reciprocal_default[102]
        getitem_1742: "f32[]" = _foreach_reciprocal_default[103]
        getitem_1743: "f32[]" = _foreach_reciprocal_default[104]
        getitem_1744: "f32[]" = _foreach_reciprocal_default[105]
        getitem_1745: "f32[]" = _foreach_reciprocal_default[106]
        getitem_1746: "f32[]" = _foreach_reciprocal_default[107]
        getitem_1747: "f32[]" = _foreach_reciprocal_default[108]
        getitem_1748: "f32[]" = _foreach_reciprocal_default[109]
        getitem_1749: "f32[]" = _foreach_reciprocal_default[110]
        getitem_1750: "f32[]" = _foreach_reciprocal_default[111]
        getitem_1751: "f32[]" = _foreach_reciprocal_default[112]
        getitem_1752: "f32[]" = _foreach_reciprocal_default[113]
        getitem_1753: "f32[]" = _foreach_reciprocal_default[114]
        getitem_1754: "f32[]" = _foreach_reciprocal_default[115]
        getitem_1755: "f32[]" = _foreach_reciprocal_default[116]
        getitem_1756: "f32[]" = _foreach_reciprocal_default[117]
        getitem_1757: "f32[]" = _foreach_reciprocal_default[118]
        getitem_1758: "f32[]" = _foreach_reciprocal_default[119]
        getitem_1759: "f32[]" = _foreach_reciprocal_default[120]
        getitem_1760: "f32[]" = _foreach_reciprocal_default[121]
        getitem_1761: "f32[]" = _foreach_reciprocal_default[122]
        getitem_1762: "f32[]" = _foreach_reciprocal_default[123]
        getitem_1763: "f32[]" = _foreach_reciprocal_default[124]
        getitem_1764: "f32[]" = _foreach_reciprocal_default[125]
        getitem_1765: "f32[]" = _foreach_reciprocal_default[126]
        getitem_1766: "f32[]" = _foreach_reciprocal_default[127]
        getitem_1767: "f32[]" = _foreach_reciprocal_default[128]
        getitem_1768: "f32[]" = _foreach_reciprocal_default[129]
        getitem_1769: "f32[]" = _foreach_reciprocal_default[130]
        getitem_1770: "f32[]" = _foreach_reciprocal_default[131]
        getitem_1771: "f32[]" = _foreach_reciprocal_default[132]
        getitem_1772: "f32[]" = _foreach_reciprocal_default[133]
        getitem_1773: "f32[]" = _foreach_reciprocal_default[134]
        getitem_1774: "f32[]" = _foreach_reciprocal_default[135]
        getitem_1775: "f32[]" = _foreach_reciprocal_default[136]
        getitem_1776: "f32[]" = _foreach_reciprocal_default[137]
        getitem_1777: "f32[]" = _foreach_reciprocal_default[138]
        getitem_1778: "f32[]" = _foreach_reciprocal_default[139]
        getitem_1779: "f32[]" = _foreach_reciprocal_default[140]
        getitem_1780: "f32[]" = _foreach_reciprocal_default[141]
        getitem_1781: "f32[]" = _foreach_reciprocal_default[142]
        getitem_1782: "f32[]" = _foreach_reciprocal_default[143]
        getitem_1783: "f32[]" = _foreach_reciprocal_default[144]
        getitem_1784: "f32[]" = _foreach_reciprocal_default[145]
        getitem_1785: "f32[]" = _foreach_reciprocal_default[146]
        getitem_1786: "f32[]" = _foreach_reciprocal_default[147]
        getitem_1787: "f32[]" = _foreach_reciprocal_default[148];  _foreach_reciprocal_default = None
        _foreach_add_scalar = torch.ops.aten._foreach_add.Scalar([getitem_2086, getitem_2087, getitem_2088, getitem_2089, getitem_2090, getitem_2091, getitem_2092, getitem_2093, getitem_2094, getitem_2095, getitem_2096, getitem_2097, getitem_2098, getitem_2099, getitem_2100, getitem_2101, getitem_2102, getitem_2103, getitem_2104, getitem_2105, getitem_2106, getitem_2107, getitem_2108, getitem_2109, getitem_2110, getitem_2111, getitem_2112, getitem_2113, getitem_2114, getitem_2115, getitem_2116, getitem_2117, getitem_2118, getitem_2119, getitem_2120, getitem_2121, getitem_2122, getitem_2123, getitem_2124, getitem_2125, getitem_2126, getitem_2127, getitem_2128, getitem_2129, getitem_2130, getitem_2131, getitem_2132, getitem_2133, getitem_2134, getitem_2135, getitem_2136, getitem_2137, getitem_2138, getitem_2139, getitem_2140, getitem_2141, getitem_2142, getitem_2143, getitem_2144, getitem_2145, getitem_2146, getitem_2147, getitem_2148, getitem_2149, getitem_2150, getitem_2151, getitem_2152, getitem_2153, getitem_2154, getitem_2155, getitem_2156, getitem_2157, getitem_2158, getitem_2159, getitem_2160, getitem_2161, getitem_2162, getitem_2163, getitem_2164, getitem_2165, getitem_2166, getitem_2167, getitem_2168, getitem_2169, getitem_2170, getitem_2171, getitem_2172, getitem_2173, getitem_2174, getitem_2175, getitem_2176, getitem_2177, getitem_2178, getitem_2179, getitem_2180, getitem_2181, getitem_2182, getitem_2183, getitem_2184, getitem_2185, getitem_2186, getitem_2187, getitem_2188, getitem_2189, getitem_2190, getitem_2191, getitem_2192, getitem_2193, getitem_2194, getitem_2195, getitem_2196, getitem_2197, getitem_2198, getitem_2199, getitem_2200, getitem_2201, getitem_2202, getitem_2203, getitem_2204, getitem_2205, getitem_2206, getitem_2207, getitem_2208, getitem_2209, getitem_2210, getitem_2211, getitem_2212, getitem_2213, getitem_2214, getitem_2215, getitem_2216, getitem_2217, getitem_2218, getitem_2219, getitem_2220, getitem_2221, getitem_2222, getitem_2223, getitem_2224, getitem_2225, getitem_2226, getitem_2227, getitem_2228, getitem_2229, getitem_2230, getitem_2231, getitem_2232, getitem_2233, getitem_2234], 1e-08);  getitem_2086 = getitem_2087 = getitem_2088 = getitem_2089 = getitem_2090 = getitem_2091 = getitem_2092 = getitem_2093 = getitem_2094 = getitem_2095 = getitem_2096 = getitem_2097 = getitem_2098 = getitem_2099 = getitem_2100 = getitem_2101 = getitem_2102 = getitem_2103 = getitem_2104 = getitem_2105 = getitem_2106 = getitem_2107 = getitem_2108 = getitem_2109 = getitem_2110 = getitem_2111 = getitem_2112 = getitem_2113 = getitem_2114 = getitem_2115 = getitem_2116 = getitem_2117 = getitem_2118 = getitem_2119 = getitem_2120 = getitem_2121 = getitem_2122 = getitem_2123 = getitem_2124 = getitem_2125 = getitem_2126 = getitem_2127 = getitem_2128 = getitem_2129 = getitem_2130 = getitem_2131 = getitem_2132 = getitem_2133 = getitem_2134 = getitem_2135 = getitem_2136 = getitem_2137 = getitem_2138 = getitem_2139 = getitem_2140 = getitem_2141 = getitem_2142 = getitem_2143 = getitem_2144 = getitem_2145 = getitem_2146 = getitem_2147 = getitem_2148 = getitem_2149 = getitem_2150 = getitem_2151 = getitem_2152 = getitem_2153 = getitem_2154 = getitem_2155 = getitem_2156 = getitem_2157 = getitem_2158 = getitem_2159 = getitem_2160 = getitem_2161 = getitem_2162 = getitem_2163 = getitem_2164 = getitem_2165 = getitem_2166 = getitem_2167 = getitem_2168 = getitem_2169 = getitem_2170 = getitem_2171 = getitem_2172 = getitem_2173 = getitem_2174 = getitem_2175 = getitem_2176 = getitem_2177 = getitem_2178 = getitem_2179 = getitem_2180 = getitem_2181 = getitem_2182 = getitem_2183 = getitem_2184 = getitem_2185 = getitem_2186 = getitem_2187 = getitem_2188 = getitem_2189 = getitem_2190 = getitem_2191 = getitem_2192 = getitem_2193 = getitem_2194 = getitem_2195 = getitem_2196 = getitem_2197 = getitem_2198 = getitem_2199 = getitem_2200 = getitem_2201 = getitem_2202 = getitem_2203 = getitem_2204 = getitem_2205 = getitem_2206 = getitem_2207 = getitem_2208 = getitem_2209 = getitem_2210 = getitem_2211 = getitem_2212 = getitem_2213 = getitem_2214 = getitem_2215 = getitem_2216 = getitem_2217 = getitem_2218 = getitem_2219 = getitem_2220 = getitem_2221 = getitem_2222 = getitem_2223 = getitem_2224 = getitem_2225 = getitem_2226 = getitem_2227 = getitem_2228 = getitem_2229 = getitem_2230 = getitem_2231 = getitem_2232 = getitem_2233 = getitem_2234 = None
        getitem_2235: "f32[50257, 768]" = _foreach_add_scalar[0]
        getitem_2236: "f32[1024, 768]" = _foreach_add_scalar[1]
        getitem_2237: "f32[768]" = _foreach_add_scalar[2]
        getitem_2238: "f32[768]" = _foreach_add_scalar[3]
        getitem_2239: "f32[768, 2304]" = _foreach_add_scalar[4]
        getitem_2240: "f32[2304]" = _foreach_add_scalar[5]
        getitem_2241: "f32[768, 768]" = _foreach_add_scalar[6]
        getitem_2242: "f32[768]" = _foreach_add_scalar[7]
        getitem_2243: "f32[768]" = _foreach_add_scalar[8]
        getitem_2244: "f32[768]" = _foreach_add_scalar[9]
        getitem_2245: "f32[768, 3072]" = _foreach_add_scalar[10]
        getitem_2246: "f32[3072]" = _foreach_add_scalar[11]
        getitem_2247: "f32[3072, 768]" = _foreach_add_scalar[12]
        getitem_2248: "f32[768]" = _foreach_add_scalar[13]
        getitem_2249: "f32[768]" = _foreach_add_scalar[14]
        getitem_2250: "f32[768]" = _foreach_add_scalar[15]
        getitem_2251: "f32[768, 2304]" = _foreach_add_scalar[16]
        getitem_2252: "f32[2304]" = _foreach_add_scalar[17]
        getitem_2253: "f32[768, 768]" = _foreach_add_scalar[18]
        getitem_2254: "f32[768]" = _foreach_add_scalar[19]
        getitem_2255: "f32[768]" = _foreach_add_scalar[20]
        getitem_2256: "f32[768]" = _foreach_add_scalar[21]
        getitem_2257: "f32[768, 3072]" = _foreach_add_scalar[22]
        getitem_2258: "f32[3072]" = _foreach_add_scalar[23]
        getitem_2259: "f32[3072, 768]" = _foreach_add_scalar[24]
        getitem_2260: "f32[768]" = _foreach_add_scalar[25]
        getitem_2261: "f32[768]" = _foreach_add_scalar[26]
        getitem_2262: "f32[768]" = _foreach_add_scalar[27]
        getitem_2263: "f32[768, 2304]" = _foreach_add_scalar[28]
        getitem_2264: "f32[2304]" = _foreach_add_scalar[29]
        getitem_2265: "f32[768, 768]" = _foreach_add_scalar[30]
        getitem_2266: "f32[768]" = _foreach_add_scalar[31]
        getitem_2267: "f32[768]" = _foreach_add_scalar[32]
        getitem_2268: "f32[768]" = _foreach_add_scalar[33]
        getitem_2269: "f32[768, 3072]" = _foreach_add_scalar[34]
        getitem_2270: "f32[3072]" = _foreach_add_scalar[35]
        getitem_2271: "f32[3072, 768]" = _foreach_add_scalar[36]
        getitem_2272: "f32[768]" = _foreach_add_scalar[37]
        getitem_2273: "f32[768]" = _foreach_add_scalar[38]
        getitem_2274: "f32[768]" = _foreach_add_scalar[39]
        getitem_2275: "f32[768, 2304]" = _foreach_add_scalar[40]
        getitem_2276: "f32[2304]" = _foreach_add_scalar[41]
        getitem_2277: "f32[768, 768]" = _foreach_add_scalar[42]
        getitem_2278: "f32[768]" = _foreach_add_scalar[43]
        getitem_2279: "f32[768]" = _foreach_add_scalar[44]
        getitem_2280: "f32[768]" = _foreach_add_scalar[45]
        getitem_2281: "f32[768, 3072]" = _foreach_add_scalar[46]
        getitem_2282: "f32[3072]" = _foreach_add_scalar[47]
        getitem_2283: "f32[3072, 768]" = _foreach_add_scalar[48]
        getitem_2284: "f32[768]" = _foreach_add_scalar[49]
        getitem_2285: "f32[768]" = _foreach_add_scalar[50]
        getitem_2286: "f32[768]" = _foreach_add_scalar[51]
        getitem_2287: "f32[768, 2304]" = _foreach_add_scalar[52]
        getitem_2288: "f32[2304]" = _foreach_add_scalar[53]
        getitem_2289: "f32[768, 768]" = _foreach_add_scalar[54]
        getitem_2290: "f32[768]" = _foreach_add_scalar[55]
        getitem_2291: "f32[768]" = _foreach_add_scalar[56]
        getitem_2292: "f32[768]" = _foreach_add_scalar[57]
        getitem_2293: "f32[768, 3072]" = _foreach_add_scalar[58]
        getitem_2294: "f32[3072]" = _foreach_add_scalar[59]
        getitem_2295: "f32[3072, 768]" = _foreach_add_scalar[60]
        getitem_2296: "f32[768]" = _foreach_add_scalar[61]
        getitem_2297: "f32[768]" = _foreach_add_scalar[62]
        getitem_2298: "f32[768]" = _foreach_add_scalar[63]
        getitem_2299: "f32[768, 2304]" = _foreach_add_scalar[64]
        getitem_2300: "f32[2304]" = _foreach_add_scalar[65]
        getitem_2301: "f32[768, 768]" = _foreach_add_scalar[66]
        getitem_2302: "f32[768]" = _foreach_add_scalar[67]
        getitem_2303: "f32[768]" = _foreach_add_scalar[68]
        getitem_2304: "f32[768]" = _foreach_add_scalar[69]
        getitem_2305: "f32[768, 3072]" = _foreach_add_scalar[70]
        getitem_2306: "f32[3072]" = _foreach_add_scalar[71]
        getitem_2307: "f32[3072, 768]" = _foreach_add_scalar[72]
        getitem_2308: "f32[768]" = _foreach_add_scalar[73]
        getitem_2309: "f32[768]" = _foreach_add_scalar[74]
        getitem_2310: "f32[768]" = _foreach_add_scalar[75]
        getitem_2311: "f32[768, 2304]" = _foreach_add_scalar[76]
        getitem_2312: "f32[2304]" = _foreach_add_scalar[77]
        getitem_2313: "f32[768, 768]" = _foreach_add_scalar[78]
        getitem_2314: "f32[768]" = _foreach_add_scalar[79]
        getitem_2315: "f32[768]" = _foreach_add_scalar[80]
        getitem_2316: "f32[768]" = _foreach_add_scalar[81]
        getitem_2317: "f32[768, 3072]" = _foreach_add_scalar[82]
        getitem_2318: "f32[3072]" = _foreach_add_scalar[83]
        getitem_2319: "f32[3072, 768]" = _foreach_add_scalar[84]
        getitem_2320: "f32[768]" = _foreach_add_scalar[85]
        getitem_2321: "f32[768]" = _foreach_add_scalar[86]
        getitem_2322: "f32[768]" = _foreach_add_scalar[87]
        getitem_2323: "f32[768, 2304]" = _foreach_add_scalar[88]
        getitem_2324: "f32[2304]" = _foreach_add_scalar[89]
        getitem_2325: "f32[768, 768]" = _foreach_add_scalar[90]
        getitem_2326: "f32[768]" = _foreach_add_scalar[91]
        getitem_2327: "f32[768]" = _foreach_add_scalar[92]
        getitem_2328: "f32[768]" = _foreach_add_scalar[93]
        getitem_2329: "f32[768, 3072]" = _foreach_add_scalar[94]
        getitem_2330: "f32[3072]" = _foreach_add_scalar[95]
        getitem_2331: "f32[3072, 768]" = _foreach_add_scalar[96]
        getitem_2332: "f32[768]" = _foreach_add_scalar[97]
        getitem_2333: "f32[768]" = _foreach_add_scalar[98]
        getitem_2334: "f32[768]" = _foreach_add_scalar[99]
        getitem_2335: "f32[768, 2304]" = _foreach_add_scalar[100]
        getitem_2336: "f32[2304]" = _foreach_add_scalar[101]
        getitem_2337: "f32[768, 768]" = _foreach_add_scalar[102]
        getitem_2338: "f32[768]" = _foreach_add_scalar[103]
        getitem_2339: "f32[768]" = _foreach_add_scalar[104]
        getitem_2340: "f32[768]" = _foreach_add_scalar[105]
        getitem_2341: "f32[768, 3072]" = _foreach_add_scalar[106]
        getitem_2342: "f32[3072]" = _foreach_add_scalar[107]
        getitem_2343: "f32[3072, 768]" = _foreach_add_scalar[108]
        getitem_2344: "f32[768]" = _foreach_add_scalar[109]
        getitem_2345: "f32[768]" = _foreach_add_scalar[110]
        getitem_2346: "f32[768]" = _foreach_add_scalar[111]
        getitem_2347: "f32[768, 2304]" = _foreach_add_scalar[112]
        getitem_2348: "f32[2304]" = _foreach_add_scalar[113]
        getitem_2349: "f32[768, 768]" = _foreach_add_scalar[114]
        getitem_2350: "f32[768]" = _foreach_add_scalar[115]
        getitem_2351: "f32[768]" = _foreach_add_scalar[116]
        getitem_2352: "f32[768]" = _foreach_add_scalar[117]
        getitem_2353: "f32[768, 3072]" = _foreach_add_scalar[118]
        getitem_2354: "f32[3072]" = _foreach_add_scalar[119]
        getitem_2355: "f32[3072, 768]" = _foreach_add_scalar[120]
        getitem_2356: "f32[768]" = _foreach_add_scalar[121]
        getitem_2357: "f32[768]" = _foreach_add_scalar[122]
        getitem_2358: "f32[768]" = _foreach_add_scalar[123]
        getitem_2359: "f32[768, 2304]" = _foreach_add_scalar[124]
        getitem_2360: "f32[2304]" = _foreach_add_scalar[125]
        getitem_2361: "f32[768, 768]" = _foreach_add_scalar[126]
        getitem_2362: "f32[768]" = _foreach_add_scalar[127]
        getitem_2363: "f32[768]" = _foreach_add_scalar[128]
        getitem_2364: "f32[768]" = _foreach_add_scalar[129]
        getitem_2365: "f32[768, 3072]" = _foreach_add_scalar[130]
        getitem_2366: "f32[3072]" = _foreach_add_scalar[131]
        getitem_2367: "f32[3072, 768]" = _foreach_add_scalar[132]
        getitem_2368: "f32[768]" = _foreach_add_scalar[133]
        getitem_2369: "f32[768]" = _foreach_add_scalar[134]
        getitem_2370: "f32[768]" = _foreach_add_scalar[135]
        getitem_2371: "f32[768, 2304]" = _foreach_add_scalar[136]
        getitem_2372: "f32[2304]" = _foreach_add_scalar[137]
        getitem_2373: "f32[768, 768]" = _foreach_add_scalar[138]
        getitem_2374: "f32[768]" = _foreach_add_scalar[139]
        getitem_2375: "f32[768]" = _foreach_add_scalar[140]
        getitem_2376: "f32[768]" = _foreach_add_scalar[141]
        getitem_2377: "f32[768, 3072]" = _foreach_add_scalar[142]
        getitem_2378: "f32[3072]" = _foreach_add_scalar[143]
        getitem_2379: "f32[3072, 768]" = _foreach_add_scalar[144]
        getitem_2380: "f32[768]" = _foreach_add_scalar[145]
        getitem_2381: "f32[768]" = _foreach_add_scalar[146]
        getitem_2382: "f32[768]" = _foreach_add_scalar[147]
        getitem_2383: "f32[2, 768]" = _foreach_add_scalar[148];  _foreach_add_scalar = None
        return (getitem, getitem_1, getitem_2, getitem_3, getitem_4, getitem_5, getitem_6, getitem_7, getitem_8, getitem_9, getitem_10, getitem_11, getitem_12, getitem_13, getitem_14, getitem_15, getitem_16, getitem_17, getitem_18, getitem_19, getitem_20, getitem_21, getitem_22, getitem_23, getitem_24, getitem_25, getitem_26, getitem_27, getitem_28, getitem_29, getitem_30, getitem_31, getitem_32, getitem_33, getitem_34, getitem_35, getitem_36, getitem_37, getitem_38, getitem_39, getitem_40, getitem_41, getitem_42, getitem_43, getitem_44, getitem_45, getitem_46, getitem_47, getitem_48, getitem_49, getitem_50, getitem_51, getitem_52, getitem_53, getitem_54, getitem_55, getitem_56, getitem_57, getitem_58, getitem_59, getitem_60, getitem_61, getitem_62, getitem_63, getitem_64, getitem_65, getitem_66, getitem_67, getitem_68, getitem_69, getitem_70, getitem_71, getitem_72, getitem_73, getitem_74, getitem_75, getitem_76, getitem_77, getitem_78, getitem_79, getitem_80, getitem_81, getitem_82, getitem_83, getitem_84, getitem_85, getitem_86, getitem_87, getitem_88, getitem_89, getitem_90, getitem_91, getitem_92, getitem_93, getitem_94, getitem_95, getitem_96, getitem_97, getitem_98, getitem_99, getitem_100, getitem_101, getitem_102, getitem_103, getitem_104, getitem_105, getitem_106, getitem_107, getitem_108, getitem_109, getitem_110, getitem_111, getitem_112, getitem_113, getitem_114, getitem_115, getitem_116, getitem_117, getitem_118, getitem_119, getitem_120, getitem_121, getitem_122, getitem_123, getitem_124, getitem_125, getitem_126, getitem_127, getitem_128, getitem_129, getitem_130, getitem_131, getitem_132, getitem_133, getitem_134, getitem_135, getitem_136, getitem_137, getitem_138, getitem_139, getitem_140, getitem_141, getitem_142, getitem_143, getitem_144, getitem_145, getitem_146, getitem_147, getitem_148, getitem_1639, getitem_1640, getitem_1641, getitem_1642, getitem_1643, getitem_1644, getitem_1645, getitem_1646, getitem_1647, getitem_1648, getitem_1649, getitem_1650, getitem_1651, getitem_1652, getitem_1653, getitem_1654, getitem_1655, getitem_1656, getitem_1657, getitem_1658, getitem_1659, getitem_1660, getitem_1661, getitem_1662, getitem_1663, getitem_1664, getitem_1665, getitem_1666, getitem_1667, getitem_1668, getitem_1669, getitem_1670, getitem_1671, getitem_1672, getitem_1673, getitem_1674, getitem_1675, getitem_1676, getitem_1677, getitem_1678, getitem_1679, getitem_1680, getitem_1681, getitem_1682, getitem_1683, getitem_1684, getitem_1685, getitem_1686, getitem_1687, getitem_1688, getitem_1689, getitem_1690, getitem_1691, getitem_1692, getitem_1693, getitem_1694, getitem_1695, getitem_1696, getitem_1697, getitem_1698, getitem_1699, getitem_1700, getitem_1701, getitem_1702, getitem_1703, getitem_1704, getitem_1705, getitem_1706, getitem_1707, getitem_1708, getitem_1709, getitem_1710, getitem_1711, getitem_1712, getitem_1713, getitem_1714, getitem_1715, getitem_1716, getitem_1717, getitem_1718, getitem_1719, getitem_1720, getitem_1721, getitem_1722, getitem_1723, getitem_1724, getitem_1725, getitem_1726, getitem_1727, getitem_1728, getitem_1729, getitem_1730, getitem_1731, getitem_1732, getitem_1733, getitem_1734, getitem_1735, getitem_1736, getitem_1737, getitem_1738, getitem_1739, getitem_1740, getitem_1741, getitem_1742, getitem_1743, getitem_1744, getitem_1745, getitem_1746, getitem_1747, getitem_1748, getitem_1749, getitem_1750, getitem_1751, getitem_1752, getitem_1753, getitem_1754, getitem_1755, getitem_1756, getitem_1757, getitem_1758, getitem_1759, getitem_1760, getitem_1761, getitem_1762, getitem_1763, getitem_1764, getitem_1765, getitem_1766, getitem_1767, getitem_1768, getitem_1769, getitem_1770, getitem_1771, getitem_1772, getitem_1773, getitem_1774, getitem_1775, getitem_1776, getitem_1777, getitem_1778, getitem_1779, getitem_1780, getitem_1781, getitem_1782, getitem_1783, getitem_1784, getitem_1785, getitem_1786, getitem_1787, getitem_2235, getitem_2236, getitem_2237, getitem_2238, getitem_2239, getitem_2240, getitem_2241, getitem_2242, getitem_2243, getitem_2244, getitem_2245, getitem_2246, getitem_2247, getitem_2248, getitem_2249, getitem_2250, getitem_2251, getitem_2252, getitem_2253, getitem_2254, getitem_2255, getitem_2256, getitem_2257, getitem_2258, getitem_2259, getitem_2260, getitem_2261, getitem_2262, getitem_2263, getitem_2264, getitem_2265, getitem_2266, getitem_2267, getitem_2268, getitem_2269, getitem_2270, getitem_2271, getitem_2272, getitem_2273, getitem_2274, getitem_2275, getitem_2276, getitem_2277, getitem_2278, getitem_2279, getitem_2280, getitem_2281, getitem_2282, getitem_2283, getitem_2284, getitem_2285, getitem_2286, getitem_2287, getitem_2288, getitem_2289, getitem_2290, getitem_2291, getitem_2292, getitem_2293, getitem_2294, getitem_2295, getitem_2296, getitem_2297, getitem_2298, getitem_2299, getitem_2300, getitem_2301, getitem_2302, getitem_2303, getitem_2304, getitem_2305, getitem_2306, getitem_2307, getitem_2308, getitem_2309, getitem_2310, getitem_2311, getitem_2312, getitem_2313, getitem_2314, getitem_2315, getitem_2316, getitem_2317, getitem_2318, getitem_2319, getitem_2320, getitem_2321, getitem_2322, getitem_2323, getitem_2324, getitem_2325, getitem_2326, getitem_2327, getitem_2328, getitem_2329, getitem_2330, getitem_2331, getitem_2332, getitem_2333, getitem_2334, getitem_2335, getitem_2336, getitem_2337, getitem_2338, getitem_2339, getitem_2340, getitem_2341, getitem_2342, getitem_2343, getitem_2344, getitem_2345, getitem_2346, getitem_2347, getitem_2348, getitem_2349, getitem_2350, getitem_2351, getitem_2352, getitem_2353, getitem_2354, getitem_2355, getitem_2356, getitem_2357, getitem_2358, getitem_2359, getitem_2360, getitem_2361, getitem_2362, getitem_2363, getitem_2364, getitem_2365, getitem_2366, getitem_2367, getitem_2368, getitem_2369, getitem_2370, getitem_2371, getitem_2372, getitem_2373, getitem_2374, getitem_2375, getitem_2376, getitem_2377, getitem_2378, getitem_2379, getitem_2380, getitem_2381, getitem_2382, getitem_2383)


def _default_make_inputs():
    return [
    torch.randn([50257, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2, 768], dtype=torch.float32, device='cuda'),
    torch.randn([50257, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2, 768], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([50257, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2, 768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
