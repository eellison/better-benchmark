"""
Standalone repro captured via capture_hook.
Label: tlparse_timm_s7_g77
Pattern hash: d8763ad0ce9b
Shape hash: 4718abe3
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg672_1: "f32[64, 3, 3, 3]", arg673_1: "f32[64]", arg674_1: "f32[64]", arg675_1: "f32[64, 3, 1, 1]", arg676_1: "f32[64]", arg677_1: "f32[64]", arg678_1: "f32[96, 64, 3, 3]", arg679_1: "f32[96]", arg680_1: "f32[96]", arg681_1: "f32[96, 64, 1, 1]", arg682_1: "f32[96]", arg683_1: "f32[96]", arg684_1: "f32[96]", arg685_1: "f32[96]", arg686_1: "f32[96, 96, 3, 3]", arg687_1: "f32[96]", arg688_1: "f32[96]", arg689_1: "f32[96, 96, 1, 1]", arg690_1: "f32[96]", arg691_1: "f32[96]", arg692_1: "f32[192, 96, 3, 3]", arg693_1: "f32[192]", arg694_1: "f32[192]", arg695_1: "f32[192, 96, 1, 1]", arg696_1: "f32[192]", arg697_1: "f32[192]", arg698_1: "f32[192]", arg699_1: "f32[192]", arg700_1: "f32[192, 192, 3, 3]", arg701_1: "f32[192]", arg702_1: "f32[192]", arg703_1: "f32[192, 192, 1, 1]", arg704_1: "f32[192]", arg705_1: "f32[192]", arg706_1: "f32[192]", arg707_1: "f32[192]", arg708_1: "f32[192, 192, 3, 3]", arg709_1: "f32[192]", arg710_1: "f32[192]", arg711_1: "f32[192, 192, 1, 1]", arg712_1: "f32[192]", arg713_1: "f32[192]", arg714_1: "f32[192]", arg715_1: "f32[192]", arg716_1: "f32[192, 192, 3, 3]", arg717_1: "f32[192]", arg718_1: "f32[192]", arg719_1: "f32[192, 192, 1, 1]", arg720_1: "f32[192]", arg721_1: "f32[192]", arg722_1: "f32[384, 192, 3, 3]", arg723_1: "f32[384]", arg724_1: "f32[384]", arg725_1: "f32[384, 192, 1, 1]", arg726_1: "f32[384]", arg727_1: "f32[384]", arg728_1: "f32[384]", arg729_1: "f32[384]", arg730_1: "f32[384, 384, 3, 3]", arg731_1: "f32[384]", arg732_1: "f32[384]", arg733_1: "f32[384, 384, 1, 1]", arg734_1: "f32[384]", arg735_1: "f32[384]", arg736_1: "f32[384]", arg737_1: "f32[384]", arg738_1: "f32[384, 384, 3, 3]", arg739_1: "f32[384]", arg740_1: "f32[384]", arg741_1: "f32[384, 384, 1, 1]", arg742_1: "f32[384]", arg743_1: "f32[384]", arg744_1: "f32[384]", arg745_1: "f32[384]", arg746_1: "f32[384, 384, 3, 3]", arg747_1: "f32[384]", arg748_1: "f32[384]", arg749_1: "f32[384, 384, 1, 1]", arg750_1: "f32[384]", arg751_1: "f32[384]", arg752_1: "f32[384]", arg753_1: "f32[384]", arg754_1: "f32[384, 384, 3, 3]", arg755_1: "f32[384]", arg756_1: "f32[384]", arg757_1: "f32[384, 384, 1, 1]", arg758_1: "f32[384]", arg759_1: "f32[384]", arg760_1: "f32[384]", arg761_1: "f32[384]", arg762_1: "f32[384, 384, 3, 3]", arg763_1: "f32[384]", arg764_1: "f32[384]", arg765_1: "f32[384, 384, 1, 1]", arg766_1: "f32[384]", arg767_1: "f32[384]", arg768_1: "f32[384]", arg769_1: "f32[384]", arg770_1: "f32[384, 384, 3, 3]", arg771_1: "f32[384]", arg772_1: "f32[384]", arg773_1: "f32[384, 384, 1, 1]", arg774_1: "f32[384]", arg775_1: "f32[384]", arg776_1: "f32[384]", arg777_1: "f32[384]", arg778_1: "f32[384, 384, 3, 3]", arg779_1: "f32[384]", arg780_1: "f32[384]", arg781_1: "f32[384, 384, 1, 1]", arg782_1: "f32[384]", arg783_1: "f32[384]", arg784_1: "f32[384]", arg785_1: "f32[384]", arg786_1: "f32[384, 384, 3, 3]", arg787_1: "f32[384]", arg788_1: "f32[384]", arg789_1: "f32[384, 384, 1, 1]", arg790_1: "f32[384]", arg791_1: "f32[384]", arg792_1: "f32[384]", arg793_1: "f32[384]", arg794_1: "f32[384, 384, 3, 3]", arg795_1: "f32[384]", arg796_1: "f32[384]", arg797_1: "f32[384, 384, 1, 1]", arg798_1: "f32[384]", arg799_1: "f32[384]", arg800_1: "f32[384]", arg801_1: "f32[384]", arg802_1: "f32[384, 384, 3, 3]", arg803_1: "f32[384]", arg804_1: "f32[384]", arg805_1: "f32[384, 384, 1, 1]", arg806_1: "f32[384]", arg807_1: "f32[384]", arg808_1: "f32[384]", arg809_1: "f32[384]", arg810_1: "f32[384, 384, 3, 3]", arg811_1: "f32[384]", arg812_1: "f32[384]", arg813_1: "f32[384, 384, 1, 1]", arg814_1: "f32[384]", arg815_1: "f32[384]", arg816_1: "f32[384]", arg817_1: "f32[384]", arg818_1: "f32[384, 384, 3, 3]", arg819_1: "f32[384]", arg820_1: "f32[384]", arg821_1: "f32[384, 384, 1, 1]", arg822_1: "f32[384]", arg823_1: "f32[384]", arg824_1: "f32[384]", arg825_1: "f32[384]", arg826_1: "f32[384, 384, 3, 3]", arg827_1: "f32[384]", arg828_1: "f32[384]", arg829_1: "f32[384, 384, 1, 1]", arg830_1: "f32[384]", arg831_1: "f32[384]", arg832_1: "f32[1408, 384, 3, 3]", arg833_1: "f32[1408]", arg834_1: "f32[1408]", arg835_1: "f32[1408, 384, 1, 1]", arg836_1: "f32[1408]", arg837_1: "f32[1408]", arg838_1: "f32[1000, 1408]", arg839_1: "f32[1000]", arg171_1: "f32[64, 3, 3, 3]", arg169_1: "f32[64]", arg340_1: "f32[64]", arg341_1: "f32[64, 3, 1, 1]", arg342_1: "f32[64]", arg343_1: "f32[64]", arg344_1: "f32[96, 64, 3, 3]", arg345_1: "f32[96]", arg346_1: "f32[96]", arg347_1: "f32[96, 64, 1, 1]", arg348_1: "f32[96]", arg349_1: "f32[96]", arg350_1: "f32[96]", arg351_1: "f32[96]", arg352_1: "f32[96, 96, 3, 3]", arg353_1: "f32[96]", arg354_1: "f32[96]", arg355_1: "f32[96, 96, 1, 1]", arg356_1: "f32[96]", arg357_1: "f32[96]", arg358_1: "f32[192, 96, 3, 3]", arg359_1: "f32[192]", arg360_1: "f32[192]", arg361_1: "f32[192, 96, 1, 1]", arg362_1: "f32[192]", arg363_1: "f32[192]", arg364_1: "f32[192]", arg365_1: "f32[192]", arg366_1: "f32[192, 192, 3, 3]", arg367_1: "f32[192]", arg368_1: "f32[192]", arg369_1: "f32[192, 192, 1, 1]", arg370_1: "f32[192]", arg371_1: "f32[192]", arg372_1: "f32[192]", arg373_1: "f32[192]", arg374_1: "f32[192, 192, 3, 3]", arg375_1: "f32[192]", arg376_1: "f32[192]", arg377_1: "f32[192, 192, 1, 1]", arg378_1: "f32[192]", arg379_1: "f32[192]", arg380_1: "f32[192]", arg381_1: "f32[192]", arg382_1: "f32[192, 192, 3, 3]", arg383_1: "f32[192]", arg384_1: "f32[192]", arg385_1: "f32[192, 192, 1, 1]", arg386_1: "f32[192]", arg387_1: "f32[192]", arg388_1: "f32[384, 192, 3, 3]", arg389_1: "f32[384]", arg390_1: "f32[384]", arg391_1: "f32[384, 192, 1, 1]", arg392_1: "f32[384]", arg393_1: "f32[384]", arg394_1: "f32[384]", arg395_1: "f32[384]", arg396_1: "f32[384, 384, 3, 3]", arg397_1: "f32[384]", arg398_1: "f32[384]", arg399_1: "f32[384, 384, 1, 1]", arg400_1: "f32[384]", arg401_1: "f32[384]", arg402_1: "f32[384]", arg403_1: "f32[384]", arg404_1: "f32[384, 384, 3, 3]", arg405_1: "f32[384]", arg406_1: "f32[384]", arg407_1: "f32[384, 384, 1, 1]", arg408_1: "f32[384]", arg409_1: "f32[384]", arg410_1: "f32[384]", arg411_1: "f32[384]", arg412_1: "f32[384, 384, 3, 3]", arg413_1: "f32[384]", arg414_1: "f32[384]", arg415_1: "f32[384, 384, 1, 1]", arg416_1: "f32[384]", arg417_1: "f32[384]", arg418_1: "f32[384]", arg419_1: "f32[384]", arg420_1: "f32[384, 384, 3, 3]", arg421_1: "f32[384]", arg422_1: "f32[384]", arg423_1: "f32[384, 384, 1, 1]", arg424_1: "f32[384]", arg425_1: "f32[384]", arg426_1: "f32[384]", arg427_1: "f32[384]", arg428_1: "f32[384, 384, 3, 3]", arg429_1: "f32[384]", arg430_1: "f32[384]", arg431_1: "f32[384, 384, 1, 1]", arg432_1: "f32[384]", arg433_1: "f32[384]", arg434_1: "f32[384]", arg435_1: "f32[384]", arg436_1: "f32[384, 384, 3, 3]", arg437_1: "f32[384]", arg438_1: "f32[384]", arg439_1: "f32[384, 384, 1, 1]", arg440_1: "f32[384]", arg441_1: "f32[384]", arg442_1: "f32[384]", arg443_1: "f32[384]", arg444_1: "f32[384, 384, 3, 3]", arg445_1: "f32[384]", arg446_1: "f32[384]", arg447_1: "f32[384, 384, 1, 1]", arg448_1: "f32[384]", arg449_1: "f32[384]", arg450_1: "f32[384]", arg451_1: "f32[384]", arg452_1: "f32[384, 384, 3, 3]", arg453_1: "f32[384]", arg454_1: "f32[384]", arg455_1: "f32[384, 384, 1, 1]", arg456_1: "f32[384]", arg457_1: "f32[384]", arg458_1: "f32[384]", arg459_1: "f32[384]", arg460_1: "f32[384, 384, 3, 3]", arg461_1: "f32[384]", arg462_1: "f32[384]", arg463_1: "f32[384, 384, 1, 1]", arg464_1: "f32[384]", arg465_1: "f32[384]", arg466_1: "f32[384]", arg467_1: "f32[384]", arg468_1: "f32[384, 384, 3, 3]", arg469_1: "f32[384]", arg470_1: "f32[384]", arg471_1: "f32[384, 384, 1, 1]", arg472_1: "f32[384]", arg473_1: "f32[384]", arg474_1: "f32[384]", arg475_1: "f32[384]", arg476_1: "f32[384, 384, 3, 3]", arg477_1: "f32[384]", arg478_1: "f32[384]", arg479_1: "f32[384, 384, 1, 1]", arg480_1: "f32[384]", arg481_1: "f32[384]", arg482_1: "f32[384]", arg483_1: "f32[384]", arg484_1: "f32[384, 384, 3, 3]", arg485_1: "f32[384]", arg486_1: "f32[384]", arg487_1: "f32[384, 384, 1, 1]", arg488_1: "f32[384]", arg489_1: "f32[384]", arg490_1: "f32[384]", arg491_1: "f32[384]", arg492_1: "f32[384, 384, 3, 3]", arg493_1: "f32[384]", arg494_1: "f32[384]", arg495_1: "f32[384, 384, 1, 1]", arg496_1: "f32[384]", arg497_1: "f32[384]", arg498_1: "f32[1408, 384, 3, 3]", arg499_1: "f32[1408]", arg500_1: "f32[1408]", arg501_1: "f32[1408, 384, 1, 1]", arg502_1: "f32[1408]", arg503_1: "f32[1408]", arg504_1: "f32[1000, 1408]", arg505_1: "f32[1000]", getitem_1680: "f32[]", getitem_1681: "f32[]", getitem_1682: "f32[]", getitem_1683: "f32[]", getitem_1684: "f32[]", getitem_1685: "f32[]", getitem_1686: "f32[]", getitem_1687: "f32[]", getitem_1688: "f32[]", getitem_1689: "f32[]", getitem_1690: "f32[]", getitem_1691: "f32[]", getitem_1692: "f32[]", getitem_1693: "f32[]", getitem_1694: "f32[]", getitem_1695: "f32[]", getitem_1696: "f32[]", getitem_1697: "f32[]", getitem_1698: "f32[]", getitem_1699: "f32[]", getitem_1700: "f32[]", getitem_1701: "f32[]", getitem_1702: "f32[]", getitem_1703: "f32[]", getitem_1704: "f32[]", getitem_1705: "f32[]", getitem_1706: "f32[]", getitem_1707: "f32[]", getitem_1708: "f32[]", getitem_1709: "f32[]", getitem_1710: "f32[]", getitem_1711: "f32[]", getitem_1712: "f32[]", getitem_1713: "f32[]", getitem_1714: "f32[]", getitem_1715: "f32[]", getitem_1716: "f32[]", getitem_1717: "f32[]", getitem_1718: "f32[]", getitem_1719: "f32[]", getitem_1720: "f32[]", getitem_1721: "f32[]", getitem_1722: "f32[]", getitem_1723: "f32[]", getitem_1724: "f32[]", getitem_1725: "f32[]", getitem_1726: "f32[]", getitem_1727: "f32[]", getitem_1728: "f32[]", getitem_1729: "f32[]", getitem_1730: "f32[]", getitem_1731: "f32[]", getitem_1732: "f32[]", getitem_1733: "f32[]", getitem_1734: "f32[]", getitem_1735: "f32[]", getitem_1736: "f32[]", getitem_1737: "f32[]", getitem_1738: "f32[]", getitem_1739: "f32[]", getitem_1740: "f32[]", getitem_1741: "f32[]", getitem_1742: "f32[]", getitem_1743: "f32[]", getitem_1744: "f32[]", getitem_1745: "f32[]", getitem_1746: "f32[]", getitem_1747: "f32[]", getitem_1748: "f32[]", getitem_1749: "f32[]", getitem_1750: "f32[]", getitem_1751: "f32[]", getitem_1752: "f32[]", getitem_1753: "f32[]", getitem_1754: "f32[]", getitem_1755: "f32[]", getitem_1756: "f32[]", getitem_1757: "f32[]", getitem_1758: "f32[]", getitem_1759: "f32[]", getitem_1760: "f32[]", getitem_1761: "f32[]", getitem_1762: "f32[]", getitem_1763: "f32[]", getitem_1764: "f32[]", getitem_1765: "f32[]", getitem_1766: "f32[]", getitem_1767: "f32[]", getitem_1768: "f32[]", getitem_1769: "f32[]", getitem_1770: "f32[]", getitem_1771: "f32[]", getitem_1772: "f32[]", getitem_1773: "f32[]", getitem_1774: "f32[]", getitem_1775: "f32[]", getitem_1776: "f32[]", getitem_1777: "f32[]", getitem_1778: "f32[]", getitem_1779: "f32[]", getitem_1780: "f32[]", getitem_1781: "f32[]", getitem_1782: "f32[]", getitem_1783: "f32[]", getitem_1784: "f32[]", getitem_1785: "f32[]", getitem_1786: "f32[]", getitem_1787: "f32[]", getitem_1788: "f32[]", getitem_1789: "f32[]", getitem_1790: "f32[]", getitem_1791: "f32[]", getitem_1792: "f32[]", getitem_1793: "f32[]", getitem_1794: "f32[]", getitem_1795: "f32[]", getitem_1796: "f32[]", getitem_1797: "f32[]", getitem_1798: "f32[]", getitem_1799: "f32[]", getitem_1800: "f32[]", getitem_1801: "f32[]", getitem_1802: "f32[]", getitem_1803: "f32[]", getitem_1804: "f32[]", getitem_1805: "f32[]", getitem_1806: "f32[]", getitem_1807: "f32[]", getitem_1808: "f32[]", getitem_1809: "f32[]", getitem_1810: "f32[]", getitem_1811: "f32[]", getitem_1812: "f32[]", getitem_1813: "f32[]", getitem_1814: "f32[]", getitem_1815: "f32[]", getitem_1816: "f32[]", getitem_1817: "f32[]", getitem_1818: "f32[]", getitem_1819: "f32[]", getitem_1820: "f32[]", getitem_1821: "f32[]", getitem_1822: "f32[]", getitem_1823: "f32[]", getitem_1824: "f32[]", getitem_1825: "f32[]", getitem_1826: "f32[]", getitem_1827: "f32[]", getitem_1828: "f32[]", getitem_1829: "f32[]", getitem_1830: "f32[]", getitem_1831: "f32[]", getitem_1832: "f32[]", getitem_1833: "f32[]", getitem_1834: "f32[]", getitem_1835: "f32[]", getitem_1836: "f32[]", getitem_1837: "f32[]", getitem_1838: "f32[]", getitem_1839: "f32[]", getitem_1840: "f32[]", getitem_1841: "f32[]", getitem_1842: "f32[]", getitem_1843: "f32[]", getitem_1844: "f32[]", getitem_1845: "f32[]", getitem_1846: "f32[]", getitem_1847: "f32[]", getitem_2352: "f32[64, 3, 3, 3]", getitem_2353: "f32[64]", getitem_2354: "f32[64]", getitem_2355: "f32[64, 3, 1, 1]", getitem_2356: "f32[64]", getitem_2357: "f32[64]", getitem_2358: "f32[96, 64, 3, 3]", getitem_2359: "f32[96]", getitem_2360: "f32[96]", getitem_2361: "f32[96, 64, 1, 1]", getitem_2362: "f32[96]", getitem_2363: "f32[96]", getitem_2364: "f32[96]", getitem_2365: "f32[96]", getitem_2366: "f32[96, 96, 3, 3]", getitem_2367: "f32[96]", getitem_2368: "f32[96]", getitem_2369: "f32[96, 96, 1, 1]", getitem_2370: "f32[96]", getitem_2371: "f32[96]", getitem_2372: "f32[192, 96, 3, 3]", getitem_2373: "f32[192]", getitem_2374: "f32[192]", getitem_2375: "f32[192, 96, 1, 1]", getitem_2376: "f32[192]", getitem_2377: "f32[192]", getitem_2378: "f32[192]", getitem_2379: "f32[192]", getitem_2380: "f32[192, 192, 3, 3]", getitem_2381: "f32[192]", getitem_2382: "f32[192]", getitem_2383: "f32[192, 192, 1, 1]", getitem_2384: "f32[192]", getitem_2385: "f32[192]", getitem_2386: "f32[192]", getitem_2387: "f32[192]", getitem_2388: "f32[192, 192, 3, 3]", getitem_2389: "f32[192]", getitem_2390: "f32[192]", getitem_2391: "f32[192, 192, 1, 1]", getitem_2392: "f32[192]", getitem_2393: "f32[192]", getitem_2394: "f32[192]", getitem_2395: "f32[192]", getitem_2396: "f32[192, 192, 3, 3]", getitem_2397: "f32[192]", getitem_2398: "f32[192]", getitem_2399: "f32[192, 192, 1, 1]", getitem_2400: "f32[192]", getitem_2401: "f32[192]", getitem_2402: "f32[384, 192, 3, 3]", getitem_2403: "f32[384]", getitem_2404: "f32[384]", getitem_2405: "f32[384, 192, 1, 1]", getitem_2406: "f32[384]", getitem_2407: "f32[384]", getitem_2408: "f32[384]", getitem_2409: "f32[384]", getitem_2410: "f32[384, 384, 3, 3]", getitem_2411: "f32[384]", getitem_2412: "f32[384]", getitem_2413: "f32[384, 384, 1, 1]", getitem_2414: "f32[384]", getitem_2415: "f32[384]", getitem_2416: "f32[384]", getitem_2417: "f32[384]", getitem_2418: "f32[384, 384, 3, 3]", getitem_2419: "f32[384]", getitem_2420: "f32[384]", getitem_2421: "f32[384, 384, 1, 1]", getitem_2422: "f32[384]", getitem_2423: "f32[384]", getitem_2424: "f32[384]", getitem_2425: "f32[384]", getitem_2426: "f32[384, 384, 3, 3]", getitem_2427: "f32[384]", getitem_2428: "f32[384]", getitem_2429: "f32[384, 384, 1, 1]", getitem_2430: "f32[384]", getitem_2431: "f32[384]", getitem_2432: "f32[384]", getitem_2433: "f32[384]", getitem_2434: "f32[384, 384, 3, 3]", getitem_2435: "f32[384]", getitem_2436: "f32[384]", getitem_2437: "f32[384, 384, 1, 1]", getitem_2438: "f32[384]", getitem_2439: "f32[384]", getitem_2440: "f32[384]", getitem_2441: "f32[384]", getitem_2442: "f32[384, 384, 3, 3]", getitem_2443: "f32[384]", getitem_2444: "f32[384]", getitem_2445: "f32[384, 384, 1, 1]", getitem_2446: "f32[384]", getitem_2447: "f32[384]", getitem_2448: "f32[384]", getitem_2449: "f32[384]", getitem_2450: "f32[384, 384, 3, 3]", getitem_2451: "f32[384]", getitem_2452: "f32[384]", getitem_2453: "f32[384, 384, 1, 1]", getitem_2454: "f32[384]", getitem_2455: "f32[384]", getitem_2456: "f32[384]", getitem_2457: "f32[384]", getitem_2458: "f32[384, 384, 3, 3]", getitem_2459: "f32[384]", getitem_2460: "f32[384]", getitem_2461: "f32[384, 384, 1, 1]", getitem_2462: "f32[384]", getitem_2463: "f32[384]", getitem_2464: "f32[384]", getitem_2465: "f32[384]", getitem_2466: "f32[384, 384, 3, 3]", getitem_2467: "f32[384]", getitem_2468: "f32[384]", getitem_2469: "f32[384, 384, 1, 1]", getitem_2470: "f32[384]", getitem_2471: "f32[384]", getitem_2472: "f32[384]", getitem_2473: "f32[384]", getitem_2474: "f32[384, 384, 3, 3]", getitem_2475: "f32[384]", getitem_2476: "f32[384]", getitem_2477: "f32[384, 384, 1, 1]", getitem_2478: "f32[384]", getitem_2479: "f32[384]", getitem_2480: "f32[384]", getitem_2481: "f32[384]", getitem_2482: "f32[384, 384, 3, 3]", getitem_2483: "f32[384]", getitem_2484: "f32[384]", getitem_2485: "f32[384, 384, 1, 1]", getitem_2486: "f32[384]", getitem_2487: "f32[384]", getitem_2488: "f32[384]", getitem_2489: "f32[384]", getitem_2490: "f32[384, 384, 3, 3]", getitem_2491: "f32[384]", getitem_2492: "f32[384]", getitem_2493: "f32[384, 384, 1, 1]", getitem_2494: "f32[384]", getitem_2495: "f32[384]", getitem_2496: "f32[384]", getitem_2497: "f32[384]", getitem_2498: "f32[384, 384, 3, 3]", getitem_2499: "f32[384]", getitem_2500: "f32[384]", getitem_2501: "f32[384, 384, 1, 1]", getitem_2502: "f32[384]", getitem_2503: "f32[384]", getitem_2504: "f32[384]", getitem_2505: "f32[384]", getitem_2506: "f32[384, 384, 3, 3]", getitem_2507: "f32[384]", getitem_2508: "f32[384]", getitem_2509: "f32[384, 384, 1, 1]", getitem_2510: "f32[384]", getitem_2511: "f32[384]", getitem_2512: "f32[1408, 384, 3, 3]", getitem_2513: "f32[1408]", getitem_2514: "f32[1408]", getitem_2515: "f32[1408, 384, 1, 1]", getitem_2516: "f32[1408]", getitem_2517: "f32[1408]", getitem_2518: "f32[1000, 1408]", getitem_2519: "f32[1000]"):
        # No stacktrace found for following nodes
        _foreach_sub_list = torch.ops.aten._foreach_sub.List([arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1, arg728_1, arg729_1, arg730_1, arg731_1, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, arg737_1, arg738_1, arg739_1, arg740_1, arg741_1, arg742_1, arg743_1, arg744_1, arg745_1, arg746_1, arg747_1, arg748_1, arg749_1, arg750_1, arg751_1, arg752_1, arg753_1, arg754_1, arg755_1, arg756_1, arg757_1, arg758_1, arg759_1, arg760_1, arg761_1, arg762_1, arg763_1, arg764_1, arg765_1, arg766_1, arg767_1, arg768_1, arg769_1, arg770_1, arg771_1, arg772_1, arg773_1, arg774_1, arg775_1, arg776_1, arg777_1, arg778_1, arg779_1, arg780_1, arg781_1, arg782_1, arg783_1, arg784_1, arg785_1, arg786_1, arg787_1, arg788_1, arg789_1, arg790_1, arg791_1, arg792_1, arg793_1, arg794_1, arg795_1, arg796_1, arg797_1, arg798_1, arg799_1, arg800_1, arg801_1, arg802_1, arg803_1, arg804_1, arg805_1, arg806_1, arg807_1, arg808_1, arg809_1, arg810_1, arg811_1, arg812_1, arg813_1, arg814_1, arg815_1, arg816_1, arg817_1, arg818_1, arg819_1, arg820_1, arg821_1, arg822_1, arg823_1, arg824_1, arg825_1, arg826_1, arg827_1, arg828_1, arg829_1, arg830_1, arg831_1, arg832_1, arg833_1, arg834_1, arg835_1, arg836_1, arg837_1, arg838_1, arg839_1], [arg171_1, arg169_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1]);  arg672_1 = arg673_1 = arg674_1 = arg675_1 = arg676_1 = arg677_1 = arg678_1 = arg679_1 = arg680_1 = arg681_1 = arg682_1 = arg683_1 = arg684_1 = arg685_1 = arg686_1 = arg687_1 = arg688_1 = arg689_1 = arg690_1 = arg691_1 = arg692_1 = arg693_1 = arg694_1 = arg695_1 = arg696_1 = arg697_1 = arg698_1 = arg699_1 = arg700_1 = arg701_1 = arg702_1 = arg703_1 = arg704_1 = arg705_1 = arg706_1 = arg707_1 = arg708_1 = arg709_1 = arg710_1 = arg711_1 = arg712_1 = arg713_1 = arg714_1 = arg715_1 = arg716_1 = arg717_1 = arg718_1 = arg719_1 = arg720_1 = arg721_1 = arg722_1 = arg723_1 = arg724_1 = arg725_1 = arg726_1 = arg727_1 = arg728_1 = arg729_1 = arg730_1 = arg731_1 = arg732_1 = arg733_1 = arg734_1 = arg735_1 = arg736_1 = arg737_1 = arg738_1 = arg739_1 = arg740_1 = arg741_1 = arg742_1 = arg743_1 = arg744_1 = arg745_1 = arg746_1 = arg747_1 = arg748_1 = arg749_1 = arg750_1 = arg751_1 = arg752_1 = arg753_1 = arg754_1 = arg755_1 = arg756_1 = arg757_1 = arg758_1 = arg759_1 = arg760_1 = arg761_1 = arg762_1 = arg763_1 = arg764_1 = arg765_1 = arg766_1 = arg767_1 = arg768_1 = arg769_1 = arg770_1 = arg771_1 = arg772_1 = arg773_1 = arg774_1 = arg775_1 = arg776_1 = arg777_1 = arg778_1 = arg779_1 = arg780_1 = arg781_1 = arg782_1 = arg783_1 = arg784_1 = arg785_1 = arg786_1 = arg787_1 = arg788_1 = arg789_1 = arg790_1 = arg791_1 = arg792_1 = arg793_1 = arg794_1 = arg795_1 = arg796_1 = arg797_1 = arg798_1 = arg799_1 = arg800_1 = arg801_1 = arg802_1 = arg803_1 = arg804_1 = arg805_1 = arg806_1 = arg807_1 = arg808_1 = arg809_1 = arg810_1 = arg811_1 = arg812_1 = arg813_1 = arg814_1 = arg815_1 = arg816_1 = arg817_1 = arg818_1 = arg819_1 = arg820_1 = arg821_1 = arg822_1 = arg823_1 = arg824_1 = arg825_1 = arg826_1 = arg827_1 = arg828_1 = arg829_1 = arg830_1 = arg831_1 = arg832_1 = arg833_1 = arg834_1 = arg835_1 = arg836_1 = arg837_1 = arg838_1 = arg839_1 = arg171_1 = arg169_1 = arg340_1 = arg341_1 = arg342_1 = arg343_1 = arg344_1 = arg345_1 = arg346_1 = arg347_1 = arg348_1 = arg349_1 = arg350_1 = arg351_1 = arg352_1 = arg353_1 = arg354_1 = arg355_1 = arg356_1 = arg357_1 = arg358_1 = arg359_1 = arg360_1 = arg361_1 = arg362_1 = arg363_1 = arg364_1 = arg365_1 = arg366_1 = arg367_1 = arg368_1 = arg369_1 = arg370_1 = arg371_1 = arg372_1 = arg373_1 = arg374_1 = arg375_1 = arg376_1 = arg377_1 = arg378_1 = arg379_1 = arg380_1 = arg381_1 = arg382_1 = arg383_1 = arg384_1 = arg385_1 = arg386_1 = arg387_1 = arg388_1 = arg389_1 = arg390_1 = arg391_1 = arg392_1 = arg393_1 = arg394_1 = arg395_1 = arg396_1 = arg397_1 = arg398_1 = arg399_1 = arg400_1 = arg401_1 = arg402_1 = arg403_1 = arg404_1 = arg405_1 = arg406_1 = arg407_1 = arg408_1 = arg409_1 = arg410_1 = arg411_1 = arg412_1 = arg413_1 = arg414_1 = arg415_1 = arg416_1 = arg417_1 = arg418_1 = arg419_1 = arg420_1 = arg421_1 = arg422_1 = arg423_1 = arg424_1 = arg425_1 = arg426_1 = arg427_1 = arg428_1 = arg429_1 = arg430_1 = arg431_1 = arg432_1 = arg433_1 = arg434_1 = arg435_1 = arg436_1 = arg437_1 = arg438_1 = arg439_1 = arg440_1 = arg441_1 = arg442_1 = arg443_1 = arg444_1 = arg445_1 = arg446_1 = arg447_1 = arg448_1 = arg449_1 = arg450_1 = arg451_1 = arg452_1 = arg453_1 = arg454_1 = arg455_1 = arg456_1 = arg457_1 = arg458_1 = arg459_1 = arg460_1 = arg461_1 = arg462_1 = arg463_1 = arg464_1 = arg465_1 = arg466_1 = arg467_1 = arg468_1 = arg469_1 = arg470_1 = arg471_1 = arg472_1 = arg473_1 = arg474_1 = arg475_1 = arg476_1 = arg477_1 = arg478_1 = arg479_1 = arg480_1 = arg481_1 = arg482_1 = arg483_1 = arg484_1 = arg485_1 = arg486_1 = arg487_1 = arg488_1 = arg489_1 = arg490_1 = arg491_1 = arg492_1 = arg493_1 = arg494_1 = arg495_1 = arg496_1 = arg497_1 = arg498_1 = arg499_1 = arg500_1 = arg501_1 = arg502_1 = arg503_1 = arg504_1 = arg505_1 = None
        getitem: "f32[64, 3, 3, 3]" = _foreach_sub_list[0]
        getitem_1: "f32[64]" = _foreach_sub_list[1]
        getitem_2: "f32[64]" = _foreach_sub_list[2]
        getitem_3: "f32[64, 3, 1, 1]" = _foreach_sub_list[3]
        getitem_4: "f32[64]" = _foreach_sub_list[4]
        getitem_5: "f32[64]" = _foreach_sub_list[5]
        getitem_6: "f32[96, 64, 3, 3]" = _foreach_sub_list[6]
        getitem_7: "f32[96]" = _foreach_sub_list[7]
        getitem_8: "f32[96]" = _foreach_sub_list[8]
        getitem_9: "f32[96, 64, 1, 1]" = _foreach_sub_list[9]
        getitem_10: "f32[96]" = _foreach_sub_list[10]
        getitem_11: "f32[96]" = _foreach_sub_list[11]
        getitem_12: "f32[96]" = _foreach_sub_list[12]
        getitem_13: "f32[96]" = _foreach_sub_list[13]
        getitem_14: "f32[96, 96, 3, 3]" = _foreach_sub_list[14]
        getitem_15: "f32[96]" = _foreach_sub_list[15]
        getitem_16: "f32[96]" = _foreach_sub_list[16]
        getitem_17: "f32[96, 96, 1, 1]" = _foreach_sub_list[17]
        getitem_18: "f32[96]" = _foreach_sub_list[18]
        getitem_19: "f32[96]" = _foreach_sub_list[19]
        getitem_20: "f32[192, 96, 3, 3]" = _foreach_sub_list[20]
        getitem_21: "f32[192]" = _foreach_sub_list[21]
        getitem_22: "f32[192]" = _foreach_sub_list[22]
        getitem_23: "f32[192, 96, 1, 1]" = _foreach_sub_list[23]
        getitem_24: "f32[192]" = _foreach_sub_list[24]
        getitem_25: "f32[192]" = _foreach_sub_list[25]
        getitem_26: "f32[192]" = _foreach_sub_list[26]
        getitem_27: "f32[192]" = _foreach_sub_list[27]
        getitem_28: "f32[192, 192, 3, 3]" = _foreach_sub_list[28]
        getitem_29: "f32[192]" = _foreach_sub_list[29]
        getitem_30: "f32[192]" = _foreach_sub_list[30]
        getitem_31: "f32[192, 192, 1, 1]" = _foreach_sub_list[31]
        getitem_32: "f32[192]" = _foreach_sub_list[32]
        getitem_33: "f32[192]" = _foreach_sub_list[33]
        getitem_34: "f32[192]" = _foreach_sub_list[34]
        getitem_35: "f32[192]" = _foreach_sub_list[35]
        getitem_36: "f32[192, 192, 3, 3]" = _foreach_sub_list[36]
        getitem_37: "f32[192]" = _foreach_sub_list[37]
        getitem_38: "f32[192]" = _foreach_sub_list[38]
        getitem_39: "f32[192, 192, 1, 1]" = _foreach_sub_list[39]
        getitem_40: "f32[192]" = _foreach_sub_list[40]
        getitem_41: "f32[192]" = _foreach_sub_list[41]
        getitem_42: "f32[192]" = _foreach_sub_list[42]
        getitem_43: "f32[192]" = _foreach_sub_list[43]
        getitem_44: "f32[192, 192, 3, 3]" = _foreach_sub_list[44]
        getitem_45: "f32[192]" = _foreach_sub_list[45]
        getitem_46: "f32[192]" = _foreach_sub_list[46]
        getitem_47: "f32[192, 192, 1, 1]" = _foreach_sub_list[47]
        getitem_48: "f32[192]" = _foreach_sub_list[48]
        getitem_49: "f32[192]" = _foreach_sub_list[49]
        getitem_50: "f32[384, 192, 3, 3]" = _foreach_sub_list[50]
        getitem_51: "f32[384]" = _foreach_sub_list[51]
        getitem_52: "f32[384]" = _foreach_sub_list[52]
        getitem_53: "f32[384, 192, 1, 1]" = _foreach_sub_list[53]
        getitem_54: "f32[384]" = _foreach_sub_list[54]
        getitem_55: "f32[384]" = _foreach_sub_list[55]
        getitem_56: "f32[384]" = _foreach_sub_list[56]
        getitem_57: "f32[384]" = _foreach_sub_list[57]
        getitem_58: "f32[384, 384, 3, 3]" = _foreach_sub_list[58]
        getitem_59: "f32[384]" = _foreach_sub_list[59]
        getitem_60: "f32[384]" = _foreach_sub_list[60]
        getitem_61: "f32[384, 384, 1, 1]" = _foreach_sub_list[61]
        getitem_62: "f32[384]" = _foreach_sub_list[62]
        getitem_63: "f32[384]" = _foreach_sub_list[63]
        getitem_64: "f32[384]" = _foreach_sub_list[64]
        getitem_65: "f32[384]" = _foreach_sub_list[65]
        getitem_66: "f32[384, 384, 3, 3]" = _foreach_sub_list[66]
        getitem_67: "f32[384]" = _foreach_sub_list[67]
        getitem_68: "f32[384]" = _foreach_sub_list[68]
        getitem_69: "f32[384, 384, 1, 1]" = _foreach_sub_list[69]
        getitem_70: "f32[384]" = _foreach_sub_list[70]
        getitem_71: "f32[384]" = _foreach_sub_list[71]
        getitem_72: "f32[384]" = _foreach_sub_list[72]
        getitem_73: "f32[384]" = _foreach_sub_list[73]
        getitem_74: "f32[384, 384, 3, 3]" = _foreach_sub_list[74]
        getitem_75: "f32[384]" = _foreach_sub_list[75]
        getitem_76: "f32[384]" = _foreach_sub_list[76]
        getitem_77: "f32[384, 384, 1, 1]" = _foreach_sub_list[77]
        getitem_78: "f32[384]" = _foreach_sub_list[78]
        getitem_79: "f32[384]" = _foreach_sub_list[79]
        getitem_80: "f32[384]" = _foreach_sub_list[80]
        getitem_81: "f32[384]" = _foreach_sub_list[81]
        getitem_82: "f32[384, 384, 3, 3]" = _foreach_sub_list[82]
        getitem_83: "f32[384]" = _foreach_sub_list[83]
        getitem_84: "f32[384]" = _foreach_sub_list[84]
        getitem_85: "f32[384, 384, 1, 1]" = _foreach_sub_list[85]
        getitem_86: "f32[384]" = _foreach_sub_list[86]
        getitem_87: "f32[384]" = _foreach_sub_list[87]
        getitem_88: "f32[384]" = _foreach_sub_list[88]
        getitem_89: "f32[384]" = _foreach_sub_list[89]
        getitem_90: "f32[384, 384, 3, 3]" = _foreach_sub_list[90]
        getitem_91: "f32[384]" = _foreach_sub_list[91]
        getitem_92: "f32[384]" = _foreach_sub_list[92]
        getitem_93: "f32[384, 384, 1, 1]" = _foreach_sub_list[93]
        getitem_94: "f32[384]" = _foreach_sub_list[94]
        getitem_95: "f32[384]" = _foreach_sub_list[95]
        getitem_96: "f32[384]" = _foreach_sub_list[96]
        getitem_97: "f32[384]" = _foreach_sub_list[97]
        getitem_98: "f32[384, 384, 3, 3]" = _foreach_sub_list[98]
        getitem_99: "f32[384]" = _foreach_sub_list[99]
        getitem_100: "f32[384]" = _foreach_sub_list[100]
        getitem_101: "f32[384, 384, 1, 1]" = _foreach_sub_list[101]
        getitem_102: "f32[384]" = _foreach_sub_list[102]
        getitem_103: "f32[384]" = _foreach_sub_list[103]
        getitem_104: "f32[384]" = _foreach_sub_list[104]
        getitem_105: "f32[384]" = _foreach_sub_list[105]
        getitem_106: "f32[384, 384, 3, 3]" = _foreach_sub_list[106]
        getitem_107: "f32[384]" = _foreach_sub_list[107]
        getitem_108: "f32[384]" = _foreach_sub_list[108]
        getitem_109: "f32[384, 384, 1, 1]" = _foreach_sub_list[109]
        getitem_110: "f32[384]" = _foreach_sub_list[110]
        getitem_111: "f32[384]" = _foreach_sub_list[111]
        getitem_112: "f32[384]" = _foreach_sub_list[112]
        getitem_113: "f32[384]" = _foreach_sub_list[113]
        getitem_114: "f32[384, 384, 3, 3]" = _foreach_sub_list[114]
        getitem_115: "f32[384]" = _foreach_sub_list[115]
        getitem_116: "f32[384]" = _foreach_sub_list[116]
        getitem_117: "f32[384, 384, 1, 1]" = _foreach_sub_list[117]
        getitem_118: "f32[384]" = _foreach_sub_list[118]
        getitem_119: "f32[384]" = _foreach_sub_list[119]
        getitem_120: "f32[384]" = _foreach_sub_list[120]
        getitem_121: "f32[384]" = _foreach_sub_list[121]
        getitem_122: "f32[384, 384, 3, 3]" = _foreach_sub_list[122]
        getitem_123: "f32[384]" = _foreach_sub_list[123]
        getitem_124: "f32[384]" = _foreach_sub_list[124]
        getitem_125: "f32[384, 384, 1, 1]" = _foreach_sub_list[125]
        getitem_126: "f32[384]" = _foreach_sub_list[126]
        getitem_127: "f32[384]" = _foreach_sub_list[127]
        getitem_128: "f32[384]" = _foreach_sub_list[128]
        getitem_129: "f32[384]" = _foreach_sub_list[129]
        getitem_130: "f32[384, 384, 3, 3]" = _foreach_sub_list[130]
        getitem_131: "f32[384]" = _foreach_sub_list[131]
        getitem_132: "f32[384]" = _foreach_sub_list[132]
        getitem_133: "f32[384, 384, 1, 1]" = _foreach_sub_list[133]
        getitem_134: "f32[384]" = _foreach_sub_list[134]
        getitem_135: "f32[384]" = _foreach_sub_list[135]
        getitem_136: "f32[384]" = _foreach_sub_list[136]
        getitem_137: "f32[384]" = _foreach_sub_list[137]
        getitem_138: "f32[384, 384, 3, 3]" = _foreach_sub_list[138]
        getitem_139: "f32[384]" = _foreach_sub_list[139]
        getitem_140: "f32[384]" = _foreach_sub_list[140]
        getitem_141: "f32[384, 384, 1, 1]" = _foreach_sub_list[141]
        getitem_142: "f32[384]" = _foreach_sub_list[142]
        getitem_143: "f32[384]" = _foreach_sub_list[143]
        getitem_144: "f32[384]" = _foreach_sub_list[144]
        getitem_145: "f32[384]" = _foreach_sub_list[145]
        getitem_146: "f32[384, 384, 3, 3]" = _foreach_sub_list[146]
        getitem_147: "f32[384]" = _foreach_sub_list[147]
        getitem_148: "f32[384]" = _foreach_sub_list[148]
        getitem_149: "f32[384, 384, 1, 1]" = _foreach_sub_list[149]
        getitem_150: "f32[384]" = _foreach_sub_list[150]
        getitem_151: "f32[384]" = _foreach_sub_list[151]
        getitem_152: "f32[384]" = _foreach_sub_list[152]
        getitem_153: "f32[384]" = _foreach_sub_list[153]
        getitem_154: "f32[384, 384, 3, 3]" = _foreach_sub_list[154]
        getitem_155: "f32[384]" = _foreach_sub_list[155]
        getitem_156: "f32[384]" = _foreach_sub_list[156]
        getitem_157: "f32[384, 384, 1, 1]" = _foreach_sub_list[157]
        getitem_158: "f32[384]" = _foreach_sub_list[158]
        getitem_159: "f32[384]" = _foreach_sub_list[159]
        getitem_160: "f32[1408, 384, 3, 3]" = _foreach_sub_list[160]
        getitem_161: "f32[1408]" = _foreach_sub_list[161]
        getitem_162: "f32[1408]" = _foreach_sub_list[162]
        getitem_163: "f32[1408, 384, 1, 1]" = _foreach_sub_list[163]
        getitem_164: "f32[1408]" = _foreach_sub_list[164]
        getitem_165: "f32[1408]" = _foreach_sub_list[165]
        getitem_166: "f32[1000, 1408]" = _foreach_sub_list[166]
        getitem_167: "f32[1000]" = _foreach_sub_list[167];  _foreach_sub_list = None
        _foreach_reciprocal_default = torch.ops.aten._foreach_reciprocal.default([getitem_1680, getitem_1681, getitem_1682, getitem_1683, getitem_1684, getitem_1685, getitem_1686, getitem_1687, getitem_1688, getitem_1689, getitem_1690, getitem_1691, getitem_1692, getitem_1693, getitem_1694, getitem_1695, getitem_1696, getitem_1697, getitem_1698, getitem_1699, getitem_1700, getitem_1701, getitem_1702, getitem_1703, getitem_1704, getitem_1705, getitem_1706, getitem_1707, getitem_1708, getitem_1709, getitem_1710, getitem_1711, getitem_1712, getitem_1713, getitem_1714, getitem_1715, getitem_1716, getitem_1717, getitem_1718, getitem_1719, getitem_1720, getitem_1721, getitem_1722, getitem_1723, getitem_1724, getitem_1725, getitem_1726, getitem_1727, getitem_1728, getitem_1729, getitem_1730, getitem_1731, getitem_1732, getitem_1733, getitem_1734, getitem_1735, getitem_1736, getitem_1737, getitem_1738, getitem_1739, getitem_1740, getitem_1741, getitem_1742, getitem_1743, getitem_1744, getitem_1745, getitem_1746, getitem_1747, getitem_1748, getitem_1749, getitem_1750, getitem_1751, getitem_1752, getitem_1753, getitem_1754, getitem_1755, getitem_1756, getitem_1757, getitem_1758, getitem_1759, getitem_1760, getitem_1761, getitem_1762, getitem_1763, getitem_1764, getitem_1765, getitem_1766, getitem_1767, getitem_1768, getitem_1769, getitem_1770, getitem_1771, getitem_1772, getitem_1773, getitem_1774, getitem_1775, getitem_1776, getitem_1777, getitem_1778, getitem_1779, getitem_1780, getitem_1781, getitem_1782, getitem_1783, getitem_1784, getitem_1785, getitem_1786, getitem_1787, getitem_1788, getitem_1789, getitem_1790, getitem_1791, getitem_1792, getitem_1793, getitem_1794, getitem_1795, getitem_1796, getitem_1797, getitem_1798, getitem_1799, getitem_1800, getitem_1801, getitem_1802, getitem_1803, getitem_1804, getitem_1805, getitem_1806, getitem_1807, getitem_1808, getitem_1809, getitem_1810, getitem_1811, getitem_1812, getitem_1813, getitem_1814, getitem_1815, getitem_1816, getitem_1817, getitem_1818, getitem_1819, getitem_1820, getitem_1821, getitem_1822, getitem_1823, getitem_1824, getitem_1825, getitem_1826, getitem_1827, getitem_1828, getitem_1829, getitem_1830, getitem_1831, getitem_1832, getitem_1833, getitem_1834, getitem_1835, getitem_1836, getitem_1837, getitem_1838, getitem_1839, getitem_1840, getitem_1841, getitem_1842, getitem_1843, getitem_1844, getitem_1845, getitem_1846, getitem_1847]);  getitem_1680 = getitem_1681 = getitem_1682 = getitem_1683 = getitem_1684 = getitem_1685 = getitem_1686 = getitem_1687 = getitem_1688 = getitem_1689 = getitem_1690 = getitem_1691 = getitem_1692 = getitem_1693 = getitem_1694 = getitem_1695 = getitem_1696 = getitem_1697 = getitem_1698 = getitem_1699 = getitem_1700 = getitem_1701 = getitem_1702 = getitem_1703 = getitem_1704 = getitem_1705 = getitem_1706 = getitem_1707 = getitem_1708 = getitem_1709 = getitem_1710 = getitem_1711 = getitem_1712 = getitem_1713 = getitem_1714 = getitem_1715 = getitem_1716 = getitem_1717 = getitem_1718 = getitem_1719 = getitem_1720 = getitem_1721 = getitem_1722 = getitem_1723 = getitem_1724 = getitem_1725 = getitem_1726 = getitem_1727 = getitem_1728 = getitem_1729 = getitem_1730 = getitem_1731 = getitem_1732 = getitem_1733 = getitem_1734 = getitem_1735 = getitem_1736 = getitem_1737 = getitem_1738 = getitem_1739 = getitem_1740 = getitem_1741 = getitem_1742 = getitem_1743 = getitem_1744 = getitem_1745 = getitem_1746 = getitem_1747 = getitem_1748 = getitem_1749 = getitem_1750 = getitem_1751 = getitem_1752 = getitem_1753 = getitem_1754 = getitem_1755 = getitem_1756 = getitem_1757 = getitem_1758 = getitem_1759 = getitem_1760 = getitem_1761 = getitem_1762 = getitem_1763 = getitem_1764 = getitem_1765 = getitem_1766 = getitem_1767 = getitem_1768 = getitem_1769 = getitem_1770 = getitem_1771 = getitem_1772 = getitem_1773 = getitem_1774 = getitem_1775 = getitem_1776 = getitem_1777 = getitem_1778 = getitem_1779 = getitem_1780 = getitem_1781 = getitem_1782 = getitem_1783 = getitem_1784 = getitem_1785 = getitem_1786 = getitem_1787 = getitem_1788 = getitem_1789 = getitem_1790 = getitem_1791 = getitem_1792 = getitem_1793 = getitem_1794 = getitem_1795 = getitem_1796 = getitem_1797 = getitem_1798 = getitem_1799 = getitem_1800 = getitem_1801 = getitem_1802 = getitem_1803 = getitem_1804 = getitem_1805 = getitem_1806 = getitem_1807 = getitem_1808 = getitem_1809 = getitem_1810 = getitem_1811 = getitem_1812 = getitem_1813 = getitem_1814 = getitem_1815 = getitem_1816 = getitem_1817 = getitem_1818 = getitem_1819 = getitem_1820 = getitem_1821 = getitem_1822 = getitem_1823 = getitem_1824 = getitem_1825 = getitem_1826 = getitem_1827 = getitem_1828 = getitem_1829 = getitem_1830 = getitem_1831 = getitem_1832 = getitem_1833 = getitem_1834 = getitem_1835 = getitem_1836 = getitem_1837 = getitem_1838 = getitem_1839 = getitem_1840 = getitem_1841 = getitem_1842 = getitem_1843 = getitem_1844 = getitem_1845 = getitem_1846 = getitem_1847 = None
        getitem_1848: "f32[]" = _foreach_reciprocal_default[0]
        getitem_1849: "f32[]" = _foreach_reciprocal_default[1]
        getitem_1850: "f32[]" = _foreach_reciprocal_default[2]
        getitem_1851: "f32[]" = _foreach_reciprocal_default[3]
        getitem_1852: "f32[]" = _foreach_reciprocal_default[4]
        getitem_1853: "f32[]" = _foreach_reciprocal_default[5]
        getitem_1854: "f32[]" = _foreach_reciprocal_default[6]
        getitem_1855: "f32[]" = _foreach_reciprocal_default[7]
        getitem_1856: "f32[]" = _foreach_reciprocal_default[8]
        getitem_1857: "f32[]" = _foreach_reciprocal_default[9]
        getitem_1858: "f32[]" = _foreach_reciprocal_default[10]
        getitem_1859: "f32[]" = _foreach_reciprocal_default[11]
        getitem_1860: "f32[]" = _foreach_reciprocal_default[12]
        getitem_1861: "f32[]" = _foreach_reciprocal_default[13]
        getitem_1862: "f32[]" = _foreach_reciprocal_default[14]
        getitem_1863: "f32[]" = _foreach_reciprocal_default[15]
        getitem_1864: "f32[]" = _foreach_reciprocal_default[16]
        getitem_1865: "f32[]" = _foreach_reciprocal_default[17]
        getitem_1866: "f32[]" = _foreach_reciprocal_default[18]
        getitem_1867: "f32[]" = _foreach_reciprocal_default[19]
        getitem_1868: "f32[]" = _foreach_reciprocal_default[20]
        getitem_1869: "f32[]" = _foreach_reciprocal_default[21]
        getitem_1870: "f32[]" = _foreach_reciprocal_default[22]
        getitem_1871: "f32[]" = _foreach_reciprocal_default[23]
        getitem_1872: "f32[]" = _foreach_reciprocal_default[24]
        getitem_1873: "f32[]" = _foreach_reciprocal_default[25]
        getitem_1874: "f32[]" = _foreach_reciprocal_default[26]
        getitem_1875: "f32[]" = _foreach_reciprocal_default[27]
        getitem_1876: "f32[]" = _foreach_reciprocal_default[28]
        getitem_1877: "f32[]" = _foreach_reciprocal_default[29]
        getitem_1878: "f32[]" = _foreach_reciprocal_default[30]
        getitem_1879: "f32[]" = _foreach_reciprocal_default[31]
        getitem_1880: "f32[]" = _foreach_reciprocal_default[32]
        getitem_1881: "f32[]" = _foreach_reciprocal_default[33]
        getitem_1882: "f32[]" = _foreach_reciprocal_default[34]
        getitem_1883: "f32[]" = _foreach_reciprocal_default[35]
        getitem_1884: "f32[]" = _foreach_reciprocal_default[36]
        getitem_1885: "f32[]" = _foreach_reciprocal_default[37]
        getitem_1886: "f32[]" = _foreach_reciprocal_default[38]
        getitem_1887: "f32[]" = _foreach_reciprocal_default[39]
        getitem_1888: "f32[]" = _foreach_reciprocal_default[40]
        getitem_1889: "f32[]" = _foreach_reciprocal_default[41]
        getitem_1890: "f32[]" = _foreach_reciprocal_default[42]
        getitem_1891: "f32[]" = _foreach_reciprocal_default[43]
        getitem_1892: "f32[]" = _foreach_reciprocal_default[44]
        getitem_1893: "f32[]" = _foreach_reciprocal_default[45]
        getitem_1894: "f32[]" = _foreach_reciprocal_default[46]
        getitem_1895: "f32[]" = _foreach_reciprocal_default[47]
        getitem_1896: "f32[]" = _foreach_reciprocal_default[48]
        getitem_1897: "f32[]" = _foreach_reciprocal_default[49]
        getitem_1898: "f32[]" = _foreach_reciprocal_default[50]
        getitem_1899: "f32[]" = _foreach_reciprocal_default[51]
        getitem_1900: "f32[]" = _foreach_reciprocal_default[52]
        getitem_1901: "f32[]" = _foreach_reciprocal_default[53]
        getitem_1902: "f32[]" = _foreach_reciprocal_default[54]
        getitem_1903: "f32[]" = _foreach_reciprocal_default[55]
        getitem_1904: "f32[]" = _foreach_reciprocal_default[56]
        getitem_1905: "f32[]" = _foreach_reciprocal_default[57]
        getitem_1906: "f32[]" = _foreach_reciprocal_default[58]
        getitem_1907: "f32[]" = _foreach_reciprocal_default[59]
        getitem_1908: "f32[]" = _foreach_reciprocal_default[60]
        getitem_1909: "f32[]" = _foreach_reciprocal_default[61]
        getitem_1910: "f32[]" = _foreach_reciprocal_default[62]
        getitem_1911: "f32[]" = _foreach_reciprocal_default[63]
        getitem_1912: "f32[]" = _foreach_reciprocal_default[64]
        getitem_1913: "f32[]" = _foreach_reciprocal_default[65]
        getitem_1914: "f32[]" = _foreach_reciprocal_default[66]
        getitem_1915: "f32[]" = _foreach_reciprocal_default[67]
        getitem_1916: "f32[]" = _foreach_reciprocal_default[68]
        getitem_1917: "f32[]" = _foreach_reciprocal_default[69]
        getitem_1918: "f32[]" = _foreach_reciprocal_default[70]
        getitem_1919: "f32[]" = _foreach_reciprocal_default[71]
        getitem_1920: "f32[]" = _foreach_reciprocal_default[72]
        getitem_1921: "f32[]" = _foreach_reciprocal_default[73]
        getitem_1922: "f32[]" = _foreach_reciprocal_default[74]
        getitem_1923: "f32[]" = _foreach_reciprocal_default[75]
        getitem_1924: "f32[]" = _foreach_reciprocal_default[76]
        getitem_1925: "f32[]" = _foreach_reciprocal_default[77]
        getitem_1926: "f32[]" = _foreach_reciprocal_default[78]
        getitem_1927: "f32[]" = _foreach_reciprocal_default[79]
        getitem_1928: "f32[]" = _foreach_reciprocal_default[80]
        getitem_1929: "f32[]" = _foreach_reciprocal_default[81]
        getitem_1930: "f32[]" = _foreach_reciprocal_default[82]
        getitem_1931: "f32[]" = _foreach_reciprocal_default[83]
        getitem_1932: "f32[]" = _foreach_reciprocal_default[84]
        getitem_1933: "f32[]" = _foreach_reciprocal_default[85]
        getitem_1934: "f32[]" = _foreach_reciprocal_default[86]
        getitem_1935: "f32[]" = _foreach_reciprocal_default[87]
        getitem_1936: "f32[]" = _foreach_reciprocal_default[88]
        getitem_1937: "f32[]" = _foreach_reciprocal_default[89]
        getitem_1938: "f32[]" = _foreach_reciprocal_default[90]
        getitem_1939: "f32[]" = _foreach_reciprocal_default[91]
        getitem_1940: "f32[]" = _foreach_reciprocal_default[92]
        getitem_1941: "f32[]" = _foreach_reciprocal_default[93]
        getitem_1942: "f32[]" = _foreach_reciprocal_default[94]
        getitem_1943: "f32[]" = _foreach_reciprocal_default[95]
        getitem_1944: "f32[]" = _foreach_reciprocal_default[96]
        getitem_1945: "f32[]" = _foreach_reciprocal_default[97]
        getitem_1946: "f32[]" = _foreach_reciprocal_default[98]
        getitem_1947: "f32[]" = _foreach_reciprocal_default[99]
        getitem_1948: "f32[]" = _foreach_reciprocal_default[100]
        getitem_1949: "f32[]" = _foreach_reciprocal_default[101]
        getitem_1950: "f32[]" = _foreach_reciprocal_default[102]
        getitem_1951: "f32[]" = _foreach_reciprocal_default[103]
        getitem_1952: "f32[]" = _foreach_reciprocal_default[104]
        getitem_1953: "f32[]" = _foreach_reciprocal_default[105]
        getitem_1954: "f32[]" = _foreach_reciprocal_default[106]
        getitem_1955: "f32[]" = _foreach_reciprocal_default[107]
        getitem_1956: "f32[]" = _foreach_reciprocal_default[108]
        getitem_1957: "f32[]" = _foreach_reciprocal_default[109]
        getitem_1958: "f32[]" = _foreach_reciprocal_default[110]
        getitem_1959: "f32[]" = _foreach_reciprocal_default[111]
        getitem_1960: "f32[]" = _foreach_reciprocal_default[112]
        getitem_1961: "f32[]" = _foreach_reciprocal_default[113]
        getitem_1962: "f32[]" = _foreach_reciprocal_default[114]
        getitem_1963: "f32[]" = _foreach_reciprocal_default[115]
        getitem_1964: "f32[]" = _foreach_reciprocal_default[116]
        getitem_1965: "f32[]" = _foreach_reciprocal_default[117]
        getitem_1966: "f32[]" = _foreach_reciprocal_default[118]
        getitem_1967: "f32[]" = _foreach_reciprocal_default[119]
        getitem_1968: "f32[]" = _foreach_reciprocal_default[120]
        getitem_1969: "f32[]" = _foreach_reciprocal_default[121]
        getitem_1970: "f32[]" = _foreach_reciprocal_default[122]
        getitem_1971: "f32[]" = _foreach_reciprocal_default[123]
        getitem_1972: "f32[]" = _foreach_reciprocal_default[124]
        getitem_1973: "f32[]" = _foreach_reciprocal_default[125]
        getitem_1974: "f32[]" = _foreach_reciprocal_default[126]
        getitem_1975: "f32[]" = _foreach_reciprocal_default[127]
        getitem_1976: "f32[]" = _foreach_reciprocal_default[128]
        getitem_1977: "f32[]" = _foreach_reciprocal_default[129]
        getitem_1978: "f32[]" = _foreach_reciprocal_default[130]
        getitem_1979: "f32[]" = _foreach_reciprocal_default[131]
        getitem_1980: "f32[]" = _foreach_reciprocal_default[132]
        getitem_1981: "f32[]" = _foreach_reciprocal_default[133]
        getitem_1982: "f32[]" = _foreach_reciprocal_default[134]
        getitem_1983: "f32[]" = _foreach_reciprocal_default[135]
        getitem_1984: "f32[]" = _foreach_reciprocal_default[136]
        getitem_1985: "f32[]" = _foreach_reciprocal_default[137]
        getitem_1986: "f32[]" = _foreach_reciprocal_default[138]
        getitem_1987: "f32[]" = _foreach_reciprocal_default[139]
        getitem_1988: "f32[]" = _foreach_reciprocal_default[140]
        getitem_1989: "f32[]" = _foreach_reciprocal_default[141]
        getitem_1990: "f32[]" = _foreach_reciprocal_default[142]
        getitem_1991: "f32[]" = _foreach_reciprocal_default[143]
        getitem_1992: "f32[]" = _foreach_reciprocal_default[144]
        getitem_1993: "f32[]" = _foreach_reciprocal_default[145]
        getitem_1994: "f32[]" = _foreach_reciprocal_default[146]
        getitem_1995: "f32[]" = _foreach_reciprocal_default[147]
        getitem_1996: "f32[]" = _foreach_reciprocal_default[148]
        getitem_1997: "f32[]" = _foreach_reciprocal_default[149]
        getitem_1998: "f32[]" = _foreach_reciprocal_default[150]
        getitem_1999: "f32[]" = _foreach_reciprocal_default[151]
        getitem_2000: "f32[]" = _foreach_reciprocal_default[152]
        getitem_2001: "f32[]" = _foreach_reciprocal_default[153]
        getitem_2002: "f32[]" = _foreach_reciprocal_default[154]
        getitem_2003: "f32[]" = _foreach_reciprocal_default[155]
        getitem_2004: "f32[]" = _foreach_reciprocal_default[156]
        getitem_2005: "f32[]" = _foreach_reciprocal_default[157]
        getitem_2006: "f32[]" = _foreach_reciprocal_default[158]
        getitem_2007: "f32[]" = _foreach_reciprocal_default[159]
        getitem_2008: "f32[]" = _foreach_reciprocal_default[160]
        getitem_2009: "f32[]" = _foreach_reciprocal_default[161]
        getitem_2010: "f32[]" = _foreach_reciprocal_default[162]
        getitem_2011: "f32[]" = _foreach_reciprocal_default[163]
        getitem_2012: "f32[]" = _foreach_reciprocal_default[164]
        getitem_2013: "f32[]" = _foreach_reciprocal_default[165]
        getitem_2014: "f32[]" = _foreach_reciprocal_default[166]
        getitem_2015: "f32[]" = _foreach_reciprocal_default[167];  _foreach_reciprocal_default = None
        _foreach_add_scalar = torch.ops.aten._foreach_add.Scalar([getitem_2352, getitem_2353, getitem_2354, getitem_2355, getitem_2356, getitem_2357, getitem_2358, getitem_2359, getitem_2360, getitem_2361, getitem_2362, getitem_2363, getitem_2364, getitem_2365, getitem_2366, getitem_2367, getitem_2368, getitem_2369, getitem_2370, getitem_2371, getitem_2372, getitem_2373, getitem_2374, getitem_2375, getitem_2376, getitem_2377, getitem_2378, getitem_2379, getitem_2380, getitem_2381, getitem_2382, getitem_2383, getitem_2384, getitem_2385, getitem_2386, getitem_2387, getitem_2388, getitem_2389, getitem_2390, getitem_2391, getitem_2392, getitem_2393, getitem_2394, getitem_2395, getitem_2396, getitem_2397, getitem_2398, getitem_2399, getitem_2400, getitem_2401, getitem_2402, getitem_2403, getitem_2404, getitem_2405, getitem_2406, getitem_2407, getitem_2408, getitem_2409, getitem_2410, getitem_2411, getitem_2412, getitem_2413, getitem_2414, getitem_2415, getitem_2416, getitem_2417, getitem_2418, getitem_2419, getitem_2420, getitem_2421, getitem_2422, getitem_2423, getitem_2424, getitem_2425, getitem_2426, getitem_2427, getitem_2428, getitem_2429, getitem_2430, getitem_2431, getitem_2432, getitem_2433, getitem_2434, getitem_2435, getitem_2436, getitem_2437, getitem_2438, getitem_2439, getitem_2440, getitem_2441, getitem_2442, getitem_2443, getitem_2444, getitem_2445, getitem_2446, getitem_2447, getitem_2448, getitem_2449, getitem_2450, getitem_2451, getitem_2452, getitem_2453, getitem_2454, getitem_2455, getitem_2456, getitem_2457, getitem_2458, getitem_2459, getitem_2460, getitem_2461, getitem_2462, getitem_2463, getitem_2464, getitem_2465, getitem_2466, getitem_2467, getitem_2468, getitem_2469, getitem_2470, getitem_2471, getitem_2472, getitem_2473, getitem_2474, getitem_2475, getitem_2476, getitem_2477, getitem_2478, getitem_2479, getitem_2480, getitem_2481, getitem_2482, getitem_2483, getitem_2484, getitem_2485, getitem_2486, getitem_2487, getitem_2488, getitem_2489, getitem_2490, getitem_2491, getitem_2492, getitem_2493, getitem_2494, getitem_2495, getitem_2496, getitem_2497, getitem_2498, getitem_2499, getitem_2500, getitem_2501, getitem_2502, getitem_2503, getitem_2504, getitem_2505, getitem_2506, getitem_2507, getitem_2508, getitem_2509, getitem_2510, getitem_2511, getitem_2512, getitem_2513, getitem_2514, getitem_2515, getitem_2516, getitem_2517, getitem_2518, getitem_2519], 1e-08);  getitem_2352 = getitem_2353 = getitem_2354 = getitem_2355 = getitem_2356 = getitem_2357 = getitem_2358 = getitem_2359 = getitem_2360 = getitem_2361 = getitem_2362 = getitem_2363 = getitem_2364 = getitem_2365 = getitem_2366 = getitem_2367 = getitem_2368 = getitem_2369 = getitem_2370 = getitem_2371 = getitem_2372 = getitem_2373 = getitem_2374 = getitem_2375 = getitem_2376 = getitem_2377 = getitem_2378 = getitem_2379 = getitem_2380 = getitem_2381 = getitem_2382 = getitem_2383 = getitem_2384 = getitem_2385 = getitem_2386 = getitem_2387 = getitem_2388 = getitem_2389 = getitem_2390 = getitem_2391 = getitem_2392 = getitem_2393 = getitem_2394 = getitem_2395 = getitem_2396 = getitem_2397 = getitem_2398 = getitem_2399 = getitem_2400 = getitem_2401 = getitem_2402 = getitem_2403 = getitem_2404 = getitem_2405 = getitem_2406 = getitem_2407 = getitem_2408 = getitem_2409 = getitem_2410 = getitem_2411 = getitem_2412 = getitem_2413 = getitem_2414 = getitem_2415 = getitem_2416 = getitem_2417 = getitem_2418 = getitem_2419 = getitem_2420 = getitem_2421 = getitem_2422 = getitem_2423 = getitem_2424 = getitem_2425 = getitem_2426 = getitem_2427 = getitem_2428 = getitem_2429 = getitem_2430 = getitem_2431 = getitem_2432 = getitem_2433 = getitem_2434 = getitem_2435 = getitem_2436 = getitem_2437 = getitem_2438 = getitem_2439 = getitem_2440 = getitem_2441 = getitem_2442 = getitem_2443 = getitem_2444 = getitem_2445 = getitem_2446 = getitem_2447 = getitem_2448 = getitem_2449 = getitem_2450 = getitem_2451 = getitem_2452 = getitem_2453 = getitem_2454 = getitem_2455 = getitem_2456 = getitem_2457 = getitem_2458 = getitem_2459 = getitem_2460 = getitem_2461 = getitem_2462 = getitem_2463 = getitem_2464 = getitem_2465 = getitem_2466 = getitem_2467 = getitem_2468 = getitem_2469 = getitem_2470 = getitem_2471 = getitem_2472 = getitem_2473 = getitem_2474 = getitem_2475 = getitem_2476 = getitem_2477 = getitem_2478 = getitem_2479 = getitem_2480 = getitem_2481 = getitem_2482 = getitem_2483 = getitem_2484 = getitem_2485 = getitem_2486 = getitem_2487 = getitem_2488 = getitem_2489 = getitem_2490 = getitem_2491 = getitem_2492 = getitem_2493 = getitem_2494 = getitem_2495 = getitem_2496 = getitem_2497 = getitem_2498 = getitem_2499 = getitem_2500 = getitem_2501 = getitem_2502 = getitem_2503 = getitem_2504 = getitem_2505 = getitem_2506 = getitem_2507 = getitem_2508 = getitem_2509 = getitem_2510 = getitem_2511 = getitem_2512 = getitem_2513 = getitem_2514 = getitem_2515 = getitem_2516 = getitem_2517 = getitem_2518 = getitem_2519 = None
        getitem_2520: "f32[64, 3, 3, 3]" = _foreach_add_scalar[0]
        getitem_2521: "f32[64]" = _foreach_add_scalar[1]
        getitem_2522: "f32[64]" = _foreach_add_scalar[2]
        getitem_2523: "f32[64, 3, 1, 1]" = _foreach_add_scalar[3]
        getitem_2524: "f32[64]" = _foreach_add_scalar[4]
        getitem_2525: "f32[64]" = _foreach_add_scalar[5]
        getitem_2526: "f32[96, 64, 3, 3]" = _foreach_add_scalar[6]
        getitem_2527: "f32[96]" = _foreach_add_scalar[7]
        getitem_2528: "f32[96]" = _foreach_add_scalar[8]
        getitem_2529: "f32[96, 64, 1, 1]" = _foreach_add_scalar[9]
        getitem_2530: "f32[96]" = _foreach_add_scalar[10]
        getitem_2531: "f32[96]" = _foreach_add_scalar[11]
        getitem_2532: "f32[96]" = _foreach_add_scalar[12]
        getitem_2533: "f32[96]" = _foreach_add_scalar[13]
        getitem_2534: "f32[96, 96, 3, 3]" = _foreach_add_scalar[14]
        getitem_2535: "f32[96]" = _foreach_add_scalar[15]
        getitem_2536: "f32[96]" = _foreach_add_scalar[16]
        getitem_2537: "f32[96, 96, 1, 1]" = _foreach_add_scalar[17]
        getitem_2538: "f32[96]" = _foreach_add_scalar[18]
        getitem_2539: "f32[96]" = _foreach_add_scalar[19]
        getitem_2540: "f32[192, 96, 3, 3]" = _foreach_add_scalar[20]
        getitem_2541: "f32[192]" = _foreach_add_scalar[21]
        getitem_2542: "f32[192]" = _foreach_add_scalar[22]
        getitem_2543: "f32[192, 96, 1, 1]" = _foreach_add_scalar[23]
        getitem_2544: "f32[192]" = _foreach_add_scalar[24]
        getitem_2545: "f32[192]" = _foreach_add_scalar[25]
        getitem_2546: "f32[192]" = _foreach_add_scalar[26]
        getitem_2547: "f32[192]" = _foreach_add_scalar[27]
        getitem_2548: "f32[192, 192, 3, 3]" = _foreach_add_scalar[28]
        getitem_2549: "f32[192]" = _foreach_add_scalar[29]
        getitem_2550: "f32[192]" = _foreach_add_scalar[30]
        getitem_2551: "f32[192, 192, 1, 1]" = _foreach_add_scalar[31]
        getitem_2552: "f32[192]" = _foreach_add_scalar[32]
        getitem_2553: "f32[192]" = _foreach_add_scalar[33]
        getitem_2554: "f32[192]" = _foreach_add_scalar[34]
        getitem_2555: "f32[192]" = _foreach_add_scalar[35]
        getitem_2556: "f32[192, 192, 3, 3]" = _foreach_add_scalar[36]
        getitem_2557: "f32[192]" = _foreach_add_scalar[37]
        getitem_2558: "f32[192]" = _foreach_add_scalar[38]
        getitem_2559: "f32[192, 192, 1, 1]" = _foreach_add_scalar[39]
        getitem_2560: "f32[192]" = _foreach_add_scalar[40]
        getitem_2561: "f32[192]" = _foreach_add_scalar[41]
        getitem_2562: "f32[192]" = _foreach_add_scalar[42]
        getitem_2563: "f32[192]" = _foreach_add_scalar[43]
        getitem_2564: "f32[192, 192, 3, 3]" = _foreach_add_scalar[44]
        getitem_2565: "f32[192]" = _foreach_add_scalar[45]
        getitem_2566: "f32[192]" = _foreach_add_scalar[46]
        getitem_2567: "f32[192, 192, 1, 1]" = _foreach_add_scalar[47]
        getitem_2568: "f32[192]" = _foreach_add_scalar[48]
        getitem_2569: "f32[192]" = _foreach_add_scalar[49]
        getitem_2570: "f32[384, 192, 3, 3]" = _foreach_add_scalar[50]
        getitem_2571: "f32[384]" = _foreach_add_scalar[51]
        getitem_2572: "f32[384]" = _foreach_add_scalar[52]
        getitem_2573: "f32[384, 192, 1, 1]" = _foreach_add_scalar[53]
        getitem_2574: "f32[384]" = _foreach_add_scalar[54]
        getitem_2575: "f32[384]" = _foreach_add_scalar[55]
        getitem_2576: "f32[384]" = _foreach_add_scalar[56]
        getitem_2577: "f32[384]" = _foreach_add_scalar[57]
        getitem_2578: "f32[384, 384, 3, 3]" = _foreach_add_scalar[58]
        getitem_2579: "f32[384]" = _foreach_add_scalar[59]
        getitem_2580: "f32[384]" = _foreach_add_scalar[60]
        getitem_2581: "f32[384, 384, 1, 1]" = _foreach_add_scalar[61]
        getitem_2582: "f32[384]" = _foreach_add_scalar[62]
        getitem_2583: "f32[384]" = _foreach_add_scalar[63]
        getitem_2584: "f32[384]" = _foreach_add_scalar[64]
        getitem_2585: "f32[384]" = _foreach_add_scalar[65]
        getitem_2586: "f32[384, 384, 3, 3]" = _foreach_add_scalar[66]
        getitem_2587: "f32[384]" = _foreach_add_scalar[67]
        getitem_2588: "f32[384]" = _foreach_add_scalar[68]
        getitem_2589: "f32[384, 384, 1, 1]" = _foreach_add_scalar[69]
        getitem_2590: "f32[384]" = _foreach_add_scalar[70]
        getitem_2591: "f32[384]" = _foreach_add_scalar[71]
        getitem_2592: "f32[384]" = _foreach_add_scalar[72]
        getitem_2593: "f32[384]" = _foreach_add_scalar[73]
        getitem_2594: "f32[384, 384, 3, 3]" = _foreach_add_scalar[74]
        getitem_2595: "f32[384]" = _foreach_add_scalar[75]
        getitem_2596: "f32[384]" = _foreach_add_scalar[76]
        getitem_2597: "f32[384, 384, 1, 1]" = _foreach_add_scalar[77]
        getitem_2598: "f32[384]" = _foreach_add_scalar[78]
        getitem_2599: "f32[384]" = _foreach_add_scalar[79]
        getitem_2600: "f32[384]" = _foreach_add_scalar[80]
        getitem_2601: "f32[384]" = _foreach_add_scalar[81]
        getitem_2602: "f32[384, 384, 3, 3]" = _foreach_add_scalar[82]
        getitem_2603: "f32[384]" = _foreach_add_scalar[83]
        getitem_2604: "f32[384]" = _foreach_add_scalar[84]
        getitem_2605: "f32[384, 384, 1, 1]" = _foreach_add_scalar[85]
        getitem_2606: "f32[384]" = _foreach_add_scalar[86]
        getitem_2607: "f32[384]" = _foreach_add_scalar[87]
        getitem_2608: "f32[384]" = _foreach_add_scalar[88]
        getitem_2609: "f32[384]" = _foreach_add_scalar[89]
        getitem_2610: "f32[384, 384, 3, 3]" = _foreach_add_scalar[90]
        getitem_2611: "f32[384]" = _foreach_add_scalar[91]
        getitem_2612: "f32[384]" = _foreach_add_scalar[92]
        getitem_2613: "f32[384, 384, 1, 1]" = _foreach_add_scalar[93]
        getitem_2614: "f32[384]" = _foreach_add_scalar[94]
        getitem_2615: "f32[384]" = _foreach_add_scalar[95]
        getitem_2616: "f32[384]" = _foreach_add_scalar[96]
        getitem_2617: "f32[384]" = _foreach_add_scalar[97]
        getitem_2618: "f32[384, 384, 3, 3]" = _foreach_add_scalar[98]
        getitem_2619: "f32[384]" = _foreach_add_scalar[99]
        getitem_2620: "f32[384]" = _foreach_add_scalar[100]
        getitem_2621: "f32[384, 384, 1, 1]" = _foreach_add_scalar[101]
        getitem_2622: "f32[384]" = _foreach_add_scalar[102]
        getitem_2623: "f32[384]" = _foreach_add_scalar[103]
        getitem_2624: "f32[384]" = _foreach_add_scalar[104]
        getitem_2625: "f32[384]" = _foreach_add_scalar[105]
        getitem_2626: "f32[384, 384, 3, 3]" = _foreach_add_scalar[106]
        getitem_2627: "f32[384]" = _foreach_add_scalar[107]
        getitem_2628: "f32[384]" = _foreach_add_scalar[108]
        getitem_2629: "f32[384, 384, 1, 1]" = _foreach_add_scalar[109]
        getitem_2630: "f32[384]" = _foreach_add_scalar[110]
        getitem_2631: "f32[384]" = _foreach_add_scalar[111]
        getitem_2632: "f32[384]" = _foreach_add_scalar[112]
        getitem_2633: "f32[384]" = _foreach_add_scalar[113]
        getitem_2634: "f32[384, 384, 3, 3]" = _foreach_add_scalar[114]
        getitem_2635: "f32[384]" = _foreach_add_scalar[115]
        getitem_2636: "f32[384]" = _foreach_add_scalar[116]
        getitem_2637: "f32[384, 384, 1, 1]" = _foreach_add_scalar[117]
        getitem_2638: "f32[384]" = _foreach_add_scalar[118]
        getitem_2639: "f32[384]" = _foreach_add_scalar[119]
        getitem_2640: "f32[384]" = _foreach_add_scalar[120]
        getitem_2641: "f32[384]" = _foreach_add_scalar[121]
        getitem_2642: "f32[384, 384, 3, 3]" = _foreach_add_scalar[122]
        getitem_2643: "f32[384]" = _foreach_add_scalar[123]
        getitem_2644: "f32[384]" = _foreach_add_scalar[124]
        getitem_2645: "f32[384, 384, 1, 1]" = _foreach_add_scalar[125]
        getitem_2646: "f32[384]" = _foreach_add_scalar[126]
        getitem_2647: "f32[384]" = _foreach_add_scalar[127]
        getitem_2648: "f32[384]" = _foreach_add_scalar[128]
        getitem_2649: "f32[384]" = _foreach_add_scalar[129]
        getitem_2650: "f32[384, 384, 3, 3]" = _foreach_add_scalar[130]
        getitem_2651: "f32[384]" = _foreach_add_scalar[131]
        getitem_2652: "f32[384]" = _foreach_add_scalar[132]
        getitem_2653: "f32[384, 384, 1, 1]" = _foreach_add_scalar[133]
        getitem_2654: "f32[384]" = _foreach_add_scalar[134]
        getitem_2655: "f32[384]" = _foreach_add_scalar[135]
        getitem_2656: "f32[384]" = _foreach_add_scalar[136]
        getitem_2657: "f32[384]" = _foreach_add_scalar[137]
        getitem_2658: "f32[384, 384, 3, 3]" = _foreach_add_scalar[138]
        getitem_2659: "f32[384]" = _foreach_add_scalar[139]
        getitem_2660: "f32[384]" = _foreach_add_scalar[140]
        getitem_2661: "f32[384, 384, 1, 1]" = _foreach_add_scalar[141]
        getitem_2662: "f32[384]" = _foreach_add_scalar[142]
        getitem_2663: "f32[384]" = _foreach_add_scalar[143]
        getitem_2664: "f32[384]" = _foreach_add_scalar[144]
        getitem_2665: "f32[384]" = _foreach_add_scalar[145]
        getitem_2666: "f32[384, 384, 3, 3]" = _foreach_add_scalar[146]
        getitem_2667: "f32[384]" = _foreach_add_scalar[147]
        getitem_2668: "f32[384]" = _foreach_add_scalar[148]
        getitem_2669: "f32[384, 384, 1, 1]" = _foreach_add_scalar[149]
        getitem_2670: "f32[384]" = _foreach_add_scalar[150]
        getitem_2671: "f32[384]" = _foreach_add_scalar[151]
        getitem_2672: "f32[384]" = _foreach_add_scalar[152]
        getitem_2673: "f32[384]" = _foreach_add_scalar[153]
        getitem_2674: "f32[384, 384, 3, 3]" = _foreach_add_scalar[154]
        getitem_2675: "f32[384]" = _foreach_add_scalar[155]
        getitem_2676: "f32[384]" = _foreach_add_scalar[156]
        getitem_2677: "f32[384, 384, 1, 1]" = _foreach_add_scalar[157]
        getitem_2678: "f32[384]" = _foreach_add_scalar[158]
        getitem_2679: "f32[384]" = _foreach_add_scalar[159]
        getitem_2680: "f32[1408, 384, 3, 3]" = _foreach_add_scalar[160]
        getitem_2681: "f32[1408]" = _foreach_add_scalar[161]
        getitem_2682: "f32[1408]" = _foreach_add_scalar[162]
        getitem_2683: "f32[1408, 384, 1, 1]" = _foreach_add_scalar[163]
        getitem_2684: "f32[1408]" = _foreach_add_scalar[164]
        getitem_2685: "f32[1408]" = _foreach_add_scalar[165]
        getitem_2686: "f32[1000, 1408]" = _foreach_add_scalar[166]
        getitem_2687: "f32[1000]" = _foreach_add_scalar[167];  _foreach_add_scalar = None
        return (getitem, getitem_1, getitem_2, getitem_3, getitem_4, getitem_5, getitem_6, getitem_7, getitem_8, getitem_9, getitem_10, getitem_11, getitem_12, getitem_13, getitem_14, getitem_15, getitem_16, getitem_17, getitem_18, getitem_19, getitem_20, getitem_21, getitem_22, getitem_23, getitem_24, getitem_25, getitem_26, getitem_27, getitem_28, getitem_29, getitem_30, getitem_31, getitem_32, getitem_33, getitem_34, getitem_35, getitem_36, getitem_37, getitem_38, getitem_39, getitem_40, getitem_41, getitem_42, getitem_43, getitem_44, getitem_45, getitem_46, getitem_47, getitem_48, getitem_49, getitem_50, getitem_51, getitem_52, getitem_53, getitem_54, getitem_55, getitem_56, getitem_57, getitem_58, getitem_59, getitem_60, getitem_61, getitem_62, getitem_63, getitem_64, getitem_65, getitem_66, getitem_67, getitem_68, getitem_69, getitem_70, getitem_71, getitem_72, getitem_73, getitem_74, getitem_75, getitem_76, getitem_77, getitem_78, getitem_79, getitem_80, getitem_81, getitem_82, getitem_83, getitem_84, getitem_85, getitem_86, getitem_87, getitem_88, getitem_89, getitem_90, getitem_91, getitem_92, getitem_93, getitem_94, getitem_95, getitem_96, getitem_97, getitem_98, getitem_99, getitem_100, getitem_101, getitem_102, getitem_103, getitem_104, getitem_105, getitem_106, getitem_107, getitem_108, getitem_109, getitem_110, getitem_111, getitem_112, getitem_113, getitem_114, getitem_115, getitem_116, getitem_117, getitem_118, getitem_119, getitem_120, getitem_121, getitem_122, getitem_123, getitem_124, getitem_125, getitem_126, getitem_127, getitem_128, getitem_129, getitem_130, getitem_131, getitem_132, getitem_133, getitem_134, getitem_135, getitem_136, getitem_137, getitem_138, getitem_139, getitem_140, getitem_141, getitem_142, getitem_143, getitem_144, getitem_145, getitem_146, getitem_147, getitem_148, getitem_149, getitem_150, getitem_151, getitem_152, getitem_153, getitem_154, getitem_155, getitem_156, getitem_157, getitem_158, getitem_159, getitem_160, getitem_161, getitem_162, getitem_163, getitem_164, getitem_165, getitem_166, getitem_167, getitem_1848, getitem_1849, getitem_1850, getitem_1851, getitem_1852, getitem_1853, getitem_1854, getitem_1855, getitem_1856, getitem_1857, getitem_1858, getitem_1859, getitem_1860, getitem_1861, getitem_1862, getitem_1863, getitem_1864, getitem_1865, getitem_1866, getitem_1867, getitem_1868, getitem_1869, getitem_1870, getitem_1871, getitem_1872, getitem_1873, getitem_1874, getitem_1875, getitem_1876, getitem_1877, getitem_1878, getitem_1879, getitem_1880, getitem_1881, getitem_1882, getitem_1883, getitem_1884, getitem_1885, getitem_1886, getitem_1887, getitem_1888, getitem_1889, getitem_1890, getitem_1891, getitem_1892, getitem_1893, getitem_1894, getitem_1895, getitem_1896, getitem_1897, getitem_1898, getitem_1899, getitem_1900, getitem_1901, getitem_1902, getitem_1903, getitem_1904, getitem_1905, getitem_1906, getitem_1907, getitem_1908, getitem_1909, getitem_1910, getitem_1911, getitem_1912, getitem_1913, getitem_1914, getitem_1915, getitem_1916, getitem_1917, getitem_1918, getitem_1919, getitem_1920, getitem_1921, getitem_1922, getitem_1923, getitem_1924, getitem_1925, getitem_1926, getitem_1927, getitem_1928, getitem_1929, getitem_1930, getitem_1931, getitem_1932, getitem_1933, getitem_1934, getitem_1935, getitem_1936, getitem_1937, getitem_1938, getitem_1939, getitem_1940, getitem_1941, getitem_1942, getitem_1943, getitem_1944, getitem_1945, getitem_1946, getitem_1947, getitem_1948, getitem_1949, getitem_1950, getitem_1951, getitem_1952, getitem_1953, getitem_1954, getitem_1955, getitem_1956, getitem_1957, getitem_1958, getitem_1959, getitem_1960, getitem_1961, getitem_1962, getitem_1963, getitem_1964, getitem_1965, getitem_1966, getitem_1967, getitem_1968, getitem_1969, getitem_1970, getitem_1971, getitem_1972, getitem_1973, getitem_1974, getitem_1975, getitem_1976, getitem_1977, getitem_1978, getitem_1979, getitem_1980, getitem_1981, getitem_1982, getitem_1983, getitem_1984, getitem_1985, getitem_1986, getitem_1987, getitem_1988, getitem_1989, getitem_1990, getitem_1991, getitem_1992, getitem_1993, getitem_1994, getitem_1995, getitem_1996, getitem_1997, getitem_1998, getitem_1999, getitem_2000, getitem_2001, getitem_2002, getitem_2003, getitem_2004, getitem_2005, getitem_2006, getitem_2007, getitem_2008, getitem_2009, getitem_2010, getitem_2011, getitem_2012, getitem_2013, getitem_2014, getitem_2015, getitem_2520, getitem_2521, getitem_2522, getitem_2523, getitem_2524, getitem_2525, getitem_2526, getitem_2527, getitem_2528, getitem_2529, getitem_2530, getitem_2531, getitem_2532, getitem_2533, getitem_2534, getitem_2535, getitem_2536, getitem_2537, getitem_2538, getitem_2539, getitem_2540, getitem_2541, getitem_2542, getitem_2543, getitem_2544, getitem_2545, getitem_2546, getitem_2547, getitem_2548, getitem_2549, getitem_2550, getitem_2551, getitem_2552, getitem_2553, getitem_2554, getitem_2555, getitem_2556, getitem_2557, getitem_2558, getitem_2559, getitem_2560, getitem_2561, getitem_2562, getitem_2563, getitem_2564, getitem_2565, getitem_2566, getitem_2567, getitem_2568, getitem_2569, getitem_2570, getitem_2571, getitem_2572, getitem_2573, getitem_2574, getitem_2575, getitem_2576, getitem_2577, getitem_2578, getitem_2579, getitem_2580, getitem_2581, getitem_2582, getitem_2583, getitem_2584, getitem_2585, getitem_2586, getitem_2587, getitem_2588, getitem_2589, getitem_2590, getitem_2591, getitem_2592, getitem_2593, getitem_2594, getitem_2595, getitem_2596, getitem_2597, getitem_2598, getitem_2599, getitem_2600, getitem_2601, getitem_2602, getitem_2603, getitem_2604, getitem_2605, getitem_2606, getitem_2607, getitem_2608, getitem_2609, getitem_2610, getitem_2611, getitem_2612, getitem_2613, getitem_2614, getitem_2615, getitem_2616, getitem_2617, getitem_2618, getitem_2619, getitem_2620, getitem_2621, getitem_2622, getitem_2623, getitem_2624, getitem_2625, getitem_2626, getitem_2627, getitem_2628, getitem_2629, getitem_2630, getitem_2631, getitem_2632, getitem_2633, getitem_2634, getitem_2635, getitem_2636, getitem_2637, getitem_2638, getitem_2639, getitem_2640, getitem_2641, getitem_2642, getitem_2643, getitem_2644, getitem_2645, getitem_2646, getitem_2647, getitem_2648, getitem_2649, getitem_2650, getitem_2651, getitem_2652, getitem_2653, getitem_2654, getitem_2655, getitem_2656, getitem_2657, getitem_2658, getitem_2659, getitem_2660, getitem_2661, getitem_2662, getitem_2663, getitem_2664, getitem_2665, getitem_2666, getitem_2667, getitem_2668, getitem_2669, getitem_2670, getitem_2671, getitem_2672, getitem_2673, getitem_2674, getitem_2675, getitem_2676, getitem_2677, getitem_2678, getitem_2679, getitem_2680, getitem_2681, getitem_2682, getitem_2683, getitem_2684, getitem_2685, getitem_2686, getitem_2687)


def _default_make_inputs():
    return [
    torch.randn([64, 3, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 3, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([96, 64, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96, 64, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96, 96, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96, 96, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([192, 96, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 96, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([384, 192, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([1408, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([1408], dtype=torch.float32, device='cuda'),
    torch.randn([1408], dtype=torch.float32, device='cuda'),
    torch.randn([1408, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1408], dtype=torch.float32, device='cuda'),
    torch.randn([1408], dtype=torch.float32, device='cuda'),
    torch.randn([1000, 1408], dtype=torch.float32, device='cuda'),
    torch.randn([1000], dtype=torch.float32, device='cuda'),
    torch.randn([64, 3, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 3, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([96, 64, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96, 64, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96, 96, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96, 96, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([192, 96, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 96, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([384, 192, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([1408, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([1408], dtype=torch.float32, device='cuda'),
    torch.randn([1408], dtype=torch.float32, device='cuda'),
    torch.randn([1408, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1408], dtype=torch.float32, device='cuda'),
    torch.randn([1408], dtype=torch.float32, device='cuda'),
    torch.randn([1000, 1408], dtype=torch.float32, device='cuda'),
    torch.randn([1000], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([64, 3, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 3, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([96, 64, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96, 64, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96, 96, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96, 96, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([192, 96, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 96, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([384, 192, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([1408, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([1408], dtype=torch.float32, device='cuda'),
    torch.randn([1408], dtype=torch.float32, device='cuda'),
    torch.randn([1408, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1408], dtype=torch.float32, device='cuda'),
    torch.randn([1408], dtype=torch.float32, device='cuda'),
    torch.randn([1000, 1408], dtype=torch.float32, device='cuda'),
    torch.randn([1000], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
