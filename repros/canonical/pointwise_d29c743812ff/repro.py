"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s9_g77
Pattern hash: d29c743812ff
Shape hash: 6ab25b36
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_1505: "f32[]", getitem_1506: "f32[]", getitem_1507: "f32[]", getitem_1508: "f32[]", getitem_1509: "f32[]", getitem_1510: "f32[]", getitem_1511: "f32[]", getitem_1512: "f32[]", getitem_1513: "f32[]", getitem_1514: "f32[]", getitem_1515: "f32[]", getitem_1516: "f32[]", getitem_1517: "f32[]", getitem_1518: "f32[]", getitem_1519: "f32[]", getitem_1520: "f32[]", getitem_1521: "f32[]", getitem_1522: "f32[]", getitem_1523: "f32[]", getitem_1524: "f32[]", getitem_1525: "f32[]", getitem_1526: "f32[]", getitem_1527: "f32[]", getitem_1528: "f32[]", getitem_1529: "f32[]", getitem_1530: "f32[]", getitem_1531: "f32[]", getitem_1532: "f32[]", getitem_1533: "f32[]", getitem_1534: "f32[]", getitem_1535: "f32[]", getitem_1536: "f32[]", getitem_1537: "f32[]", getitem_1538: "f32[]", getitem_1539: "f32[]", getitem_1540: "f32[]", getitem_1541: "f32[]", getitem_1542: "f32[]", getitem_1543: "f32[]", getitem_1544: "f32[]", getitem_1545: "f32[]", getitem_1546: "f32[]", getitem_1547: "f32[]", getitem_1548: "f32[]", getitem_1549: "f32[]", getitem_1550: "f32[]", getitem_1551: "f32[]", getitem_1552: "f32[]", getitem_1553: "f32[]", getitem_1554: "f32[]", getitem_1555: "f32[]", getitem_1556: "f32[]", getitem_1557: "f32[]", getitem_1558: "f32[]", getitem_1559: "f32[]", getitem_1560: "f32[]", getitem_1561: "f32[]", getitem_1562: "f32[]", getitem_1563: "f32[]", getitem_1564: "f32[]", getitem_1565: "f32[]", getitem_1566: "f32[]", getitem_1567: "f32[]", getitem_1568: "f32[]", getitem_1569: "f32[]", getitem_1570: "f32[]", getitem_1571: "f32[]", getitem_1572: "f32[]", getitem_1573: "f32[]", getitem_1574: "f32[]", getitem_1575: "f32[]", getitem_1576: "f32[]", getitem_1577: "f32[]", getitem_1578: "f32[]", getitem_1579: "f32[]", getitem_1580: "f32[]", getitem_1581: "f32[]", getitem_1582: "f32[]", getitem_1583: "f32[]", getitem_1584: "f32[]", getitem_1585: "f32[]", getitem_1586: "f32[]", getitem_1587: "f32[]", getitem_1588: "f32[]", getitem_1589: "f32[]", getitem_1590: "f32[]", getitem_1591: "f32[]", getitem_1592: "f32[]", getitem_1593: "f32[]", getitem_1594: "f32[]", getitem_1595: "f32[]", getitem_1596: "f32[]", getitem_1597: "f32[]", getitem_1598: "f32[]", getitem_1599: "f32[]", getitem_1600: "f32[]", getitem_1601: "f32[]", getitem_1602: "f32[]", getitem_1603: "f32[]", getitem_1604: "f32[]", getitem_1605: "f32[]", getitem_1606: "f32[]", getitem_1607: "f32[]", getitem_1608: "f32[]", getitem_1609: "f32[]", getitem_1610: "f32[]", getitem_1611: "f32[]", getitem_1612: "f32[]", getitem_1613: "f32[]", getitem_1614: "f32[]", getitem_1615: "f32[]", getitem_1616: "f32[]", getitem_1617: "f32[]", getitem_1618: "f32[]", getitem_1619: "f32[]", getitem_1620: "f32[]", getitem_1621: "f32[]", getitem_1622: "f32[]", getitem_1623: "f32[]", getitem_1624: "f32[]", getitem_1625: "f32[]", getitem_1626: "f32[]", getitem_1627: "f32[]", getitem_1628: "f32[]", getitem_1629: "f32[]", getitem_1630: "f32[]", getitem_1631: "f32[]", getitem_1632: "f32[]", getitem_1633: "f32[]", getitem_1634: "f32[]", getitem_1635: "f32[]", getitem_1636: "f32[]", getitem_1637: "f32[]", getitem_1638: "f32[]", getitem_1639: "f32[]", getitem_1640: "f32[]", getitem_1641: "f32[]", getitem_1642: "f32[]", getitem_1643: "f32[]", getitem_1644: "f32[]", getitem_1645: "f32[]", getitem_1646: "f32[]", getitem_1647: "f32[]", getitem_1648: "f32[]", getitem_1649: "f32[]", getitem_1650: "f32[]", getitem_1651: "f32[]", getitem_1652: "f32[]", getitem_1653: "f32[]", getitem_1654: "f32[]", getitem_1655: "f32[]", getitem_1656: "f32[]", getitem_1657: "f32[]", getitem_1658: "f32[]", getitem_1659: "f32[]", getitem_1660: "f32[]", getitem_1661: "f32[]", getitem_1662: "f32[]", getitem_1663: "f32[]", getitem_1664: "f32[]", getitem_1665: "f32[]", getitem_1666: "f32[]", getitem_1667: "f32[]", getitem_1668: "f32[]", getitem_1669: "f32[]", getitem_1670: "f32[]", getitem_1671: "f32[]", getitem_1672: "f32[]", getitem_1673: "f32[]", getitem_1674: "f32[]", getitem_1675: "f32[]", getitem_1676: "f32[]", getitem_1677: "f32[]", getitem_1678: "f32[]", getitem_1679: "f32[]", getitem_1680: "f32[]", getitem_1681: "f32[]", getitem_1682: "f32[]", getitem_1683: "f32[]", getitem_1684: "f32[]", getitem_1685: "f32[]", getitem_1686: "f32[]", getitem_1687: "f32[]", getitem_1688: "f32[]", getitem_1689: "f32[]", getitem_1690: "f32[]", getitem_1691: "f32[]", getitem_1692: "f32[]", getitem_1693: "f32[]", getitem_1694: "f32[]", getitem_1695: "f32[]", getitem_1696: "f32[]", getitem_1697: "f32[]", getitem_1698: "f32[]", getitem_1699: "f32[]", getitem_1700: "f32[]", getitem_1701: "f32[]", getitem_1702: "f32[]", getitem_1703: "f32[]", getitem_1704: "f32[]", getitem_1705: "f32[]", getitem_1706: "f32[]", getitem_1707: "f32[]", getitem_1708: "f32[]", getitem_1709: "f32[]", getitem_1710: "f32[]", getitem_1711: "f32[]", getitem_1712: "f32[]", getitem_1713: "f32[]", getitem_1714: "f32[]", getitem_1715: "f32[]", getitem_1716: "f32[]", getitem_1717: "f32[]", getitem_1718: "f32[]", getitem_1719: "f32[]", getitem_1720: "f32[]", getitem_1721: "f32[]", getitem_1722: "f32[]", getitem_1723: "f32[]", getitem_1724: "f32[]", getitem_1725: "f32[]", getitem_1726: "f32[]", getitem_1727: "f32[]", getitem_1728: "f32[]", getitem_1729: "f32[]", getitem_1730: "f32[]", getitem_1731: "f32[]", getitem_1732: "f32[]", getitem_1733: "f32[]", getitem_1734: "f32[]", getitem_1735: "f32[]", getitem_1736: "f32[]", getitem_1737: "f32[]", getitem_1738: "f32[]", getitem_1739: "f32[]", getitem_1740: "f32[]", getitem_1741: "f32[]", getitem_1742: "f32[]", getitem_1743: "f32[]", getitem_1744: "f32[]", getitem_1745: "f32[]", getitem_1746: "f32[]", getitem_1747: "f32[]", getitem_1748: "f32[]", getitem_1749: "f32[]", getitem_1750: "f32[]", getitem_1751: "f32[]", getitem_1752: "f32[]", getitem_1753: "f32[]", getitem_1754: "f32[]", getitem_1755: "f32[]", getitem_1756: "f32[]", getitem_1757: "f32[]", getitem_1758: "f32[]", getitem_1759: "f32[]", getitem_1760: "f32[]", getitem_1761: "f32[]", getitem_1762: "f32[]", getitem_1763: "f32[]", getitem_1764: "f32[]", getitem_1765: "f32[]", getitem_1766: "f32[]", getitem_1767: "f32[]", getitem_1768: "f32[]", getitem_1769: "f32[]", getitem_1770: "f32[]", getitem_1771: "f32[]", getitem_1772: "f32[]", getitem_1773: "f32[]", getitem_1774: "f32[]", getitem_1775: "f32[]", getitem_1776: "f32[]", getitem_1777: "f32[]", getitem_1778: "f32[]", getitem_1779: "f32[]", getitem_1780: "f32[]", getitem_1781: "f32[]", getitem_1782: "f32[]", getitem_1783: "f32[]", getitem_1784: "f32[]", getitem_1785: "f32[]", getitem_1786: "f32[]", getitem_1787: "f32[]", getitem_1788: "f32[]", getitem_1789: "f32[]", getitem_1790: "f32[]", getitem_1791: "f32[]", getitem_1792: "f32[]", getitem_1793: "f32[]", getitem_1794: "f32[]", getitem_1795: "f32[]", getitem_1796: "f32[]", getitem_1797: "f32[]", getitem_1798: "f32[]", getitem_1799: "f32[]", getitem_1800: "f32[]", getitem_1801: "f32[]", getitem_1802: "f32[]", getitem_1803: "f32[]", getitem_1804: "f32[]", getitem_1805: "f32[]", getitem_2709: "f32[]", getitem_2710: "f32[]", getitem_2711: "f32[]", getitem_2712: "f32[]", getitem_2713: "f32[]", getitem_2714: "f32[]", getitem_2715: "f32[]", getitem_2716: "f32[]", getitem_2717: "f32[]", getitem_2718: "f32[]", getitem_2719: "f32[]", getitem_2720: "f32[]", getitem_2721: "f32[]", getitem_2722: "f32[]", getitem_2723: "f32[]", getitem_2724: "f32[]", getitem_2725: "f32[]", getitem_2726: "f32[]", getitem_2727: "f32[]", getitem_2728: "f32[]", getitem_2729: "f32[]", getitem_2730: "f32[]", getitem_2731: "f32[]", getitem_2732: "f32[]", getitem_2733: "f32[]", getitem_2734: "f32[]", getitem_2735: "f32[]", getitem_2736: "f32[]", getitem_2737: "f32[]", getitem_2738: "f32[]", getitem_2739: "f32[]", getitem_2740: "f32[]", getitem_2741: "f32[]", getitem_2742: "f32[]", getitem_2743: "f32[]", getitem_2744: "f32[]", getitem_2745: "f32[]", getitem_2746: "f32[]", getitem_2747: "f32[]", getitem_2748: "f32[]", getitem_2749: "f32[]", getitem_2750: "f32[]", getitem_2751: "f32[]", getitem_2752: "f32[]", getitem_2753: "f32[]", getitem_2754: "f32[]", getitem_2755: "f32[]", getitem_2756: "f32[]", getitem_2757: "f32[]", getitem_2758: "f32[]", getitem_2759: "f32[]", getitem_2760: "f32[]", getitem_2761: "f32[]", getitem_2762: "f32[]", getitem_2763: "f32[]", getitem_2764: "f32[]", getitem_2765: "f32[]", getitem_2766: "f32[]", getitem_2767: "f32[]", getitem_2768: "f32[]", getitem_2769: "f32[]", getitem_2770: "f32[]", getitem_2771: "f32[]", getitem_2772: "f32[]", getitem_2773: "f32[]", getitem_2774: "f32[]", getitem_2775: "f32[]", getitem_2776: "f32[]", getitem_2777: "f32[]", getitem_2778: "f32[]", getitem_2779: "f32[]", getitem_2780: "f32[]", getitem_2781: "f32[]", getitem_2782: "f32[]", getitem_2783: "f32[]", getitem_2784: "f32[]", getitem_2785: "f32[]", getitem_2786: "f32[]", getitem_2787: "f32[]", getitem_2788: "f32[]", getitem_2789: "f32[]", getitem_2790: "f32[]", getitem_2791: "f32[]", getitem_2792: "f32[]", getitem_2793: "f32[]", getitem_2794: "f32[]", getitem_2795: "f32[]", getitem_2796: "f32[]", getitem_2797: "f32[]", getitem_2798: "f32[]", getitem_2799: "f32[]", getitem_2800: "f32[]", getitem_2801: "f32[]", getitem_2802: "f32[]", getitem_2803: "f32[]", getitem_2804: "f32[]", getitem_2805: "f32[]", getitem_2806: "f32[]", getitem_2807: "f32[]", getitem_2808: "f32[]", getitem_2809: "f32[]", getitem_2810: "f32[]", getitem_2811: "f32[]", getitem_2812: "f32[]", getitem_2813: "f32[]", getitem_2814: "f32[]", getitem_2815: "f32[]", getitem_2816: "f32[]", getitem_2817: "f32[]", getitem_2818: "f32[]", getitem_2819: "f32[]", getitem_2820: "f32[]", getitem_2821: "f32[]", getitem_2822: "f32[]", getitem_2823: "f32[]", getitem_2824: "f32[]", getitem_2825: "f32[]", getitem_2826: "f32[]", getitem_2827: "f32[]", getitem_2828: "f32[]", getitem_2829: "f32[]", getitem_2830: "f32[]", getitem_2831: "f32[]", getitem_2832: "f32[]", getitem_2833: "f32[]", getitem_2834: "f32[]", getitem_2835: "f32[]", getitem_2836: "f32[]", getitem_2837: "f32[]", getitem_2838: "f32[]", getitem_2839: "f32[]", getitem_2840: "f32[]", getitem_2841: "f32[]", getitem_2842: "f32[]", getitem_2843: "f32[]", getitem_2844: "f32[]", getitem_2845: "f32[]", getitem_2846: "f32[]", getitem_2847: "f32[]", getitem_2848: "f32[]", getitem_2849: "f32[]", getitem_2850: "f32[]", getitem_2851: "f32[]", getitem_2852: "f32[]", getitem_2853: "f32[]", getitem_2854: "f32[]", getitem_2855: "f32[]", getitem_2856: "f32[]", getitem_2857: "f32[]", getitem_2858: "f32[]", getitem_2859: "f32[]", getitem_2860: "f32[]", getitem_2861: "f32[]", getitem_2862: "f32[]", getitem_2863: "f32[]", getitem_2864: "f32[]", getitem_2865: "f32[]", getitem_2866: "f32[]", getitem_2867: "f32[]", getitem_2868: "f32[]", getitem_2869: "f32[]", getitem_2870: "f32[]", getitem_2871: "f32[]", getitem_2872: "f32[]", getitem_2873: "f32[]", getitem_2874: "f32[]", getitem_2875: "f32[]", getitem_2876: "f32[]", getitem_2877: "f32[]", getitem_2878: "f32[]", getitem_2879: "f32[]", getitem_2880: "f32[]", getitem_2881: "f32[]", getitem_2882: "f32[]", getitem_2883: "f32[]", getitem_2884: "f32[]", getitem_2885: "f32[]", getitem_2886: "f32[]", getitem_2887: "f32[]", getitem_2888: "f32[]", getitem_2889: "f32[]", getitem_2890: "f32[]", getitem_2891: "f32[]", getitem_2892: "f32[]", getitem_2893: "f32[]", getitem_2894: "f32[]", getitem_2895: "f32[]", getitem_2896: "f32[]", getitem_2897: "f32[]", getitem_2898: "f32[]", getitem_2899: "f32[]", getitem_2900: "f32[]", getitem_2901: "f32[]", getitem_2902: "f32[]", getitem_2903: "f32[]", getitem_2904: "f32[]", getitem_2905: "f32[]", getitem_2906: "f32[]", getitem_2907: "f32[]", getitem_2908: "f32[]", getitem_2909: "f32[]", getitem_2910: "f32[]", getitem_2911: "f32[]", getitem_2912: "f32[]", getitem_2913: "f32[]", getitem_2914: "f32[]", getitem_2915: "f32[]", getitem_2916: "f32[]", getitem_2917: "f32[]", getitem_2918: "f32[]", getitem_2919: "f32[]", getitem_2920: "f32[]", getitem_2921: "f32[]", getitem_2922: "f32[]", getitem_2923: "f32[]", getitem_2924: "f32[]", getitem_2925: "f32[]", getitem_2926: "f32[]", getitem_2927: "f32[]", getitem_2928: "f32[]", getitem_2929: "f32[]", getitem_2930: "f32[]", getitem_2931: "f32[]", getitem_2932: "f32[]", getitem_2933: "f32[]", getitem_2934: "f32[]", getitem_2935: "f32[]", getitem_2936: "f32[]", getitem_2937: "f32[]", getitem_2938: "f32[]", getitem_2939: "f32[]", getitem_2940: "f32[]", getitem_2941: "f32[]", getitem_2942: "f32[]", getitem_2943: "f32[]", getitem_2944: "f32[]", getitem_2945: "f32[]", getitem_2946: "f32[]", getitem_2947: "f32[]", getitem_2948: "f32[]", getitem_2949: "f32[]", getitem_2950: "f32[]", getitem_2951: "f32[]", getitem_2952: "f32[]", getitem_2953: "f32[]", getitem_2954: "f32[]", getitem_2955: "f32[]", getitem_2956: "f32[]", getitem_2957: "f32[]", getitem_2958: "f32[]", getitem_2959: "f32[]", getitem_2960: "f32[]", getitem_2961: "f32[]", getitem_2962: "f32[]", getitem_2963: "f32[]", getitem_2964: "f32[]", getitem_2965: "f32[]", getitem_2966: "f32[]", getitem_2967: "f32[]", getitem_2968: "f32[]", getitem_2969: "f32[]", getitem_2970: "f32[]", getitem_2971: "f32[]", getitem_2972: "f32[]", getitem_2973: "f32[]", getitem_2974: "f32[]", getitem_2975: "f32[]", getitem_2976: "f32[]", getitem_2977: "f32[]", getitem_2978: "f32[]", getitem_2979: "f32[]", getitem_2980: "f32[]", getitem_2981: "f32[]", getitem_2982: "f32[]", getitem_2983: "f32[]", getitem_2984: "f32[]", getitem_2985: "f32[]", getitem_2986: "f32[]", getitem_2987: "f32[]", getitem_2988: "f32[]", getitem_2989: "f32[]", getitem_2990: "f32[]", getitem_2991: "f32[]", getitem_2992: "f32[]", getitem_2993: "f32[]", getitem_2994: "f32[]", getitem_2995: "f32[]", getitem_2996: "f32[]", getitem_2997: "f32[]", getitem_2998: "f32[]", getitem_2999: "f32[]", getitem_3000: "f32[]", getitem_3001: "f32[]", getitem_3002: "f32[]", getitem_3003: "f32[]", getitem_3004: "f32[]", getitem_3005: "f32[]", getitem_3006: "f32[]", getitem_3007: "f32[]", getitem_3008: "f32[]", getitem_3009: "f32[]", getitem_1204: "f32[768]", getitem_1205: "f32[50, 768]", getitem_1206: "f32[768, 512]", getitem_1207: "f32[768, 3, 32, 32]", getitem_1208: "f32[768]", getitem_1209: "f32[768]", getitem_1210: "f32[2304, 768]", getitem_1211: "f32[2304]", getitem_1212: "f32[768, 768]", getitem_1213: "f32[768]", getitem_1214: "f32[3072, 768]", getitem_1215: "f32[3072]", getitem_1216: "f32[768, 3072]", getitem_1217: "f32[768]", getitem_1218: "f32[768]", getitem_1219: "f32[768]", getitem_1220: "f32[768]", getitem_1221: "f32[768]", getitem_1222: "f32[2304, 768]", getitem_1223: "f32[2304]", getitem_1224: "f32[768, 768]", getitem_1225: "f32[768]", getitem_1226: "f32[3072, 768]", getitem_1227: "f32[3072]", getitem_1228: "f32[768, 3072]", getitem_1229: "f32[768]", getitem_1230: "f32[768]", getitem_1231: "f32[768]", getitem_1232: "f32[768]", getitem_1233: "f32[768]", getitem_1234: "f32[2304, 768]", getitem_1235: "f32[2304]", getitem_1236: "f32[768, 768]", getitem_1237: "f32[768]", getitem_1238: "f32[3072, 768]", getitem_1239: "f32[3072]", getitem_1240: "f32[768, 3072]", getitem_1241: "f32[768]", getitem_1242: "f32[768]", getitem_1243: "f32[768]", getitem_1244: "f32[768]", getitem_1245: "f32[768]", getitem_1246: "f32[2304, 768]", getitem_1247: "f32[2304]", getitem_1248: "f32[768, 768]", getitem_1249: "f32[768]", getitem_1250: "f32[3072, 768]", getitem_1251: "f32[3072]", getitem_1252: "f32[768, 3072]", getitem_1253: "f32[768]", getitem_1254: "f32[768]", getitem_1255: "f32[768]", getitem_1256: "f32[768]", getitem_1257: "f32[768]", getitem_1258: "f32[2304, 768]", getitem_1259: "f32[2304]", getitem_1260: "f32[768, 768]", getitem_1261: "f32[768]", getitem_1262: "f32[3072, 768]", getitem_1263: "f32[3072]", getitem_1264: "f32[768, 3072]", getitem_1265: "f32[768]", getitem_1266: "f32[768]", getitem_1267: "f32[768]", getitem_1268: "f32[768]", getitem_1269: "f32[768]", getitem_1270: "f32[2304, 768]", getitem_1271: "f32[2304]", getitem_1272: "f32[768, 768]", getitem_1273: "f32[768]", getitem_1274: "f32[3072, 768]", getitem_1275: "f32[3072]", getitem_1276: "f32[768, 3072]", getitem_1277: "f32[768]", getitem_1278: "f32[768]", getitem_1279: "f32[768]", getitem_1280: "f32[768]", getitem_1281: "f32[768]", getitem_1282: "f32[2304, 768]", getitem_1283: "f32[2304]", getitem_1284: "f32[768, 768]", getitem_1285: "f32[768]", getitem_1286: "f32[3072, 768]", getitem_1287: "f32[3072]", getitem_1288: "f32[768, 3072]", getitem_1289: "f32[768]", getitem_1290: "f32[768]", getitem_1291: "f32[768]", getitem_1292: "f32[768]", getitem_1293: "f32[768]", getitem_1294: "f32[2304, 768]", getitem_1295: "f32[2304]", getitem_1296: "f32[768, 768]", getitem_1297: "f32[768]", getitem_1298: "f32[3072, 768]", getitem_1299: "f32[3072]", getitem_1300: "f32[768, 3072]", getitem_1301: "f32[768]", getitem_1302: "f32[768]", getitem_1303: "f32[768]", getitem_1304: "f32[768]", getitem_1305: "f32[768]", getitem_1306: "f32[2304, 768]", getitem_1307: "f32[2304]", getitem_1308: "f32[768, 768]", getitem_1309: "f32[768]", getitem_1310: "f32[3072, 768]", getitem_1311: "f32[3072]", getitem_1312: "f32[768, 3072]", getitem_1313: "f32[768]", getitem_1314: "f32[768]", getitem_1315: "f32[768]", getitem_1316: "f32[768]", getitem_1317: "f32[768]", getitem_1318: "f32[2304, 768]", getitem_1319: "f32[2304]", getitem_1320: "f32[768, 768]", getitem_1321: "f32[768]", getitem_1322: "f32[3072, 768]", getitem_1323: "f32[3072]", getitem_1324: "f32[768, 3072]", getitem_1325: "f32[768]", getitem_1326: "f32[768]", getitem_1327: "f32[768]", getitem_1328: "f32[768]", getitem_1329: "f32[768]", getitem_1330: "f32[2304, 768]", getitem_1331: "f32[2304]", getitem_1332: "f32[768, 768]", getitem_1333: "f32[768]", getitem_1334: "f32[3072, 768]", getitem_1335: "f32[3072]", getitem_1336: "f32[768, 3072]", getitem_1337: "f32[768]", getitem_1338: "f32[768]", getitem_1339: "f32[768]", getitem_1340: "f32[768]", getitem_1341: "f32[768]", getitem_1342: "f32[2304, 768]", getitem_1343: "f32[2304]", getitem_1344: "f32[768, 768]", getitem_1345: "f32[768]", getitem_1346: "f32[3072, 768]", getitem_1347: "f32[3072]", getitem_1348: "f32[768, 3072]", getitem_1349: "f32[768]", getitem_1350: "f32[768]", getitem_1351: "f32[768]", getitem_1352: "f32[768]", getitem_1353: "f32[768]", getitem_1354: "f32[768]", getitem_1355: "f32[768]", getitem_1356: "f32[77, 512]", getitem_1357: "f32[49408, 512]", getitem_1358: "f32[1536, 512]", getitem_1359: "f32[1536]", getitem_1360: "f32[512, 512]", getitem_1361: "f32[512]", getitem_1362: "f32[2048, 512]", getitem_1363: "f32[2048]", getitem_1364: "f32[512, 2048]", getitem_1365: "f32[512]", getitem_1366: "f32[512]", getitem_1367: "f32[512]", getitem_1368: "f32[512]", getitem_1369: "f32[512]", getitem_1370: "f32[1536, 512]", getitem_1371: "f32[1536]", getitem_1372: "f32[512, 512]", getitem_1373: "f32[512]", getitem_1374: "f32[2048, 512]", getitem_1375: "f32[2048]", getitem_1376: "f32[512, 2048]", getitem_1377: "f32[512]", getitem_1378: "f32[512]", getitem_1379: "f32[512]", getitem_1380: "f32[512]", getitem_1381: "f32[512]", getitem_1382: "f32[1536, 512]", getitem_1383: "f32[1536]", getitem_1384: "f32[512, 512]", getitem_1385: "f32[512]", getitem_1386: "f32[2048, 512]", getitem_1387: "f32[2048]", getitem_1388: "f32[512, 2048]", getitem_1389: "f32[512]", getitem_1390: "f32[512]", getitem_1391: "f32[512]", getitem_1392: "f32[512]", getitem_1393: "f32[512]", getitem_1394: "f32[1536, 512]", getitem_1395: "f32[1536]", getitem_1396: "f32[512, 512]", getitem_1397: "f32[512]", getitem_1398: "f32[2048, 512]", getitem_1399: "f32[2048]", getitem_1400: "f32[512, 2048]", getitem_1401: "f32[512]", getitem_1402: "f32[512]", getitem_1403: "f32[512]", getitem_1404: "f32[512]", getitem_1405: "f32[512]", getitem_1406: "f32[1536, 512]", getitem_1407: "f32[1536]", getitem_1408: "f32[512, 512]", getitem_1409: "f32[512]", getitem_1410: "f32[2048, 512]", getitem_1411: "f32[2048]", getitem_1412: "f32[512, 2048]", getitem_1413: "f32[512]", getitem_1414: "f32[512]", getitem_1415: "f32[512]", getitem_1416: "f32[512]", getitem_1417: "f32[512]", getitem_1418: "f32[1536, 512]", getitem_1419: "f32[1536]", getitem_1420: "f32[512, 512]", getitem_1421: "f32[512]", getitem_1422: "f32[2048, 512]", getitem_1423: "f32[2048]", getitem_1424: "f32[512, 2048]", getitem_1425: "f32[512]", getitem_1426: "f32[512]", getitem_1427: "f32[512]", getitem_1428: "f32[512]", getitem_1429: "f32[512]", getitem_1430: "f32[1536, 512]", getitem_1431: "f32[1536]", getitem_1432: "f32[512, 512]", getitem_1433: "f32[512]", getitem_1434: "f32[2048, 512]", getitem_1435: "f32[2048]", getitem_1436: "f32[512, 2048]", getitem_1437: "f32[512]", getitem_1438: "f32[512]", getitem_1439: "f32[512]", getitem_1440: "f32[512]", getitem_1441: "f32[512]", getitem_1442: "f32[1536, 512]", getitem_1443: "f32[1536]", getitem_1444: "f32[512, 512]", getitem_1445: "f32[512]", getitem_1446: "f32[2048, 512]", getitem_1447: "f32[2048]", getitem_1448: "f32[512, 2048]", getitem_1449: "f32[512]", getitem_1450: "f32[512]", getitem_1451: "f32[512]", getitem_1452: "f32[512]", getitem_1453: "f32[512]", getitem_1454: "f32[1536, 512]", getitem_1455: "f32[1536]", getitem_1456: "f32[512, 512]", getitem_1457: "f32[512]", getitem_1458: "f32[2048, 512]", getitem_1459: "f32[2048]", getitem_1460: "f32[512, 2048]", getitem_1461: "f32[512]", getitem_1462: "f32[512]", getitem_1463: "f32[512]", getitem_1464: "f32[512]", getitem_1465: "f32[512]", getitem_1466: "f32[1536, 512]", getitem_1467: "f32[1536]", getitem_1468: "f32[512, 512]", getitem_1469: "f32[512]", getitem_1470: "f32[2048, 512]", getitem_1471: "f32[2048]", getitem_1472: "f32[512, 2048]", getitem_1473: "f32[512]", getitem_1474: "f32[512]", getitem_1475: "f32[512]", getitem_1476: "f32[512]", getitem_1477: "f32[512]", getitem_1478: "f32[1536, 512]", getitem_1479: "f32[1536]", getitem_1480: "f32[512, 512]", getitem_1481: "f32[512]", getitem_1482: "f32[2048, 512]", getitem_1483: "f32[2048]", getitem_1484: "f32[512, 2048]", getitem_1485: "f32[512]", getitem_1486: "f32[512]", getitem_1487: "f32[512]", getitem_1488: "f32[512]", getitem_1489: "f32[512]", getitem_1490: "f32[1536, 512]", getitem_1491: "f32[1536]", getitem_1492: "f32[512, 512]", getitem_1493: "f32[512]", getitem_1494: "f32[2048, 512]", getitem_1495: "f32[2048]", getitem_1496: "f32[512, 2048]", getitem_1497: "f32[512]", getitem_1498: "f32[512]", getitem_1499: "f32[512]", getitem_1500: "f32[512]", getitem_1501: "f32[512]", getitem_1502: "f32[512]", getitem_1503: "f32[512]", getitem_1504: "f32[512, 512]"):
        # No stacktrace found for following nodes
        _foreach_sub_scalar = torch.ops.aten._foreach_sub.Scalar([getitem_1505, getitem_1506, getitem_1507, getitem_1508, getitem_1509, getitem_1510, getitem_1511, getitem_1512, getitem_1513, getitem_1514, getitem_1515, getitem_1516, getitem_1517, getitem_1518, getitem_1519, getitem_1520, getitem_1521, getitem_1522, getitem_1523, getitem_1524, getitem_1525, getitem_1526, getitem_1527, getitem_1528, getitem_1529, getitem_1530, getitem_1531, getitem_1532, getitem_1533, getitem_1534, getitem_1535, getitem_1536, getitem_1537, getitem_1538, getitem_1539, getitem_1540, getitem_1541, getitem_1542, getitem_1543, getitem_1544, getitem_1545, getitem_1546, getitem_1547, getitem_1548, getitem_1549, getitem_1550, getitem_1551, getitem_1552, getitem_1553, getitem_1554, getitem_1555, getitem_1556, getitem_1557, getitem_1558, getitem_1559, getitem_1560, getitem_1561, getitem_1562, getitem_1563, getitem_1564, getitem_1565, getitem_1566, getitem_1567, getitem_1568, getitem_1569, getitem_1570, getitem_1571, getitem_1572, getitem_1573, getitem_1574, getitem_1575, getitem_1576, getitem_1577, getitem_1578, getitem_1579, getitem_1580, getitem_1581, getitem_1582, getitem_1583, getitem_1584, getitem_1585, getitem_1586, getitem_1587, getitem_1588, getitem_1589, getitem_1590, getitem_1591, getitem_1592, getitem_1593, getitem_1594, getitem_1595, getitem_1596, getitem_1597, getitem_1598, getitem_1599, getitem_1600, getitem_1601, getitem_1602, getitem_1603, getitem_1604, getitem_1605, getitem_1606, getitem_1607, getitem_1608, getitem_1609, getitem_1610, getitem_1611, getitem_1612, getitem_1613, getitem_1614, getitem_1615, getitem_1616, getitem_1617, getitem_1618, getitem_1619, getitem_1620, getitem_1621, getitem_1622, getitem_1623, getitem_1624, getitem_1625, getitem_1626, getitem_1627, getitem_1628, getitem_1629, getitem_1630, getitem_1631, getitem_1632, getitem_1633, getitem_1634, getitem_1635, getitem_1636, getitem_1637, getitem_1638, getitem_1639, getitem_1640, getitem_1641, getitem_1642, getitem_1643, getitem_1644, getitem_1645, getitem_1646, getitem_1647, getitem_1648, getitem_1649, getitem_1650, getitem_1651, getitem_1652, getitem_1653, getitem_1654, getitem_1655, getitem_1656, getitem_1657, getitem_1658, getitem_1659, getitem_1660, getitem_1661, getitem_1662, getitem_1663, getitem_1664, getitem_1665, getitem_1666, getitem_1667, getitem_1668, getitem_1669, getitem_1670, getitem_1671, getitem_1672, getitem_1673, getitem_1674, getitem_1675, getitem_1676, getitem_1677, getitem_1678, getitem_1679, getitem_1680, getitem_1681, getitem_1682, getitem_1683, getitem_1684, getitem_1685, getitem_1686, getitem_1687, getitem_1688, getitem_1689, getitem_1690, getitem_1691, getitem_1692, getitem_1693, getitem_1694, getitem_1695, getitem_1696, getitem_1697, getitem_1698, getitem_1699, getitem_1700, getitem_1701, getitem_1702, getitem_1703, getitem_1704, getitem_1705, getitem_1706, getitem_1707, getitem_1708, getitem_1709, getitem_1710, getitem_1711, getitem_1712, getitem_1713, getitem_1714, getitem_1715, getitem_1716, getitem_1717, getitem_1718, getitem_1719, getitem_1720, getitem_1721, getitem_1722, getitem_1723, getitem_1724, getitem_1725, getitem_1726, getitem_1727, getitem_1728, getitem_1729, getitem_1730, getitem_1731, getitem_1732, getitem_1733, getitem_1734, getitem_1735, getitem_1736, getitem_1737, getitem_1738, getitem_1739, getitem_1740, getitem_1741, getitem_1742, getitem_1743, getitem_1744, getitem_1745, getitem_1746, getitem_1747, getitem_1748, getitem_1749, getitem_1750, getitem_1751, getitem_1752, getitem_1753, getitem_1754, getitem_1755, getitem_1756, getitem_1757, getitem_1758, getitem_1759, getitem_1760, getitem_1761, getitem_1762, getitem_1763, getitem_1764, getitem_1765, getitem_1766, getitem_1767, getitem_1768, getitem_1769, getitem_1770, getitem_1771, getitem_1772, getitem_1773, getitem_1774, getitem_1775, getitem_1776, getitem_1777, getitem_1778, getitem_1779, getitem_1780, getitem_1781, getitem_1782, getitem_1783, getitem_1784, getitem_1785, getitem_1786, getitem_1787, getitem_1788, getitem_1789, getitem_1790, getitem_1791, getitem_1792, getitem_1793, getitem_1794, getitem_1795, getitem_1796, getitem_1797, getitem_1798, getitem_1799, getitem_1800, getitem_1801, getitem_1802, getitem_1803, getitem_1804, getitem_1805], 1);  getitem_1505 = getitem_1506 = getitem_1507 = getitem_1508 = getitem_1509 = getitem_1510 = getitem_1511 = getitem_1512 = getitem_1513 = getitem_1514 = getitem_1515 = getitem_1516 = getitem_1517 = getitem_1518 = getitem_1519 = getitem_1520 = getitem_1521 = getitem_1522 = getitem_1523 = getitem_1524 = getitem_1525 = getitem_1526 = getitem_1527 = getitem_1528 = getitem_1529 = getitem_1530 = getitem_1531 = getitem_1532 = getitem_1533 = getitem_1534 = getitem_1535 = getitem_1536 = getitem_1537 = getitem_1538 = getitem_1539 = getitem_1540 = getitem_1541 = getitem_1542 = getitem_1543 = getitem_1544 = getitem_1545 = getitem_1546 = getitem_1547 = getitem_1548 = getitem_1549 = getitem_1550 = getitem_1551 = getitem_1552 = getitem_1553 = getitem_1554 = getitem_1555 = getitem_1556 = getitem_1557 = getitem_1558 = getitem_1559 = getitem_1560 = getitem_1561 = getitem_1562 = getitem_1563 = getitem_1564 = getitem_1565 = getitem_1566 = getitem_1567 = getitem_1568 = getitem_1569 = getitem_1570 = getitem_1571 = getitem_1572 = getitem_1573 = getitem_1574 = getitem_1575 = getitem_1576 = getitem_1577 = getitem_1578 = getitem_1579 = getitem_1580 = getitem_1581 = getitem_1582 = getitem_1583 = getitem_1584 = getitem_1585 = getitem_1586 = getitem_1587 = getitem_1588 = getitem_1589 = getitem_1590 = getitem_1591 = getitem_1592 = getitem_1593 = getitem_1594 = getitem_1595 = getitem_1596 = getitem_1597 = getitem_1598 = getitem_1599 = getitem_1600 = getitem_1601 = getitem_1602 = getitem_1603 = getitem_1604 = getitem_1605 = getitem_1606 = getitem_1607 = getitem_1608 = getitem_1609 = getitem_1610 = getitem_1611 = getitem_1612 = getitem_1613 = getitem_1614 = getitem_1615 = getitem_1616 = getitem_1617 = getitem_1618 = getitem_1619 = getitem_1620 = getitem_1621 = getitem_1622 = getitem_1623 = getitem_1624 = getitem_1625 = getitem_1626 = getitem_1627 = getitem_1628 = getitem_1629 = getitem_1630 = getitem_1631 = getitem_1632 = getitem_1633 = getitem_1634 = getitem_1635 = getitem_1636 = getitem_1637 = getitem_1638 = getitem_1639 = getitem_1640 = getitem_1641 = getitem_1642 = getitem_1643 = getitem_1644 = getitem_1645 = getitem_1646 = getitem_1647 = getitem_1648 = getitem_1649 = getitem_1650 = getitem_1651 = getitem_1652 = getitem_1653 = getitem_1654 = getitem_1655 = getitem_1656 = getitem_1657 = getitem_1658 = getitem_1659 = getitem_1660 = getitem_1661 = getitem_1662 = getitem_1663 = getitem_1664 = getitem_1665 = getitem_1666 = getitem_1667 = getitem_1668 = getitem_1669 = getitem_1670 = getitem_1671 = getitem_1672 = getitem_1673 = getitem_1674 = getitem_1675 = getitem_1676 = getitem_1677 = getitem_1678 = getitem_1679 = getitem_1680 = getitem_1681 = getitem_1682 = getitem_1683 = getitem_1684 = getitem_1685 = getitem_1686 = getitem_1687 = getitem_1688 = getitem_1689 = getitem_1690 = getitem_1691 = getitem_1692 = getitem_1693 = getitem_1694 = getitem_1695 = getitem_1696 = getitem_1697 = getitem_1698 = getitem_1699 = getitem_1700 = getitem_1701 = getitem_1702 = getitem_1703 = getitem_1704 = getitem_1705 = getitem_1706 = getitem_1707 = getitem_1708 = getitem_1709 = getitem_1710 = getitem_1711 = getitem_1712 = getitem_1713 = getitem_1714 = getitem_1715 = getitem_1716 = getitem_1717 = getitem_1718 = getitem_1719 = getitem_1720 = getitem_1721 = getitem_1722 = getitem_1723 = getitem_1724 = getitem_1725 = getitem_1726 = getitem_1727 = getitem_1728 = getitem_1729 = getitem_1730 = getitem_1731 = getitem_1732 = getitem_1733 = getitem_1734 = getitem_1735 = getitem_1736 = getitem_1737 = getitem_1738 = getitem_1739 = getitem_1740 = getitem_1741 = getitem_1742 = getitem_1743 = getitem_1744 = getitem_1745 = getitem_1746 = getitem_1747 = getitem_1748 = getitem_1749 = getitem_1750 = getitem_1751 = getitem_1752 = getitem_1753 = getitem_1754 = getitem_1755 = getitem_1756 = getitem_1757 = getitem_1758 = getitem_1759 = getitem_1760 = getitem_1761 = getitem_1762 = getitem_1763 = getitem_1764 = getitem_1765 = getitem_1766 = getitem_1767 = getitem_1768 = getitem_1769 = getitem_1770 = getitem_1771 = getitem_1772 = getitem_1773 = getitem_1774 = getitem_1775 = getitem_1776 = getitem_1777 = getitem_1778 = getitem_1779 = getitem_1780 = getitem_1781 = getitem_1782 = getitem_1783 = getitem_1784 = getitem_1785 = getitem_1786 = getitem_1787 = getitem_1788 = getitem_1789 = getitem_1790 = getitem_1791 = getitem_1792 = getitem_1793 = getitem_1794 = getitem_1795 = getitem_1796 = getitem_1797 = getitem_1798 = getitem_1799 = getitem_1800 = getitem_1801 = getitem_1802 = getitem_1803 = getitem_1804 = getitem_1805 = None
        getitem: "f32[]" = _foreach_sub_scalar[0]
        getitem_1806: "f32[]" = _foreach_sub_scalar[1]
        getitem_1807: "f32[]" = _foreach_sub_scalar[2]
        getitem_1808: "f32[]" = _foreach_sub_scalar[3]
        getitem_1809: "f32[]" = _foreach_sub_scalar[4]
        getitem_1810: "f32[]" = _foreach_sub_scalar[5]
        getitem_1811: "f32[]" = _foreach_sub_scalar[6]
        getitem_1812: "f32[]" = _foreach_sub_scalar[7]
        getitem_1813: "f32[]" = _foreach_sub_scalar[8]
        getitem_1814: "f32[]" = _foreach_sub_scalar[9]
        getitem_1815: "f32[]" = _foreach_sub_scalar[10]
        getitem_1816: "f32[]" = _foreach_sub_scalar[11]
        getitem_1817: "f32[]" = _foreach_sub_scalar[12]
        getitem_1818: "f32[]" = _foreach_sub_scalar[13]
        getitem_1819: "f32[]" = _foreach_sub_scalar[14]
        getitem_1820: "f32[]" = _foreach_sub_scalar[15]
        getitem_1821: "f32[]" = _foreach_sub_scalar[16]
        getitem_1822: "f32[]" = _foreach_sub_scalar[17]
        getitem_1823: "f32[]" = _foreach_sub_scalar[18]
        getitem_1824: "f32[]" = _foreach_sub_scalar[19]
        getitem_1825: "f32[]" = _foreach_sub_scalar[20]
        getitem_1826: "f32[]" = _foreach_sub_scalar[21]
        getitem_1827: "f32[]" = _foreach_sub_scalar[22]
        getitem_1828: "f32[]" = _foreach_sub_scalar[23]
        getitem_1829: "f32[]" = _foreach_sub_scalar[24]
        getitem_1830: "f32[]" = _foreach_sub_scalar[25]
        getitem_1831: "f32[]" = _foreach_sub_scalar[26]
        getitem_1832: "f32[]" = _foreach_sub_scalar[27]
        getitem_1833: "f32[]" = _foreach_sub_scalar[28]
        getitem_1834: "f32[]" = _foreach_sub_scalar[29]
        getitem_1835: "f32[]" = _foreach_sub_scalar[30]
        getitem_1836: "f32[]" = _foreach_sub_scalar[31]
        getitem_1837: "f32[]" = _foreach_sub_scalar[32]
        getitem_1838: "f32[]" = _foreach_sub_scalar[33]
        getitem_1839: "f32[]" = _foreach_sub_scalar[34]
        getitem_1840: "f32[]" = _foreach_sub_scalar[35]
        getitem_1841: "f32[]" = _foreach_sub_scalar[36]
        getitem_1842: "f32[]" = _foreach_sub_scalar[37]
        getitem_1843: "f32[]" = _foreach_sub_scalar[38]
        getitem_1844: "f32[]" = _foreach_sub_scalar[39]
        getitem_1845: "f32[]" = _foreach_sub_scalar[40]
        getitem_1846: "f32[]" = _foreach_sub_scalar[41]
        getitem_1847: "f32[]" = _foreach_sub_scalar[42]
        getitem_1848: "f32[]" = _foreach_sub_scalar[43]
        getitem_1849: "f32[]" = _foreach_sub_scalar[44]
        getitem_1850: "f32[]" = _foreach_sub_scalar[45]
        getitem_1851: "f32[]" = _foreach_sub_scalar[46]
        getitem_1852: "f32[]" = _foreach_sub_scalar[47]
        getitem_1853: "f32[]" = _foreach_sub_scalar[48]
        getitem_1854: "f32[]" = _foreach_sub_scalar[49]
        getitem_1855: "f32[]" = _foreach_sub_scalar[50]
        getitem_1856: "f32[]" = _foreach_sub_scalar[51]
        getitem_1857: "f32[]" = _foreach_sub_scalar[52]
        getitem_1858: "f32[]" = _foreach_sub_scalar[53]
        getitem_1859: "f32[]" = _foreach_sub_scalar[54]
        getitem_1860: "f32[]" = _foreach_sub_scalar[55]
        getitem_1861: "f32[]" = _foreach_sub_scalar[56]
        getitem_1862: "f32[]" = _foreach_sub_scalar[57]
        getitem_1863: "f32[]" = _foreach_sub_scalar[58]
        getitem_1864: "f32[]" = _foreach_sub_scalar[59]
        getitem_1865: "f32[]" = _foreach_sub_scalar[60]
        getitem_1866: "f32[]" = _foreach_sub_scalar[61]
        getitem_1867: "f32[]" = _foreach_sub_scalar[62]
        getitem_1868: "f32[]" = _foreach_sub_scalar[63]
        getitem_1869: "f32[]" = _foreach_sub_scalar[64]
        getitem_1870: "f32[]" = _foreach_sub_scalar[65]
        getitem_1871: "f32[]" = _foreach_sub_scalar[66]
        getitem_1872: "f32[]" = _foreach_sub_scalar[67]
        getitem_1873: "f32[]" = _foreach_sub_scalar[68]
        getitem_1874: "f32[]" = _foreach_sub_scalar[69]
        getitem_1875: "f32[]" = _foreach_sub_scalar[70]
        getitem_1876: "f32[]" = _foreach_sub_scalar[71]
        getitem_1877: "f32[]" = _foreach_sub_scalar[72]
        getitem_1878: "f32[]" = _foreach_sub_scalar[73]
        getitem_1879: "f32[]" = _foreach_sub_scalar[74]
        getitem_1880: "f32[]" = _foreach_sub_scalar[75]
        getitem_1881: "f32[]" = _foreach_sub_scalar[76]
        getitem_1882: "f32[]" = _foreach_sub_scalar[77]
        getitem_1883: "f32[]" = _foreach_sub_scalar[78]
        getitem_1884: "f32[]" = _foreach_sub_scalar[79]
        getitem_1885: "f32[]" = _foreach_sub_scalar[80]
        getitem_1886: "f32[]" = _foreach_sub_scalar[81]
        getitem_1887: "f32[]" = _foreach_sub_scalar[82]
        getitem_1888: "f32[]" = _foreach_sub_scalar[83]
        getitem_1889: "f32[]" = _foreach_sub_scalar[84]
        getitem_1890: "f32[]" = _foreach_sub_scalar[85]
        getitem_1891: "f32[]" = _foreach_sub_scalar[86]
        getitem_1892: "f32[]" = _foreach_sub_scalar[87]
        getitem_1893: "f32[]" = _foreach_sub_scalar[88]
        getitem_1894: "f32[]" = _foreach_sub_scalar[89]
        getitem_1895: "f32[]" = _foreach_sub_scalar[90]
        getitem_1896: "f32[]" = _foreach_sub_scalar[91]
        getitem_1897: "f32[]" = _foreach_sub_scalar[92]
        getitem_1898: "f32[]" = _foreach_sub_scalar[93]
        getitem_1899: "f32[]" = _foreach_sub_scalar[94]
        getitem_1900: "f32[]" = _foreach_sub_scalar[95]
        getitem_1901: "f32[]" = _foreach_sub_scalar[96]
        getitem_1902: "f32[]" = _foreach_sub_scalar[97]
        getitem_1903: "f32[]" = _foreach_sub_scalar[98]
        getitem_1904: "f32[]" = _foreach_sub_scalar[99]
        getitem_1905: "f32[]" = _foreach_sub_scalar[100]
        getitem_1906: "f32[]" = _foreach_sub_scalar[101]
        getitem_1907: "f32[]" = _foreach_sub_scalar[102]
        getitem_1908: "f32[]" = _foreach_sub_scalar[103]
        getitem_1909: "f32[]" = _foreach_sub_scalar[104]
        getitem_1910: "f32[]" = _foreach_sub_scalar[105]
        getitem_1911: "f32[]" = _foreach_sub_scalar[106]
        getitem_1912: "f32[]" = _foreach_sub_scalar[107]
        getitem_1913: "f32[]" = _foreach_sub_scalar[108]
        getitem_1914: "f32[]" = _foreach_sub_scalar[109]
        getitem_1915: "f32[]" = _foreach_sub_scalar[110]
        getitem_1916: "f32[]" = _foreach_sub_scalar[111]
        getitem_1917: "f32[]" = _foreach_sub_scalar[112]
        getitem_1918: "f32[]" = _foreach_sub_scalar[113]
        getitem_1919: "f32[]" = _foreach_sub_scalar[114]
        getitem_1920: "f32[]" = _foreach_sub_scalar[115]
        getitem_1921: "f32[]" = _foreach_sub_scalar[116]
        getitem_1922: "f32[]" = _foreach_sub_scalar[117]
        getitem_1923: "f32[]" = _foreach_sub_scalar[118]
        getitem_1924: "f32[]" = _foreach_sub_scalar[119]
        getitem_1925: "f32[]" = _foreach_sub_scalar[120]
        getitem_1926: "f32[]" = _foreach_sub_scalar[121]
        getitem_1927: "f32[]" = _foreach_sub_scalar[122]
        getitem_1928: "f32[]" = _foreach_sub_scalar[123]
        getitem_1929: "f32[]" = _foreach_sub_scalar[124]
        getitem_1930: "f32[]" = _foreach_sub_scalar[125]
        getitem_1931: "f32[]" = _foreach_sub_scalar[126]
        getitem_1932: "f32[]" = _foreach_sub_scalar[127]
        getitem_1933: "f32[]" = _foreach_sub_scalar[128]
        getitem_1934: "f32[]" = _foreach_sub_scalar[129]
        getitem_1935: "f32[]" = _foreach_sub_scalar[130]
        getitem_1936: "f32[]" = _foreach_sub_scalar[131]
        getitem_1937: "f32[]" = _foreach_sub_scalar[132]
        getitem_1938: "f32[]" = _foreach_sub_scalar[133]
        getitem_1939: "f32[]" = _foreach_sub_scalar[134]
        getitem_1940: "f32[]" = _foreach_sub_scalar[135]
        getitem_1941: "f32[]" = _foreach_sub_scalar[136]
        getitem_1942: "f32[]" = _foreach_sub_scalar[137]
        getitem_1943: "f32[]" = _foreach_sub_scalar[138]
        getitem_1944: "f32[]" = _foreach_sub_scalar[139]
        getitem_1945: "f32[]" = _foreach_sub_scalar[140]
        getitem_1946: "f32[]" = _foreach_sub_scalar[141]
        getitem_1947: "f32[]" = _foreach_sub_scalar[142]
        getitem_1948: "f32[]" = _foreach_sub_scalar[143]
        getitem_1949: "f32[]" = _foreach_sub_scalar[144]
        getitem_1950: "f32[]" = _foreach_sub_scalar[145]
        getitem_1951: "f32[]" = _foreach_sub_scalar[146]
        getitem_1952: "f32[]" = _foreach_sub_scalar[147]
        getitem_1953: "f32[]" = _foreach_sub_scalar[148]
        getitem_1954: "f32[]" = _foreach_sub_scalar[149]
        getitem_1955: "f32[]" = _foreach_sub_scalar[150]
        getitem_1956: "f32[]" = _foreach_sub_scalar[151]
        getitem_1957: "f32[]" = _foreach_sub_scalar[152]
        getitem_1958: "f32[]" = _foreach_sub_scalar[153]
        getitem_1959: "f32[]" = _foreach_sub_scalar[154]
        getitem_1960: "f32[]" = _foreach_sub_scalar[155]
        getitem_1961: "f32[]" = _foreach_sub_scalar[156]
        getitem_1962: "f32[]" = _foreach_sub_scalar[157]
        getitem_1963: "f32[]" = _foreach_sub_scalar[158]
        getitem_1964: "f32[]" = _foreach_sub_scalar[159]
        getitem_1965: "f32[]" = _foreach_sub_scalar[160]
        getitem_1966: "f32[]" = _foreach_sub_scalar[161]
        getitem_1967: "f32[]" = _foreach_sub_scalar[162]
        getitem_1968: "f32[]" = _foreach_sub_scalar[163]
        getitem_1969: "f32[]" = _foreach_sub_scalar[164]
        getitem_1970: "f32[]" = _foreach_sub_scalar[165]
        getitem_1971: "f32[]" = _foreach_sub_scalar[166]
        getitem_1972: "f32[]" = _foreach_sub_scalar[167]
        getitem_1973: "f32[]" = _foreach_sub_scalar[168]
        getitem_1974: "f32[]" = _foreach_sub_scalar[169]
        getitem_1975: "f32[]" = _foreach_sub_scalar[170]
        getitem_1976: "f32[]" = _foreach_sub_scalar[171]
        getitem_1977: "f32[]" = _foreach_sub_scalar[172]
        getitem_1978: "f32[]" = _foreach_sub_scalar[173]
        getitem_1979: "f32[]" = _foreach_sub_scalar[174]
        getitem_1980: "f32[]" = _foreach_sub_scalar[175]
        getitem_1981: "f32[]" = _foreach_sub_scalar[176]
        getitem_1982: "f32[]" = _foreach_sub_scalar[177]
        getitem_1983: "f32[]" = _foreach_sub_scalar[178]
        getitem_1984: "f32[]" = _foreach_sub_scalar[179]
        getitem_1985: "f32[]" = _foreach_sub_scalar[180]
        getitem_1986: "f32[]" = _foreach_sub_scalar[181]
        getitem_1987: "f32[]" = _foreach_sub_scalar[182]
        getitem_1988: "f32[]" = _foreach_sub_scalar[183]
        getitem_1989: "f32[]" = _foreach_sub_scalar[184]
        getitem_1990: "f32[]" = _foreach_sub_scalar[185]
        getitem_1991: "f32[]" = _foreach_sub_scalar[186]
        getitem_1992: "f32[]" = _foreach_sub_scalar[187]
        getitem_1993: "f32[]" = _foreach_sub_scalar[188]
        getitem_1994: "f32[]" = _foreach_sub_scalar[189]
        getitem_1995: "f32[]" = _foreach_sub_scalar[190]
        getitem_1996: "f32[]" = _foreach_sub_scalar[191]
        getitem_1997: "f32[]" = _foreach_sub_scalar[192]
        getitem_1998: "f32[]" = _foreach_sub_scalar[193]
        getitem_1999: "f32[]" = _foreach_sub_scalar[194]
        getitem_2000: "f32[]" = _foreach_sub_scalar[195]
        getitem_2001: "f32[]" = _foreach_sub_scalar[196]
        getitem_2002: "f32[]" = _foreach_sub_scalar[197]
        getitem_2003: "f32[]" = _foreach_sub_scalar[198]
        getitem_2004: "f32[]" = _foreach_sub_scalar[199]
        getitem_2005: "f32[]" = _foreach_sub_scalar[200]
        getitem_2006: "f32[]" = _foreach_sub_scalar[201]
        getitem_2007: "f32[]" = _foreach_sub_scalar[202]
        getitem_2008: "f32[]" = _foreach_sub_scalar[203]
        getitem_2009: "f32[]" = _foreach_sub_scalar[204]
        getitem_2010: "f32[]" = _foreach_sub_scalar[205]
        getitem_2011: "f32[]" = _foreach_sub_scalar[206]
        getitem_2012: "f32[]" = _foreach_sub_scalar[207]
        getitem_2013: "f32[]" = _foreach_sub_scalar[208]
        getitem_2014: "f32[]" = _foreach_sub_scalar[209]
        getitem_2015: "f32[]" = _foreach_sub_scalar[210]
        getitem_2016: "f32[]" = _foreach_sub_scalar[211]
        getitem_2017: "f32[]" = _foreach_sub_scalar[212]
        getitem_2018: "f32[]" = _foreach_sub_scalar[213]
        getitem_2019: "f32[]" = _foreach_sub_scalar[214]
        getitem_2020: "f32[]" = _foreach_sub_scalar[215]
        getitem_2021: "f32[]" = _foreach_sub_scalar[216]
        getitem_2022: "f32[]" = _foreach_sub_scalar[217]
        getitem_2023: "f32[]" = _foreach_sub_scalar[218]
        getitem_2024: "f32[]" = _foreach_sub_scalar[219]
        getitem_2025: "f32[]" = _foreach_sub_scalar[220]
        getitem_2026: "f32[]" = _foreach_sub_scalar[221]
        getitem_2027: "f32[]" = _foreach_sub_scalar[222]
        getitem_2028: "f32[]" = _foreach_sub_scalar[223]
        getitem_2029: "f32[]" = _foreach_sub_scalar[224]
        getitem_2030: "f32[]" = _foreach_sub_scalar[225]
        getitem_2031: "f32[]" = _foreach_sub_scalar[226]
        getitem_2032: "f32[]" = _foreach_sub_scalar[227]
        getitem_2033: "f32[]" = _foreach_sub_scalar[228]
        getitem_2034: "f32[]" = _foreach_sub_scalar[229]
        getitem_2035: "f32[]" = _foreach_sub_scalar[230]
        getitem_2036: "f32[]" = _foreach_sub_scalar[231]
        getitem_2037: "f32[]" = _foreach_sub_scalar[232]
        getitem_2038: "f32[]" = _foreach_sub_scalar[233]
        getitem_2039: "f32[]" = _foreach_sub_scalar[234]
        getitem_2040: "f32[]" = _foreach_sub_scalar[235]
        getitem_2041: "f32[]" = _foreach_sub_scalar[236]
        getitem_2042: "f32[]" = _foreach_sub_scalar[237]
        getitem_2043: "f32[]" = _foreach_sub_scalar[238]
        getitem_2044: "f32[]" = _foreach_sub_scalar[239]
        getitem_2045: "f32[]" = _foreach_sub_scalar[240]
        getitem_2046: "f32[]" = _foreach_sub_scalar[241]
        getitem_2047: "f32[]" = _foreach_sub_scalar[242]
        getitem_2048: "f32[]" = _foreach_sub_scalar[243]
        getitem_2049: "f32[]" = _foreach_sub_scalar[244]
        getitem_2050: "f32[]" = _foreach_sub_scalar[245]
        getitem_2051: "f32[]" = _foreach_sub_scalar[246]
        getitem_2052: "f32[]" = _foreach_sub_scalar[247]
        getitem_2053: "f32[]" = _foreach_sub_scalar[248]
        getitem_2054: "f32[]" = _foreach_sub_scalar[249]
        getitem_2055: "f32[]" = _foreach_sub_scalar[250]
        getitem_2056: "f32[]" = _foreach_sub_scalar[251]
        getitem_2057: "f32[]" = _foreach_sub_scalar[252]
        getitem_2058: "f32[]" = _foreach_sub_scalar[253]
        getitem_2059: "f32[]" = _foreach_sub_scalar[254]
        getitem_2060: "f32[]" = _foreach_sub_scalar[255]
        getitem_2061: "f32[]" = _foreach_sub_scalar[256]
        getitem_2062: "f32[]" = _foreach_sub_scalar[257]
        getitem_2063: "f32[]" = _foreach_sub_scalar[258]
        getitem_2064: "f32[]" = _foreach_sub_scalar[259]
        getitem_2065: "f32[]" = _foreach_sub_scalar[260]
        getitem_2066: "f32[]" = _foreach_sub_scalar[261]
        getitem_2067: "f32[]" = _foreach_sub_scalar[262]
        getitem_2068: "f32[]" = _foreach_sub_scalar[263]
        getitem_2069: "f32[]" = _foreach_sub_scalar[264]
        getitem_2070: "f32[]" = _foreach_sub_scalar[265]
        getitem_2071: "f32[]" = _foreach_sub_scalar[266]
        getitem_2072: "f32[]" = _foreach_sub_scalar[267]
        getitem_2073: "f32[]" = _foreach_sub_scalar[268]
        getitem_2074: "f32[]" = _foreach_sub_scalar[269]
        getitem_2075: "f32[]" = _foreach_sub_scalar[270]
        getitem_2076: "f32[]" = _foreach_sub_scalar[271]
        getitem_2077: "f32[]" = _foreach_sub_scalar[272]
        getitem_2078: "f32[]" = _foreach_sub_scalar[273]
        getitem_2079: "f32[]" = _foreach_sub_scalar[274]
        getitem_2080: "f32[]" = _foreach_sub_scalar[275]
        getitem_2081: "f32[]" = _foreach_sub_scalar[276]
        getitem_2082: "f32[]" = _foreach_sub_scalar[277]
        getitem_2083: "f32[]" = _foreach_sub_scalar[278]
        getitem_2084: "f32[]" = _foreach_sub_scalar[279]
        getitem_2085: "f32[]" = _foreach_sub_scalar[280]
        getitem_2086: "f32[]" = _foreach_sub_scalar[281]
        getitem_2087: "f32[]" = _foreach_sub_scalar[282]
        getitem_2088: "f32[]" = _foreach_sub_scalar[283]
        getitem_2089: "f32[]" = _foreach_sub_scalar[284]
        getitem_2090: "f32[]" = _foreach_sub_scalar[285]
        getitem_2091: "f32[]" = _foreach_sub_scalar[286]
        getitem_2092: "f32[]" = _foreach_sub_scalar[287]
        getitem_2093: "f32[]" = _foreach_sub_scalar[288]
        getitem_2094: "f32[]" = _foreach_sub_scalar[289]
        getitem_2095: "f32[]" = _foreach_sub_scalar[290]
        getitem_2096: "f32[]" = _foreach_sub_scalar[291]
        getitem_2097: "f32[]" = _foreach_sub_scalar[292]
        getitem_2098: "f32[]" = _foreach_sub_scalar[293]
        getitem_2099: "f32[]" = _foreach_sub_scalar[294]
        getitem_2100: "f32[]" = _foreach_sub_scalar[295]
        getitem_2101: "f32[]" = _foreach_sub_scalar[296]
        getitem_2102: "f32[]" = _foreach_sub_scalar[297]
        getitem_2103: "f32[]" = _foreach_sub_scalar[298]
        getitem_2104: "f32[]" = _foreach_sub_scalar[299]
        getitem_2105: "f32[]" = _foreach_sub_scalar[300];  _foreach_sub_scalar = None
        _foreach_sqrt_default = torch.ops.aten._foreach_sqrt.default([getitem_2709, getitem_2710, getitem_2711, getitem_2712, getitem_2713, getitem_2714, getitem_2715, getitem_2716, getitem_2717, getitem_2718, getitem_2719, getitem_2720, getitem_2721, getitem_2722, getitem_2723, getitem_2724, getitem_2725, getitem_2726, getitem_2727, getitem_2728, getitem_2729, getitem_2730, getitem_2731, getitem_2732, getitem_2733, getitem_2734, getitem_2735, getitem_2736, getitem_2737, getitem_2738, getitem_2739, getitem_2740, getitem_2741, getitem_2742, getitem_2743, getitem_2744, getitem_2745, getitem_2746, getitem_2747, getitem_2748, getitem_2749, getitem_2750, getitem_2751, getitem_2752, getitem_2753, getitem_2754, getitem_2755, getitem_2756, getitem_2757, getitem_2758, getitem_2759, getitem_2760, getitem_2761, getitem_2762, getitem_2763, getitem_2764, getitem_2765, getitem_2766, getitem_2767, getitem_2768, getitem_2769, getitem_2770, getitem_2771, getitem_2772, getitem_2773, getitem_2774, getitem_2775, getitem_2776, getitem_2777, getitem_2778, getitem_2779, getitem_2780, getitem_2781, getitem_2782, getitem_2783, getitem_2784, getitem_2785, getitem_2786, getitem_2787, getitem_2788, getitem_2789, getitem_2790, getitem_2791, getitem_2792, getitem_2793, getitem_2794, getitem_2795, getitem_2796, getitem_2797, getitem_2798, getitem_2799, getitem_2800, getitem_2801, getitem_2802, getitem_2803, getitem_2804, getitem_2805, getitem_2806, getitem_2807, getitem_2808, getitem_2809, getitem_2810, getitem_2811, getitem_2812, getitem_2813, getitem_2814, getitem_2815, getitem_2816, getitem_2817, getitem_2818, getitem_2819, getitem_2820, getitem_2821, getitem_2822, getitem_2823, getitem_2824, getitem_2825, getitem_2826, getitem_2827, getitem_2828, getitem_2829, getitem_2830, getitem_2831, getitem_2832, getitem_2833, getitem_2834, getitem_2835, getitem_2836, getitem_2837, getitem_2838, getitem_2839, getitem_2840, getitem_2841, getitem_2842, getitem_2843, getitem_2844, getitem_2845, getitem_2846, getitem_2847, getitem_2848, getitem_2849, getitem_2850, getitem_2851, getitem_2852, getitem_2853, getitem_2854, getitem_2855, getitem_2856, getitem_2857, getitem_2858, getitem_2859, getitem_2860, getitem_2861, getitem_2862, getitem_2863, getitem_2864, getitem_2865, getitem_2866, getitem_2867, getitem_2868, getitem_2869, getitem_2870, getitem_2871, getitem_2872, getitem_2873, getitem_2874, getitem_2875, getitem_2876, getitem_2877, getitem_2878, getitem_2879, getitem_2880, getitem_2881, getitem_2882, getitem_2883, getitem_2884, getitem_2885, getitem_2886, getitem_2887, getitem_2888, getitem_2889, getitem_2890, getitem_2891, getitem_2892, getitem_2893, getitem_2894, getitem_2895, getitem_2896, getitem_2897, getitem_2898, getitem_2899, getitem_2900, getitem_2901, getitem_2902, getitem_2903, getitem_2904, getitem_2905, getitem_2906, getitem_2907, getitem_2908, getitem_2909, getitem_2910, getitem_2911, getitem_2912, getitem_2913, getitem_2914, getitem_2915, getitem_2916, getitem_2917, getitem_2918, getitem_2919, getitem_2920, getitem_2921, getitem_2922, getitem_2923, getitem_2924, getitem_2925, getitem_2926, getitem_2927, getitem_2928, getitem_2929, getitem_2930, getitem_2931, getitem_2932, getitem_2933, getitem_2934, getitem_2935, getitem_2936, getitem_2937, getitem_2938, getitem_2939, getitem_2940, getitem_2941, getitem_2942, getitem_2943, getitem_2944, getitem_2945, getitem_2946, getitem_2947, getitem_2948, getitem_2949, getitem_2950, getitem_2951, getitem_2952, getitem_2953, getitem_2954, getitem_2955, getitem_2956, getitem_2957, getitem_2958, getitem_2959, getitem_2960, getitem_2961, getitem_2962, getitem_2963, getitem_2964, getitem_2965, getitem_2966, getitem_2967, getitem_2968, getitem_2969, getitem_2970, getitem_2971, getitem_2972, getitem_2973, getitem_2974, getitem_2975, getitem_2976, getitem_2977, getitem_2978, getitem_2979, getitem_2980, getitem_2981, getitem_2982, getitem_2983, getitem_2984, getitem_2985, getitem_2986, getitem_2987, getitem_2988, getitem_2989, getitem_2990, getitem_2991, getitem_2992, getitem_2993, getitem_2994, getitem_2995, getitem_2996, getitem_2997, getitem_2998, getitem_2999, getitem_3000, getitem_3001, getitem_3002, getitem_3003, getitem_3004, getitem_3005, getitem_3006, getitem_3007, getitem_3008, getitem_3009]);  getitem_2709 = getitem_2710 = getitem_2711 = getitem_2712 = getitem_2713 = getitem_2714 = getitem_2715 = getitem_2716 = getitem_2717 = getitem_2718 = getitem_2719 = getitem_2720 = getitem_2721 = getitem_2722 = getitem_2723 = getitem_2724 = getitem_2725 = getitem_2726 = getitem_2727 = getitem_2728 = getitem_2729 = getitem_2730 = getitem_2731 = getitem_2732 = getitem_2733 = getitem_2734 = getitem_2735 = getitem_2736 = getitem_2737 = getitem_2738 = getitem_2739 = getitem_2740 = getitem_2741 = getitem_2742 = getitem_2743 = getitem_2744 = getitem_2745 = getitem_2746 = getitem_2747 = getitem_2748 = getitem_2749 = getitem_2750 = getitem_2751 = getitem_2752 = getitem_2753 = getitem_2754 = getitem_2755 = getitem_2756 = getitem_2757 = getitem_2758 = getitem_2759 = getitem_2760 = getitem_2761 = getitem_2762 = getitem_2763 = getitem_2764 = getitem_2765 = getitem_2766 = getitem_2767 = getitem_2768 = getitem_2769 = getitem_2770 = getitem_2771 = getitem_2772 = getitem_2773 = getitem_2774 = getitem_2775 = getitem_2776 = getitem_2777 = getitem_2778 = getitem_2779 = getitem_2780 = getitem_2781 = getitem_2782 = getitem_2783 = getitem_2784 = getitem_2785 = getitem_2786 = getitem_2787 = getitem_2788 = getitem_2789 = getitem_2790 = getitem_2791 = getitem_2792 = getitem_2793 = getitem_2794 = getitem_2795 = getitem_2796 = getitem_2797 = getitem_2798 = getitem_2799 = getitem_2800 = getitem_2801 = getitem_2802 = getitem_2803 = getitem_2804 = getitem_2805 = getitem_2806 = getitem_2807 = getitem_2808 = getitem_2809 = getitem_2810 = getitem_2811 = getitem_2812 = getitem_2813 = getitem_2814 = getitem_2815 = getitem_2816 = getitem_2817 = getitem_2818 = getitem_2819 = getitem_2820 = getitem_2821 = getitem_2822 = getitem_2823 = getitem_2824 = getitem_2825 = getitem_2826 = getitem_2827 = getitem_2828 = getitem_2829 = getitem_2830 = getitem_2831 = getitem_2832 = getitem_2833 = getitem_2834 = getitem_2835 = getitem_2836 = getitem_2837 = getitem_2838 = getitem_2839 = getitem_2840 = getitem_2841 = getitem_2842 = getitem_2843 = getitem_2844 = getitem_2845 = getitem_2846 = getitem_2847 = getitem_2848 = getitem_2849 = getitem_2850 = getitem_2851 = getitem_2852 = getitem_2853 = getitem_2854 = getitem_2855 = getitem_2856 = getitem_2857 = getitem_2858 = getitem_2859 = getitem_2860 = getitem_2861 = getitem_2862 = getitem_2863 = getitem_2864 = getitem_2865 = getitem_2866 = getitem_2867 = getitem_2868 = getitem_2869 = getitem_2870 = getitem_2871 = getitem_2872 = getitem_2873 = getitem_2874 = getitem_2875 = getitem_2876 = getitem_2877 = getitem_2878 = getitem_2879 = getitem_2880 = getitem_2881 = getitem_2882 = getitem_2883 = getitem_2884 = getitem_2885 = getitem_2886 = getitem_2887 = getitem_2888 = getitem_2889 = getitem_2890 = getitem_2891 = getitem_2892 = getitem_2893 = getitem_2894 = getitem_2895 = getitem_2896 = getitem_2897 = getitem_2898 = getitem_2899 = getitem_2900 = getitem_2901 = getitem_2902 = getitem_2903 = getitem_2904 = getitem_2905 = getitem_2906 = getitem_2907 = getitem_2908 = getitem_2909 = getitem_2910 = getitem_2911 = getitem_2912 = getitem_2913 = getitem_2914 = getitem_2915 = getitem_2916 = getitem_2917 = getitem_2918 = getitem_2919 = getitem_2920 = getitem_2921 = getitem_2922 = getitem_2923 = getitem_2924 = getitem_2925 = getitem_2926 = getitem_2927 = getitem_2928 = getitem_2929 = getitem_2930 = getitem_2931 = getitem_2932 = getitem_2933 = getitem_2934 = getitem_2935 = getitem_2936 = getitem_2937 = getitem_2938 = getitem_2939 = getitem_2940 = getitem_2941 = getitem_2942 = getitem_2943 = getitem_2944 = getitem_2945 = getitem_2946 = getitem_2947 = getitem_2948 = getitem_2949 = getitem_2950 = getitem_2951 = getitem_2952 = getitem_2953 = getitem_2954 = getitem_2955 = getitem_2956 = getitem_2957 = getitem_2958 = getitem_2959 = getitem_2960 = getitem_2961 = getitem_2962 = getitem_2963 = getitem_2964 = getitem_2965 = getitem_2966 = getitem_2967 = getitem_2968 = getitem_2969 = getitem_2970 = getitem_2971 = getitem_2972 = getitem_2973 = getitem_2974 = getitem_2975 = getitem_2976 = getitem_2977 = getitem_2978 = getitem_2979 = getitem_2980 = getitem_2981 = getitem_2982 = getitem_2983 = getitem_2984 = getitem_2985 = getitem_2986 = getitem_2987 = getitem_2988 = getitem_2989 = getitem_2990 = getitem_2991 = getitem_2992 = getitem_2993 = getitem_2994 = getitem_2995 = getitem_2996 = getitem_2997 = getitem_2998 = getitem_2999 = getitem_3000 = getitem_3001 = getitem_3002 = getitem_3003 = getitem_3004 = getitem_3005 = getitem_3006 = getitem_3007 = getitem_3008 = getitem_3009 = None
        getitem_3010: "f32[]" = _foreach_sqrt_default[0]
        getitem_3011: "f32[]" = _foreach_sqrt_default[1]
        getitem_3012: "f32[]" = _foreach_sqrt_default[2]
        getitem_3013: "f32[]" = _foreach_sqrt_default[3]
        getitem_3014: "f32[]" = _foreach_sqrt_default[4]
        getitem_3015: "f32[]" = _foreach_sqrt_default[5]
        getitem_3016: "f32[]" = _foreach_sqrt_default[6]
        getitem_3017: "f32[]" = _foreach_sqrt_default[7]
        getitem_3018: "f32[]" = _foreach_sqrt_default[8]
        getitem_3019: "f32[]" = _foreach_sqrt_default[9]
        getitem_3020: "f32[]" = _foreach_sqrt_default[10]
        getitem_3021: "f32[]" = _foreach_sqrt_default[11]
        getitem_3022: "f32[]" = _foreach_sqrt_default[12]
        getitem_3023: "f32[]" = _foreach_sqrt_default[13]
        getitem_3024: "f32[]" = _foreach_sqrt_default[14]
        getitem_3025: "f32[]" = _foreach_sqrt_default[15]
        getitem_3026: "f32[]" = _foreach_sqrt_default[16]
        getitem_3027: "f32[]" = _foreach_sqrt_default[17]
        getitem_3028: "f32[]" = _foreach_sqrt_default[18]
        getitem_3029: "f32[]" = _foreach_sqrt_default[19]
        getitem_3030: "f32[]" = _foreach_sqrt_default[20]
        getitem_3031: "f32[]" = _foreach_sqrt_default[21]
        getitem_3032: "f32[]" = _foreach_sqrt_default[22]
        getitem_3033: "f32[]" = _foreach_sqrt_default[23]
        getitem_3034: "f32[]" = _foreach_sqrt_default[24]
        getitem_3035: "f32[]" = _foreach_sqrt_default[25]
        getitem_3036: "f32[]" = _foreach_sqrt_default[26]
        getitem_3037: "f32[]" = _foreach_sqrt_default[27]
        getitem_3038: "f32[]" = _foreach_sqrt_default[28]
        getitem_3039: "f32[]" = _foreach_sqrt_default[29]
        getitem_3040: "f32[]" = _foreach_sqrt_default[30]
        getitem_3041: "f32[]" = _foreach_sqrt_default[31]
        getitem_3042: "f32[]" = _foreach_sqrt_default[32]
        getitem_3043: "f32[]" = _foreach_sqrt_default[33]
        getitem_3044: "f32[]" = _foreach_sqrt_default[34]
        getitem_3045: "f32[]" = _foreach_sqrt_default[35]
        getitem_3046: "f32[]" = _foreach_sqrt_default[36]
        getitem_3047: "f32[]" = _foreach_sqrt_default[37]
        getitem_3048: "f32[]" = _foreach_sqrt_default[38]
        getitem_3049: "f32[]" = _foreach_sqrt_default[39]
        getitem_3050: "f32[]" = _foreach_sqrt_default[40]
        getitem_3051: "f32[]" = _foreach_sqrt_default[41]
        getitem_3052: "f32[]" = _foreach_sqrt_default[42]
        getitem_3053: "f32[]" = _foreach_sqrt_default[43]
        getitem_3054: "f32[]" = _foreach_sqrt_default[44]
        getitem_3055: "f32[]" = _foreach_sqrt_default[45]
        getitem_3056: "f32[]" = _foreach_sqrt_default[46]
        getitem_3057: "f32[]" = _foreach_sqrt_default[47]
        getitem_3058: "f32[]" = _foreach_sqrt_default[48]
        getitem_3059: "f32[]" = _foreach_sqrt_default[49]
        getitem_3060: "f32[]" = _foreach_sqrt_default[50]
        getitem_3061: "f32[]" = _foreach_sqrt_default[51]
        getitem_3062: "f32[]" = _foreach_sqrt_default[52]
        getitem_3063: "f32[]" = _foreach_sqrt_default[53]
        getitem_3064: "f32[]" = _foreach_sqrt_default[54]
        getitem_3065: "f32[]" = _foreach_sqrt_default[55]
        getitem_3066: "f32[]" = _foreach_sqrt_default[56]
        getitem_3067: "f32[]" = _foreach_sqrt_default[57]
        getitem_3068: "f32[]" = _foreach_sqrt_default[58]
        getitem_3069: "f32[]" = _foreach_sqrt_default[59]
        getitem_3070: "f32[]" = _foreach_sqrt_default[60]
        getitem_3071: "f32[]" = _foreach_sqrt_default[61]
        getitem_3072: "f32[]" = _foreach_sqrt_default[62]
        getitem_3073: "f32[]" = _foreach_sqrt_default[63]
        getitem_3074: "f32[]" = _foreach_sqrt_default[64]
        getitem_3075: "f32[]" = _foreach_sqrt_default[65]
        getitem_3076: "f32[]" = _foreach_sqrt_default[66]
        getitem_3077: "f32[]" = _foreach_sqrt_default[67]
        getitem_3078: "f32[]" = _foreach_sqrt_default[68]
        getitem_3079: "f32[]" = _foreach_sqrt_default[69]
        getitem_3080: "f32[]" = _foreach_sqrt_default[70]
        getitem_3081: "f32[]" = _foreach_sqrt_default[71]
        getitem_3082: "f32[]" = _foreach_sqrt_default[72]
        getitem_3083: "f32[]" = _foreach_sqrt_default[73]
        getitem_3084: "f32[]" = _foreach_sqrt_default[74]
        getitem_3085: "f32[]" = _foreach_sqrt_default[75]
        getitem_3086: "f32[]" = _foreach_sqrt_default[76]
        getitem_3087: "f32[]" = _foreach_sqrt_default[77]
        getitem_3088: "f32[]" = _foreach_sqrt_default[78]
        getitem_3089: "f32[]" = _foreach_sqrt_default[79]
        getitem_3090: "f32[]" = _foreach_sqrt_default[80]
        getitem_3091: "f32[]" = _foreach_sqrt_default[81]
        getitem_3092: "f32[]" = _foreach_sqrt_default[82]
        getitem_3093: "f32[]" = _foreach_sqrt_default[83]
        getitem_3094: "f32[]" = _foreach_sqrt_default[84]
        getitem_3095: "f32[]" = _foreach_sqrt_default[85]
        getitem_3096: "f32[]" = _foreach_sqrt_default[86]
        getitem_3097: "f32[]" = _foreach_sqrt_default[87]
        getitem_3098: "f32[]" = _foreach_sqrt_default[88]
        getitem_3099: "f32[]" = _foreach_sqrt_default[89]
        getitem_3100: "f32[]" = _foreach_sqrt_default[90]
        getitem_3101: "f32[]" = _foreach_sqrt_default[91]
        getitem_3102: "f32[]" = _foreach_sqrt_default[92]
        getitem_3103: "f32[]" = _foreach_sqrt_default[93]
        getitem_3104: "f32[]" = _foreach_sqrt_default[94]
        getitem_3105: "f32[]" = _foreach_sqrt_default[95]
        getitem_3106: "f32[]" = _foreach_sqrt_default[96]
        getitem_3107: "f32[]" = _foreach_sqrt_default[97]
        getitem_3108: "f32[]" = _foreach_sqrt_default[98]
        getitem_3109: "f32[]" = _foreach_sqrt_default[99]
        getitem_3110: "f32[]" = _foreach_sqrt_default[100]
        getitem_3111: "f32[]" = _foreach_sqrt_default[101]
        getitem_3112: "f32[]" = _foreach_sqrt_default[102]
        getitem_3113: "f32[]" = _foreach_sqrt_default[103]
        getitem_3114: "f32[]" = _foreach_sqrt_default[104]
        getitem_3115: "f32[]" = _foreach_sqrt_default[105]
        getitem_3116: "f32[]" = _foreach_sqrt_default[106]
        getitem_3117: "f32[]" = _foreach_sqrt_default[107]
        getitem_3118: "f32[]" = _foreach_sqrt_default[108]
        getitem_3119: "f32[]" = _foreach_sqrt_default[109]
        getitem_3120: "f32[]" = _foreach_sqrt_default[110]
        getitem_3121: "f32[]" = _foreach_sqrt_default[111]
        getitem_3122: "f32[]" = _foreach_sqrt_default[112]
        getitem_3123: "f32[]" = _foreach_sqrt_default[113]
        getitem_3124: "f32[]" = _foreach_sqrt_default[114]
        getitem_3125: "f32[]" = _foreach_sqrt_default[115]
        getitem_3126: "f32[]" = _foreach_sqrt_default[116]
        getitem_3127: "f32[]" = _foreach_sqrt_default[117]
        getitem_3128: "f32[]" = _foreach_sqrt_default[118]
        getitem_3129: "f32[]" = _foreach_sqrt_default[119]
        getitem_3130: "f32[]" = _foreach_sqrt_default[120]
        getitem_3131: "f32[]" = _foreach_sqrt_default[121]
        getitem_3132: "f32[]" = _foreach_sqrt_default[122]
        getitem_3133: "f32[]" = _foreach_sqrt_default[123]
        getitem_3134: "f32[]" = _foreach_sqrt_default[124]
        getitem_3135: "f32[]" = _foreach_sqrt_default[125]
        getitem_3136: "f32[]" = _foreach_sqrt_default[126]
        getitem_3137: "f32[]" = _foreach_sqrt_default[127]
        getitem_3138: "f32[]" = _foreach_sqrt_default[128]
        getitem_3139: "f32[]" = _foreach_sqrt_default[129]
        getitem_3140: "f32[]" = _foreach_sqrt_default[130]
        getitem_3141: "f32[]" = _foreach_sqrt_default[131]
        getitem_3142: "f32[]" = _foreach_sqrt_default[132]
        getitem_3143: "f32[]" = _foreach_sqrt_default[133]
        getitem_3144: "f32[]" = _foreach_sqrt_default[134]
        getitem_3145: "f32[]" = _foreach_sqrt_default[135]
        getitem_3146: "f32[]" = _foreach_sqrt_default[136]
        getitem_3147: "f32[]" = _foreach_sqrt_default[137]
        getitem_3148: "f32[]" = _foreach_sqrt_default[138]
        getitem_3149: "f32[]" = _foreach_sqrt_default[139]
        getitem_3150: "f32[]" = _foreach_sqrt_default[140]
        getitem_3151: "f32[]" = _foreach_sqrt_default[141]
        getitem_3152: "f32[]" = _foreach_sqrt_default[142]
        getitem_3153: "f32[]" = _foreach_sqrt_default[143]
        getitem_3154: "f32[]" = _foreach_sqrt_default[144]
        getitem_3155: "f32[]" = _foreach_sqrt_default[145]
        getitem_3156: "f32[]" = _foreach_sqrt_default[146]
        getitem_3157: "f32[]" = _foreach_sqrt_default[147]
        getitem_3158: "f32[]" = _foreach_sqrt_default[148]
        getitem_3159: "f32[]" = _foreach_sqrt_default[149]
        getitem_3160: "f32[]" = _foreach_sqrt_default[150]
        getitem_3161: "f32[]" = _foreach_sqrt_default[151]
        getitem_3162: "f32[]" = _foreach_sqrt_default[152]
        getitem_3163: "f32[]" = _foreach_sqrt_default[153]
        getitem_3164: "f32[]" = _foreach_sqrt_default[154]
        getitem_3165: "f32[]" = _foreach_sqrt_default[155]
        getitem_3166: "f32[]" = _foreach_sqrt_default[156]
        getitem_3167: "f32[]" = _foreach_sqrt_default[157]
        getitem_3168: "f32[]" = _foreach_sqrt_default[158]
        getitem_3169: "f32[]" = _foreach_sqrt_default[159]
        getitem_3170: "f32[]" = _foreach_sqrt_default[160]
        getitem_3171: "f32[]" = _foreach_sqrt_default[161]
        getitem_3172: "f32[]" = _foreach_sqrt_default[162]
        getitem_3173: "f32[]" = _foreach_sqrt_default[163]
        getitem_3174: "f32[]" = _foreach_sqrt_default[164]
        getitem_3175: "f32[]" = _foreach_sqrt_default[165]
        getitem_3176: "f32[]" = _foreach_sqrt_default[166]
        getitem_3177: "f32[]" = _foreach_sqrt_default[167]
        getitem_3178: "f32[]" = _foreach_sqrt_default[168]
        getitem_3179: "f32[]" = _foreach_sqrt_default[169]
        getitem_3180: "f32[]" = _foreach_sqrt_default[170]
        getitem_3181: "f32[]" = _foreach_sqrt_default[171]
        getitem_3182: "f32[]" = _foreach_sqrt_default[172]
        getitem_3183: "f32[]" = _foreach_sqrt_default[173]
        getitem_3184: "f32[]" = _foreach_sqrt_default[174]
        getitem_3185: "f32[]" = _foreach_sqrt_default[175]
        getitem_3186: "f32[]" = _foreach_sqrt_default[176]
        getitem_3187: "f32[]" = _foreach_sqrt_default[177]
        getitem_3188: "f32[]" = _foreach_sqrt_default[178]
        getitem_3189: "f32[]" = _foreach_sqrt_default[179]
        getitem_3190: "f32[]" = _foreach_sqrt_default[180]
        getitem_3191: "f32[]" = _foreach_sqrt_default[181]
        getitem_3192: "f32[]" = _foreach_sqrt_default[182]
        getitem_3193: "f32[]" = _foreach_sqrt_default[183]
        getitem_3194: "f32[]" = _foreach_sqrt_default[184]
        getitem_3195: "f32[]" = _foreach_sqrt_default[185]
        getitem_3196: "f32[]" = _foreach_sqrt_default[186]
        getitem_3197: "f32[]" = _foreach_sqrt_default[187]
        getitem_3198: "f32[]" = _foreach_sqrt_default[188]
        getitem_3199: "f32[]" = _foreach_sqrt_default[189]
        getitem_3200: "f32[]" = _foreach_sqrt_default[190]
        getitem_3201: "f32[]" = _foreach_sqrt_default[191]
        getitem_3202: "f32[]" = _foreach_sqrt_default[192]
        getitem_3203: "f32[]" = _foreach_sqrt_default[193]
        getitem_3204: "f32[]" = _foreach_sqrt_default[194]
        getitem_3205: "f32[]" = _foreach_sqrt_default[195]
        getitem_3206: "f32[]" = _foreach_sqrt_default[196]
        getitem_3207: "f32[]" = _foreach_sqrt_default[197]
        getitem_3208: "f32[]" = _foreach_sqrt_default[198]
        getitem_3209: "f32[]" = _foreach_sqrt_default[199]
        getitem_3210: "f32[]" = _foreach_sqrt_default[200]
        getitem_3211: "f32[]" = _foreach_sqrt_default[201]
        getitem_3212: "f32[]" = _foreach_sqrt_default[202]
        getitem_3213: "f32[]" = _foreach_sqrt_default[203]
        getitem_3214: "f32[]" = _foreach_sqrt_default[204]
        getitem_3215: "f32[]" = _foreach_sqrt_default[205]
        getitem_3216: "f32[]" = _foreach_sqrt_default[206]
        getitem_3217: "f32[]" = _foreach_sqrt_default[207]
        getitem_3218: "f32[]" = _foreach_sqrt_default[208]
        getitem_3219: "f32[]" = _foreach_sqrt_default[209]
        getitem_3220: "f32[]" = _foreach_sqrt_default[210]
        getitem_3221: "f32[]" = _foreach_sqrt_default[211]
        getitem_3222: "f32[]" = _foreach_sqrt_default[212]
        getitem_3223: "f32[]" = _foreach_sqrt_default[213]
        getitem_3224: "f32[]" = _foreach_sqrt_default[214]
        getitem_3225: "f32[]" = _foreach_sqrt_default[215]
        getitem_3226: "f32[]" = _foreach_sqrt_default[216]
        getitem_3227: "f32[]" = _foreach_sqrt_default[217]
        getitem_3228: "f32[]" = _foreach_sqrt_default[218]
        getitem_3229: "f32[]" = _foreach_sqrt_default[219]
        getitem_3230: "f32[]" = _foreach_sqrt_default[220]
        getitem_3231: "f32[]" = _foreach_sqrt_default[221]
        getitem_3232: "f32[]" = _foreach_sqrt_default[222]
        getitem_3233: "f32[]" = _foreach_sqrt_default[223]
        getitem_3234: "f32[]" = _foreach_sqrt_default[224]
        getitem_3235: "f32[]" = _foreach_sqrt_default[225]
        getitem_3236: "f32[]" = _foreach_sqrt_default[226]
        getitem_3237: "f32[]" = _foreach_sqrt_default[227]
        getitem_3238: "f32[]" = _foreach_sqrt_default[228]
        getitem_3239: "f32[]" = _foreach_sqrt_default[229]
        getitem_3240: "f32[]" = _foreach_sqrt_default[230]
        getitem_3241: "f32[]" = _foreach_sqrt_default[231]
        getitem_3242: "f32[]" = _foreach_sqrt_default[232]
        getitem_3243: "f32[]" = _foreach_sqrt_default[233]
        getitem_3244: "f32[]" = _foreach_sqrt_default[234]
        getitem_3245: "f32[]" = _foreach_sqrt_default[235]
        getitem_3246: "f32[]" = _foreach_sqrt_default[236]
        getitem_3247: "f32[]" = _foreach_sqrt_default[237]
        getitem_3248: "f32[]" = _foreach_sqrt_default[238]
        getitem_3249: "f32[]" = _foreach_sqrt_default[239]
        getitem_3250: "f32[]" = _foreach_sqrt_default[240]
        getitem_3251: "f32[]" = _foreach_sqrt_default[241]
        getitem_3252: "f32[]" = _foreach_sqrt_default[242]
        getitem_3253: "f32[]" = _foreach_sqrt_default[243]
        getitem_3254: "f32[]" = _foreach_sqrt_default[244]
        getitem_3255: "f32[]" = _foreach_sqrt_default[245]
        getitem_3256: "f32[]" = _foreach_sqrt_default[246]
        getitem_3257: "f32[]" = _foreach_sqrt_default[247]
        getitem_3258: "f32[]" = _foreach_sqrt_default[248]
        getitem_3259: "f32[]" = _foreach_sqrt_default[249]
        getitem_3260: "f32[]" = _foreach_sqrt_default[250]
        getitem_3261: "f32[]" = _foreach_sqrt_default[251]
        getitem_3262: "f32[]" = _foreach_sqrt_default[252]
        getitem_3263: "f32[]" = _foreach_sqrt_default[253]
        getitem_3264: "f32[]" = _foreach_sqrt_default[254]
        getitem_3265: "f32[]" = _foreach_sqrt_default[255]
        getitem_3266: "f32[]" = _foreach_sqrt_default[256]
        getitem_3267: "f32[]" = _foreach_sqrt_default[257]
        getitem_3268: "f32[]" = _foreach_sqrt_default[258]
        getitem_3269: "f32[]" = _foreach_sqrt_default[259]
        getitem_3270: "f32[]" = _foreach_sqrt_default[260]
        getitem_3271: "f32[]" = _foreach_sqrt_default[261]
        getitem_3272: "f32[]" = _foreach_sqrt_default[262]
        getitem_3273: "f32[]" = _foreach_sqrt_default[263]
        getitem_3274: "f32[]" = _foreach_sqrt_default[264]
        getitem_3275: "f32[]" = _foreach_sqrt_default[265]
        getitem_3276: "f32[]" = _foreach_sqrt_default[266]
        getitem_3277: "f32[]" = _foreach_sqrt_default[267]
        getitem_3278: "f32[]" = _foreach_sqrt_default[268]
        getitem_3279: "f32[]" = _foreach_sqrt_default[269]
        getitem_3280: "f32[]" = _foreach_sqrt_default[270]
        getitem_3281: "f32[]" = _foreach_sqrt_default[271]
        getitem_3282: "f32[]" = _foreach_sqrt_default[272]
        getitem_3283: "f32[]" = _foreach_sqrt_default[273]
        getitem_3284: "f32[]" = _foreach_sqrt_default[274]
        getitem_3285: "f32[]" = _foreach_sqrt_default[275]
        getitem_3286: "f32[]" = _foreach_sqrt_default[276]
        getitem_3287: "f32[]" = _foreach_sqrt_default[277]
        getitem_3288: "f32[]" = _foreach_sqrt_default[278]
        getitem_3289: "f32[]" = _foreach_sqrt_default[279]
        getitem_3290: "f32[]" = _foreach_sqrt_default[280]
        getitem_3291: "f32[]" = _foreach_sqrt_default[281]
        getitem_3292: "f32[]" = _foreach_sqrt_default[282]
        getitem_3293: "f32[]" = _foreach_sqrt_default[283]
        getitem_3294: "f32[]" = _foreach_sqrt_default[284]
        getitem_3295: "f32[]" = _foreach_sqrt_default[285]
        getitem_3296: "f32[]" = _foreach_sqrt_default[286]
        getitem_3297: "f32[]" = _foreach_sqrt_default[287]
        getitem_3298: "f32[]" = _foreach_sqrt_default[288]
        getitem_3299: "f32[]" = _foreach_sqrt_default[289]
        getitem_3300: "f32[]" = _foreach_sqrt_default[290]
        getitem_3301: "f32[]" = _foreach_sqrt_default[291]
        getitem_3302: "f32[]" = _foreach_sqrt_default[292]
        getitem_3303: "f32[]" = _foreach_sqrt_default[293]
        getitem_3304: "f32[]" = _foreach_sqrt_default[294]
        getitem_3305: "f32[]" = _foreach_sqrt_default[295]
        getitem_3306: "f32[]" = _foreach_sqrt_default[296]
        getitem_3307: "f32[]" = _foreach_sqrt_default[297]
        getitem_3308: "f32[]" = _foreach_sqrt_default[298]
        getitem_3309: "f32[]" = _foreach_sqrt_default[299]
        getitem_3310: "f32[]" = _foreach_sqrt_default[300];  _foreach_sqrt_default = None
        _foreach_sqrt_default_1 = torch.ops.aten._foreach_sqrt.default([getitem_1204, getitem_1205, getitem_1206, getitem_1207, getitem_1208, getitem_1209, getitem_1210, getitem_1211, getitem_1212, getitem_1213, getitem_1214, getitem_1215, getitem_1216, getitem_1217, getitem_1218, getitem_1219, getitem_1220, getitem_1221, getitem_1222, getitem_1223, getitem_1224, getitem_1225, getitem_1226, getitem_1227, getitem_1228, getitem_1229, getitem_1230, getitem_1231, getitem_1232, getitem_1233, getitem_1234, getitem_1235, getitem_1236, getitem_1237, getitem_1238, getitem_1239, getitem_1240, getitem_1241, getitem_1242, getitem_1243, getitem_1244, getitem_1245, getitem_1246, getitem_1247, getitem_1248, getitem_1249, getitem_1250, getitem_1251, getitem_1252, getitem_1253, getitem_1254, getitem_1255, getitem_1256, getitem_1257, getitem_1258, getitem_1259, getitem_1260, getitem_1261, getitem_1262, getitem_1263, getitem_1264, getitem_1265, getitem_1266, getitem_1267, getitem_1268, getitem_1269, getitem_1270, getitem_1271, getitem_1272, getitem_1273, getitem_1274, getitem_1275, getitem_1276, getitem_1277, getitem_1278, getitem_1279, getitem_1280, getitem_1281, getitem_1282, getitem_1283, getitem_1284, getitem_1285, getitem_1286, getitem_1287, getitem_1288, getitem_1289, getitem_1290, getitem_1291, getitem_1292, getitem_1293, getitem_1294, getitem_1295, getitem_1296, getitem_1297, getitem_1298, getitem_1299, getitem_1300, getitem_1301, getitem_1302, getitem_1303, getitem_1304, getitem_1305, getitem_1306, getitem_1307, getitem_1308, getitem_1309, getitem_1310, getitem_1311, getitem_1312, getitem_1313, getitem_1314, getitem_1315, getitem_1316, getitem_1317, getitem_1318, getitem_1319, getitem_1320, getitem_1321, getitem_1322, getitem_1323, getitem_1324, getitem_1325, getitem_1326, getitem_1327, getitem_1328, getitem_1329, getitem_1330, getitem_1331, getitem_1332, getitem_1333, getitem_1334, getitem_1335, getitem_1336, getitem_1337, getitem_1338, getitem_1339, getitem_1340, getitem_1341, getitem_1342, getitem_1343, getitem_1344, getitem_1345, getitem_1346, getitem_1347, getitem_1348, getitem_1349, getitem_1350, getitem_1351, getitem_1352, getitem_1353, getitem_1354, getitem_1355, getitem_1356, getitem_1357, getitem_1358, getitem_1359, getitem_1360, getitem_1361, getitem_1362, getitem_1363, getitem_1364, getitem_1365, getitem_1366, getitem_1367, getitem_1368, getitem_1369, getitem_1370, getitem_1371, getitem_1372, getitem_1373, getitem_1374, getitem_1375, getitem_1376, getitem_1377, getitem_1378, getitem_1379, getitem_1380, getitem_1381, getitem_1382, getitem_1383, getitem_1384, getitem_1385, getitem_1386, getitem_1387, getitem_1388, getitem_1389, getitem_1390, getitem_1391, getitem_1392, getitem_1393, getitem_1394, getitem_1395, getitem_1396, getitem_1397, getitem_1398, getitem_1399, getitem_1400, getitem_1401, getitem_1402, getitem_1403, getitem_1404, getitem_1405, getitem_1406, getitem_1407, getitem_1408, getitem_1409, getitem_1410, getitem_1411, getitem_1412, getitem_1413, getitem_1414, getitem_1415, getitem_1416, getitem_1417, getitem_1418, getitem_1419, getitem_1420, getitem_1421, getitem_1422, getitem_1423, getitem_1424, getitem_1425, getitem_1426, getitem_1427, getitem_1428, getitem_1429, getitem_1430, getitem_1431, getitem_1432, getitem_1433, getitem_1434, getitem_1435, getitem_1436, getitem_1437, getitem_1438, getitem_1439, getitem_1440, getitem_1441, getitem_1442, getitem_1443, getitem_1444, getitem_1445, getitem_1446, getitem_1447, getitem_1448, getitem_1449, getitem_1450, getitem_1451, getitem_1452, getitem_1453, getitem_1454, getitem_1455, getitem_1456, getitem_1457, getitem_1458, getitem_1459, getitem_1460, getitem_1461, getitem_1462, getitem_1463, getitem_1464, getitem_1465, getitem_1466, getitem_1467, getitem_1468, getitem_1469, getitem_1470, getitem_1471, getitem_1472, getitem_1473, getitem_1474, getitem_1475, getitem_1476, getitem_1477, getitem_1478, getitem_1479, getitem_1480, getitem_1481, getitem_1482, getitem_1483, getitem_1484, getitem_1485, getitem_1486, getitem_1487, getitem_1488, getitem_1489, getitem_1490, getitem_1491, getitem_1492, getitem_1493, getitem_1494, getitem_1495, getitem_1496, getitem_1497, getitem_1498, getitem_1499, getitem_1500, getitem_1501, getitem_1502, getitem_1503, getitem_1504]);  getitem_1204 = getitem_1205 = getitem_1206 = getitem_1207 = getitem_1208 = getitem_1209 = getitem_1210 = getitem_1211 = getitem_1212 = getitem_1213 = getitem_1214 = getitem_1215 = getitem_1216 = getitem_1217 = getitem_1218 = getitem_1219 = getitem_1220 = getitem_1221 = getitem_1222 = getitem_1223 = getitem_1224 = getitem_1225 = getitem_1226 = getitem_1227 = getitem_1228 = getitem_1229 = getitem_1230 = getitem_1231 = getitem_1232 = getitem_1233 = getitem_1234 = getitem_1235 = getitem_1236 = getitem_1237 = getitem_1238 = getitem_1239 = getitem_1240 = getitem_1241 = getitem_1242 = getitem_1243 = getitem_1244 = getitem_1245 = getitem_1246 = getitem_1247 = getitem_1248 = getitem_1249 = getitem_1250 = getitem_1251 = getitem_1252 = getitem_1253 = getitem_1254 = getitem_1255 = getitem_1256 = getitem_1257 = getitem_1258 = getitem_1259 = getitem_1260 = getitem_1261 = getitem_1262 = getitem_1263 = getitem_1264 = getitem_1265 = getitem_1266 = getitem_1267 = getitem_1268 = getitem_1269 = getitem_1270 = getitem_1271 = getitem_1272 = getitem_1273 = getitem_1274 = getitem_1275 = getitem_1276 = getitem_1277 = getitem_1278 = getitem_1279 = getitem_1280 = getitem_1281 = getitem_1282 = getitem_1283 = getitem_1284 = getitem_1285 = getitem_1286 = getitem_1287 = getitem_1288 = getitem_1289 = getitem_1290 = getitem_1291 = getitem_1292 = getitem_1293 = getitem_1294 = getitem_1295 = getitem_1296 = getitem_1297 = getitem_1298 = getitem_1299 = getitem_1300 = getitem_1301 = getitem_1302 = getitem_1303 = getitem_1304 = getitem_1305 = getitem_1306 = getitem_1307 = getitem_1308 = getitem_1309 = getitem_1310 = getitem_1311 = getitem_1312 = getitem_1313 = getitem_1314 = getitem_1315 = getitem_1316 = getitem_1317 = getitem_1318 = getitem_1319 = getitem_1320 = getitem_1321 = getitem_1322 = getitem_1323 = getitem_1324 = getitem_1325 = getitem_1326 = getitem_1327 = getitem_1328 = getitem_1329 = getitem_1330 = getitem_1331 = getitem_1332 = getitem_1333 = getitem_1334 = getitem_1335 = getitem_1336 = getitem_1337 = getitem_1338 = getitem_1339 = getitem_1340 = getitem_1341 = getitem_1342 = getitem_1343 = getitem_1344 = getitem_1345 = getitem_1346 = getitem_1347 = getitem_1348 = getitem_1349 = getitem_1350 = getitem_1351 = getitem_1352 = getitem_1353 = getitem_1354 = getitem_1355 = getitem_1356 = getitem_1357 = getitem_1358 = getitem_1359 = getitem_1360 = getitem_1361 = getitem_1362 = getitem_1363 = getitem_1364 = getitem_1365 = getitem_1366 = getitem_1367 = getitem_1368 = getitem_1369 = getitem_1370 = getitem_1371 = getitem_1372 = getitem_1373 = getitem_1374 = getitem_1375 = getitem_1376 = getitem_1377 = getitem_1378 = getitem_1379 = getitem_1380 = getitem_1381 = getitem_1382 = getitem_1383 = getitem_1384 = getitem_1385 = getitem_1386 = getitem_1387 = getitem_1388 = getitem_1389 = getitem_1390 = getitem_1391 = getitem_1392 = getitem_1393 = getitem_1394 = getitem_1395 = getitem_1396 = getitem_1397 = getitem_1398 = getitem_1399 = getitem_1400 = getitem_1401 = getitem_1402 = getitem_1403 = getitem_1404 = getitem_1405 = getitem_1406 = getitem_1407 = getitem_1408 = getitem_1409 = getitem_1410 = getitem_1411 = getitem_1412 = getitem_1413 = getitem_1414 = getitem_1415 = getitem_1416 = getitem_1417 = getitem_1418 = getitem_1419 = getitem_1420 = getitem_1421 = getitem_1422 = getitem_1423 = getitem_1424 = getitem_1425 = getitem_1426 = getitem_1427 = getitem_1428 = getitem_1429 = getitem_1430 = getitem_1431 = getitem_1432 = getitem_1433 = getitem_1434 = getitem_1435 = getitem_1436 = getitem_1437 = getitem_1438 = getitem_1439 = getitem_1440 = getitem_1441 = getitem_1442 = getitem_1443 = getitem_1444 = getitem_1445 = getitem_1446 = getitem_1447 = getitem_1448 = getitem_1449 = getitem_1450 = getitem_1451 = getitem_1452 = getitem_1453 = getitem_1454 = getitem_1455 = getitem_1456 = getitem_1457 = getitem_1458 = getitem_1459 = getitem_1460 = getitem_1461 = getitem_1462 = getitem_1463 = getitem_1464 = getitem_1465 = getitem_1466 = getitem_1467 = getitem_1468 = getitem_1469 = getitem_1470 = getitem_1471 = getitem_1472 = getitem_1473 = getitem_1474 = getitem_1475 = getitem_1476 = getitem_1477 = getitem_1478 = getitem_1479 = getitem_1480 = getitem_1481 = getitem_1482 = getitem_1483 = getitem_1484 = getitem_1485 = getitem_1486 = getitem_1487 = getitem_1488 = getitem_1489 = getitem_1490 = getitem_1491 = getitem_1492 = getitem_1493 = getitem_1494 = getitem_1495 = getitem_1496 = getitem_1497 = getitem_1498 = getitem_1499 = getitem_1500 = getitem_1501 = getitem_1502 = getitem_1503 = getitem_1504 = None
        getitem_2106: "f32[768]" = _foreach_sqrt_default_1[0]
        getitem_2107: "f32[50, 768]" = _foreach_sqrt_default_1[1]
        getitem_2108: "f32[768, 512]" = _foreach_sqrt_default_1[2]
        getitem_2109: "f32[768, 3, 32, 32]" = _foreach_sqrt_default_1[3]
        getitem_2110: "f32[768]" = _foreach_sqrt_default_1[4]
        getitem_2111: "f32[768]" = _foreach_sqrt_default_1[5]
        getitem_2112: "f32[2304, 768]" = _foreach_sqrt_default_1[6]
        getitem_2113: "f32[2304]" = _foreach_sqrt_default_1[7]
        getitem_2114: "f32[768, 768]" = _foreach_sqrt_default_1[8]
        getitem_2115: "f32[768]" = _foreach_sqrt_default_1[9]
        getitem_2116: "f32[3072, 768]" = _foreach_sqrt_default_1[10]
        getitem_2117: "f32[3072]" = _foreach_sqrt_default_1[11]
        getitem_2118: "f32[768, 3072]" = _foreach_sqrt_default_1[12]
        getitem_2119: "f32[768]" = _foreach_sqrt_default_1[13]
        getitem_2120: "f32[768]" = _foreach_sqrt_default_1[14]
        getitem_2121: "f32[768]" = _foreach_sqrt_default_1[15]
        getitem_2122: "f32[768]" = _foreach_sqrt_default_1[16]
        getitem_2123: "f32[768]" = _foreach_sqrt_default_1[17]
        getitem_2124: "f32[2304, 768]" = _foreach_sqrt_default_1[18]
        getitem_2125: "f32[2304]" = _foreach_sqrt_default_1[19]
        getitem_2126: "f32[768, 768]" = _foreach_sqrt_default_1[20]
        getitem_2127: "f32[768]" = _foreach_sqrt_default_1[21]
        getitem_2128: "f32[3072, 768]" = _foreach_sqrt_default_1[22]
        getitem_2129: "f32[3072]" = _foreach_sqrt_default_1[23]
        getitem_2130: "f32[768, 3072]" = _foreach_sqrt_default_1[24]
        getitem_2131: "f32[768]" = _foreach_sqrt_default_1[25]
        getitem_2132: "f32[768]" = _foreach_sqrt_default_1[26]
        getitem_2133: "f32[768]" = _foreach_sqrt_default_1[27]
        getitem_2134: "f32[768]" = _foreach_sqrt_default_1[28]
        getitem_2135: "f32[768]" = _foreach_sqrt_default_1[29]
        getitem_2136: "f32[2304, 768]" = _foreach_sqrt_default_1[30]
        getitem_2137: "f32[2304]" = _foreach_sqrt_default_1[31]
        getitem_2138: "f32[768, 768]" = _foreach_sqrt_default_1[32]
        getitem_2139: "f32[768]" = _foreach_sqrt_default_1[33]
        getitem_2140: "f32[3072, 768]" = _foreach_sqrt_default_1[34]
        getitem_2141: "f32[3072]" = _foreach_sqrt_default_1[35]
        getitem_2142: "f32[768, 3072]" = _foreach_sqrt_default_1[36]
        getitem_2143: "f32[768]" = _foreach_sqrt_default_1[37]
        getitem_2144: "f32[768]" = _foreach_sqrt_default_1[38]
        getitem_2145: "f32[768]" = _foreach_sqrt_default_1[39]
        getitem_2146: "f32[768]" = _foreach_sqrt_default_1[40]
        getitem_2147: "f32[768]" = _foreach_sqrt_default_1[41]
        getitem_2148: "f32[2304, 768]" = _foreach_sqrt_default_1[42]
        getitem_2149: "f32[2304]" = _foreach_sqrt_default_1[43]
        getitem_2150: "f32[768, 768]" = _foreach_sqrt_default_1[44]
        getitem_2151: "f32[768]" = _foreach_sqrt_default_1[45]
        getitem_2152: "f32[3072, 768]" = _foreach_sqrt_default_1[46]
        getitem_2153: "f32[3072]" = _foreach_sqrt_default_1[47]
        getitem_2154: "f32[768, 3072]" = _foreach_sqrt_default_1[48]
        getitem_2155: "f32[768]" = _foreach_sqrt_default_1[49]
        getitem_2156: "f32[768]" = _foreach_sqrt_default_1[50]
        getitem_2157: "f32[768]" = _foreach_sqrt_default_1[51]
        getitem_2158: "f32[768]" = _foreach_sqrt_default_1[52]
        getitem_2159: "f32[768]" = _foreach_sqrt_default_1[53]
        getitem_2160: "f32[2304, 768]" = _foreach_sqrt_default_1[54]
        getitem_2161: "f32[2304]" = _foreach_sqrt_default_1[55]
        getitem_2162: "f32[768, 768]" = _foreach_sqrt_default_1[56]
        getitem_2163: "f32[768]" = _foreach_sqrt_default_1[57]
        getitem_2164: "f32[3072, 768]" = _foreach_sqrt_default_1[58]
        getitem_2165: "f32[3072]" = _foreach_sqrt_default_1[59]
        getitem_2166: "f32[768, 3072]" = _foreach_sqrt_default_1[60]
        getitem_2167: "f32[768]" = _foreach_sqrt_default_1[61]
        getitem_2168: "f32[768]" = _foreach_sqrt_default_1[62]
        getitem_2169: "f32[768]" = _foreach_sqrt_default_1[63]
        getitem_2170: "f32[768]" = _foreach_sqrt_default_1[64]
        getitem_2171: "f32[768]" = _foreach_sqrt_default_1[65]
        getitem_2172: "f32[2304, 768]" = _foreach_sqrt_default_1[66]
        getitem_2173: "f32[2304]" = _foreach_sqrt_default_1[67]
        getitem_2174: "f32[768, 768]" = _foreach_sqrt_default_1[68]
        getitem_2175: "f32[768]" = _foreach_sqrt_default_1[69]
        getitem_2176: "f32[3072, 768]" = _foreach_sqrt_default_1[70]
        getitem_2177: "f32[3072]" = _foreach_sqrt_default_1[71]
        getitem_2178: "f32[768, 3072]" = _foreach_sqrt_default_1[72]
        getitem_2179: "f32[768]" = _foreach_sqrt_default_1[73]
        getitem_2180: "f32[768]" = _foreach_sqrt_default_1[74]
        getitem_2181: "f32[768]" = _foreach_sqrt_default_1[75]
        getitem_2182: "f32[768]" = _foreach_sqrt_default_1[76]
        getitem_2183: "f32[768]" = _foreach_sqrt_default_1[77]
        getitem_2184: "f32[2304, 768]" = _foreach_sqrt_default_1[78]
        getitem_2185: "f32[2304]" = _foreach_sqrt_default_1[79]
        getitem_2186: "f32[768, 768]" = _foreach_sqrt_default_1[80]
        getitem_2187: "f32[768]" = _foreach_sqrt_default_1[81]
        getitem_2188: "f32[3072, 768]" = _foreach_sqrt_default_1[82]
        getitem_2189: "f32[3072]" = _foreach_sqrt_default_1[83]
        getitem_2190: "f32[768, 3072]" = _foreach_sqrt_default_1[84]
        getitem_2191: "f32[768]" = _foreach_sqrt_default_1[85]
        getitem_2192: "f32[768]" = _foreach_sqrt_default_1[86]
        getitem_2193: "f32[768]" = _foreach_sqrt_default_1[87]
        getitem_2194: "f32[768]" = _foreach_sqrt_default_1[88]
        getitem_2195: "f32[768]" = _foreach_sqrt_default_1[89]
        getitem_2196: "f32[2304, 768]" = _foreach_sqrt_default_1[90]
        getitem_2197: "f32[2304]" = _foreach_sqrt_default_1[91]
        getitem_2198: "f32[768, 768]" = _foreach_sqrt_default_1[92]
        getitem_2199: "f32[768]" = _foreach_sqrt_default_1[93]
        getitem_2200: "f32[3072, 768]" = _foreach_sqrt_default_1[94]
        getitem_2201: "f32[3072]" = _foreach_sqrt_default_1[95]
        getitem_2202: "f32[768, 3072]" = _foreach_sqrt_default_1[96]
        getitem_2203: "f32[768]" = _foreach_sqrt_default_1[97]
        getitem_2204: "f32[768]" = _foreach_sqrt_default_1[98]
        getitem_2205: "f32[768]" = _foreach_sqrt_default_1[99]
        getitem_2206: "f32[768]" = _foreach_sqrt_default_1[100]
        getitem_2207: "f32[768]" = _foreach_sqrt_default_1[101]
        getitem_2208: "f32[2304, 768]" = _foreach_sqrt_default_1[102]
        getitem_2209: "f32[2304]" = _foreach_sqrt_default_1[103]
        getitem_2210: "f32[768, 768]" = _foreach_sqrt_default_1[104]
        getitem_2211: "f32[768]" = _foreach_sqrt_default_1[105]
        getitem_2212: "f32[3072, 768]" = _foreach_sqrt_default_1[106]
        getitem_2213: "f32[3072]" = _foreach_sqrt_default_1[107]
        getitem_2214: "f32[768, 3072]" = _foreach_sqrt_default_1[108]
        getitem_2215: "f32[768]" = _foreach_sqrt_default_1[109]
        getitem_2216: "f32[768]" = _foreach_sqrt_default_1[110]
        getitem_2217: "f32[768]" = _foreach_sqrt_default_1[111]
        getitem_2218: "f32[768]" = _foreach_sqrt_default_1[112]
        getitem_2219: "f32[768]" = _foreach_sqrt_default_1[113]
        getitem_2220: "f32[2304, 768]" = _foreach_sqrt_default_1[114]
        getitem_2221: "f32[2304]" = _foreach_sqrt_default_1[115]
        getitem_2222: "f32[768, 768]" = _foreach_sqrt_default_1[116]
        getitem_2223: "f32[768]" = _foreach_sqrt_default_1[117]
        getitem_2224: "f32[3072, 768]" = _foreach_sqrt_default_1[118]
        getitem_2225: "f32[3072]" = _foreach_sqrt_default_1[119]
        getitem_2226: "f32[768, 3072]" = _foreach_sqrt_default_1[120]
        getitem_2227: "f32[768]" = _foreach_sqrt_default_1[121]
        getitem_2228: "f32[768]" = _foreach_sqrt_default_1[122]
        getitem_2229: "f32[768]" = _foreach_sqrt_default_1[123]
        getitem_2230: "f32[768]" = _foreach_sqrt_default_1[124]
        getitem_2231: "f32[768]" = _foreach_sqrt_default_1[125]
        getitem_2232: "f32[2304, 768]" = _foreach_sqrt_default_1[126]
        getitem_2233: "f32[2304]" = _foreach_sqrt_default_1[127]
        getitem_2234: "f32[768, 768]" = _foreach_sqrt_default_1[128]
        getitem_2235: "f32[768]" = _foreach_sqrt_default_1[129]
        getitem_2236: "f32[3072, 768]" = _foreach_sqrt_default_1[130]
        getitem_2237: "f32[3072]" = _foreach_sqrt_default_1[131]
        getitem_2238: "f32[768, 3072]" = _foreach_sqrt_default_1[132]
        getitem_2239: "f32[768]" = _foreach_sqrt_default_1[133]
        getitem_2240: "f32[768]" = _foreach_sqrt_default_1[134]
        getitem_2241: "f32[768]" = _foreach_sqrt_default_1[135]
        getitem_2242: "f32[768]" = _foreach_sqrt_default_1[136]
        getitem_2243: "f32[768]" = _foreach_sqrt_default_1[137]
        getitem_2244: "f32[2304, 768]" = _foreach_sqrt_default_1[138]
        getitem_2245: "f32[2304]" = _foreach_sqrt_default_1[139]
        getitem_2246: "f32[768, 768]" = _foreach_sqrt_default_1[140]
        getitem_2247: "f32[768]" = _foreach_sqrt_default_1[141]
        getitem_2248: "f32[3072, 768]" = _foreach_sqrt_default_1[142]
        getitem_2249: "f32[3072]" = _foreach_sqrt_default_1[143]
        getitem_2250: "f32[768, 3072]" = _foreach_sqrt_default_1[144]
        getitem_2251: "f32[768]" = _foreach_sqrt_default_1[145]
        getitem_2252: "f32[768]" = _foreach_sqrt_default_1[146]
        getitem_2253: "f32[768]" = _foreach_sqrt_default_1[147]
        getitem_2254: "f32[768]" = _foreach_sqrt_default_1[148]
        getitem_2255: "f32[768]" = _foreach_sqrt_default_1[149]
        getitem_2256: "f32[768]" = _foreach_sqrt_default_1[150]
        getitem_2257: "f32[768]" = _foreach_sqrt_default_1[151]
        getitem_2258: "f32[77, 512]" = _foreach_sqrt_default_1[152]
        getitem_2259: "f32[49408, 512]" = _foreach_sqrt_default_1[153]
        getitem_2260: "f32[1536, 512]" = _foreach_sqrt_default_1[154]
        getitem_2261: "f32[1536]" = _foreach_sqrt_default_1[155]
        getitem_2262: "f32[512, 512]" = _foreach_sqrt_default_1[156]
        getitem_2263: "f32[512]" = _foreach_sqrt_default_1[157]
        getitem_2264: "f32[2048, 512]" = _foreach_sqrt_default_1[158]
        getitem_2265: "f32[2048]" = _foreach_sqrt_default_1[159]
        getitem_2266: "f32[512, 2048]" = _foreach_sqrt_default_1[160]
        getitem_2267: "f32[512]" = _foreach_sqrt_default_1[161]
        getitem_2268: "f32[512]" = _foreach_sqrt_default_1[162]
        getitem_2269: "f32[512]" = _foreach_sqrt_default_1[163]
        getitem_2270: "f32[512]" = _foreach_sqrt_default_1[164]
        getitem_2271: "f32[512]" = _foreach_sqrt_default_1[165]
        getitem_2272: "f32[1536, 512]" = _foreach_sqrt_default_1[166]
        getitem_2273: "f32[1536]" = _foreach_sqrt_default_1[167]
        getitem_2274: "f32[512, 512]" = _foreach_sqrt_default_1[168]
        getitem_2275: "f32[512]" = _foreach_sqrt_default_1[169]
        getitem_2276: "f32[2048, 512]" = _foreach_sqrt_default_1[170]
        getitem_2277: "f32[2048]" = _foreach_sqrt_default_1[171]
        getitem_2278: "f32[512, 2048]" = _foreach_sqrt_default_1[172]
        getitem_2279: "f32[512]" = _foreach_sqrt_default_1[173]
        getitem_2280: "f32[512]" = _foreach_sqrt_default_1[174]
        getitem_2281: "f32[512]" = _foreach_sqrt_default_1[175]
        getitem_2282: "f32[512]" = _foreach_sqrt_default_1[176]
        getitem_2283: "f32[512]" = _foreach_sqrt_default_1[177]
        getitem_2284: "f32[1536, 512]" = _foreach_sqrt_default_1[178]
        getitem_2285: "f32[1536]" = _foreach_sqrt_default_1[179]
        getitem_2286: "f32[512, 512]" = _foreach_sqrt_default_1[180]
        getitem_2287: "f32[512]" = _foreach_sqrt_default_1[181]
        getitem_2288: "f32[2048, 512]" = _foreach_sqrt_default_1[182]
        getitem_2289: "f32[2048]" = _foreach_sqrt_default_1[183]
        getitem_2290: "f32[512, 2048]" = _foreach_sqrt_default_1[184]
        getitem_2291: "f32[512]" = _foreach_sqrt_default_1[185]
        getitem_2292: "f32[512]" = _foreach_sqrt_default_1[186]
        getitem_2293: "f32[512]" = _foreach_sqrt_default_1[187]
        getitem_2294: "f32[512]" = _foreach_sqrt_default_1[188]
        getitem_2295: "f32[512]" = _foreach_sqrt_default_1[189]
        getitem_2296: "f32[1536, 512]" = _foreach_sqrt_default_1[190]
        getitem_2297: "f32[1536]" = _foreach_sqrt_default_1[191]
        getitem_2298: "f32[512, 512]" = _foreach_sqrt_default_1[192]
        getitem_2299: "f32[512]" = _foreach_sqrt_default_1[193]
        getitem_2300: "f32[2048, 512]" = _foreach_sqrt_default_1[194]
        getitem_2301: "f32[2048]" = _foreach_sqrt_default_1[195]
        getitem_2302: "f32[512, 2048]" = _foreach_sqrt_default_1[196]
        getitem_2303: "f32[512]" = _foreach_sqrt_default_1[197]
        getitem_2304: "f32[512]" = _foreach_sqrt_default_1[198]
        getitem_2305: "f32[512]" = _foreach_sqrt_default_1[199]
        getitem_2306: "f32[512]" = _foreach_sqrt_default_1[200]
        getitem_2307: "f32[512]" = _foreach_sqrt_default_1[201]
        getitem_2308: "f32[1536, 512]" = _foreach_sqrt_default_1[202]
        getitem_2309: "f32[1536]" = _foreach_sqrt_default_1[203]
        getitem_2310: "f32[512, 512]" = _foreach_sqrt_default_1[204]
        getitem_2311: "f32[512]" = _foreach_sqrt_default_1[205]
        getitem_2312: "f32[2048, 512]" = _foreach_sqrt_default_1[206]
        getitem_2313: "f32[2048]" = _foreach_sqrt_default_1[207]
        getitem_2314: "f32[512, 2048]" = _foreach_sqrt_default_1[208]
        getitem_2315: "f32[512]" = _foreach_sqrt_default_1[209]
        getitem_2316: "f32[512]" = _foreach_sqrt_default_1[210]
        getitem_2317: "f32[512]" = _foreach_sqrt_default_1[211]
        getitem_2318: "f32[512]" = _foreach_sqrt_default_1[212]
        getitem_2319: "f32[512]" = _foreach_sqrt_default_1[213]
        getitem_2320: "f32[1536, 512]" = _foreach_sqrt_default_1[214]
        getitem_2321: "f32[1536]" = _foreach_sqrt_default_1[215]
        getitem_2322: "f32[512, 512]" = _foreach_sqrt_default_1[216]
        getitem_2323: "f32[512]" = _foreach_sqrt_default_1[217]
        getitem_2324: "f32[2048, 512]" = _foreach_sqrt_default_1[218]
        getitem_2325: "f32[2048]" = _foreach_sqrt_default_1[219]
        getitem_2326: "f32[512, 2048]" = _foreach_sqrt_default_1[220]
        getitem_2327: "f32[512]" = _foreach_sqrt_default_1[221]
        getitem_2328: "f32[512]" = _foreach_sqrt_default_1[222]
        getitem_2329: "f32[512]" = _foreach_sqrt_default_1[223]
        getitem_2330: "f32[512]" = _foreach_sqrt_default_1[224]
        getitem_2331: "f32[512]" = _foreach_sqrt_default_1[225]
        getitem_2332: "f32[1536, 512]" = _foreach_sqrt_default_1[226]
        getitem_2333: "f32[1536]" = _foreach_sqrt_default_1[227]
        getitem_2334: "f32[512, 512]" = _foreach_sqrt_default_1[228]
        getitem_2335: "f32[512]" = _foreach_sqrt_default_1[229]
        getitem_2336: "f32[2048, 512]" = _foreach_sqrt_default_1[230]
        getitem_2337: "f32[2048]" = _foreach_sqrt_default_1[231]
        getitem_2338: "f32[512, 2048]" = _foreach_sqrt_default_1[232]
        getitem_2339: "f32[512]" = _foreach_sqrt_default_1[233]
        getitem_2340: "f32[512]" = _foreach_sqrt_default_1[234]
        getitem_2341: "f32[512]" = _foreach_sqrt_default_1[235]
        getitem_2342: "f32[512]" = _foreach_sqrt_default_1[236]
        getitem_2343: "f32[512]" = _foreach_sqrt_default_1[237]
        getitem_2344: "f32[1536, 512]" = _foreach_sqrt_default_1[238]
        getitem_2345: "f32[1536]" = _foreach_sqrt_default_1[239]
        getitem_2346: "f32[512, 512]" = _foreach_sqrt_default_1[240]
        getitem_2347: "f32[512]" = _foreach_sqrt_default_1[241]
        getitem_2348: "f32[2048, 512]" = _foreach_sqrt_default_1[242]
        getitem_2349: "f32[2048]" = _foreach_sqrt_default_1[243]
        getitem_2350: "f32[512, 2048]" = _foreach_sqrt_default_1[244]
        getitem_2351: "f32[512]" = _foreach_sqrt_default_1[245]
        getitem_2352: "f32[512]" = _foreach_sqrt_default_1[246]
        getitem_2353: "f32[512]" = _foreach_sqrt_default_1[247]
        getitem_2354: "f32[512]" = _foreach_sqrt_default_1[248]
        getitem_2355: "f32[512]" = _foreach_sqrt_default_1[249]
        getitem_2356: "f32[1536, 512]" = _foreach_sqrt_default_1[250]
        getitem_2357: "f32[1536]" = _foreach_sqrt_default_1[251]
        getitem_2358: "f32[512, 512]" = _foreach_sqrt_default_1[252]
        getitem_2359: "f32[512]" = _foreach_sqrt_default_1[253]
        getitem_2360: "f32[2048, 512]" = _foreach_sqrt_default_1[254]
        getitem_2361: "f32[2048]" = _foreach_sqrt_default_1[255]
        getitem_2362: "f32[512, 2048]" = _foreach_sqrt_default_1[256]
        getitem_2363: "f32[512]" = _foreach_sqrt_default_1[257]
        getitem_2364: "f32[512]" = _foreach_sqrt_default_1[258]
        getitem_2365: "f32[512]" = _foreach_sqrt_default_1[259]
        getitem_2366: "f32[512]" = _foreach_sqrt_default_1[260]
        getitem_2367: "f32[512]" = _foreach_sqrt_default_1[261]
        getitem_2368: "f32[1536, 512]" = _foreach_sqrt_default_1[262]
        getitem_2369: "f32[1536]" = _foreach_sqrt_default_1[263]
        getitem_2370: "f32[512, 512]" = _foreach_sqrt_default_1[264]
        getitem_2371: "f32[512]" = _foreach_sqrt_default_1[265]
        getitem_2372: "f32[2048, 512]" = _foreach_sqrt_default_1[266]
        getitem_2373: "f32[2048]" = _foreach_sqrt_default_1[267]
        getitem_2374: "f32[512, 2048]" = _foreach_sqrt_default_1[268]
        getitem_2375: "f32[512]" = _foreach_sqrt_default_1[269]
        getitem_2376: "f32[512]" = _foreach_sqrt_default_1[270]
        getitem_2377: "f32[512]" = _foreach_sqrt_default_1[271]
        getitem_2378: "f32[512]" = _foreach_sqrt_default_1[272]
        getitem_2379: "f32[512]" = _foreach_sqrt_default_1[273]
        getitem_2380: "f32[1536, 512]" = _foreach_sqrt_default_1[274]
        getitem_2381: "f32[1536]" = _foreach_sqrt_default_1[275]
        getitem_2382: "f32[512, 512]" = _foreach_sqrt_default_1[276]
        getitem_2383: "f32[512]" = _foreach_sqrt_default_1[277]
        getitem_2384: "f32[2048, 512]" = _foreach_sqrt_default_1[278]
        getitem_2385: "f32[2048]" = _foreach_sqrt_default_1[279]
        getitem_2386: "f32[512, 2048]" = _foreach_sqrt_default_1[280]
        getitem_2387: "f32[512]" = _foreach_sqrt_default_1[281]
        getitem_2388: "f32[512]" = _foreach_sqrt_default_1[282]
        getitem_2389: "f32[512]" = _foreach_sqrt_default_1[283]
        getitem_2390: "f32[512]" = _foreach_sqrt_default_1[284]
        getitem_2391: "f32[512]" = _foreach_sqrt_default_1[285]
        getitem_2392: "f32[1536, 512]" = _foreach_sqrt_default_1[286]
        getitem_2393: "f32[1536]" = _foreach_sqrt_default_1[287]
        getitem_2394: "f32[512, 512]" = _foreach_sqrt_default_1[288]
        getitem_2395: "f32[512]" = _foreach_sqrt_default_1[289]
        getitem_2396: "f32[2048, 512]" = _foreach_sqrt_default_1[290]
        getitem_2397: "f32[2048]" = _foreach_sqrt_default_1[291]
        getitem_2398: "f32[512, 2048]" = _foreach_sqrt_default_1[292]
        getitem_2399: "f32[512]" = _foreach_sqrt_default_1[293]
        getitem_2400: "f32[512]" = _foreach_sqrt_default_1[294]
        getitem_2401: "f32[512]" = _foreach_sqrt_default_1[295]
        getitem_2402: "f32[512]" = _foreach_sqrt_default_1[296]
        getitem_2403: "f32[512]" = _foreach_sqrt_default_1[297]
        getitem_2404: "f32[512]" = _foreach_sqrt_default_1[298]
        getitem_2405: "f32[512]" = _foreach_sqrt_default_1[299]
        getitem_2406: "f32[512, 512]" = _foreach_sqrt_default_1[300];  _foreach_sqrt_default_1 = None
        return (getitem, getitem_1806, getitem_1807, getitem_1808, getitem_1809, getitem_1810, getitem_1811, getitem_1812, getitem_1813, getitem_1814, getitem_1815, getitem_1816, getitem_1817, getitem_1818, getitem_1819, getitem_1820, getitem_1821, getitem_1822, getitem_1823, getitem_1824, getitem_1825, getitem_1826, getitem_1827, getitem_1828, getitem_1829, getitem_1830, getitem_1831, getitem_1832, getitem_1833, getitem_1834, getitem_1835, getitem_1836, getitem_1837, getitem_1838, getitem_1839, getitem_1840, getitem_1841, getitem_1842, getitem_1843, getitem_1844, getitem_1845, getitem_1846, getitem_1847, getitem_1848, getitem_1849, getitem_1850, getitem_1851, getitem_1852, getitem_1853, getitem_1854, getitem_1855, getitem_1856, getitem_1857, getitem_1858, getitem_1859, getitem_1860, getitem_1861, getitem_1862, getitem_1863, getitem_1864, getitem_1865, getitem_1866, getitem_1867, getitem_1868, getitem_1869, getitem_1870, getitem_1871, getitem_1872, getitem_1873, getitem_1874, getitem_1875, getitem_1876, getitem_1877, getitem_1878, getitem_1879, getitem_1880, getitem_1881, getitem_1882, getitem_1883, getitem_1884, getitem_1885, getitem_1886, getitem_1887, getitem_1888, getitem_1889, getitem_1890, getitem_1891, getitem_1892, getitem_1893, getitem_1894, getitem_1895, getitem_1896, getitem_1897, getitem_1898, getitem_1899, getitem_1900, getitem_1901, getitem_1902, getitem_1903, getitem_1904, getitem_1905, getitem_1906, getitem_1907, getitem_1908, getitem_1909, getitem_1910, getitem_1911, getitem_1912, getitem_1913, getitem_1914, getitem_1915, getitem_1916, getitem_1917, getitem_1918, getitem_1919, getitem_1920, getitem_1921, getitem_1922, getitem_1923, getitem_1924, getitem_1925, getitem_1926, getitem_1927, getitem_1928, getitem_1929, getitem_1930, getitem_1931, getitem_1932, getitem_1933, getitem_1934, getitem_1935, getitem_1936, getitem_1937, getitem_1938, getitem_1939, getitem_1940, getitem_1941, getitem_1942, getitem_1943, getitem_1944, getitem_1945, getitem_1946, getitem_1947, getitem_1948, getitem_1949, getitem_1950, getitem_1951, getitem_1952, getitem_1953, getitem_1954, getitem_1955, getitem_1956, getitem_1957, getitem_1958, getitem_1959, getitem_1960, getitem_1961, getitem_1962, getitem_1963, getitem_1964, getitem_1965, getitem_1966, getitem_1967, getitem_1968, getitem_1969, getitem_1970, getitem_1971, getitem_1972, getitem_1973, getitem_1974, getitem_1975, getitem_1976, getitem_1977, getitem_1978, getitem_1979, getitem_1980, getitem_1981, getitem_1982, getitem_1983, getitem_1984, getitem_1985, getitem_1986, getitem_1987, getitem_1988, getitem_1989, getitem_1990, getitem_1991, getitem_1992, getitem_1993, getitem_1994, getitem_1995, getitem_1996, getitem_1997, getitem_1998, getitem_1999, getitem_2000, getitem_2001, getitem_2002, getitem_2003, getitem_2004, getitem_2005, getitem_2006, getitem_2007, getitem_2008, getitem_2009, getitem_2010, getitem_2011, getitem_2012, getitem_2013, getitem_2014, getitem_2015, getitem_2016, getitem_2017, getitem_2018, getitem_2019, getitem_2020, getitem_2021, getitem_2022, getitem_2023, getitem_2024, getitem_2025, getitem_2026, getitem_2027, getitem_2028, getitem_2029, getitem_2030, getitem_2031, getitem_2032, getitem_2033, getitem_2034, getitem_2035, getitem_2036, getitem_2037, getitem_2038, getitem_2039, getitem_2040, getitem_2041, getitem_2042, getitem_2043, getitem_2044, getitem_2045, getitem_2046, getitem_2047, getitem_2048, getitem_2049, getitem_2050, getitem_2051, getitem_2052, getitem_2053, getitem_2054, getitem_2055, getitem_2056, getitem_2057, getitem_2058, getitem_2059, getitem_2060, getitem_2061, getitem_2062, getitem_2063, getitem_2064, getitem_2065, getitem_2066, getitem_2067, getitem_2068, getitem_2069, getitem_2070, getitem_2071, getitem_2072, getitem_2073, getitem_2074, getitem_2075, getitem_2076, getitem_2077, getitem_2078, getitem_2079, getitem_2080, getitem_2081, getitem_2082, getitem_2083, getitem_2084, getitem_2085, getitem_2086, getitem_2087, getitem_2088, getitem_2089, getitem_2090, getitem_2091, getitem_2092, getitem_2093, getitem_2094, getitem_2095, getitem_2096, getitem_2097, getitem_2098, getitem_2099, getitem_2100, getitem_2101, getitem_2102, getitem_2103, getitem_2104, getitem_2105, getitem_3010, getitem_3011, getitem_3012, getitem_3013, getitem_3014, getitem_3015, getitem_3016, getitem_3017, getitem_3018, getitem_3019, getitem_3020, getitem_3021, getitem_3022, getitem_3023, getitem_3024, getitem_3025, getitem_3026, getitem_3027, getitem_3028, getitem_3029, getitem_3030, getitem_3031, getitem_3032, getitem_3033, getitem_3034, getitem_3035, getitem_3036, getitem_3037, getitem_3038, getitem_3039, getitem_3040, getitem_3041, getitem_3042, getitem_3043, getitem_3044, getitem_3045, getitem_3046, getitem_3047, getitem_3048, getitem_3049, getitem_3050, getitem_3051, getitem_3052, getitem_3053, getitem_3054, getitem_3055, getitem_3056, getitem_3057, getitem_3058, getitem_3059, getitem_3060, getitem_3061, getitem_3062, getitem_3063, getitem_3064, getitem_3065, getitem_3066, getitem_3067, getitem_3068, getitem_3069, getitem_3070, getitem_3071, getitem_3072, getitem_3073, getitem_3074, getitem_3075, getitem_3076, getitem_3077, getitem_3078, getitem_3079, getitem_3080, getitem_3081, getitem_3082, getitem_3083, getitem_3084, getitem_3085, getitem_3086, getitem_3087, getitem_3088, getitem_3089, getitem_3090, getitem_3091, getitem_3092, getitem_3093, getitem_3094, getitem_3095, getitem_3096, getitem_3097, getitem_3098, getitem_3099, getitem_3100, getitem_3101, getitem_3102, getitem_3103, getitem_3104, getitem_3105, getitem_3106, getitem_3107, getitem_3108, getitem_3109, getitem_3110, getitem_3111, getitem_3112, getitem_3113, getitem_3114, getitem_3115, getitem_3116, getitem_3117, getitem_3118, getitem_3119, getitem_3120, getitem_3121, getitem_3122, getitem_3123, getitem_3124, getitem_3125, getitem_3126, getitem_3127, getitem_3128, getitem_3129, getitem_3130, getitem_3131, getitem_3132, getitem_3133, getitem_3134, getitem_3135, getitem_3136, getitem_3137, getitem_3138, getitem_3139, getitem_3140, getitem_3141, getitem_3142, getitem_3143, getitem_3144, getitem_3145, getitem_3146, getitem_3147, getitem_3148, getitem_3149, getitem_3150, getitem_3151, getitem_3152, getitem_3153, getitem_3154, getitem_3155, getitem_3156, getitem_3157, getitem_3158, getitem_3159, getitem_3160, getitem_3161, getitem_3162, getitem_3163, getitem_3164, getitem_3165, getitem_3166, getitem_3167, getitem_3168, getitem_3169, getitem_3170, getitem_3171, getitem_3172, getitem_3173, getitem_3174, getitem_3175, getitem_3176, getitem_3177, getitem_3178, getitem_3179, getitem_3180, getitem_3181, getitem_3182, getitem_3183, getitem_3184, getitem_3185, getitem_3186, getitem_3187, getitem_3188, getitem_3189, getitem_3190, getitem_3191, getitem_3192, getitem_3193, getitem_3194, getitem_3195, getitem_3196, getitem_3197, getitem_3198, getitem_3199, getitem_3200, getitem_3201, getitem_3202, getitem_3203, getitem_3204, getitem_3205, getitem_3206, getitem_3207, getitem_3208, getitem_3209, getitem_3210, getitem_3211, getitem_3212, getitem_3213, getitem_3214, getitem_3215, getitem_3216, getitem_3217, getitem_3218, getitem_3219, getitem_3220, getitem_3221, getitem_3222, getitem_3223, getitem_3224, getitem_3225, getitem_3226, getitem_3227, getitem_3228, getitem_3229, getitem_3230, getitem_3231, getitem_3232, getitem_3233, getitem_3234, getitem_3235, getitem_3236, getitem_3237, getitem_3238, getitem_3239, getitem_3240, getitem_3241, getitem_3242, getitem_3243, getitem_3244, getitem_3245, getitem_3246, getitem_3247, getitem_3248, getitem_3249, getitem_3250, getitem_3251, getitem_3252, getitem_3253, getitem_3254, getitem_3255, getitem_3256, getitem_3257, getitem_3258, getitem_3259, getitem_3260, getitem_3261, getitem_3262, getitem_3263, getitem_3264, getitem_3265, getitem_3266, getitem_3267, getitem_3268, getitem_3269, getitem_3270, getitem_3271, getitem_3272, getitem_3273, getitem_3274, getitem_3275, getitem_3276, getitem_3277, getitem_3278, getitem_3279, getitem_3280, getitem_3281, getitem_3282, getitem_3283, getitem_3284, getitem_3285, getitem_3286, getitem_3287, getitem_3288, getitem_3289, getitem_3290, getitem_3291, getitem_3292, getitem_3293, getitem_3294, getitem_3295, getitem_3296, getitem_3297, getitem_3298, getitem_3299, getitem_3300, getitem_3301, getitem_3302, getitem_3303, getitem_3304, getitem_3305, getitem_3306, getitem_3307, getitem_3308, getitem_3309, getitem_3310, getitem_2106, getitem_2107, getitem_2108, getitem_2109, getitem_2110, getitem_2111, getitem_2112, getitem_2113, getitem_2114, getitem_2115, getitem_2116, getitem_2117, getitem_2118, getitem_2119, getitem_2120, getitem_2121, getitem_2122, getitem_2123, getitem_2124, getitem_2125, getitem_2126, getitem_2127, getitem_2128, getitem_2129, getitem_2130, getitem_2131, getitem_2132, getitem_2133, getitem_2134, getitem_2135, getitem_2136, getitem_2137, getitem_2138, getitem_2139, getitem_2140, getitem_2141, getitem_2142, getitem_2143, getitem_2144, getitem_2145, getitem_2146, getitem_2147, getitem_2148, getitem_2149, getitem_2150, getitem_2151, getitem_2152, getitem_2153, getitem_2154, getitem_2155, getitem_2156, getitem_2157, getitem_2158, getitem_2159, getitem_2160, getitem_2161, getitem_2162, getitem_2163, getitem_2164, getitem_2165, getitem_2166, getitem_2167, getitem_2168, getitem_2169, getitem_2170, getitem_2171, getitem_2172, getitem_2173, getitem_2174, getitem_2175, getitem_2176, getitem_2177, getitem_2178, getitem_2179, getitem_2180, getitem_2181, getitem_2182, getitem_2183, getitem_2184, getitem_2185, getitem_2186, getitem_2187, getitem_2188, getitem_2189, getitem_2190, getitem_2191, getitem_2192, getitem_2193, getitem_2194, getitem_2195, getitem_2196, getitem_2197, getitem_2198, getitem_2199, getitem_2200, getitem_2201, getitem_2202, getitem_2203, getitem_2204, getitem_2205, getitem_2206, getitem_2207, getitem_2208, getitem_2209, getitem_2210, getitem_2211, getitem_2212, getitem_2213, getitem_2214, getitem_2215, getitem_2216, getitem_2217, getitem_2218, getitem_2219, getitem_2220, getitem_2221, getitem_2222, getitem_2223, getitem_2224, getitem_2225, getitem_2226, getitem_2227, getitem_2228, getitem_2229, getitem_2230, getitem_2231, getitem_2232, getitem_2233, getitem_2234, getitem_2235, getitem_2236, getitem_2237, getitem_2238, getitem_2239, getitem_2240, getitem_2241, getitem_2242, getitem_2243, getitem_2244, getitem_2245, getitem_2246, getitem_2247, getitem_2248, getitem_2249, getitem_2250, getitem_2251, getitem_2252, getitem_2253, getitem_2254, getitem_2255, getitem_2256, getitem_2257, getitem_2258, getitem_2259, getitem_2260, getitem_2261, getitem_2262, getitem_2263, getitem_2264, getitem_2265, getitem_2266, getitem_2267, getitem_2268, getitem_2269, getitem_2270, getitem_2271, getitem_2272, getitem_2273, getitem_2274, getitem_2275, getitem_2276, getitem_2277, getitem_2278, getitem_2279, getitem_2280, getitem_2281, getitem_2282, getitem_2283, getitem_2284, getitem_2285, getitem_2286, getitem_2287, getitem_2288, getitem_2289, getitem_2290, getitem_2291, getitem_2292, getitem_2293, getitem_2294, getitem_2295, getitem_2296, getitem_2297, getitem_2298, getitem_2299, getitem_2300, getitem_2301, getitem_2302, getitem_2303, getitem_2304, getitem_2305, getitem_2306, getitem_2307, getitem_2308, getitem_2309, getitem_2310, getitem_2311, getitem_2312, getitem_2313, getitem_2314, getitem_2315, getitem_2316, getitem_2317, getitem_2318, getitem_2319, getitem_2320, getitem_2321, getitem_2322, getitem_2323, getitem_2324, getitem_2325, getitem_2326, getitem_2327, getitem_2328, getitem_2329, getitem_2330, getitem_2331, getitem_2332, getitem_2333, getitem_2334, getitem_2335, getitem_2336, getitem_2337, getitem_2338, getitem_2339, getitem_2340, getitem_2341, getitem_2342, getitem_2343, getitem_2344, getitem_2345, getitem_2346, getitem_2347, getitem_2348, getitem_2349, getitem_2350, getitem_2351, getitem_2352, getitem_2353, getitem_2354, getitem_2355, getitem_2356, getitem_2357, getitem_2358, getitem_2359, getitem_2360, getitem_2361, getitem_2362, getitem_2363, getitem_2364, getitem_2365, getitem_2366, getitem_2367, getitem_2368, getitem_2369, getitem_2370, getitem_2371, getitem_2372, getitem_2373, getitem_2374, getitem_2375, getitem_2376, getitem_2377, getitem_2378, getitem_2379, getitem_2380, getitem_2381, getitem_2382, getitem_2383, getitem_2384, getitem_2385, getitem_2386, getitem_2387, getitem_2388, getitem_2389, getitem_2390, getitem_2391, getitem_2392, getitem_2393, getitem_2394, getitem_2395, getitem_2396, getitem_2397, getitem_2398, getitem_2399, getitem_2400, getitem_2401, getitem_2402, getitem_2403, getitem_2404, getitem_2405, getitem_2406)


def _default_make_inputs():
    return [
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
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
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
