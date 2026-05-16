"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s4_g59
Pattern hash: f3c2f193df94
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
    def forward(self, arg524_1: "f32[32128, 512]", arg525_1: "f32[512, 512]", arg526_1: "f32[512, 512]", arg527_1: "f32[512, 512]", arg528_1: "f32[512, 512]", arg529_1: "f32[32, 8]", arg530_1: "f32[512]", arg531_1: "f32[2048, 512]", arg532_1: "f32[512, 2048]", arg533_1: "f32[512]", arg534_1: "f32[512, 512]", arg535_1: "f32[512, 512]", arg536_1: "f32[512, 512]", arg537_1: "f32[512, 512]", arg538_1: "f32[512]", arg539_1: "f32[2048, 512]", arg540_1: "f32[512, 2048]", arg541_1: "f32[512]", arg542_1: "f32[512, 512]", arg543_1: "f32[512, 512]", arg544_1: "f32[512, 512]", arg545_1: "f32[512, 512]", arg546_1: "f32[512]", arg547_1: "f32[2048, 512]", arg548_1: "f32[512, 2048]", arg549_1: "f32[512]", arg550_1: "f32[512, 512]", arg551_1: "f32[512, 512]", arg552_1: "f32[512, 512]", arg553_1: "f32[512, 512]", arg554_1: "f32[512]", arg555_1: "f32[2048, 512]", arg556_1: "f32[512, 2048]", arg557_1: "f32[512]", arg558_1: "f32[512, 512]", arg559_1: "f32[512, 512]", arg560_1: "f32[512, 512]", arg561_1: "f32[512, 512]", arg562_1: "f32[512]", arg563_1: "f32[2048, 512]", arg564_1: "f32[512, 2048]", arg565_1: "f32[512]", arg566_1: "f32[512, 512]", arg567_1: "f32[512, 512]", arg568_1: "f32[512, 512]", arg569_1: "f32[512, 512]", arg570_1: "f32[512]", arg571_1: "f32[2048, 512]", arg572_1: "f32[512, 2048]", arg573_1: "f32[512]", arg574_1: "f32[512]", arg575_1: "f32[512, 512]", arg576_1: "f32[512, 512]", arg577_1: "f32[512, 512]", arg578_1: "f32[512, 512]", arg579_1: "f32[32, 8]", arg580_1: "f32[512]", arg581_1: "f32[512, 512]", arg582_1: "f32[512, 512]", arg583_1: "f32[512, 512]", arg584_1: "f32[512, 512]", arg585_1: "f32[512]", arg586_1: "f32[2048, 512]", arg587_1: "f32[512, 2048]", arg588_1: "f32[512]", arg589_1: "f32[512, 512]", arg590_1: "f32[512, 512]", arg591_1: "f32[512, 512]", arg592_1: "f32[512, 512]", arg593_1: "f32[512]", arg594_1: "f32[512, 512]", arg595_1: "f32[512, 512]", arg596_1: "f32[512, 512]", arg597_1: "f32[512, 512]", arg598_1: "f32[512]", arg599_1: "f32[2048, 512]", arg600_1: "f32[512, 2048]", arg601_1: "f32[512]", arg602_1: "f32[512, 512]", arg603_1: "f32[512, 512]", arg604_1: "f32[512, 512]", arg605_1: "f32[512, 512]", arg606_1: "f32[512]", arg607_1: "f32[512, 512]", arg608_1: "f32[512, 512]", arg609_1: "f32[512, 512]", arg610_1: "f32[512, 512]", arg611_1: "f32[512]", arg612_1: "f32[2048, 512]", arg613_1: "f32[512, 2048]", arg614_1: "f32[512]", arg615_1: "f32[512, 512]", arg616_1: "f32[512, 512]", arg617_1: "f32[512, 512]", arg618_1: "f32[512, 512]", arg619_1: "f32[512]", arg620_1: "f32[512, 512]", arg621_1: "f32[512, 512]", arg622_1: "f32[512, 512]", arg623_1: "f32[512, 512]", arg624_1: "f32[512]", arg625_1: "f32[2048, 512]", arg626_1: "f32[512, 2048]", arg627_1: "f32[512]", arg628_1: "f32[512, 512]", arg629_1: "f32[512, 512]", arg630_1: "f32[512, 512]", arg631_1: "f32[512, 512]", arg632_1: "f32[512]", arg633_1: "f32[512, 512]", arg634_1: "f32[512, 512]", arg635_1: "f32[512, 512]", arg636_1: "f32[512, 512]", arg637_1: "f32[512]", arg638_1: "f32[2048, 512]", arg639_1: "f32[512, 2048]", arg640_1: "f32[512]", arg641_1: "f32[512, 512]", arg642_1: "f32[512, 512]", arg643_1: "f32[512, 512]", arg644_1: "f32[512, 512]", arg645_1: "f32[512]", arg646_1: "f32[512, 512]", arg647_1: "f32[512, 512]", arg648_1: "f32[512, 512]", arg649_1: "f32[512, 512]", arg650_1: "f32[512]", arg651_1: "f32[2048, 512]", arg652_1: "f32[512, 2048]", arg653_1: "f32[512]", arg654_1: "f32[512]", arg134_1: "f32[32128, 512]", arg132_1: "f32[512, 512]", arg266_1: "f32[512, 512]", arg267_1: "f32[512, 512]", arg268_1: "f32[512, 512]", arg269_1: "f32[32, 8]", arg270_1: "f32[512]", arg271_1: "f32[2048, 512]", arg272_1: "f32[512, 2048]", arg273_1: "f32[512]", arg274_1: "f32[512, 512]", arg275_1: "f32[512, 512]", arg276_1: "f32[512, 512]", arg277_1: "f32[512, 512]", arg278_1: "f32[512]", arg279_1: "f32[2048, 512]", arg280_1: "f32[512, 2048]", arg281_1: "f32[512]", arg282_1: "f32[512, 512]", arg283_1: "f32[512, 512]", arg284_1: "f32[512, 512]", arg285_1: "f32[512, 512]", arg286_1: "f32[512]", arg287_1: "f32[2048, 512]", arg288_1: "f32[512, 2048]", arg289_1: "f32[512]", arg290_1: "f32[512, 512]", arg291_1: "f32[512, 512]", arg292_1: "f32[512, 512]", arg293_1: "f32[512, 512]", arg294_1: "f32[512]", arg295_1: "f32[2048, 512]", arg296_1: "f32[512, 2048]", arg297_1: "f32[512]", arg298_1: "f32[512, 512]", arg299_1: "f32[512, 512]", arg300_1: "f32[512, 512]", arg301_1: "f32[512, 512]", arg302_1: "f32[512]", arg303_1: "f32[2048, 512]", arg304_1: "f32[512, 2048]", arg305_1: "f32[512]", arg306_1: "f32[512, 512]", arg307_1: "f32[512, 512]", arg308_1: "f32[512, 512]", arg309_1: "f32[512, 512]", arg310_1: "f32[512]", arg311_1: "f32[2048, 512]", arg312_1: "f32[512, 2048]", arg313_1: "f32[512]", arg314_1: "f32[512]", arg315_1: "f32[512, 512]", arg316_1: "f32[512, 512]", arg317_1: "f32[512, 512]", arg318_1: "f32[512, 512]", arg319_1: "f32[32, 8]", arg320_1: "f32[512]", arg321_1: "f32[512, 512]", arg322_1: "f32[512, 512]", arg323_1: "f32[512, 512]", arg324_1: "f32[512, 512]", arg325_1: "f32[512]", arg326_1: "f32[2048, 512]", arg327_1: "f32[512, 2048]", arg328_1: "f32[512]", arg329_1: "f32[512, 512]", arg330_1: "f32[512, 512]", arg331_1: "f32[512, 512]", arg332_1: "f32[512, 512]", arg333_1: "f32[512]", arg334_1: "f32[512, 512]", arg335_1: "f32[512, 512]", arg336_1: "f32[512, 512]", arg337_1: "f32[512, 512]", arg338_1: "f32[512]", arg339_1: "f32[2048, 512]", arg340_1: "f32[512, 2048]", arg341_1: "f32[512]", arg342_1: "f32[512, 512]", arg343_1: "f32[512, 512]", arg344_1: "f32[512, 512]", arg345_1: "f32[512, 512]", arg346_1: "f32[512]", arg347_1: "f32[512, 512]", arg348_1: "f32[512, 512]", arg349_1: "f32[512, 512]", arg350_1: "f32[512, 512]", arg351_1: "f32[512]", arg352_1: "f32[2048, 512]", arg353_1: "f32[512, 2048]", arg354_1: "f32[512]", arg355_1: "f32[512, 512]", arg356_1: "f32[512, 512]", arg357_1: "f32[512, 512]", arg358_1: "f32[512, 512]", arg359_1: "f32[512]", arg360_1: "f32[512, 512]", arg361_1: "f32[512, 512]", arg362_1: "f32[512, 512]", arg363_1: "f32[512, 512]", arg364_1: "f32[512]", arg365_1: "f32[2048, 512]", arg366_1: "f32[512, 2048]", arg367_1: "f32[512]", arg368_1: "f32[512, 512]", arg369_1: "f32[512, 512]", arg370_1: "f32[512, 512]", arg371_1: "f32[512, 512]", arg372_1: "f32[512]", arg373_1: "f32[512, 512]", arg374_1: "f32[512, 512]", arg375_1: "f32[512, 512]", arg376_1: "f32[512, 512]", arg377_1: "f32[512]", arg378_1: "f32[2048, 512]", arg379_1: "f32[512, 2048]", arg380_1: "f32[512]", arg381_1: "f32[512, 512]", arg382_1: "f32[512, 512]", arg383_1: "f32[512, 512]", arg384_1: "f32[512, 512]", arg385_1: "f32[512]", arg386_1: "f32[512, 512]", arg387_1: "f32[512, 512]", arg388_1: "f32[512, 512]", arg389_1: "f32[512, 512]", arg390_1: "f32[512]", arg391_1: "f32[2048, 512]", arg392_1: "f32[512, 2048]", arg393_1: "f32[512]", arg394_1: "f32[512]", getitem_1310: "f32[]", getitem_1311: "f32[]", getitem_1312: "f32[]", getitem_1313: "f32[]", getitem_1314: "f32[]", getitem_1315: "f32[]", getitem_1316: "f32[]", getitem_1317: "f32[]", getitem_1318: "f32[]", getitem_1319: "f32[]", getitem_1320: "f32[]", getitem_1321: "f32[]", getitem_1322: "f32[]", getitem_1323: "f32[]", getitem_1324: "f32[]", getitem_1325: "f32[]", getitem_1326: "f32[]", getitem_1327: "f32[]", getitem_1328: "f32[]", getitem_1329: "f32[]", getitem_1330: "f32[]", getitem_1331: "f32[]", getitem_1332: "f32[]", getitem_1333: "f32[]", getitem_1334: "f32[]", getitem_1335: "f32[]", getitem_1336: "f32[]", getitem_1337: "f32[]", getitem_1338: "f32[]", getitem_1339: "f32[]", getitem_1340: "f32[]", getitem_1341: "f32[]", getitem_1342: "f32[]", getitem_1343: "f32[]", getitem_1344: "f32[]", getitem_1345: "f32[]", getitem_1346: "f32[]", getitem_1347: "f32[]", getitem_1348: "f32[]", getitem_1349: "f32[]", getitem_1350: "f32[]", getitem_1351: "f32[]", getitem_1352: "f32[]", getitem_1353: "f32[]", getitem_1354: "f32[]", getitem_1355: "f32[]", getitem_1356: "f32[]", getitem_1357: "f32[]", getitem_1358: "f32[]", getitem_1359: "f32[]", getitem_1360: "f32[]", getitem_1361: "f32[]", getitem_1362: "f32[]", getitem_1363: "f32[]", getitem_1364: "f32[]", getitem_1365: "f32[]", getitem_1366: "f32[]", getitem_1367: "f32[]", getitem_1368: "f32[]", getitem_1369: "f32[]", getitem_1370: "f32[]", getitem_1371: "f32[]", getitem_1372: "f32[]", getitem_1373: "f32[]", getitem_1374: "f32[]", getitem_1375: "f32[]", getitem_1376: "f32[]", getitem_1377: "f32[]", getitem_1378: "f32[]", getitem_1379: "f32[]", getitem_1380: "f32[]", getitem_1381: "f32[]", getitem_1382: "f32[]", getitem_1383: "f32[]", getitem_1384: "f32[]", getitem_1385: "f32[]", getitem_1386: "f32[]", getitem_1387: "f32[]", getitem_1388: "f32[]", getitem_1389: "f32[]", getitem_1390: "f32[]", getitem_1391: "f32[]", getitem_1392: "f32[]", getitem_1393: "f32[]", getitem_1394: "f32[]", getitem_1395: "f32[]", getitem_1396: "f32[]", getitem_1397: "f32[]", getitem_1398: "f32[]", getitem_1399: "f32[]", getitem_1400: "f32[]", getitem_1401: "f32[]", getitem_1402: "f32[]", getitem_1403: "f32[]", getitem_1404: "f32[]", getitem_1405: "f32[]", getitem_1406: "f32[]", getitem_1407: "f32[]", getitem_1408: "f32[]", getitem_1409: "f32[]", getitem_1410: "f32[]", getitem_1411: "f32[]", getitem_1412: "f32[]", getitem_1413: "f32[]", getitem_1414: "f32[]", getitem_1415: "f32[]", getitem_1416: "f32[]", getitem_1417: "f32[]", getitem_1418: "f32[]", getitem_1419: "f32[]", getitem_1420: "f32[]", getitem_1421: "f32[]", getitem_1422: "f32[]", getitem_1423: "f32[]", getitem_1424: "f32[]", getitem_1425: "f32[]", getitem_1426: "f32[]", getitem_1427: "f32[]", getitem_1428: "f32[]", getitem_1429: "f32[]", getitem_1430: "f32[]", getitem_1431: "f32[]", getitem_1432: "f32[]", getitem_1433: "f32[]", getitem_1434: "f32[]", getitem_1435: "f32[]", getitem_1436: "f32[]", getitem_1437: "f32[]", getitem_1438: "f32[]", getitem_1439: "f32[]", getitem_1440: "f32[]", getitem_1834: "f32[32128, 512]", getitem_1835: "f32[512, 512]", getitem_1836: "f32[512, 512]", getitem_1837: "f32[512, 512]", getitem_1838: "f32[512, 512]", getitem_1839: "f32[32, 8]", getitem_1840: "f32[512]", getitem_1841: "f32[2048, 512]", getitem_1842: "f32[512, 2048]", getitem_1843: "f32[512]", getitem_1844: "f32[512, 512]", getitem_1845: "f32[512, 512]", getitem_1846: "f32[512, 512]", getitem_1847: "f32[512, 512]", getitem_1848: "f32[512]", getitem_1849: "f32[2048, 512]", getitem_1850: "f32[512, 2048]", getitem_1851: "f32[512]", getitem_1852: "f32[512, 512]", getitem_1853: "f32[512, 512]", getitem_1854: "f32[512, 512]", getitem_1855: "f32[512, 512]", getitem_1856: "f32[512]", getitem_1857: "f32[2048, 512]", getitem_1858: "f32[512, 2048]", getitem_1859: "f32[512]", getitem_1860: "f32[512, 512]", getitem_1861: "f32[512, 512]", getitem_1862: "f32[512, 512]", getitem_1863: "f32[512, 512]", getitem_1864: "f32[512]", getitem_1865: "f32[2048, 512]", getitem_1866: "f32[512, 2048]", getitem_1867: "f32[512]", getitem_1868: "f32[512, 512]", getitem_1869: "f32[512, 512]", getitem_1870: "f32[512, 512]", getitem_1871: "f32[512, 512]", getitem_1872: "f32[512]", getitem_1873: "f32[2048, 512]", getitem_1874: "f32[512, 2048]", getitem_1875: "f32[512]", getitem_1876: "f32[512, 512]", getitem_1877: "f32[512, 512]", getitem_1878: "f32[512, 512]", getitem_1879: "f32[512, 512]", getitem_1880: "f32[512]", getitem_1881: "f32[2048, 512]", getitem_1882: "f32[512, 2048]", getitem_1883: "f32[512]", getitem_1884: "f32[512]", getitem_1885: "f32[512, 512]", getitem_1886: "f32[512, 512]", getitem_1887: "f32[512, 512]", getitem_1888: "f32[512, 512]", getitem_1889: "f32[32, 8]", getitem_1890: "f32[512]", getitem_1891: "f32[512, 512]", getitem_1892: "f32[512, 512]", getitem_1893: "f32[512, 512]", getitem_1894: "f32[512, 512]", getitem_1895: "f32[512]", getitem_1896: "f32[2048, 512]", getitem_1897: "f32[512, 2048]", getitem_1898: "f32[512]", getitem_1899: "f32[512, 512]", getitem_1900: "f32[512, 512]", getitem_1901: "f32[512, 512]", getitem_1902: "f32[512, 512]", getitem_1903: "f32[512]", getitem_1904: "f32[512, 512]", getitem_1905: "f32[512, 512]", getitem_1906: "f32[512, 512]", getitem_1907: "f32[512, 512]", getitem_1908: "f32[512]", getitem_1909: "f32[2048, 512]", getitem_1910: "f32[512, 2048]", getitem_1911: "f32[512]", getitem_1912: "f32[512, 512]", getitem_1913: "f32[512, 512]", getitem_1914: "f32[512, 512]", getitem_1915: "f32[512, 512]", getitem_1916: "f32[512]", getitem_1917: "f32[512, 512]", getitem_1918: "f32[512, 512]", getitem_1919: "f32[512, 512]", getitem_1920: "f32[512, 512]", getitem_1921: "f32[512]", getitem_1922: "f32[2048, 512]", getitem_1923: "f32[512, 2048]", getitem_1924: "f32[512]", getitem_1925: "f32[512, 512]", getitem_1926: "f32[512, 512]", getitem_1927: "f32[512, 512]", getitem_1928: "f32[512, 512]", getitem_1929: "f32[512]", getitem_1930: "f32[512, 512]", getitem_1931: "f32[512, 512]", getitem_1932: "f32[512, 512]", getitem_1933: "f32[512, 512]", getitem_1934: "f32[512]", getitem_1935: "f32[2048, 512]", getitem_1936: "f32[512, 2048]", getitem_1937: "f32[512]", getitem_1938: "f32[512, 512]", getitem_1939: "f32[512, 512]", getitem_1940: "f32[512, 512]", getitem_1941: "f32[512, 512]", getitem_1942: "f32[512]", getitem_1943: "f32[512, 512]", getitem_1944: "f32[512, 512]", getitem_1945: "f32[512, 512]", getitem_1946: "f32[512, 512]", getitem_1947: "f32[512]", getitem_1948: "f32[2048, 512]", getitem_1949: "f32[512, 2048]", getitem_1950: "f32[512]", getitem_1951: "f32[512, 512]", getitem_1952: "f32[512, 512]", getitem_1953: "f32[512, 512]", getitem_1954: "f32[512, 512]", getitem_1955: "f32[512]", getitem_1956: "f32[512, 512]", getitem_1957: "f32[512, 512]", getitem_1958: "f32[512, 512]", getitem_1959: "f32[512, 512]", getitem_1960: "f32[512]", getitem_1961: "f32[2048, 512]", getitem_1962: "f32[512, 2048]", getitem_1963: "f32[512]", getitem_1964: "f32[512]"):
        # No stacktrace found for following nodes
        _foreach_sub_list = torch.ops.aten._foreach_sub.List([arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1], [arg134_1, arg132_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1]);  arg524_1 = arg525_1 = arg526_1 = arg527_1 = arg528_1 = arg529_1 = arg530_1 = arg531_1 = arg532_1 = arg533_1 = arg534_1 = arg535_1 = arg536_1 = arg537_1 = arg538_1 = arg539_1 = arg540_1 = arg541_1 = arg542_1 = arg543_1 = arg544_1 = arg545_1 = arg546_1 = arg547_1 = arg548_1 = arg549_1 = arg550_1 = arg551_1 = arg552_1 = arg553_1 = arg554_1 = arg555_1 = arg556_1 = arg557_1 = arg558_1 = arg559_1 = arg560_1 = arg561_1 = arg562_1 = arg563_1 = arg564_1 = arg565_1 = arg566_1 = arg567_1 = arg568_1 = arg569_1 = arg570_1 = arg571_1 = arg572_1 = arg573_1 = arg574_1 = arg575_1 = arg576_1 = arg577_1 = arg578_1 = arg579_1 = arg580_1 = arg581_1 = arg582_1 = arg583_1 = arg584_1 = arg585_1 = arg586_1 = arg587_1 = arg588_1 = arg589_1 = arg590_1 = arg591_1 = arg592_1 = arg593_1 = arg594_1 = arg595_1 = arg596_1 = arg597_1 = arg598_1 = arg599_1 = arg600_1 = arg601_1 = arg602_1 = arg603_1 = arg604_1 = arg605_1 = arg606_1 = arg607_1 = arg608_1 = arg609_1 = arg610_1 = arg611_1 = arg612_1 = arg613_1 = arg614_1 = arg615_1 = arg616_1 = arg617_1 = arg618_1 = arg619_1 = arg620_1 = arg621_1 = arg622_1 = arg623_1 = arg624_1 = arg625_1 = arg626_1 = arg627_1 = arg628_1 = arg629_1 = arg630_1 = arg631_1 = arg632_1 = arg633_1 = arg634_1 = arg635_1 = arg636_1 = arg637_1 = arg638_1 = arg639_1 = arg640_1 = arg641_1 = arg642_1 = arg643_1 = arg644_1 = arg645_1 = arg646_1 = arg647_1 = arg648_1 = arg649_1 = arg650_1 = arg651_1 = arg652_1 = arg653_1 = arg654_1 = arg134_1 = arg132_1 = arg266_1 = arg267_1 = arg268_1 = arg269_1 = arg270_1 = arg271_1 = arg272_1 = arg273_1 = arg274_1 = arg275_1 = arg276_1 = arg277_1 = arg278_1 = arg279_1 = arg280_1 = arg281_1 = arg282_1 = arg283_1 = arg284_1 = arg285_1 = arg286_1 = arg287_1 = arg288_1 = arg289_1 = arg290_1 = arg291_1 = arg292_1 = arg293_1 = arg294_1 = arg295_1 = arg296_1 = arg297_1 = arg298_1 = arg299_1 = arg300_1 = arg301_1 = arg302_1 = arg303_1 = arg304_1 = arg305_1 = arg306_1 = arg307_1 = arg308_1 = arg309_1 = arg310_1 = arg311_1 = arg312_1 = arg313_1 = arg314_1 = arg315_1 = arg316_1 = arg317_1 = arg318_1 = arg319_1 = arg320_1 = arg321_1 = arg322_1 = arg323_1 = arg324_1 = arg325_1 = arg326_1 = arg327_1 = arg328_1 = arg329_1 = arg330_1 = arg331_1 = arg332_1 = arg333_1 = arg334_1 = arg335_1 = arg336_1 = arg337_1 = arg338_1 = arg339_1 = arg340_1 = arg341_1 = arg342_1 = arg343_1 = arg344_1 = arg345_1 = arg346_1 = arg347_1 = arg348_1 = arg349_1 = arg350_1 = arg351_1 = arg352_1 = arg353_1 = arg354_1 = arg355_1 = arg356_1 = arg357_1 = arg358_1 = arg359_1 = arg360_1 = arg361_1 = arg362_1 = arg363_1 = arg364_1 = arg365_1 = arg366_1 = arg367_1 = arg368_1 = arg369_1 = arg370_1 = arg371_1 = arg372_1 = arg373_1 = arg374_1 = arg375_1 = arg376_1 = arg377_1 = arg378_1 = arg379_1 = arg380_1 = arg381_1 = arg382_1 = arg383_1 = arg384_1 = arg385_1 = arg386_1 = arg387_1 = arg388_1 = arg389_1 = arg390_1 = arg391_1 = arg392_1 = arg393_1 = arg394_1 = None
        getitem: "f32[32128, 512]" = _foreach_sub_list[0]
        getitem_1: "f32[512, 512]" = _foreach_sub_list[1]
        getitem_2: "f32[512, 512]" = _foreach_sub_list[2]
        getitem_3: "f32[512, 512]" = _foreach_sub_list[3]
        getitem_4: "f32[512, 512]" = _foreach_sub_list[4]
        getitem_5: "f32[32, 8]" = _foreach_sub_list[5]
        getitem_6: "f32[512]" = _foreach_sub_list[6]
        getitem_7: "f32[2048, 512]" = _foreach_sub_list[7]
        getitem_8: "f32[512, 2048]" = _foreach_sub_list[8]
        getitem_9: "f32[512]" = _foreach_sub_list[9]
        getitem_10: "f32[512, 512]" = _foreach_sub_list[10]
        getitem_11: "f32[512, 512]" = _foreach_sub_list[11]
        getitem_12: "f32[512, 512]" = _foreach_sub_list[12]
        getitem_13: "f32[512, 512]" = _foreach_sub_list[13]
        getitem_14: "f32[512]" = _foreach_sub_list[14]
        getitem_15: "f32[2048, 512]" = _foreach_sub_list[15]
        getitem_16: "f32[512, 2048]" = _foreach_sub_list[16]
        getitem_17: "f32[512]" = _foreach_sub_list[17]
        getitem_18: "f32[512, 512]" = _foreach_sub_list[18]
        getitem_19: "f32[512, 512]" = _foreach_sub_list[19]
        getitem_20: "f32[512, 512]" = _foreach_sub_list[20]
        getitem_21: "f32[512, 512]" = _foreach_sub_list[21]
        getitem_22: "f32[512]" = _foreach_sub_list[22]
        getitem_23: "f32[2048, 512]" = _foreach_sub_list[23]
        getitem_24: "f32[512, 2048]" = _foreach_sub_list[24]
        getitem_25: "f32[512]" = _foreach_sub_list[25]
        getitem_26: "f32[512, 512]" = _foreach_sub_list[26]
        getitem_27: "f32[512, 512]" = _foreach_sub_list[27]
        getitem_28: "f32[512, 512]" = _foreach_sub_list[28]
        getitem_29: "f32[512, 512]" = _foreach_sub_list[29]
        getitem_30: "f32[512]" = _foreach_sub_list[30]
        getitem_31: "f32[2048, 512]" = _foreach_sub_list[31]
        getitem_32: "f32[512, 2048]" = _foreach_sub_list[32]
        getitem_33: "f32[512]" = _foreach_sub_list[33]
        getitem_34: "f32[512, 512]" = _foreach_sub_list[34]
        getitem_35: "f32[512, 512]" = _foreach_sub_list[35]
        getitem_36: "f32[512, 512]" = _foreach_sub_list[36]
        getitem_37: "f32[512, 512]" = _foreach_sub_list[37]
        getitem_38: "f32[512]" = _foreach_sub_list[38]
        getitem_39: "f32[2048, 512]" = _foreach_sub_list[39]
        getitem_40: "f32[512, 2048]" = _foreach_sub_list[40]
        getitem_41: "f32[512]" = _foreach_sub_list[41]
        getitem_42: "f32[512, 512]" = _foreach_sub_list[42]
        getitem_43: "f32[512, 512]" = _foreach_sub_list[43]
        getitem_44: "f32[512, 512]" = _foreach_sub_list[44]
        getitem_45: "f32[512, 512]" = _foreach_sub_list[45]
        getitem_46: "f32[512]" = _foreach_sub_list[46]
        getitem_47: "f32[2048, 512]" = _foreach_sub_list[47]
        getitem_48: "f32[512, 2048]" = _foreach_sub_list[48]
        getitem_49: "f32[512]" = _foreach_sub_list[49]
        getitem_50: "f32[512]" = _foreach_sub_list[50]
        getitem_51: "f32[512, 512]" = _foreach_sub_list[51]
        getitem_52: "f32[512, 512]" = _foreach_sub_list[52]
        getitem_53: "f32[512, 512]" = _foreach_sub_list[53]
        getitem_54: "f32[512, 512]" = _foreach_sub_list[54]
        getitem_55: "f32[32, 8]" = _foreach_sub_list[55]
        getitem_56: "f32[512]" = _foreach_sub_list[56]
        getitem_57: "f32[512, 512]" = _foreach_sub_list[57]
        getitem_58: "f32[512, 512]" = _foreach_sub_list[58]
        getitem_59: "f32[512, 512]" = _foreach_sub_list[59]
        getitem_60: "f32[512, 512]" = _foreach_sub_list[60]
        getitem_61: "f32[512]" = _foreach_sub_list[61]
        getitem_62: "f32[2048, 512]" = _foreach_sub_list[62]
        getitem_63: "f32[512, 2048]" = _foreach_sub_list[63]
        getitem_64: "f32[512]" = _foreach_sub_list[64]
        getitem_65: "f32[512, 512]" = _foreach_sub_list[65]
        getitem_66: "f32[512, 512]" = _foreach_sub_list[66]
        getitem_67: "f32[512, 512]" = _foreach_sub_list[67]
        getitem_68: "f32[512, 512]" = _foreach_sub_list[68]
        getitem_69: "f32[512]" = _foreach_sub_list[69]
        getitem_70: "f32[512, 512]" = _foreach_sub_list[70]
        getitem_71: "f32[512, 512]" = _foreach_sub_list[71]
        getitem_72: "f32[512, 512]" = _foreach_sub_list[72]
        getitem_73: "f32[512, 512]" = _foreach_sub_list[73]
        getitem_74: "f32[512]" = _foreach_sub_list[74]
        getitem_75: "f32[2048, 512]" = _foreach_sub_list[75]
        getitem_76: "f32[512, 2048]" = _foreach_sub_list[76]
        getitem_77: "f32[512]" = _foreach_sub_list[77]
        getitem_78: "f32[512, 512]" = _foreach_sub_list[78]
        getitem_79: "f32[512, 512]" = _foreach_sub_list[79]
        getitem_80: "f32[512, 512]" = _foreach_sub_list[80]
        getitem_81: "f32[512, 512]" = _foreach_sub_list[81]
        getitem_82: "f32[512]" = _foreach_sub_list[82]
        getitem_83: "f32[512, 512]" = _foreach_sub_list[83]
        getitem_84: "f32[512, 512]" = _foreach_sub_list[84]
        getitem_85: "f32[512, 512]" = _foreach_sub_list[85]
        getitem_86: "f32[512, 512]" = _foreach_sub_list[86]
        getitem_87: "f32[512]" = _foreach_sub_list[87]
        getitem_88: "f32[2048, 512]" = _foreach_sub_list[88]
        getitem_89: "f32[512, 2048]" = _foreach_sub_list[89]
        getitem_90: "f32[512]" = _foreach_sub_list[90]
        getitem_91: "f32[512, 512]" = _foreach_sub_list[91]
        getitem_92: "f32[512, 512]" = _foreach_sub_list[92]
        getitem_93: "f32[512, 512]" = _foreach_sub_list[93]
        getitem_94: "f32[512, 512]" = _foreach_sub_list[94]
        getitem_95: "f32[512]" = _foreach_sub_list[95]
        getitem_96: "f32[512, 512]" = _foreach_sub_list[96]
        getitem_97: "f32[512, 512]" = _foreach_sub_list[97]
        getitem_98: "f32[512, 512]" = _foreach_sub_list[98]
        getitem_99: "f32[512, 512]" = _foreach_sub_list[99]
        getitem_100: "f32[512]" = _foreach_sub_list[100]
        getitem_101: "f32[2048, 512]" = _foreach_sub_list[101]
        getitem_102: "f32[512, 2048]" = _foreach_sub_list[102]
        getitem_103: "f32[512]" = _foreach_sub_list[103]
        getitem_104: "f32[512, 512]" = _foreach_sub_list[104]
        getitem_105: "f32[512, 512]" = _foreach_sub_list[105]
        getitem_106: "f32[512, 512]" = _foreach_sub_list[106]
        getitem_107: "f32[512, 512]" = _foreach_sub_list[107]
        getitem_108: "f32[512]" = _foreach_sub_list[108]
        getitem_109: "f32[512, 512]" = _foreach_sub_list[109]
        getitem_110: "f32[512, 512]" = _foreach_sub_list[110]
        getitem_111: "f32[512, 512]" = _foreach_sub_list[111]
        getitem_112: "f32[512, 512]" = _foreach_sub_list[112]
        getitem_113: "f32[512]" = _foreach_sub_list[113]
        getitem_114: "f32[2048, 512]" = _foreach_sub_list[114]
        getitem_115: "f32[512, 2048]" = _foreach_sub_list[115]
        getitem_116: "f32[512]" = _foreach_sub_list[116]
        getitem_117: "f32[512, 512]" = _foreach_sub_list[117]
        getitem_118: "f32[512, 512]" = _foreach_sub_list[118]
        getitem_119: "f32[512, 512]" = _foreach_sub_list[119]
        getitem_120: "f32[512, 512]" = _foreach_sub_list[120]
        getitem_121: "f32[512]" = _foreach_sub_list[121]
        getitem_122: "f32[512, 512]" = _foreach_sub_list[122]
        getitem_123: "f32[512, 512]" = _foreach_sub_list[123]
        getitem_124: "f32[512, 512]" = _foreach_sub_list[124]
        getitem_125: "f32[512, 512]" = _foreach_sub_list[125]
        getitem_126: "f32[512]" = _foreach_sub_list[126]
        getitem_127: "f32[2048, 512]" = _foreach_sub_list[127]
        getitem_128: "f32[512, 2048]" = _foreach_sub_list[128]
        getitem_129: "f32[512]" = _foreach_sub_list[129]
        getitem_130: "f32[512]" = _foreach_sub_list[130];  _foreach_sub_list = None
        _foreach_reciprocal_default = torch.ops.aten._foreach_reciprocal.default([getitem_1310, getitem_1311, getitem_1312, getitem_1313, getitem_1314, getitem_1315, getitem_1316, getitem_1317, getitem_1318, getitem_1319, getitem_1320, getitem_1321, getitem_1322, getitem_1323, getitem_1324, getitem_1325, getitem_1326, getitem_1327, getitem_1328, getitem_1329, getitem_1330, getitem_1331, getitem_1332, getitem_1333, getitem_1334, getitem_1335, getitem_1336, getitem_1337, getitem_1338, getitem_1339, getitem_1340, getitem_1341, getitem_1342, getitem_1343, getitem_1344, getitem_1345, getitem_1346, getitem_1347, getitem_1348, getitem_1349, getitem_1350, getitem_1351, getitem_1352, getitem_1353, getitem_1354, getitem_1355, getitem_1356, getitem_1357, getitem_1358, getitem_1359, getitem_1360, getitem_1361, getitem_1362, getitem_1363, getitem_1364, getitem_1365, getitem_1366, getitem_1367, getitem_1368, getitem_1369, getitem_1370, getitem_1371, getitem_1372, getitem_1373, getitem_1374, getitem_1375, getitem_1376, getitem_1377, getitem_1378, getitem_1379, getitem_1380, getitem_1381, getitem_1382, getitem_1383, getitem_1384, getitem_1385, getitem_1386, getitem_1387, getitem_1388, getitem_1389, getitem_1390, getitem_1391, getitem_1392, getitem_1393, getitem_1394, getitem_1395, getitem_1396, getitem_1397, getitem_1398, getitem_1399, getitem_1400, getitem_1401, getitem_1402, getitem_1403, getitem_1404, getitem_1405, getitem_1406, getitem_1407, getitem_1408, getitem_1409, getitem_1410, getitem_1411, getitem_1412, getitem_1413, getitem_1414, getitem_1415, getitem_1416, getitem_1417, getitem_1418, getitem_1419, getitem_1420, getitem_1421, getitem_1422, getitem_1423, getitem_1424, getitem_1425, getitem_1426, getitem_1427, getitem_1428, getitem_1429, getitem_1430, getitem_1431, getitem_1432, getitem_1433, getitem_1434, getitem_1435, getitem_1436, getitem_1437, getitem_1438, getitem_1439, getitem_1440]);  getitem_1310 = getitem_1311 = getitem_1312 = getitem_1313 = getitem_1314 = getitem_1315 = getitem_1316 = getitem_1317 = getitem_1318 = getitem_1319 = getitem_1320 = getitem_1321 = getitem_1322 = getitem_1323 = getitem_1324 = getitem_1325 = getitem_1326 = getitem_1327 = getitem_1328 = getitem_1329 = getitem_1330 = getitem_1331 = getitem_1332 = getitem_1333 = getitem_1334 = getitem_1335 = getitem_1336 = getitem_1337 = getitem_1338 = getitem_1339 = getitem_1340 = getitem_1341 = getitem_1342 = getitem_1343 = getitem_1344 = getitem_1345 = getitem_1346 = getitem_1347 = getitem_1348 = getitem_1349 = getitem_1350 = getitem_1351 = getitem_1352 = getitem_1353 = getitem_1354 = getitem_1355 = getitem_1356 = getitem_1357 = getitem_1358 = getitem_1359 = getitem_1360 = getitem_1361 = getitem_1362 = getitem_1363 = getitem_1364 = getitem_1365 = getitem_1366 = getitem_1367 = getitem_1368 = getitem_1369 = getitem_1370 = getitem_1371 = getitem_1372 = getitem_1373 = getitem_1374 = getitem_1375 = getitem_1376 = getitem_1377 = getitem_1378 = getitem_1379 = getitem_1380 = getitem_1381 = getitem_1382 = getitem_1383 = getitem_1384 = getitem_1385 = getitem_1386 = getitem_1387 = getitem_1388 = getitem_1389 = getitem_1390 = getitem_1391 = getitem_1392 = getitem_1393 = getitem_1394 = getitem_1395 = getitem_1396 = getitem_1397 = getitem_1398 = getitem_1399 = getitem_1400 = getitem_1401 = getitem_1402 = getitem_1403 = getitem_1404 = getitem_1405 = getitem_1406 = getitem_1407 = getitem_1408 = getitem_1409 = getitem_1410 = getitem_1411 = getitem_1412 = getitem_1413 = getitem_1414 = getitem_1415 = getitem_1416 = getitem_1417 = getitem_1418 = getitem_1419 = getitem_1420 = getitem_1421 = getitem_1422 = getitem_1423 = getitem_1424 = getitem_1425 = getitem_1426 = getitem_1427 = getitem_1428 = getitem_1429 = getitem_1430 = getitem_1431 = getitem_1432 = getitem_1433 = getitem_1434 = getitem_1435 = getitem_1436 = getitem_1437 = getitem_1438 = getitem_1439 = getitem_1440 = None
        getitem_1441: "f32[]" = _foreach_reciprocal_default[0]
        getitem_1442: "f32[]" = _foreach_reciprocal_default[1]
        getitem_1443: "f32[]" = _foreach_reciprocal_default[2]
        getitem_1444: "f32[]" = _foreach_reciprocal_default[3]
        getitem_1445: "f32[]" = _foreach_reciprocal_default[4]
        getitem_1446: "f32[]" = _foreach_reciprocal_default[5]
        getitem_1447: "f32[]" = _foreach_reciprocal_default[6]
        getitem_1448: "f32[]" = _foreach_reciprocal_default[7]
        getitem_1449: "f32[]" = _foreach_reciprocal_default[8]
        getitem_1450: "f32[]" = _foreach_reciprocal_default[9]
        getitem_1451: "f32[]" = _foreach_reciprocal_default[10]
        getitem_1452: "f32[]" = _foreach_reciprocal_default[11]
        getitem_1453: "f32[]" = _foreach_reciprocal_default[12]
        getitem_1454: "f32[]" = _foreach_reciprocal_default[13]
        getitem_1455: "f32[]" = _foreach_reciprocal_default[14]
        getitem_1456: "f32[]" = _foreach_reciprocal_default[15]
        getitem_1457: "f32[]" = _foreach_reciprocal_default[16]
        getitem_1458: "f32[]" = _foreach_reciprocal_default[17]
        getitem_1459: "f32[]" = _foreach_reciprocal_default[18]
        getitem_1460: "f32[]" = _foreach_reciprocal_default[19]
        getitem_1461: "f32[]" = _foreach_reciprocal_default[20]
        getitem_1462: "f32[]" = _foreach_reciprocal_default[21]
        getitem_1463: "f32[]" = _foreach_reciprocal_default[22]
        getitem_1464: "f32[]" = _foreach_reciprocal_default[23]
        getitem_1465: "f32[]" = _foreach_reciprocal_default[24]
        getitem_1466: "f32[]" = _foreach_reciprocal_default[25]
        getitem_1467: "f32[]" = _foreach_reciprocal_default[26]
        getitem_1468: "f32[]" = _foreach_reciprocal_default[27]
        getitem_1469: "f32[]" = _foreach_reciprocal_default[28]
        getitem_1470: "f32[]" = _foreach_reciprocal_default[29]
        getitem_1471: "f32[]" = _foreach_reciprocal_default[30]
        getitem_1472: "f32[]" = _foreach_reciprocal_default[31]
        getitem_1473: "f32[]" = _foreach_reciprocal_default[32]
        getitem_1474: "f32[]" = _foreach_reciprocal_default[33]
        getitem_1475: "f32[]" = _foreach_reciprocal_default[34]
        getitem_1476: "f32[]" = _foreach_reciprocal_default[35]
        getitem_1477: "f32[]" = _foreach_reciprocal_default[36]
        getitem_1478: "f32[]" = _foreach_reciprocal_default[37]
        getitem_1479: "f32[]" = _foreach_reciprocal_default[38]
        getitem_1480: "f32[]" = _foreach_reciprocal_default[39]
        getitem_1481: "f32[]" = _foreach_reciprocal_default[40]
        getitem_1482: "f32[]" = _foreach_reciprocal_default[41]
        getitem_1483: "f32[]" = _foreach_reciprocal_default[42]
        getitem_1484: "f32[]" = _foreach_reciprocal_default[43]
        getitem_1485: "f32[]" = _foreach_reciprocal_default[44]
        getitem_1486: "f32[]" = _foreach_reciprocal_default[45]
        getitem_1487: "f32[]" = _foreach_reciprocal_default[46]
        getitem_1488: "f32[]" = _foreach_reciprocal_default[47]
        getitem_1489: "f32[]" = _foreach_reciprocal_default[48]
        getitem_1490: "f32[]" = _foreach_reciprocal_default[49]
        getitem_1491: "f32[]" = _foreach_reciprocal_default[50]
        getitem_1492: "f32[]" = _foreach_reciprocal_default[51]
        getitem_1493: "f32[]" = _foreach_reciprocal_default[52]
        getitem_1494: "f32[]" = _foreach_reciprocal_default[53]
        getitem_1495: "f32[]" = _foreach_reciprocal_default[54]
        getitem_1496: "f32[]" = _foreach_reciprocal_default[55]
        getitem_1497: "f32[]" = _foreach_reciprocal_default[56]
        getitem_1498: "f32[]" = _foreach_reciprocal_default[57]
        getitem_1499: "f32[]" = _foreach_reciprocal_default[58]
        getitem_1500: "f32[]" = _foreach_reciprocal_default[59]
        getitem_1501: "f32[]" = _foreach_reciprocal_default[60]
        getitem_1502: "f32[]" = _foreach_reciprocal_default[61]
        getitem_1503: "f32[]" = _foreach_reciprocal_default[62]
        getitem_1504: "f32[]" = _foreach_reciprocal_default[63]
        getitem_1505: "f32[]" = _foreach_reciprocal_default[64]
        getitem_1506: "f32[]" = _foreach_reciprocal_default[65]
        getitem_1507: "f32[]" = _foreach_reciprocal_default[66]
        getitem_1508: "f32[]" = _foreach_reciprocal_default[67]
        getitem_1509: "f32[]" = _foreach_reciprocal_default[68]
        getitem_1510: "f32[]" = _foreach_reciprocal_default[69]
        getitem_1511: "f32[]" = _foreach_reciprocal_default[70]
        getitem_1512: "f32[]" = _foreach_reciprocal_default[71]
        getitem_1513: "f32[]" = _foreach_reciprocal_default[72]
        getitem_1514: "f32[]" = _foreach_reciprocal_default[73]
        getitem_1515: "f32[]" = _foreach_reciprocal_default[74]
        getitem_1516: "f32[]" = _foreach_reciprocal_default[75]
        getitem_1517: "f32[]" = _foreach_reciprocal_default[76]
        getitem_1518: "f32[]" = _foreach_reciprocal_default[77]
        getitem_1519: "f32[]" = _foreach_reciprocal_default[78]
        getitem_1520: "f32[]" = _foreach_reciprocal_default[79]
        getitem_1521: "f32[]" = _foreach_reciprocal_default[80]
        getitem_1522: "f32[]" = _foreach_reciprocal_default[81]
        getitem_1523: "f32[]" = _foreach_reciprocal_default[82]
        getitem_1524: "f32[]" = _foreach_reciprocal_default[83]
        getitem_1525: "f32[]" = _foreach_reciprocal_default[84]
        getitem_1526: "f32[]" = _foreach_reciprocal_default[85]
        getitem_1527: "f32[]" = _foreach_reciprocal_default[86]
        getitem_1528: "f32[]" = _foreach_reciprocal_default[87]
        getitem_1529: "f32[]" = _foreach_reciprocal_default[88]
        getitem_1530: "f32[]" = _foreach_reciprocal_default[89]
        getitem_1531: "f32[]" = _foreach_reciprocal_default[90]
        getitem_1532: "f32[]" = _foreach_reciprocal_default[91]
        getitem_1533: "f32[]" = _foreach_reciprocal_default[92]
        getitem_1534: "f32[]" = _foreach_reciprocal_default[93]
        getitem_1535: "f32[]" = _foreach_reciprocal_default[94]
        getitem_1536: "f32[]" = _foreach_reciprocal_default[95]
        getitem_1537: "f32[]" = _foreach_reciprocal_default[96]
        getitem_1538: "f32[]" = _foreach_reciprocal_default[97]
        getitem_1539: "f32[]" = _foreach_reciprocal_default[98]
        getitem_1540: "f32[]" = _foreach_reciprocal_default[99]
        getitem_1541: "f32[]" = _foreach_reciprocal_default[100]
        getitem_1542: "f32[]" = _foreach_reciprocal_default[101]
        getitem_1543: "f32[]" = _foreach_reciprocal_default[102]
        getitem_1544: "f32[]" = _foreach_reciprocal_default[103]
        getitem_1545: "f32[]" = _foreach_reciprocal_default[104]
        getitem_1546: "f32[]" = _foreach_reciprocal_default[105]
        getitem_1547: "f32[]" = _foreach_reciprocal_default[106]
        getitem_1548: "f32[]" = _foreach_reciprocal_default[107]
        getitem_1549: "f32[]" = _foreach_reciprocal_default[108]
        getitem_1550: "f32[]" = _foreach_reciprocal_default[109]
        getitem_1551: "f32[]" = _foreach_reciprocal_default[110]
        getitem_1552: "f32[]" = _foreach_reciprocal_default[111]
        getitem_1553: "f32[]" = _foreach_reciprocal_default[112]
        getitem_1554: "f32[]" = _foreach_reciprocal_default[113]
        getitem_1555: "f32[]" = _foreach_reciprocal_default[114]
        getitem_1556: "f32[]" = _foreach_reciprocal_default[115]
        getitem_1557: "f32[]" = _foreach_reciprocal_default[116]
        getitem_1558: "f32[]" = _foreach_reciprocal_default[117]
        getitem_1559: "f32[]" = _foreach_reciprocal_default[118]
        getitem_1560: "f32[]" = _foreach_reciprocal_default[119]
        getitem_1561: "f32[]" = _foreach_reciprocal_default[120]
        getitem_1562: "f32[]" = _foreach_reciprocal_default[121]
        getitem_1563: "f32[]" = _foreach_reciprocal_default[122]
        getitem_1564: "f32[]" = _foreach_reciprocal_default[123]
        getitem_1565: "f32[]" = _foreach_reciprocal_default[124]
        getitem_1566: "f32[]" = _foreach_reciprocal_default[125]
        getitem_1567: "f32[]" = _foreach_reciprocal_default[126]
        getitem_1568: "f32[]" = _foreach_reciprocal_default[127]
        getitem_1569: "f32[]" = _foreach_reciprocal_default[128]
        getitem_1570: "f32[]" = _foreach_reciprocal_default[129]
        getitem_1571: "f32[]" = _foreach_reciprocal_default[130];  _foreach_reciprocal_default = None
        _foreach_add_scalar = torch.ops.aten._foreach_add.Scalar([getitem_1834, getitem_1835, getitem_1836, getitem_1837, getitem_1838, getitem_1839, getitem_1840, getitem_1841, getitem_1842, getitem_1843, getitem_1844, getitem_1845, getitem_1846, getitem_1847, getitem_1848, getitem_1849, getitem_1850, getitem_1851, getitem_1852, getitem_1853, getitem_1854, getitem_1855, getitem_1856, getitem_1857, getitem_1858, getitem_1859, getitem_1860, getitem_1861, getitem_1862, getitem_1863, getitem_1864, getitem_1865, getitem_1866, getitem_1867, getitem_1868, getitem_1869, getitem_1870, getitem_1871, getitem_1872, getitem_1873, getitem_1874, getitem_1875, getitem_1876, getitem_1877, getitem_1878, getitem_1879, getitem_1880, getitem_1881, getitem_1882, getitem_1883, getitem_1884, getitem_1885, getitem_1886, getitem_1887, getitem_1888, getitem_1889, getitem_1890, getitem_1891, getitem_1892, getitem_1893, getitem_1894, getitem_1895, getitem_1896, getitem_1897, getitem_1898, getitem_1899, getitem_1900, getitem_1901, getitem_1902, getitem_1903, getitem_1904, getitem_1905, getitem_1906, getitem_1907, getitem_1908, getitem_1909, getitem_1910, getitem_1911, getitem_1912, getitem_1913, getitem_1914, getitem_1915, getitem_1916, getitem_1917, getitem_1918, getitem_1919, getitem_1920, getitem_1921, getitem_1922, getitem_1923, getitem_1924, getitem_1925, getitem_1926, getitem_1927, getitem_1928, getitem_1929, getitem_1930, getitem_1931, getitem_1932, getitem_1933, getitem_1934, getitem_1935, getitem_1936, getitem_1937, getitem_1938, getitem_1939, getitem_1940, getitem_1941, getitem_1942, getitem_1943, getitem_1944, getitem_1945, getitem_1946, getitem_1947, getitem_1948, getitem_1949, getitem_1950, getitem_1951, getitem_1952, getitem_1953, getitem_1954, getitem_1955, getitem_1956, getitem_1957, getitem_1958, getitem_1959, getitem_1960, getitem_1961, getitem_1962, getitem_1963, getitem_1964], 1e-08);  getitem_1834 = getitem_1835 = getitem_1836 = getitem_1837 = getitem_1838 = getitem_1839 = getitem_1840 = getitem_1841 = getitem_1842 = getitem_1843 = getitem_1844 = getitem_1845 = getitem_1846 = getitem_1847 = getitem_1848 = getitem_1849 = getitem_1850 = getitem_1851 = getitem_1852 = getitem_1853 = getitem_1854 = getitem_1855 = getitem_1856 = getitem_1857 = getitem_1858 = getitem_1859 = getitem_1860 = getitem_1861 = getitem_1862 = getitem_1863 = getitem_1864 = getitem_1865 = getitem_1866 = getitem_1867 = getitem_1868 = getitem_1869 = getitem_1870 = getitem_1871 = getitem_1872 = getitem_1873 = getitem_1874 = getitem_1875 = getitem_1876 = getitem_1877 = getitem_1878 = getitem_1879 = getitem_1880 = getitem_1881 = getitem_1882 = getitem_1883 = getitem_1884 = getitem_1885 = getitem_1886 = getitem_1887 = getitem_1888 = getitem_1889 = getitem_1890 = getitem_1891 = getitem_1892 = getitem_1893 = getitem_1894 = getitem_1895 = getitem_1896 = getitem_1897 = getitem_1898 = getitem_1899 = getitem_1900 = getitem_1901 = getitem_1902 = getitem_1903 = getitem_1904 = getitem_1905 = getitem_1906 = getitem_1907 = getitem_1908 = getitem_1909 = getitem_1910 = getitem_1911 = getitem_1912 = getitem_1913 = getitem_1914 = getitem_1915 = getitem_1916 = getitem_1917 = getitem_1918 = getitem_1919 = getitem_1920 = getitem_1921 = getitem_1922 = getitem_1923 = getitem_1924 = getitem_1925 = getitem_1926 = getitem_1927 = getitem_1928 = getitem_1929 = getitem_1930 = getitem_1931 = getitem_1932 = getitem_1933 = getitem_1934 = getitem_1935 = getitem_1936 = getitem_1937 = getitem_1938 = getitem_1939 = getitem_1940 = getitem_1941 = getitem_1942 = getitem_1943 = getitem_1944 = getitem_1945 = getitem_1946 = getitem_1947 = getitem_1948 = getitem_1949 = getitem_1950 = getitem_1951 = getitem_1952 = getitem_1953 = getitem_1954 = getitem_1955 = getitem_1956 = getitem_1957 = getitem_1958 = getitem_1959 = getitem_1960 = getitem_1961 = getitem_1962 = getitem_1963 = getitem_1964 = None
        getitem_1965: "f32[32128, 512]" = _foreach_add_scalar[0]
        getitem_1966: "f32[512, 512]" = _foreach_add_scalar[1]
        getitem_1967: "f32[512, 512]" = _foreach_add_scalar[2]
        getitem_1968: "f32[512, 512]" = _foreach_add_scalar[3]
        getitem_1969: "f32[512, 512]" = _foreach_add_scalar[4]
        getitem_1970: "f32[32, 8]" = _foreach_add_scalar[5]
        getitem_1971: "f32[512]" = _foreach_add_scalar[6]
        getitem_1972: "f32[2048, 512]" = _foreach_add_scalar[7]
        getitem_1973: "f32[512, 2048]" = _foreach_add_scalar[8]
        getitem_1974: "f32[512]" = _foreach_add_scalar[9]
        getitem_1975: "f32[512, 512]" = _foreach_add_scalar[10]
        getitem_1976: "f32[512, 512]" = _foreach_add_scalar[11]
        getitem_1977: "f32[512, 512]" = _foreach_add_scalar[12]
        getitem_1978: "f32[512, 512]" = _foreach_add_scalar[13]
        getitem_1979: "f32[512]" = _foreach_add_scalar[14]
        getitem_1980: "f32[2048, 512]" = _foreach_add_scalar[15]
        getitem_1981: "f32[512, 2048]" = _foreach_add_scalar[16]
        getitem_1982: "f32[512]" = _foreach_add_scalar[17]
        getitem_1983: "f32[512, 512]" = _foreach_add_scalar[18]
        getitem_1984: "f32[512, 512]" = _foreach_add_scalar[19]
        getitem_1985: "f32[512, 512]" = _foreach_add_scalar[20]
        getitem_1986: "f32[512, 512]" = _foreach_add_scalar[21]
        getitem_1987: "f32[512]" = _foreach_add_scalar[22]
        getitem_1988: "f32[2048, 512]" = _foreach_add_scalar[23]
        getitem_1989: "f32[512, 2048]" = _foreach_add_scalar[24]
        getitem_1990: "f32[512]" = _foreach_add_scalar[25]
        getitem_1991: "f32[512, 512]" = _foreach_add_scalar[26]
        getitem_1992: "f32[512, 512]" = _foreach_add_scalar[27]
        getitem_1993: "f32[512, 512]" = _foreach_add_scalar[28]
        getitem_1994: "f32[512, 512]" = _foreach_add_scalar[29]
        getitem_1995: "f32[512]" = _foreach_add_scalar[30]
        getitem_1996: "f32[2048, 512]" = _foreach_add_scalar[31]
        getitem_1997: "f32[512, 2048]" = _foreach_add_scalar[32]
        getitem_1998: "f32[512]" = _foreach_add_scalar[33]
        getitem_1999: "f32[512, 512]" = _foreach_add_scalar[34]
        getitem_2000: "f32[512, 512]" = _foreach_add_scalar[35]
        getitem_2001: "f32[512, 512]" = _foreach_add_scalar[36]
        getitem_2002: "f32[512, 512]" = _foreach_add_scalar[37]
        getitem_2003: "f32[512]" = _foreach_add_scalar[38]
        getitem_2004: "f32[2048, 512]" = _foreach_add_scalar[39]
        getitem_2005: "f32[512, 2048]" = _foreach_add_scalar[40]
        getitem_2006: "f32[512]" = _foreach_add_scalar[41]
        getitem_2007: "f32[512, 512]" = _foreach_add_scalar[42]
        getitem_2008: "f32[512, 512]" = _foreach_add_scalar[43]
        getitem_2009: "f32[512, 512]" = _foreach_add_scalar[44]
        getitem_2010: "f32[512, 512]" = _foreach_add_scalar[45]
        getitem_2011: "f32[512]" = _foreach_add_scalar[46]
        getitem_2012: "f32[2048, 512]" = _foreach_add_scalar[47]
        getitem_2013: "f32[512, 2048]" = _foreach_add_scalar[48]
        getitem_2014: "f32[512]" = _foreach_add_scalar[49]
        getitem_2015: "f32[512]" = _foreach_add_scalar[50]
        getitem_2016: "f32[512, 512]" = _foreach_add_scalar[51]
        getitem_2017: "f32[512, 512]" = _foreach_add_scalar[52]
        getitem_2018: "f32[512, 512]" = _foreach_add_scalar[53]
        getitem_2019: "f32[512, 512]" = _foreach_add_scalar[54]
        getitem_2020: "f32[32, 8]" = _foreach_add_scalar[55]
        getitem_2021: "f32[512]" = _foreach_add_scalar[56]
        getitem_2022: "f32[512, 512]" = _foreach_add_scalar[57]
        getitem_2023: "f32[512, 512]" = _foreach_add_scalar[58]
        getitem_2024: "f32[512, 512]" = _foreach_add_scalar[59]
        getitem_2025: "f32[512, 512]" = _foreach_add_scalar[60]
        getitem_2026: "f32[512]" = _foreach_add_scalar[61]
        getitem_2027: "f32[2048, 512]" = _foreach_add_scalar[62]
        getitem_2028: "f32[512, 2048]" = _foreach_add_scalar[63]
        getitem_2029: "f32[512]" = _foreach_add_scalar[64]
        getitem_2030: "f32[512, 512]" = _foreach_add_scalar[65]
        getitem_2031: "f32[512, 512]" = _foreach_add_scalar[66]
        getitem_2032: "f32[512, 512]" = _foreach_add_scalar[67]
        getitem_2033: "f32[512, 512]" = _foreach_add_scalar[68]
        getitem_2034: "f32[512]" = _foreach_add_scalar[69]
        getitem_2035: "f32[512, 512]" = _foreach_add_scalar[70]
        getitem_2036: "f32[512, 512]" = _foreach_add_scalar[71]
        getitem_2037: "f32[512, 512]" = _foreach_add_scalar[72]
        getitem_2038: "f32[512, 512]" = _foreach_add_scalar[73]
        getitem_2039: "f32[512]" = _foreach_add_scalar[74]
        getitem_2040: "f32[2048, 512]" = _foreach_add_scalar[75]
        getitem_2041: "f32[512, 2048]" = _foreach_add_scalar[76]
        getitem_2042: "f32[512]" = _foreach_add_scalar[77]
        getitem_2043: "f32[512, 512]" = _foreach_add_scalar[78]
        getitem_2044: "f32[512, 512]" = _foreach_add_scalar[79]
        getitem_2045: "f32[512, 512]" = _foreach_add_scalar[80]
        getitem_2046: "f32[512, 512]" = _foreach_add_scalar[81]
        getitem_2047: "f32[512]" = _foreach_add_scalar[82]
        getitem_2048: "f32[512, 512]" = _foreach_add_scalar[83]
        getitem_2049: "f32[512, 512]" = _foreach_add_scalar[84]
        getitem_2050: "f32[512, 512]" = _foreach_add_scalar[85]
        getitem_2051: "f32[512, 512]" = _foreach_add_scalar[86]
        getitem_2052: "f32[512]" = _foreach_add_scalar[87]
        getitem_2053: "f32[2048, 512]" = _foreach_add_scalar[88]
        getitem_2054: "f32[512, 2048]" = _foreach_add_scalar[89]
        getitem_2055: "f32[512]" = _foreach_add_scalar[90]
        getitem_2056: "f32[512, 512]" = _foreach_add_scalar[91]
        getitem_2057: "f32[512, 512]" = _foreach_add_scalar[92]
        getitem_2058: "f32[512, 512]" = _foreach_add_scalar[93]
        getitem_2059: "f32[512, 512]" = _foreach_add_scalar[94]
        getitem_2060: "f32[512]" = _foreach_add_scalar[95]
        getitem_2061: "f32[512, 512]" = _foreach_add_scalar[96]
        getitem_2062: "f32[512, 512]" = _foreach_add_scalar[97]
        getitem_2063: "f32[512, 512]" = _foreach_add_scalar[98]
        getitem_2064: "f32[512, 512]" = _foreach_add_scalar[99]
        getitem_2065: "f32[512]" = _foreach_add_scalar[100]
        getitem_2066: "f32[2048, 512]" = _foreach_add_scalar[101]
        getitem_2067: "f32[512, 2048]" = _foreach_add_scalar[102]
        getitem_2068: "f32[512]" = _foreach_add_scalar[103]
        getitem_2069: "f32[512, 512]" = _foreach_add_scalar[104]
        getitem_2070: "f32[512, 512]" = _foreach_add_scalar[105]
        getitem_2071: "f32[512, 512]" = _foreach_add_scalar[106]
        getitem_2072: "f32[512, 512]" = _foreach_add_scalar[107]
        getitem_2073: "f32[512]" = _foreach_add_scalar[108]
        getitem_2074: "f32[512, 512]" = _foreach_add_scalar[109]
        getitem_2075: "f32[512, 512]" = _foreach_add_scalar[110]
        getitem_2076: "f32[512, 512]" = _foreach_add_scalar[111]
        getitem_2077: "f32[512, 512]" = _foreach_add_scalar[112]
        getitem_2078: "f32[512]" = _foreach_add_scalar[113]
        getitem_2079: "f32[2048, 512]" = _foreach_add_scalar[114]
        getitem_2080: "f32[512, 2048]" = _foreach_add_scalar[115]
        getitem_2081: "f32[512]" = _foreach_add_scalar[116]
        getitem_2082: "f32[512, 512]" = _foreach_add_scalar[117]
        getitem_2083: "f32[512, 512]" = _foreach_add_scalar[118]
        getitem_2084: "f32[512, 512]" = _foreach_add_scalar[119]
        getitem_2085: "f32[512, 512]" = _foreach_add_scalar[120]
        getitem_2086: "f32[512]" = _foreach_add_scalar[121]
        getitem_2087: "f32[512, 512]" = _foreach_add_scalar[122]
        getitem_2088: "f32[512, 512]" = _foreach_add_scalar[123]
        getitem_2089: "f32[512, 512]" = _foreach_add_scalar[124]
        getitem_2090: "f32[512, 512]" = _foreach_add_scalar[125]
        getitem_2091: "f32[512]" = _foreach_add_scalar[126]
        getitem_2092: "f32[2048, 512]" = _foreach_add_scalar[127]
        getitem_2093: "f32[512, 2048]" = _foreach_add_scalar[128]
        getitem_2094: "f32[512]" = _foreach_add_scalar[129]
        getitem_2095: "f32[512]" = _foreach_add_scalar[130];  _foreach_add_scalar = None
        return (getitem, getitem_1, getitem_2, getitem_3, getitem_4, getitem_5, getitem_6, getitem_7, getitem_8, getitem_9, getitem_10, getitem_11, getitem_12, getitem_13, getitem_14, getitem_15, getitem_16, getitem_17, getitem_18, getitem_19, getitem_20, getitem_21, getitem_22, getitem_23, getitem_24, getitem_25, getitem_26, getitem_27, getitem_28, getitem_29, getitem_30, getitem_31, getitem_32, getitem_33, getitem_34, getitem_35, getitem_36, getitem_37, getitem_38, getitem_39, getitem_40, getitem_41, getitem_42, getitem_43, getitem_44, getitem_45, getitem_46, getitem_47, getitem_48, getitem_49, getitem_50, getitem_51, getitem_52, getitem_53, getitem_54, getitem_55, getitem_56, getitem_57, getitem_58, getitem_59, getitem_60, getitem_61, getitem_62, getitem_63, getitem_64, getitem_65, getitem_66, getitem_67, getitem_68, getitem_69, getitem_70, getitem_71, getitem_72, getitem_73, getitem_74, getitem_75, getitem_76, getitem_77, getitem_78, getitem_79, getitem_80, getitem_81, getitem_82, getitem_83, getitem_84, getitem_85, getitem_86, getitem_87, getitem_88, getitem_89, getitem_90, getitem_91, getitem_92, getitem_93, getitem_94, getitem_95, getitem_96, getitem_97, getitem_98, getitem_99, getitem_100, getitem_101, getitem_102, getitem_103, getitem_104, getitem_105, getitem_106, getitem_107, getitem_108, getitem_109, getitem_110, getitem_111, getitem_112, getitem_113, getitem_114, getitem_115, getitem_116, getitem_117, getitem_118, getitem_119, getitem_120, getitem_121, getitem_122, getitem_123, getitem_124, getitem_125, getitem_126, getitem_127, getitem_128, getitem_129, getitem_130, getitem_1441, getitem_1442, getitem_1443, getitem_1444, getitem_1445, getitem_1446, getitem_1447, getitem_1448, getitem_1449, getitem_1450, getitem_1451, getitem_1452, getitem_1453, getitem_1454, getitem_1455, getitem_1456, getitem_1457, getitem_1458, getitem_1459, getitem_1460, getitem_1461, getitem_1462, getitem_1463, getitem_1464, getitem_1465, getitem_1466, getitem_1467, getitem_1468, getitem_1469, getitem_1470, getitem_1471, getitem_1472, getitem_1473, getitem_1474, getitem_1475, getitem_1476, getitem_1477, getitem_1478, getitem_1479, getitem_1480, getitem_1481, getitem_1482, getitem_1483, getitem_1484, getitem_1485, getitem_1486, getitem_1487, getitem_1488, getitem_1489, getitem_1490, getitem_1491, getitem_1492, getitem_1493, getitem_1494, getitem_1495, getitem_1496, getitem_1497, getitem_1498, getitem_1499, getitem_1500, getitem_1501, getitem_1502, getitem_1503, getitem_1504, getitem_1505, getitem_1506, getitem_1507, getitem_1508, getitem_1509, getitem_1510, getitem_1511, getitem_1512, getitem_1513, getitem_1514, getitem_1515, getitem_1516, getitem_1517, getitem_1518, getitem_1519, getitem_1520, getitem_1521, getitem_1522, getitem_1523, getitem_1524, getitem_1525, getitem_1526, getitem_1527, getitem_1528, getitem_1529, getitem_1530, getitem_1531, getitem_1532, getitem_1533, getitem_1534, getitem_1535, getitem_1536, getitem_1537, getitem_1538, getitem_1539, getitem_1540, getitem_1541, getitem_1542, getitem_1543, getitem_1544, getitem_1545, getitem_1546, getitem_1547, getitem_1548, getitem_1549, getitem_1550, getitem_1551, getitem_1552, getitem_1553, getitem_1554, getitem_1555, getitem_1556, getitem_1557, getitem_1558, getitem_1559, getitem_1560, getitem_1561, getitem_1562, getitem_1563, getitem_1564, getitem_1565, getitem_1566, getitem_1567, getitem_1568, getitem_1569, getitem_1570, getitem_1571, getitem_1965, getitem_1966, getitem_1967, getitem_1968, getitem_1969, getitem_1970, getitem_1971, getitem_1972, getitem_1973, getitem_1974, getitem_1975, getitem_1976, getitem_1977, getitem_1978, getitem_1979, getitem_1980, getitem_1981, getitem_1982, getitem_1983, getitem_1984, getitem_1985, getitem_1986, getitem_1987, getitem_1988, getitem_1989, getitem_1990, getitem_1991, getitem_1992, getitem_1993, getitem_1994, getitem_1995, getitem_1996, getitem_1997, getitem_1998, getitem_1999, getitem_2000, getitem_2001, getitem_2002, getitem_2003, getitem_2004, getitem_2005, getitem_2006, getitem_2007, getitem_2008, getitem_2009, getitem_2010, getitem_2011, getitem_2012, getitem_2013, getitem_2014, getitem_2015, getitem_2016, getitem_2017, getitem_2018, getitem_2019, getitem_2020, getitem_2021, getitem_2022, getitem_2023, getitem_2024, getitem_2025, getitem_2026, getitem_2027, getitem_2028, getitem_2029, getitem_2030, getitem_2031, getitem_2032, getitem_2033, getitem_2034, getitem_2035, getitem_2036, getitem_2037, getitem_2038, getitem_2039, getitem_2040, getitem_2041, getitem_2042, getitem_2043, getitem_2044, getitem_2045, getitem_2046, getitem_2047, getitem_2048, getitem_2049, getitem_2050, getitem_2051, getitem_2052, getitem_2053, getitem_2054, getitem_2055, getitem_2056, getitem_2057, getitem_2058, getitem_2059, getitem_2060, getitem_2061, getitem_2062, getitem_2063, getitem_2064, getitem_2065, getitem_2066, getitem_2067, getitem_2068, getitem_2069, getitem_2070, getitem_2071, getitem_2072, getitem_2073, getitem_2074, getitem_2075, getitem_2076, getitem_2077, getitem_2078, getitem_2079, getitem_2080, getitem_2081, getitem_2082, getitem_2083, getitem_2084, getitem_2085, getitem_2086, getitem_2087, getitem_2088, getitem_2089, getitem_2090, getitem_2091, getitem_2092, getitem_2093, getitem_2094, getitem_2095)


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
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
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
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
