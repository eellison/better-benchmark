"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s3_g60
Pattern hash: 336749472a48
Shape hash: 428b4768
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg210_1: "f32[30522, 768]", arg208_1: "f32[512, 768]", arg620_1: "f32[1024, 768]", arg621_1: "f32[1024, 768]", arg622_1: "f32[1024, 768]", arg623_1: "f32[1024, 768]", arg624_1: "f32[2, 768]", arg625_1: "f32[768]", arg626_1: "f32[768]", arg627_1: "f32[768, 768]", arg628_1: "f32[768]", arg629_1: "f32[768, 768]", arg630_1: "f32[768]", arg631_1: "f32[768, 768]", arg632_1: "f32[768]", arg633_1: "f32[768, 768]", arg634_1: "f32[768]", arg635_1: "f32[768]", arg636_1: "f32[768]", arg637_1: "f32[3072, 768]", arg638_1: "f32[3072]", arg639_1: "f32[768, 3072]", arg640_1: "f32[768]", arg641_1: "f32[768]", arg642_1: "f32[768]", arg643_1: "f32[768, 768]", arg644_1: "f32[768]", arg645_1: "f32[768, 768]", arg646_1: "f32[768]", arg647_1: "f32[768, 768]", arg648_1: "f32[768]", arg649_1: "f32[768, 768]", arg650_1: "f32[768]", arg651_1: "f32[768]", arg652_1: "f32[768]", arg653_1: "f32[3072, 768]", arg654_1: "f32[3072]", arg655_1: "f32[768, 3072]", arg656_1: "f32[768]", arg657_1: "f32[768]", arg658_1: "f32[768]", arg659_1: "f32[768, 768]", arg660_1: "f32[768]", arg661_1: "f32[768, 768]", arg662_1: "f32[768]", arg663_1: "f32[768, 768]", arg664_1: "f32[768]", arg665_1: "f32[768, 768]", arg666_1: "f32[768]", arg667_1: "f32[768]", arg668_1: "f32[768]", arg669_1: "f32[3072, 768]", arg670_1: "f32[3072]", arg671_1: "f32[768, 3072]", arg672_1: "f32[768]", arg673_1: "f32[768]", arg674_1: "f32[768]", arg675_1: "f32[768, 768]", arg676_1: "f32[768]", arg677_1: "f32[768, 768]", arg678_1: "f32[768]", arg679_1: "f32[768, 768]", arg680_1: "f32[768]", arg681_1: "f32[768, 768]", arg682_1: "f32[768]", arg683_1: "f32[768]", arg684_1: "f32[768]", arg685_1: "f32[3072, 768]", arg686_1: "f32[3072]", arg687_1: "f32[768, 3072]", arg688_1: "f32[768]", arg689_1: "f32[768]", arg690_1: "f32[768]", arg691_1: "f32[768, 768]", arg692_1: "f32[768]", arg693_1: "f32[768, 768]", arg694_1: "f32[768]", arg695_1: "f32[768, 768]", arg696_1: "f32[768]", arg697_1: "f32[768, 768]", arg698_1: "f32[768]", arg699_1: "f32[768]", arg700_1: "f32[768]", arg701_1: "f32[3072, 768]", arg702_1: "f32[3072]", arg703_1: "f32[768, 3072]", arg704_1: "f32[768]", arg705_1: "f32[768]", arg706_1: "f32[768]", arg707_1: "f32[768, 768]", arg708_1: "f32[768]", arg709_1: "f32[768, 768]", arg710_1: "f32[768]", arg711_1: "f32[768, 768]", arg712_1: "f32[768]", arg713_1: "f32[768, 768]", arg714_1: "f32[768]", arg715_1: "f32[768]", arg716_1: "f32[768]", arg717_1: "f32[3072, 768]", arg718_1: "f32[3072]", arg719_1: "f32[768, 3072]", arg720_1: "f32[768]", arg721_1: "f32[768]", arg722_1: "f32[768]", arg723_1: "f32[768, 768]", arg724_1: "f32[768]", arg725_1: "f32[768, 768]", arg726_1: "f32[768]", arg727_1: "f32[768, 768]", arg728_1: "f32[768]", arg729_1: "f32[768, 768]", arg730_1: "f32[768]", arg731_1: "f32[768]", arg732_1: "f32[768]", arg733_1: "f32[3072, 768]", arg734_1: "f32[3072]", arg735_1: "f32[768, 3072]", arg736_1: "f32[768]", arg737_1: "f32[768]", arg738_1: "f32[768]", arg739_1: "f32[768, 768]", arg740_1: "f32[768]", arg741_1: "f32[768, 768]", arg742_1: "f32[768]", arg743_1: "f32[768, 768]", arg744_1: "f32[768]", arg745_1: "f32[768, 768]", arg746_1: "f32[768]", arg747_1: "f32[768]", arg748_1: "f32[768]", arg749_1: "f32[3072, 768]", arg750_1: "f32[3072]", arg751_1: "f32[768, 3072]", arg752_1: "f32[768]", arg753_1: "f32[768]", arg754_1: "f32[768]", arg755_1: "f32[768, 768]", arg756_1: "f32[768]", arg757_1: "f32[768, 768]", arg758_1: "f32[768]", arg759_1: "f32[768, 768]", arg760_1: "f32[768]", arg761_1: "f32[768, 768]", arg762_1: "f32[768]", arg763_1: "f32[768]", arg764_1: "f32[768]", arg765_1: "f32[3072, 768]", arg766_1: "f32[3072]", arg767_1: "f32[768, 3072]", arg768_1: "f32[768]", arg769_1: "f32[768]", arg770_1: "f32[768]", arg771_1: "f32[768, 768]", arg772_1: "f32[768]", arg773_1: "f32[768, 768]", arg774_1: "f32[768]", arg775_1: "f32[768, 768]", arg776_1: "f32[768]", arg777_1: "f32[768, 768]", arg778_1: "f32[768]", arg779_1: "f32[768]", arg780_1: "f32[768]", arg781_1: "f32[3072, 768]", arg782_1: "f32[3072]", arg783_1: "f32[768, 3072]", arg784_1: "f32[768]", arg785_1: "f32[768]", arg786_1: "f32[768]", arg787_1: "f32[768, 768]", arg788_1: "f32[768]", arg789_1: "f32[768, 768]", arg790_1: "f32[768]", arg791_1: "f32[768, 768]", arg792_1: "f32[768]", arg793_1: "f32[768, 768]", arg794_1: "f32[768]", arg795_1: "f32[768]", arg796_1: "f32[768]", arg797_1: "f32[3072, 768]", arg798_1: "f32[3072]", arg799_1: "f32[768, 3072]", arg800_1: "f32[768]", arg801_1: "f32[768]", arg802_1: "f32[768]", arg803_1: "f32[768, 768]", arg804_1: "f32[768]", arg805_1: "f32[768, 768]", arg806_1: "f32[768]", arg807_1: "f32[768, 768]", arg808_1: "f32[768]", arg809_1: "f32[768, 768]", arg810_1: "f32[768]", arg811_1: "f32[768]", arg812_1: "f32[768]", arg813_1: "f32[3072, 768]", arg814_1: "f32[3072]", arg815_1: "f32[768, 3072]", arg816_1: "f32[768]", arg817_1: "f32[768]", arg818_1: "f32[768]", arg819_1: "f32[30522]", arg820_1: "f32[768, 768]", arg821_1: "f32[768]", arg822_1: "f32[768]", arg823_1: "f32[768]", getitem_1236: "f32[]", getitem_1237: "f32[]", getitem_1238: "f32[]", getitem_1239: "f32[]", getitem_1240: "f32[]", getitem_1241: "f32[]", getitem_1242: "f32[]", getitem_1243: "f32[]", getitem_1244: "f32[]", getitem_1245: "f32[]", getitem_1246: "f32[]", getitem_1247: "f32[]", getitem_1248: "f32[]", getitem_1249: "f32[]", getitem_1250: "f32[]", getitem_1251: "f32[]", getitem_1252: "f32[]", getitem_1253: "f32[]", getitem_1254: "f32[]", getitem_1255: "f32[]", getitem_1256: "f32[]", getitem_1257: "f32[]", getitem_1258: "f32[]", getitem_1259: "f32[]", getitem_1260: "f32[]", getitem_1261: "f32[]", getitem_1262: "f32[]", getitem_1263: "f32[]", getitem_1264: "f32[]", getitem_1265: "f32[]", getitem_1266: "f32[]", getitem_1267: "f32[]", getitem_1268: "f32[]", getitem_1269: "f32[]", getitem_1270: "f32[]", getitem_1271: "f32[]", getitem_1272: "f32[]", getitem_1273: "f32[]", getitem_1274: "f32[]", getitem_1275: "f32[]", getitem_1276: "f32[]", getitem_1277: "f32[]", getitem_1278: "f32[]", getitem_1279: "f32[]", getitem_1280: "f32[]", getitem_1281: "f32[]", getitem_1282: "f32[]", getitem_1283: "f32[]", getitem_1284: "f32[]", getitem_1285: "f32[]", getitem_1286: "f32[]", getitem_1287: "f32[]", getitem_1288: "f32[]", getitem_1289: "f32[]", getitem_1290: "f32[]", getitem_1291: "f32[]", getitem_1292: "f32[]", getitem_1293: "f32[]", getitem_1294: "f32[]", getitem_1295: "f32[]", getitem_1296: "f32[]", getitem_1297: "f32[]", getitem_1298: "f32[]", getitem_1299: "f32[]", getitem_1300: "f32[]", getitem_1301: "f32[]", getitem_1302: "f32[]", getitem_1303: "f32[]", getitem_1304: "f32[]", getitem_1305: "f32[]", getitem_1306: "f32[]", getitem_1307: "f32[]", getitem_1308: "f32[]", getitem_1309: "f32[]", getitem_1310: "f32[]", getitem_1311: "f32[]", getitem_1312: "f32[]", getitem_1313: "f32[]", getitem_1314: "f32[]", getitem_1315: "f32[]", getitem_1316: "f32[]", getitem_1317: "f32[]", getitem_1318: "f32[]", getitem_1319: "f32[]", getitem_1320: "f32[]", getitem_1321: "f32[]", getitem_1322: "f32[]", getitem_1323: "f32[]", getitem_1324: "f32[]", getitem_1325: "f32[]", getitem_1326: "f32[]", getitem_1327: "f32[]", getitem_1328: "f32[]", getitem_1329: "f32[]", getitem_1330: "f32[]", getitem_1331: "f32[]", getitem_1332: "f32[]", getitem_1333: "f32[]", getitem_1334: "f32[]", getitem_1335: "f32[]", getitem_1336: "f32[]", getitem_1337: "f32[]", getitem_1338: "f32[]", getitem_1339: "f32[]", getitem_1340: "f32[]", getitem_1341: "f32[]", getitem_1342: "f32[]", getitem_1343: "f32[]", getitem_1344: "f32[]", getitem_1345: "f32[]", getitem_1346: "f32[]", getitem_1347: "f32[]", getitem_1348: "f32[]", getitem_1349: "f32[]", getitem_1350: "f32[]", getitem_1351: "f32[]", getitem_1352: "f32[]", getitem_1353: "f32[]", getitem_1354: "f32[]", getitem_1355: "f32[]", getitem_1356: "f32[]", getitem_1357: "f32[]", getitem_1358: "f32[]", getitem_1359: "f32[]", getitem_1360: "f32[]", getitem_1361: "f32[]", getitem_1362: "f32[]", getitem_1363: "f32[]", getitem_1364: "f32[]", getitem_1365: "f32[]", getitem_1366: "f32[]", getitem_1367: "f32[]", getitem_1368: "f32[]", getitem_1369: "f32[]", getitem_1370: "f32[]", getitem_1371: "f32[]", getitem_1372: "f32[]", getitem_1373: "f32[]", getitem_1374: "f32[]", getitem_1375: "f32[]", getitem_1376: "f32[]", getitem_1377: "f32[]", getitem_1378: "f32[]", getitem_1379: "f32[]", getitem_1380: "f32[]", getitem_1381: "f32[]", getitem_1382: "f32[]", getitem_1383: "f32[]", getitem_1384: "f32[]", getitem_1385: "f32[]", getitem_1386: "f32[]", getitem_1387: "f32[]", getitem_1388: "f32[]", getitem_1389: "f32[]", getitem_1390: "f32[]", getitem_1391: "f32[]", getitem_1392: "f32[]", getitem_1393: "f32[]", getitem_1394: "f32[]", getitem_1395: "f32[]", getitem_1396: "f32[]", getitem_1397: "f32[]", getitem_1398: "f32[]", getitem_1399: "f32[]", getitem_1400: "f32[]", getitem_1401: "f32[]", getitem_1402: "f32[]", getitem_1403: "f32[]", getitem_1404: "f32[]", getitem_1405: "f32[]", getitem_1406: "f32[]", getitem_1407: "f32[]", getitem_1408: "f32[]", getitem_1409: "f32[]", getitem_1410: "f32[]", getitem_1411: "f32[]", getitem_1412: "f32[]", getitem_1413: "f32[]", getitem_1414: "f32[]", getitem_1415: "f32[]", getitem_1416: "f32[]", getitem_1417: "f32[]", getitem_1418: "f32[]", getitem_1419: "f32[]", getitem_1420: "f32[]", getitem_1421: "f32[]", getitem_1422: "f32[]", getitem_1423: "f32[]", getitem_1424: "f32[]", getitem_1425: "f32[]", getitem_1426: "f32[]", getitem_1427: "f32[]", getitem_1428: "f32[]", getitem_1429: "f32[]", getitem_1430: "f32[]", getitem_1431: "f32[]", getitem_1432: "f32[]", getitem_1433: "f32[]", getitem_1434: "f32[]", getitem_1435: "f32[]", getitem_1436: "f32[]", getitem_1437: "f32[]", getitem_1438: "f32[]", getitem_1439: "f32[]", getitem_1440: "f32[]", getitem_1441: "f32[]"):
        # No stacktrace found for following nodes
        _foreach_mul_scalar = torch.ops.aten._foreach_mul.Scalar([arg210_1, arg208_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1, arg728_1, arg729_1, arg730_1, arg731_1, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, arg737_1, arg738_1, arg739_1, arg740_1, arg741_1, arg742_1, arg743_1, arg744_1, arg745_1, arg746_1, arg747_1, arg748_1, arg749_1, arg750_1, arg751_1, arg752_1, arg753_1, arg754_1, arg755_1, arg756_1, arg757_1, arg758_1, arg759_1, arg760_1, arg761_1, arg762_1, arg763_1, arg764_1, arg765_1, arg766_1, arg767_1, arg768_1, arg769_1, arg770_1, arg771_1, arg772_1, arg773_1, arg774_1, arg775_1, arg776_1, arg777_1, arg778_1, arg779_1, arg780_1, arg781_1, arg782_1, arg783_1, arg784_1, arg785_1, arg786_1, arg787_1, arg788_1, arg789_1, arg790_1, arg791_1, arg792_1, arg793_1, arg794_1, arg795_1, arg796_1, arg797_1, arg798_1, arg799_1, arg800_1, arg801_1, arg802_1, arg803_1, arg804_1, arg805_1, arg806_1, arg807_1, arg808_1, arg809_1, arg810_1, arg811_1, arg812_1, arg813_1, arg814_1, arg815_1, arg816_1, arg817_1, arg818_1, arg819_1, arg820_1, arg821_1, arg822_1, arg823_1], 0.999);  arg210_1 = arg208_1 = arg620_1 = arg621_1 = arg622_1 = arg623_1 = arg624_1 = arg625_1 = arg626_1 = arg627_1 = arg628_1 = arg629_1 = arg630_1 = arg631_1 = arg632_1 = arg633_1 = arg634_1 = arg635_1 = arg636_1 = arg637_1 = arg638_1 = arg639_1 = arg640_1 = arg641_1 = arg642_1 = arg643_1 = arg644_1 = arg645_1 = arg646_1 = arg647_1 = arg648_1 = arg649_1 = arg650_1 = arg651_1 = arg652_1 = arg653_1 = arg654_1 = arg655_1 = arg656_1 = arg657_1 = arg658_1 = arg659_1 = arg660_1 = arg661_1 = arg662_1 = arg663_1 = arg664_1 = arg665_1 = arg666_1 = arg667_1 = arg668_1 = arg669_1 = arg670_1 = arg671_1 = arg672_1 = arg673_1 = arg674_1 = arg675_1 = arg676_1 = arg677_1 = arg678_1 = arg679_1 = arg680_1 = arg681_1 = arg682_1 = arg683_1 = arg684_1 = arg685_1 = arg686_1 = arg687_1 = arg688_1 = arg689_1 = arg690_1 = arg691_1 = arg692_1 = arg693_1 = arg694_1 = arg695_1 = arg696_1 = arg697_1 = arg698_1 = arg699_1 = arg700_1 = arg701_1 = arg702_1 = arg703_1 = arg704_1 = arg705_1 = arg706_1 = arg707_1 = arg708_1 = arg709_1 = arg710_1 = arg711_1 = arg712_1 = arg713_1 = arg714_1 = arg715_1 = arg716_1 = arg717_1 = arg718_1 = arg719_1 = arg720_1 = arg721_1 = arg722_1 = arg723_1 = arg724_1 = arg725_1 = arg726_1 = arg727_1 = arg728_1 = arg729_1 = arg730_1 = arg731_1 = arg732_1 = arg733_1 = arg734_1 = arg735_1 = arg736_1 = arg737_1 = arg738_1 = arg739_1 = arg740_1 = arg741_1 = arg742_1 = arg743_1 = arg744_1 = arg745_1 = arg746_1 = arg747_1 = arg748_1 = arg749_1 = arg750_1 = arg751_1 = arg752_1 = arg753_1 = arg754_1 = arg755_1 = arg756_1 = arg757_1 = arg758_1 = arg759_1 = arg760_1 = arg761_1 = arg762_1 = arg763_1 = arg764_1 = arg765_1 = arg766_1 = arg767_1 = arg768_1 = arg769_1 = arg770_1 = arg771_1 = arg772_1 = arg773_1 = arg774_1 = arg775_1 = arg776_1 = arg777_1 = arg778_1 = arg779_1 = arg780_1 = arg781_1 = arg782_1 = arg783_1 = arg784_1 = arg785_1 = arg786_1 = arg787_1 = arg788_1 = arg789_1 = arg790_1 = arg791_1 = arg792_1 = arg793_1 = arg794_1 = arg795_1 = arg796_1 = arg797_1 = arg798_1 = arg799_1 = arg800_1 = arg801_1 = arg802_1 = arg803_1 = arg804_1 = arg805_1 = arg806_1 = arg807_1 = arg808_1 = arg809_1 = arg810_1 = arg811_1 = arg812_1 = arg813_1 = arg814_1 = arg815_1 = arg816_1 = arg817_1 = arg818_1 = arg819_1 = arg820_1 = arg821_1 = arg822_1 = arg823_1 = None
        getitem: "f32[30522, 768]" = _foreach_mul_scalar[0]
        getitem_1: "f32[512, 768]" = _foreach_mul_scalar[1]
        getitem_2: "f32[1024, 768]" = _foreach_mul_scalar[2]
        getitem_3: "f32[1024, 768]" = _foreach_mul_scalar[3]
        getitem_4: "f32[1024, 768]" = _foreach_mul_scalar[4]
        getitem_5: "f32[1024, 768]" = _foreach_mul_scalar[5]
        getitem_6: "f32[2, 768]" = _foreach_mul_scalar[6]
        getitem_7: "f32[768]" = _foreach_mul_scalar[7]
        getitem_8: "f32[768]" = _foreach_mul_scalar[8]
        getitem_9: "f32[768, 768]" = _foreach_mul_scalar[9]
        getitem_10: "f32[768]" = _foreach_mul_scalar[10]
        getitem_11: "f32[768, 768]" = _foreach_mul_scalar[11]
        getitem_12: "f32[768]" = _foreach_mul_scalar[12]
        getitem_13: "f32[768, 768]" = _foreach_mul_scalar[13]
        getitem_14: "f32[768]" = _foreach_mul_scalar[14]
        getitem_15: "f32[768, 768]" = _foreach_mul_scalar[15]
        getitem_16: "f32[768]" = _foreach_mul_scalar[16]
        getitem_17: "f32[768]" = _foreach_mul_scalar[17]
        getitem_18: "f32[768]" = _foreach_mul_scalar[18]
        getitem_19: "f32[3072, 768]" = _foreach_mul_scalar[19]
        getitem_20: "f32[3072]" = _foreach_mul_scalar[20]
        getitem_21: "f32[768, 3072]" = _foreach_mul_scalar[21]
        getitem_22: "f32[768]" = _foreach_mul_scalar[22]
        getitem_23: "f32[768]" = _foreach_mul_scalar[23]
        getitem_24: "f32[768]" = _foreach_mul_scalar[24]
        getitem_25: "f32[768, 768]" = _foreach_mul_scalar[25]
        getitem_26: "f32[768]" = _foreach_mul_scalar[26]
        getitem_27: "f32[768, 768]" = _foreach_mul_scalar[27]
        getitem_28: "f32[768]" = _foreach_mul_scalar[28]
        getitem_29: "f32[768, 768]" = _foreach_mul_scalar[29]
        getitem_30: "f32[768]" = _foreach_mul_scalar[30]
        getitem_31: "f32[768, 768]" = _foreach_mul_scalar[31]
        getitem_32: "f32[768]" = _foreach_mul_scalar[32]
        getitem_33: "f32[768]" = _foreach_mul_scalar[33]
        getitem_34: "f32[768]" = _foreach_mul_scalar[34]
        getitem_35: "f32[3072, 768]" = _foreach_mul_scalar[35]
        getitem_36: "f32[3072]" = _foreach_mul_scalar[36]
        getitem_37: "f32[768, 3072]" = _foreach_mul_scalar[37]
        getitem_38: "f32[768]" = _foreach_mul_scalar[38]
        getitem_39: "f32[768]" = _foreach_mul_scalar[39]
        getitem_40: "f32[768]" = _foreach_mul_scalar[40]
        getitem_41: "f32[768, 768]" = _foreach_mul_scalar[41]
        getitem_42: "f32[768]" = _foreach_mul_scalar[42]
        getitem_43: "f32[768, 768]" = _foreach_mul_scalar[43]
        getitem_44: "f32[768]" = _foreach_mul_scalar[44]
        getitem_45: "f32[768, 768]" = _foreach_mul_scalar[45]
        getitem_46: "f32[768]" = _foreach_mul_scalar[46]
        getitem_47: "f32[768, 768]" = _foreach_mul_scalar[47]
        getitem_48: "f32[768]" = _foreach_mul_scalar[48]
        getitem_49: "f32[768]" = _foreach_mul_scalar[49]
        getitem_50: "f32[768]" = _foreach_mul_scalar[50]
        getitem_51: "f32[3072, 768]" = _foreach_mul_scalar[51]
        getitem_52: "f32[3072]" = _foreach_mul_scalar[52]
        getitem_53: "f32[768, 3072]" = _foreach_mul_scalar[53]
        getitem_54: "f32[768]" = _foreach_mul_scalar[54]
        getitem_55: "f32[768]" = _foreach_mul_scalar[55]
        getitem_56: "f32[768]" = _foreach_mul_scalar[56]
        getitem_57: "f32[768, 768]" = _foreach_mul_scalar[57]
        getitem_58: "f32[768]" = _foreach_mul_scalar[58]
        getitem_59: "f32[768, 768]" = _foreach_mul_scalar[59]
        getitem_60: "f32[768]" = _foreach_mul_scalar[60]
        getitem_61: "f32[768, 768]" = _foreach_mul_scalar[61]
        getitem_62: "f32[768]" = _foreach_mul_scalar[62]
        getitem_63: "f32[768, 768]" = _foreach_mul_scalar[63]
        getitem_64: "f32[768]" = _foreach_mul_scalar[64]
        getitem_65: "f32[768]" = _foreach_mul_scalar[65]
        getitem_66: "f32[768]" = _foreach_mul_scalar[66]
        getitem_67: "f32[3072, 768]" = _foreach_mul_scalar[67]
        getitem_68: "f32[3072]" = _foreach_mul_scalar[68]
        getitem_69: "f32[768, 3072]" = _foreach_mul_scalar[69]
        getitem_70: "f32[768]" = _foreach_mul_scalar[70]
        getitem_71: "f32[768]" = _foreach_mul_scalar[71]
        getitem_72: "f32[768]" = _foreach_mul_scalar[72]
        getitem_73: "f32[768, 768]" = _foreach_mul_scalar[73]
        getitem_74: "f32[768]" = _foreach_mul_scalar[74]
        getitem_75: "f32[768, 768]" = _foreach_mul_scalar[75]
        getitem_76: "f32[768]" = _foreach_mul_scalar[76]
        getitem_77: "f32[768, 768]" = _foreach_mul_scalar[77]
        getitem_78: "f32[768]" = _foreach_mul_scalar[78]
        getitem_79: "f32[768, 768]" = _foreach_mul_scalar[79]
        getitem_80: "f32[768]" = _foreach_mul_scalar[80]
        getitem_81: "f32[768]" = _foreach_mul_scalar[81]
        getitem_82: "f32[768]" = _foreach_mul_scalar[82]
        getitem_83: "f32[3072, 768]" = _foreach_mul_scalar[83]
        getitem_84: "f32[3072]" = _foreach_mul_scalar[84]
        getitem_85: "f32[768, 3072]" = _foreach_mul_scalar[85]
        getitem_86: "f32[768]" = _foreach_mul_scalar[86]
        getitem_87: "f32[768]" = _foreach_mul_scalar[87]
        getitem_88: "f32[768]" = _foreach_mul_scalar[88]
        getitem_89: "f32[768, 768]" = _foreach_mul_scalar[89]
        getitem_90: "f32[768]" = _foreach_mul_scalar[90]
        getitem_91: "f32[768, 768]" = _foreach_mul_scalar[91]
        getitem_92: "f32[768]" = _foreach_mul_scalar[92]
        getitem_93: "f32[768, 768]" = _foreach_mul_scalar[93]
        getitem_94: "f32[768]" = _foreach_mul_scalar[94]
        getitem_95: "f32[768, 768]" = _foreach_mul_scalar[95]
        getitem_96: "f32[768]" = _foreach_mul_scalar[96]
        getitem_97: "f32[768]" = _foreach_mul_scalar[97]
        getitem_98: "f32[768]" = _foreach_mul_scalar[98]
        getitem_99: "f32[3072, 768]" = _foreach_mul_scalar[99]
        getitem_100: "f32[3072]" = _foreach_mul_scalar[100]
        getitem_101: "f32[768, 3072]" = _foreach_mul_scalar[101]
        getitem_102: "f32[768]" = _foreach_mul_scalar[102]
        getitem_103: "f32[768]" = _foreach_mul_scalar[103]
        getitem_104: "f32[768]" = _foreach_mul_scalar[104]
        getitem_105: "f32[768, 768]" = _foreach_mul_scalar[105]
        getitem_106: "f32[768]" = _foreach_mul_scalar[106]
        getitem_107: "f32[768, 768]" = _foreach_mul_scalar[107]
        getitem_108: "f32[768]" = _foreach_mul_scalar[108]
        getitem_109: "f32[768, 768]" = _foreach_mul_scalar[109]
        getitem_110: "f32[768]" = _foreach_mul_scalar[110]
        getitem_111: "f32[768, 768]" = _foreach_mul_scalar[111]
        getitem_112: "f32[768]" = _foreach_mul_scalar[112]
        getitem_113: "f32[768]" = _foreach_mul_scalar[113]
        getitem_114: "f32[768]" = _foreach_mul_scalar[114]
        getitem_115: "f32[3072, 768]" = _foreach_mul_scalar[115]
        getitem_116: "f32[3072]" = _foreach_mul_scalar[116]
        getitem_117: "f32[768, 3072]" = _foreach_mul_scalar[117]
        getitem_118: "f32[768]" = _foreach_mul_scalar[118]
        getitem_119: "f32[768]" = _foreach_mul_scalar[119]
        getitem_120: "f32[768]" = _foreach_mul_scalar[120]
        getitem_121: "f32[768, 768]" = _foreach_mul_scalar[121]
        getitem_122: "f32[768]" = _foreach_mul_scalar[122]
        getitem_123: "f32[768, 768]" = _foreach_mul_scalar[123]
        getitem_124: "f32[768]" = _foreach_mul_scalar[124]
        getitem_125: "f32[768, 768]" = _foreach_mul_scalar[125]
        getitem_126: "f32[768]" = _foreach_mul_scalar[126]
        getitem_127: "f32[768, 768]" = _foreach_mul_scalar[127]
        getitem_128: "f32[768]" = _foreach_mul_scalar[128]
        getitem_129: "f32[768]" = _foreach_mul_scalar[129]
        getitem_130: "f32[768]" = _foreach_mul_scalar[130]
        getitem_131: "f32[3072, 768]" = _foreach_mul_scalar[131]
        getitem_132: "f32[3072]" = _foreach_mul_scalar[132]
        getitem_133: "f32[768, 3072]" = _foreach_mul_scalar[133]
        getitem_134: "f32[768]" = _foreach_mul_scalar[134]
        getitem_135: "f32[768]" = _foreach_mul_scalar[135]
        getitem_136: "f32[768]" = _foreach_mul_scalar[136]
        getitem_137: "f32[768, 768]" = _foreach_mul_scalar[137]
        getitem_138: "f32[768]" = _foreach_mul_scalar[138]
        getitem_139: "f32[768, 768]" = _foreach_mul_scalar[139]
        getitem_140: "f32[768]" = _foreach_mul_scalar[140]
        getitem_141: "f32[768, 768]" = _foreach_mul_scalar[141]
        getitem_142: "f32[768]" = _foreach_mul_scalar[142]
        getitem_143: "f32[768, 768]" = _foreach_mul_scalar[143]
        getitem_144: "f32[768]" = _foreach_mul_scalar[144]
        getitem_145: "f32[768]" = _foreach_mul_scalar[145]
        getitem_146: "f32[768]" = _foreach_mul_scalar[146]
        getitem_147: "f32[3072, 768]" = _foreach_mul_scalar[147]
        getitem_148: "f32[3072]" = _foreach_mul_scalar[148]
        getitem_149: "f32[768, 3072]" = _foreach_mul_scalar[149]
        getitem_150: "f32[768]" = _foreach_mul_scalar[150]
        getitem_151: "f32[768]" = _foreach_mul_scalar[151]
        getitem_152: "f32[768]" = _foreach_mul_scalar[152]
        getitem_153: "f32[768, 768]" = _foreach_mul_scalar[153]
        getitem_154: "f32[768]" = _foreach_mul_scalar[154]
        getitem_155: "f32[768, 768]" = _foreach_mul_scalar[155]
        getitem_156: "f32[768]" = _foreach_mul_scalar[156]
        getitem_157: "f32[768, 768]" = _foreach_mul_scalar[157]
        getitem_158: "f32[768]" = _foreach_mul_scalar[158]
        getitem_159: "f32[768, 768]" = _foreach_mul_scalar[159]
        getitem_160: "f32[768]" = _foreach_mul_scalar[160]
        getitem_161: "f32[768]" = _foreach_mul_scalar[161]
        getitem_162: "f32[768]" = _foreach_mul_scalar[162]
        getitem_163: "f32[3072, 768]" = _foreach_mul_scalar[163]
        getitem_164: "f32[3072]" = _foreach_mul_scalar[164]
        getitem_165: "f32[768, 3072]" = _foreach_mul_scalar[165]
        getitem_166: "f32[768]" = _foreach_mul_scalar[166]
        getitem_167: "f32[768]" = _foreach_mul_scalar[167]
        getitem_168: "f32[768]" = _foreach_mul_scalar[168]
        getitem_169: "f32[768, 768]" = _foreach_mul_scalar[169]
        getitem_170: "f32[768]" = _foreach_mul_scalar[170]
        getitem_171: "f32[768, 768]" = _foreach_mul_scalar[171]
        getitem_172: "f32[768]" = _foreach_mul_scalar[172]
        getitem_173: "f32[768, 768]" = _foreach_mul_scalar[173]
        getitem_174: "f32[768]" = _foreach_mul_scalar[174]
        getitem_175: "f32[768, 768]" = _foreach_mul_scalar[175]
        getitem_176: "f32[768]" = _foreach_mul_scalar[176]
        getitem_177: "f32[768]" = _foreach_mul_scalar[177]
        getitem_178: "f32[768]" = _foreach_mul_scalar[178]
        getitem_179: "f32[3072, 768]" = _foreach_mul_scalar[179]
        getitem_180: "f32[3072]" = _foreach_mul_scalar[180]
        getitem_181: "f32[768, 3072]" = _foreach_mul_scalar[181]
        getitem_182: "f32[768]" = _foreach_mul_scalar[182]
        getitem_183: "f32[768]" = _foreach_mul_scalar[183]
        getitem_184: "f32[768]" = _foreach_mul_scalar[184]
        getitem_185: "f32[768, 768]" = _foreach_mul_scalar[185]
        getitem_186: "f32[768]" = _foreach_mul_scalar[186]
        getitem_187: "f32[768, 768]" = _foreach_mul_scalar[187]
        getitem_188: "f32[768]" = _foreach_mul_scalar[188]
        getitem_189: "f32[768, 768]" = _foreach_mul_scalar[189]
        getitem_190: "f32[768]" = _foreach_mul_scalar[190]
        getitem_191: "f32[768, 768]" = _foreach_mul_scalar[191]
        getitem_192: "f32[768]" = _foreach_mul_scalar[192]
        getitem_193: "f32[768]" = _foreach_mul_scalar[193]
        getitem_194: "f32[768]" = _foreach_mul_scalar[194]
        getitem_195: "f32[3072, 768]" = _foreach_mul_scalar[195]
        getitem_196: "f32[3072]" = _foreach_mul_scalar[196]
        getitem_197: "f32[768, 3072]" = _foreach_mul_scalar[197]
        getitem_198: "f32[768]" = _foreach_mul_scalar[198]
        getitem_199: "f32[768]" = _foreach_mul_scalar[199]
        getitem_200: "f32[768]" = _foreach_mul_scalar[200]
        getitem_201: "f32[30522]" = _foreach_mul_scalar[201]
        getitem_202: "f32[768, 768]" = _foreach_mul_scalar[202]
        getitem_203: "f32[768]" = _foreach_mul_scalar[203]
        getitem_204: "f32[768]" = _foreach_mul_scalar[204]
        getitem_205: "f32[768]" = _foreach_mul_scalar[205];  _foreach_mul_scalar = None
        _foreach_sub_scalar = torch.ops.aten._foreach_sub.Scalar([getitem_1236, getitem_1237, getitem_1238, getitem_1239, getitem_1240, getitem_1241, getitem_1242, getitem_1243, getitem_1244, getitem_1245, getitem_1246, getitem_1247, getitem_1248, getitem_1249, getitem_1250, getitem_1251, getitem_1252, getitem_1253, getitem_1254, getitem_1255, getitem_1256, getitem_1257, getitem_1258, getitem_1259, getitem_1260, getitem_1261, getitem_1262, getitem_1263, getitem_1264, getitem_1265, getitem_1266, getitem_1267, getitem_1268, getitem_1269, getitem_1270, getitem_1271, getitem_1272, getitem_1273, getitem_1274, getitem_1275, getitem_1276, getitem_1277, getitem_1278, getitem_1279, getitem_1280, getitem_1281, getitem_1282, getitem_1283, getitem_1284, getitem_1285, getitem_1286, getitem_1287, getitem_1288, getitem_1289, getitem_1290, getitem_1291, getitem_1292, getitem_1293, getitem_1294, getitem_1295, getitem_1296, getitem_1297, getitem_1298, getitem_1299, getitem_1300, getitem_1301, getitem_1302, getitem_1303, getitem_1304, getitem_1305, getitem_1306, getitem_1307, getitem_1308, getitem_1309, getitem_1310, getitem_1311, getitem_1312, getitem_1313, getitem_1314, getitem_1315, getitem_1316, getitem_1317, getitem_1318, getitem_1319, getitem_1320, getitem_1321, getitem_1322, getitem_1323, getitem_1324, getitem_1325, getitem_1326, getitem_1327, getitem_1328, getitem_1329, getitem_1330, getitem_1331, getitem_1332, getitem_1333, getitem_1334, getitem_1335, getitem_1336, getitem_1337, getitem_1338, getitem_1339, getitem_1340, getitem_1341, getitem_1342, getitem_1343, getitem_1344, getitem_1345, getitem_1346, getitem_1347, getitem_1348, getitem_1349, getitem_1350, getitem_1351, getitem_1352, getitem_1353, getitem_1354, getitem_1355, getitem_1356, getitem_1357, getitem_1358, getitem_1359, getitem_1360, getitem_1361, getitem_1362, getitem_1363, getitem_1364, getitem_1365, getitem_1366, getitem_1367, getitem_1368, getitem_1369, getitem_1370, getitem_1371, getitem_1372, getitem_1373, getitem_1374, getitem_1375, getitem_1376, getitem_1377, getitem_1378, getitem_1379, getitem_1380, getitem_1381, getitem_1382, getitem_1383, getitem_1384, getitem_1385, getitem_1386, getitem_1387, getitem_1388, getitem_1389, getitem_1390, getitem_1391, getitem_1392, getitem_1393, getitem_1394, getitem_1395, getitem_1396, getitem_1397, getitem_1398, getitem_1399, getitem_1400, getitem_1401, getitem_1402, getitem_1403, getitem_1404, getitem_1405, getitem_1406, getitem_1407, getitem_1408, getitem_1409, getitem_1410, getitem_1411, getitem_1412, getitem_1413, getitem_1414, getitem_1415, getitem_1416, getitem_1417, getitem_1418, getitem_1419, getitem_1420, getitem_1421, getitem_1422, getitem_1423, getitem_1424, getitem_1425, getitem_1426, getitem_1427, getitem_1428, getitem_1429, getitem_1430, getitem_1431, getitem_1432, getitem_1433, getitem_1434, getitem_1435, getitem_1436, getitem_1437, getitem_1438, getitem_1439, getitem_1440, getitem_1441], 1);  getitem_1236 = getitem_1237 = getitem_1238 = getitem_1239 = getitem_1240 = getitem_1241 = getitem_1242 = getitem_1243 = getitem_1244 = getitem_1245 = getitem_1246 = getitem_1247 = getitem_1248 = getitem_1249 = getitem_1250 = getitem_1251 = getitem_1252 = getitem_1253 = getitem_1254 = getitem_1255 = getitem_1256 = getitem_1257 = getitem_1258 = getitem_1259 = getitem_1260 = getitem_1261 = getitem_1262 = getitem_1263 = getitem_1264 = getitem_1265 = getitem_1266 = getitem_1267 = getitem_1268 = getitem_1269 = getitem_1270 = getitem_1271 = getitem_1272 = getitem_1273 = getitem_1274 = getitem_1275 = getitem_1276 = getitem_1277 = getitem_1278 = getitem_1279 = getitem_1280 = getitem_1281 = getitem_1282 = getitem_1283 = getitem_1284 = getitem_1285 = getitem_1286 = getitem_1287 = getitem_1288 = getitem_1289 = getitem_1290 = getitem_1291 = getitem_1292 = getitem_1293 = getitem_1294 = getitem_1295 = getitem_1296 = getitem_1297 = getitem_1298 = getitem_1299 = getitem_1300 = getitem_1301 = getitem_1302 = getitem_1303 = getitem_1304 = getitem_1305 = getitem_1306 = getitem_1307 = getitem_1308 = getitem_1309 = getitem_1310 = getitem_1311 = getitem_1312 = getitem_1313 = getitem_1314 = getitem_1315 = getitem_1316 = getitem_1317 = getitem_1318 = getitem_1319 = getitem_1320 = getitem_1321 = getitem_1322 = getitem_1323 = getitem_1324 = getitem_1325 = getitem_1326 = getitem_1327 = getitem_1328 = getitem_1329 = getitem_1330 = getitem_1331 = getitem_1332 = getitem_1333 = getitem_1334 = getitem_1335 = getitem_1336 = getitem_1337 = getitem_1338 = getitem_1339 = getitem_1340 = getitem_1341 = getitem_1342 = getitem_1343 = getitem_1344 = getitem_1345 = getitem_1346 = getitem_1347 = getitem_1348 = getitem_1349 = getitem_1350 = getitem_1351 = getitem_1352 = getitem_1353 = getitem_1354 = getitem_1355 = getitem_1356 = getitem_1357 = getitem_1358 = getitem_1359 = getitem_1360 = getitem_1361 = getitem_1362 = getitem_1363 = getitem_1364 = getitem_1365 = getitem_1366 = getitem_1367 = getitem_1368 = getitem_1369 = getitem_1370 = getitem_1371 = getitem_1372 = getitem_1373 = getitem_1374 = getitem_1375 = getitem_1376 = getitem_1377 = getitem_1378 = getitem_1379 = getitem_1380 = getitem_1381 = getitem_1382 = getitem_1383 = getitem_1384 = getitem_1385 = getitem_1386 = getitem_1387 = getitem_1388 = getitem_1389 = getitem_1390 = getitem_1391 = getitem_1392 = getitem_1393 = getitem_1394 = getitem_1395 = getitem_1396 = getitem_1397 = getitem_1398 = getitem_1399 = getitem_1400 = getitem_1401 = getitem_1402 = getitem_1403 = getitem_1404 = getitem_1405 = getitem_1406 = getitem_1407 = getitem_1408 = getitem_1409 = getitem_1410 = getitem_1411 = getitem_1412 = getitem_1413 = getitem_1414 = getitem_1415 = getitem_1416 = getitem_1417 = getitem_1418 = getitem_1419 = getitem_1420 = getitem_1421 = getitem_1422 = getitem_1423 = getitem_1424 = getitem_1425 = getitem_1426 = getitem_1427 = getitem_1428 = getitem_1429 = getitem_1430 = getitem_1431 = getitem_1432 = getitem_1433 = getitem_1434 = getitem_1435 = getitem_1436 = getitem_1437 = getitem_1438 = getitem_1439 = getitem_1440 = getitem_1441 = None
        getitem_1442: "f32[]" = _foreach_sub_scalar[0]
        getitem_1443: "f32[]" = _foreach_sub_scalar[1]
        getitem_1444: "f32[]" = _foreach_sub_scalar[2]
        getitem_1445: "f32[]" = _foreach_sub_scalar[3]
        getitem_1446: "f32[]" = _foreach_sub_scalar[4]
        getitem_1447: "f32[]" = _foreach_sub_scalar[5]
        getitem_1448: "f32[]" = _foreach_sub_scalar[6]
        getitem_1449: "f32[]" = _foreach_sub_scalar[7]
        getitem_1450: "f32[]" = _foreach_sub_scalar[8]
        getitem_1451: "f32[]" = _foreach_sub_scalar[9]
        getitem_1452: "f32[]" = _foreach_sub_scalar[10]
        getitem_1453: "f32[]" = _foreach_sub_scalar[11]
        getitem_1454: "f32[]" = _foreach_sub_scalar[12]
        getitem_1455: "f32[]" = _foreach_sub_scalar[13]
        getitem_1456: "f32[]" = _foreach_sub_scalar[14]
        getitem_1457: "f32[]" = _foreach_sub_scalar[15]
        getitem_1458: "f32[]" = _foreach_sub_scalar[16]
        getitem_1459: "f32[]" = _foreach_sub_scalar[17]
        getitem_1460: "f32[]" = _foreach_sub_scalar[18]
        getitem_1461: "f32[]" = _foreach_sub_scalar[19]
        getitem_1462: "f32[]" = _foreach_sub_scalar[20]
        getitem_1463: "f32[]" = _foreach_sub_scalar[21]
        getitem_1464: "f32[]" = _foreach_sub_scalar[22]
        getitem_1465: "f32[]" = _foreach_sub_scalar[23]
        getitem_1466: "f32[]" = _foreach_sub_scalar[24]
        getitem_1467: "f32[]" = _foreach_sub_scalar[25]
        getitem_1468: "f32[]" = _foreach_sub_scalar[26]
        getitem_1469: "f32[]" = _foreach_sub_scalar[27]
        getitem_1470: "f32[]" = _foreach_sub_scalar[28]
        getitem_1471: "f32[]" = _foreach_sub_scalar[29]
        getitem_1472: "f32[]" = _foreach_sub_scalar[30]
        getitem_1473: "f32[]" = _foreach_sub_scalar[31]
        getitem_1474: "f32[]" = _foreach_sub_scalar[32]
        getitem_1475: "f32[]" = _foreach_sub_scalar[33]
        getitem_1476: "f32[]" = _foreach_sub_scalar[34]
        getitem_1477: "f32[]" = _foreach_sub_scalar[35]
        getitem_1478: "f32[]" = _foreach_sub_scalar[36]
        getitem_1479: "f32[]" = _foreach_sub_scalar[37]
        getitem_1480: "f32[]" = _foreach_sub_scalar[38]
        getitem_1481: "f32[]" = _foreach_sub_scalar[39]
        getitem_1482: "f32[]" = _foreach_sub_scalar[40]
        getitem_1483: "f32[]" = _foreach_sub_scalar[41]
        getitem_1484: "f32[]" = _foreach_sub_scalar[42]
        getitem_1485: "f32[]" = _foreach_sub_scalar[43]
        getitem_1486: "f32[]" = _foreach_sub_scalar[44]
        getitem_1487: "f32[]" = _foreach_sub_scalar[45]
        getitem_1488: "f32[]" = _foreach_sub_scalar[46]
        getitem_1489: "f32[]" = _foreach_sub_scalar[47]
        getitem_1490: "f32[]" = _foreach_sub_scalar[48]
        getitem_1491: "f32[]" = _foreach_sub_scalar[49]
        getitem_1492: "f32[]" = _foreach_sub_scalar[50]
        getitem_1493: "f32[]" = _foreach_sub_scalar[51]
        getitem_1494: "f32[]" = _foreach_sub_scalar[52]
        getitem_1495: "f32[]" = _foreach_sub_scalar[53]
        getitem_1496: "f32[]" = _foreach_sub_scalar[54]
        getitem_1497: "f32[]" = _foreach_sub_scalar[55]
        getitem_1498: "f32[]" = _foreach_sub_scalar[56]
        getitem_1499: "f32[]" = _foreach_sub_scalar[57]
        getitem_1500: "f32[]" = _foreach_sub_scalar[58]
        getitem_1501: "f32[]" = _foreach_sub_scalar[59]
        getitem_1502: "f32[]" = _foreach_sub_scalar[60]
        getitem_1503: "f32[]" = _foreach_sub_scalar[61]
        getitem_1504: "f32[]" = _foreach_sub_scalar[62]
        getitem_1505: "f32[]" = _foreach_sub_scalar[63]
        getitem_1506: "f32[]" = _foreach_sub_scalar[64]
        getitem_1507: "f32[]" = _foreach_sub_scalar[65]
        getitem_1508: "f32[]" = _foreach_sub_scalar[66]
        getitem_1509: "f32[]" = _foreach_sub_scalar[67]
        getitem_1510: "f32[]" = _foreach_sub_scalar[68]
        getitem_1511: "f32[]" = _foreach_sub_scalar[69]
        getitem_1512: "f32[]" = _foreach_sub_scalar[70]
        getitem_1513: "f32[]" = _foreach_sub_scalar[71]
        getitem_1514: "f32[]" = _foreach_sub_scalar[72]
        getitem_1515: "f32[]" = _foreach_sub_scalar[73]
        getitem_1516: "f32[]" = _foreach_sub_scalar[74]
        getitem_1517: "f32[]" = _foreach_sub_scalar[75]
        getitem_1518: "f32[]" = _foreach_sub_scalar[76]
        getitem_1519: "f32[]" = _foreach_sub_scalar[77]
        getitem_1520: "f32[]" = _foreach_sub_scalar[78]
        getitem_1521: "f32[]" = _foreach_sub_scalar[79]
        getitem_1522: "f32[]" = _foreach_sub_scalar[80]
        getitem_1523: "f32[]" = _foreach_sub_scalar[81]
        getitem_1524: "f32[]" = _foreach_sub_scalar[82]
        getitem_1525: "f32[]" = _foreach_sub_scalar[83]
        getitem_1526: "f32[]" = _foreach_sub_scalar[84]
        getitem_1527: "f32[]" = _foreach_sub_scalar[85]
        getitem_1528: "f32[]" = _foreach_sub_scalar[86]
        getitem_1529: "f32[]" = _foreach_sub_scalar[87]
        getitem_1530: "f32[]" = _foreach_sub_scalar[88]
        getitem_1531: "f32[]" = _foreach_sub_scalar[89]
        getitem_1532: "f32[]" = _foreach_sub_scalar[90]
        getitem_1533: "f32[]" = _foreach_sub_scalar[91]
        getitem_1534: "f32[]" = _foreach_sub_scalar[92]
        getitem_1535: "f32[]" = _foreach_sub_scalar[93]
        getitem_1536: "f32[]" = _foreach_sub_scalar[94]
        getitem_1537: "f32[]" = _foreach_sub_scalar[95]
        getitem_1538: "f32[]" = _foreach_sub_scalar[96]
        getitem_1539: "f32[]" = _foreach_sub_scalar[97]
        getitem_1540: "f32[]" = _foreach_sub_scalar[98]
        getitem_1541: "f32[]" = _foreach_sub_scalar[99]
        getitem_1542: "f32[]" = _foreach_sub_scalar[100]
        getitem_1543: "f32[]" = _foreach_sub_scalar[101]
        getitem_1544: "f32[]" = _foreach_sub_scalar[102]
        getitem_1545: "f32[]" = _foreach_sub_scalar[103]
        getitem_1546: "f32[]" = _foreach_sub_scalar[104]
        getitem_1547: "f32[]" = _foreach_sub_scalar[105]
        getitem_1548: "f32[]" = _foreach_sub_scalar[106]
        getitem_1549: "f32[]" = _foreach_sub_scalar[107]
        getitem_1550: "f32[]" = _foreach_sub_scalar[108]
        getitem_1551: "f32[]" = _foreach_sub_scalar[109]
        getitem_1552: "f32[]" = _foreach_sub_scalar[110]
        getitem_1553: "f32[]" = _foreach_sub_scalar[111]
        getitem_1554: "f32[]" = _foreach_sub_scalar[112]
        getitem_1555: "f32[]" = _foreach_sub_scalar[113]
        getitem_1556: "f32[]" = _foreach_sub_scalar[114]
        getitem_1557: "f32[]" = _foreach_sub_scalar[115]
        getitem_1558: "f32[]" = _foreach_sub_scalar[116]
        getitem_1559: "f32[]" = _foreach_sub_scalar[117]
        getitem_1560: "f32[]" = _foreach_sub_scalar[118]
        getitem_1561: "f32[]" = _foreach_sub_scalar[119]
        getitem_1562: "f32[]" = _foreach_sub_scalar[120]
        getitem_1563: "f32[]" = _foreach_sub_scalar[121]
        getitem_1564: "f32[]" = _foreach_sub_scalar[122]
        getitem_1565: "f32[]" = _foreach_sub_scalar[123]
        getitem_1566: "f32[]" = _foreach_sub_scalar[124]
        getitem_1567: "f32[]" = _foreach_sub_scalar[125]
        getitem_1568: "f32[]" = _foreach_sub_scalar[126]
        getitem_1569: "f32[]" = _foreach_sub_scalar[127]
        getitem_1570: "f32[]" = _foreach_sub_scalar[128]
        getitem_1571: "f32[]" = _foreach_sub_scalar[129]
        getitem_1572: "f32[]" = _foreach_sub_scalar[130]
        getitem_1573: "f32[]" = _foreach_sub_scalar[131]
        getitem_1574: "f32[]" = _foreach_sub_scalar[132]
        getitem_1575: "f32[]" = _foreach_sub_scalar[133]
        getitem_1576: "f32[]" = _foreach_sub_scalar[134]
        getitem_1577: "f32[]" = _foreach_sub_scalar[135]
        getitem_1578: "f32[]" = _foreach_sub_scalar[136]
        getitem_1579: "f32[]" = _foreach_sub_scalar[137]
        getitem_1580: "f32[]" = _foreach_sub_scalar[138]
        getitem_1581: "f32[]" = _foreach_sub_scalar[139]
        getitem_1582: "f32[]" = _foreach_sub_scalar[140]
        getitem_1583: "f32[]" = _foreach_sub_scalar[141]
        getitem_1584: "f32[]" = _foreach_sub_scalar[142]
        getitem_1585: "f32[]" = _foreach_sub_scalar[143]
        getitem_1586: "f32[]" = _foreach_sub_scalar[144]
        getitem_1587: "f32[]" = _foreach_sub_scalar[145]
        getitem_1588: "f32[]" = _foreach_sub_scalar[146]
        getitem_1589: "f32[]" = _foreach_sub_scalar[147]
        getitem_1590: "f32[]" = _foreach_sub_scalar[148]
        getitem_1591: "f32[]" = _foreach_sub_scalar[149]
        getitem_1592: "f32[]" = _foreach_sub_scalar[150]
        getitem_1593: "f32[]" = _foreach_sub_scalar[151]
        getitem_1594: "f32[]" = _foreach_sub_scalar[152]
        getitem_1595: "f32[]" = _foreach_sub_scalar[153]
        getitem_1596: "f32[]" = _foreach_sub_scalar[154]
        getitem_1597: "f32[]" = _foreach_sub_scalar[155]
        getitem_1598: "f32[]" = _foreach_sub_scalar[156]
        getitem_1599: "f32[]" = _foreach_sub_scalar[157]
        getitem_1600: "f32[]" = _foreach_sub_scalar[158]
        getitem_1601: "f32[]" = _foreach_sub_scalar[159]
        getitem_1602: "f32[]" = _foreach_sub_scalar[160]
        getitem_1603: "f32[]" = _foreach_sub_scalar[161]
        getitem_1604: "f32[]" = _foreach_sub_scalar[162]
        getitem_1605: "f32[]" = _foreach_sub_scalar[163]
        getitem_1606: "f32[]" = _foreach_sub_scalar[164]
        getitem_1607: "f32[]" = _foreach_sub_scalar[165]
        getitem_1608: "f32[]" = _foreach_sub_scalar[166]
        getitem_1609: "f32[]" = _foreach_sub_scalar[167]
        getitem_1610: "f32[]" = _foreach_sub_scalar[168]
        getitem_1611: "f32[]" = _foreach_sub_scalar[169]
        getitem_1612: "f32[]" = _foreach_sub_scalar[170]
        getitem_1613: "f32[]" = _foreach_sub_scalar[171]
        getitem_1614: "f32[]" = _foreach_sub_scalar[172]
        getitem_1615: "f32[]" = _foreach_sub_scalar[173]
        getitem_1616: "f32[]" = _foreach_sub_scalar[174]
        getitem_1617: "f32[]" = _foreach_sub_scalar[175]
        getitem_1618: "f32[]" = _foreach_sub_scalar[176]
        getitem_1619: "f32[]" = _foreach_sub_scalar[177]
        getitem_1620: "f32[]" = _foreach_sub_scalar[178]
        getitem_1621: "f32[]" = _foreach_sub_scalar[179]
        getitem_1622: "f32[]" = _foreach_sub_scalar[180]
        getitem_1623: "f32[]" = _foreach_sub_scalar[181]
        getitem_1624: "f32[]" = _foreach_sub_scalar[182]
        getitem_1625: "f32[]" = _foreach_sub_scalar[183]
        getitem_1626: "f32[]" = _foreach_sub_scalar[184]
        getitem_1627: "f32[]" = _foreach_sub_scalar[185]
        getitem_1628: "f32[]" = _foreach_sub_scalar[186]
        getitem_1629: "f32[]" = _foreach_sub_scalar[187]
        getitem_1630: "f32[]" = _foreach_sub_scalar[188]
        getitem_1631: "f32[]" = _foreach_sub_scalar[189]
        getitem_1632: "f32[]" = _foreach_sub_scalar[190]
        getitem_1633: "f32[]" = _foreach_sub_scalar[191]
        getitem_1634: "f32[]" = _foreach_sub_scalar[192]
        getitem_1635: "f32[]" = _foreach_sub_scalar[193]
        getitem_1636: "f32[]" = _foreach_sub_scalar[194]
        getitem_1637: "f32[]" = _foreach_sub_scalar[195]
        getitem_1638: "f32[]" = _foreach_sub_scalar[196]
        getitem_1639: "f32[]" = _foreach_sub_scalar[197]
        getitem_1640: "f32[]" = _foreach_sub_scalar[198]
        getitem_1641: "f32[]" = _foreach_sub_scalar[199]
        getitem_1642: "f32[]" = _foreach_sub_scalar[200]
        getitem_1643: "f32[]" = _foreach_sub_scalar[201]
        getitem_1644: "f32[]" = _foreach_sub_scalar[202]
        getitem_1645: "f32[]" = _foreach_sub_scalar[203]
        getitem_1646: "f32[]" = _foreach_sub_scalar[204]
        getitem_1647: "f32[]" = _foreach_sub_scalar[205];  _foreach_sub_scalar = None
        return (getitem, getitem_1, getitem_2, getitem_3, getitem_4, getitem_5, getitem_6, getitem_7, getitem_8, getitem_9, getitem_10, getitem_11, getitem_12, getitem_13, getitem_14, getitem_15, getitem_16, getitem_17, getitem_18, getitem_19, getitem_20, getitem_21, getitem_22, getitem_23, getitem_24, getitem_25, getitem_26, getitem_27, getitem_28, getitem_29, getitem_30, getitem_31, getitem_32, getitem_33, getitem_34, getitem_35, getitem_36, getitem_37, getitem_38, getitem_39, getitem_40, getitem_41, getitem_42, getitem_43, getitem_44, getitem_45, getitem_46, getitem_47, getitem_48, getitem_49, getitem_50, getitem_51, getitem_52, getitem_53, getitem_54, getitem_55, getitem_56, getitem_57, getitem_58, getitem_59, getitem_60, getitem_61, getitem_62, getitem_63, getitem_64, getitem_65, getitem_66, getitem_67, getitem_68, getitem_69, getitem_70, getitem_71, getitem_72, getitem_73, getitem_74, getitem_75, getitem_76, getitem_77, getitem_78, getitem_79, getitem_80, getitem_81, getitem_82, getitem_83, getitem_84, getitem_85, getitem_86, getitem_87, getitem_88, getitem_89, getitem_90, getitem_91, getitem_92, getitem_93, getitem_94, getitem_95, getitem_96, getitem_97, getitem_98, getitem_99, getitem_100, getitem_101, getitem_102, getitem_103, getitem_104, getitem_105, getitem_106, getitem_107, getitem_108, getitem_109, getitem_110, getitem_111, getitem_112, getitem_113, getitem_114, getitem_115, getitem_116, getitem_117, getitem_118, getitem_119, getitem_120, getitem_121, getitem_122, getitem_123, getitem_124, getitem_125, getitem_126, getitem_127, getitem_128, getitem_129, getitem_130, getitem_131, getitem_132, getitem_133, getitem_134, getitem_135, getitem_136, getitem_137, getitem_138, getitem_139, getitem_140, getitem_141, getitem_142, getitem_143, getitem_144, getitem_145, getitem_146, getitem_147, getitem_148, getitem_149, getitem_150, getitem_151, getitem_152, getitem_153, getitem_154, getitem_155, getitem_156, getitem_157, getitem_158, getitem_159, getitem_160, getitem_161, getitem_162, getitem_163, getitem_164, getitem_165, getitem_166, getitem_167, getitem_168, getitem_169, getitem_170, getitem_171, getitem_172, getitem_173, getitem_174, getitem_175, getitem_176, getitem_177, getitem_178, getitem_179, getitem_180, getitem_181, getitem_182, getitem_183, getitem_184, getitem_185, getitem_186, getitem_187, getitem_188, getitem_189, getitem_190, getitem_191, getitem_192, getitem_193, getitem_194, getitem_195, getitem_196, getitem_197, getitem_198, getitem_199, getitem_200, getitem_201, getitem_202, getitem_203, getitem_204, getitem_205, getitem_1442, getitem_1443, getitem_1444, getitem_1445, getitem_1446, getitem_1447, getitem_1448, getitem_1449, getitem_1450, getitem_1451, getitem_1452, getitem_1453, getitem_1454, getitem_1455, getitem_1456, getitem_1457, getitem_1458, getitem_1459, getitem_1460, getitem_1461, getitem_1462, getitem_1463, getitem_1464, getitem_1465, getitem_1466, getitem_1467, getitem_1468, getitem_1469, getitem_1470, getitem_1471, getitem_1472, getitem_1473, getitem_1474, getitem_1475, getitem_1476, getitem_1477, getitem_1478, getitem_1479, getitem_1480, getitem_1481, getitem_1482, getitem_1483, getitem_1484, getitem_1485, getitem_1486, getitem_1487, getitem_1488, getitem_1489, getitem_1490, getitem_1491, getitem_1492, getitem_1493, getitem_1494, getitem_1495, getitem_1496, getitem_1497, getitem_1498, getitem_1499, getitem_1500, getitem_1501, getitem_1502, getitem_1503, getitem_1504, getitem_1505, getitem_1506, getitem_1507, getitem_1508, getitem_1509, getitem_1510, getitem_1511, getitem_1512, getitem_1513, getitem_1514, getitem_1515, getitem_1516, getitem_1517, getitem_1518, getitem_1519, getitem_1520, getitem_1521, getitem_1522, getitem_1523, getitem_1524, getitem_1525, getitem_1526, getitem_1527, getitem_1528, getitem_1529, getitem_1530, getitem_1531, getitem_1532, getitem_1533, getitem_1534, getitem_1535, getitem_1536, getitem_1537, getitem_1538, getitem_1539, getitem_1540, getitem_1541, getitem_1542, getitem_1543, getitem_1544, getitem_1545, getitem_1546, getitem_1547, getitem_1548, getitem_1549, getitem_1550, getitem_1551, getitem_1552, getitem_1553, getitem_1554, getitem_1555, getitem_1556, getitem_1557, getitem_1558, getitem_1559, getitem_1560, getitem_1561, getitem_1562, getitem_1563, getitem_1564, getitem_1565, getitem_1566, getitem_1567, getitem_1568, getitem_1569, getitem_1570, getitem_1571, getitem_1572, getitem_1573, getitem_1574, getitem_1575, getitem_1576, getitem_1577, getitem_1578, getitem_1579, getitem_1580, getitem_1581, getitem_1582, getitem_1583, getitem_1584, getitem_1585, getitem_1586, getitem_1587, getitem_1588, getitem_1589, getitem_1590, getitem_1591, getitem_1592, getitem_1593, getitem_1594, getitem_1595, getitem_1596, getitem_1597, getitem_1598, getitem_1599, getitem_1600, getitem_1601, getitem_1602, getitem_1603, getitem_1604, getitem_1605, getitem_1606, getitem_1607, getitem_1608, getitem_1609, getitem_1610, getitem_1611, getitem_1612, getitem_1613, getitem_1614, getitem_1615, getitem_1616, getitem_1617, getitem_1618, getitem_1619, getitem_1620, getitem_1621, getitem_1622, getitem_1623, getitem_1624, getitem_1625, getitem_1626, getitem_1627, getitem_1628, getitem_1629, getitem_1630, getitem_1631, getitem_1632, getitem_1633, getitem_1634, getitem_1635, getitem_1636, getitem_1637, getitem_1638, getitem_1639, getitem_1640, getitem_1641, getitem_1642, getitem_1643, getitem_1644, getitem_1645, getitem_1646, getitem_1647)


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
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
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
