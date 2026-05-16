"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s9_g77
Pattern hash: 89ca58cf6ccb
Shape hash: f4ea4807
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_903: "f32[768]", getitem_904: "f32[50, 768]", getitem_905: "f32[768, 512]", getitem_906: "f32[768, 3, 32, 32]", getitem_907: "f32[768]", getitem_908: "f32[768]", getitem_909: "f32[2304, 768]", getitem_910: "f32[2304]", getitem_911: "f32[768, 768]", getitem_912: "f32[768]", getitem_913: "f32[3072, 768]", getitem_914: "f32[3072]", getitem_915: "f32[768, 3072]", getitem_916: "f32[768]", getitem_917: "f32[768]", getitem_918: "f32[768]", getitem_919: "f32[768]", getitem_920: "f32[768]", getitem_921: "f32[2304, 768]", getitem_922: "f32[2304]", getitem_923: "f32[768, 768]", getitem_924: "f32[768]", getitem_925: "f32[3072, 768]", getitem_926: "f32[3072]", getitem_927: "f32[768, 3072]", getitem_928: "f32[768]", getitem_929: "f32[768]", getitem_930: "f32[768]", getitem_931: "f32[768]", getitem_932: "f32[768]", getitem_933: "f32[2304, 768]", getitem_934: "f32[2304]", getitem_935: "f32[768, 768]", getitem_936: "f32[768]", getitem_937: "f32[3072, 768]", getitem_938: "f32[3072]", getitem_939: "f32[768, 3072]", getitem_940: "f32[768]", getitem_941: "f32[768]", getitem_942: "f32[768]", getitem_943: "f32[768]", getitem_944: "f32[768]", getitem_945: "f32[2304, 768]", getitem_946: "f32[2304]", getitem_947: "f32[768, 768]", getitem_948: "f32[768]", getitem_949: "f32[3072, 768]", getitem_950: "f32[3072]", getitem_951: "f32[768, 3072]", getitem_952: "f32[768]", getitem_953: "f32[768]", getitem_954: "f32[768]", getitem_955: "f32[768]", getitem_956: "f32[768]", getitem_957: "f32[2304, 768]", getitem_958: "f32[2304]", getitem_959: "f32[768, 768]", getitem_960: "f32[768]", getitem_961: "f32[3072, 768]", getitem_962: "f32[3072]", getitem_963: "f32[768, 3072]", getitem_964: "f32[768]", getitem_965: "f32[768]", getitem_966: "f32[768]", getitem_967: "f32[768]", getitem_968: "f32[768]", getitem_969: "f32[2304, 768]", getitem_970: "f32[2304]", getitem_971: "f32[768, 768]", getitem_972: "f32[768]", getitem_973: "f32[3072, 768]", getitem_974: "f32[3072]", getitem_975: "f32[768, 3072]", getitem_976: "f32[768]", getitem_977: "f32[768]", getitem_978: "f32[768]", getitem_979: "f32[768]", getitem_980: "f32[768]", getitem_981: "f32[2304, 768]", getitem_982: "f32[2304]", getitem_983: "f32[768, 768]", getitem_984: "f32[768]", getitem_985: "f32[3072, 768]", getitem_986: "f32[3072]", getitem_987: "f32[768, 3072]", getitem_988: "f32[768]", getitem_989: "f32[768]", getitem_990: "f32[768]", getitem_991: "f32[768]", getitem_992: "f32[768]", getitem_993: "f32[2304, 768]", getitem_994: "f32[2304]", getitem_995: "f32[768, 768]", getitem_996: "f32[768]", getitem_997: "f32[3072, 768]", getitem_998: "f32[3072]", getitem_999: "f32[768, 3072]", getitem_1000: "f32[768]", getitem_1001: "f32[768]", getitem_1002: "f32[768]", getitem_1003: "f32[768]", getitem_1004: "f32[768]", getitem_1005: "f32[2304, 768]", getitem_1006: "f32[2304]", getitem_1007: "f32[768, 768]", getitem_1008: "f32[768]", getitem_1009: "f32[3072, 768]", getitem_1010: "f32[3072]", getitem_1011: "f32[768, 3072]", getitem_1012: "f32[768]", getitem_1013: "f32[768]", getitem_1014: "f32[768]", getitem_1015: "f32[768]", getitem_1016: "f32[768]", getitem_1017: "f32[2304, 768]", getitem_1018: "f32[2304]", getitem_1019: "f32[768, 768]", getitem_1020: "f32[768]", getitem_1021: "f32[3072, 768]", getitem_1022: "f32[3072]", getitem_1023: "f32[768, 3072]", getitem_1024: "f32[768]", getitem_1025: "f32[768]", getitem_1026: "f32[768]", getitem_1027: "f32[768]", getitem_1028: "f32[768]", getitem_1029: "f32[2304, 768]", getitem_1030: "f32[2304]", getitem_1031: "f32[768, 768]", getitem_1032: "f32[768]", getitem_1033: "f32[3072, 768]", getitem_1034: "f32[3072]", getitem_1035: "f32[768, 3072]", getitem_1036: "f32[768]", getitem_1037: "f32[768]", getitem_1038: "f32[768]", getitem_1039: "f32[768]", getitem_1040: "f32[768]", getitem_1041: "f32[2304, 768]", getitem_1042: "f32[2304]", getitem_1043: "f32[768, 768]", getitem_1044: "f32[768]", getitem_1045: "f32[3072, 768]", getitem_1046: "f32[3072]", getitem_1047: "f32[768, 3072]", getitem_1048: "f32[768]", getitem_1049: "f32[768]", getitem_1050: "f32[768]", getitem_1051: "f32[768]", getitem_1052: "f32[768]", getitem_1053: "f32[768]", getitem_1054: "f32[768]", getitem_1055: "f32[77, 512]", getitem_1056: "f32[49408, 512]", getitem_1057: "f32[1536, 512]", getitem_1058: "f32[1536]", getitem_1059: "f32[512, 512]", getitem_1060: "f32[512]", getitem_1061: "f32[2048, 512]", getitem_1062: "f32[2048]", getitem_1063: "f32[512, 2048]", getitem_1064: "f32[512]", getitem_1065: "f32[512]", getitem_1066: "f32[512]", getitem_1067: "f32[512]", getitem_1068: "f32[512]", getitem_1069: "f32[1536, 512]", getitem_1070: "f32[1536]", getitem_1071: "f32[512, 512]", getitem_1072: "f32[512]", getitem_1073: "f32[2048, 512]", getitem_1074: "f32[2048]", getitem_1075: "f32[512, 2048]", getitem_1076: "f32[512]", getitem_1077: "f32[512]", getitem_1078: "f32[512]", getitem_1079: "f32[512]", getitem_1080: "f32[512]", getitem_1081: "f32[1536, 512]", getitem_1082: "f32[1536]", getitem_1083: "f32[512, 512]", getitem_1084: "f32[512]", getitem_1085: "f32[2048, 512]", getitem_1086: "f32[2048]", getitem_1087: "f32[512, 2048]", getitem_1088: "f32[512]", getitem_1089: "f32[512]", getitem_1090: "f32[512]", getitem_1091: "f32[512]", getitem_1092: "f32[512]", getitem_1093: "f32[1536, 512]", getitem_1094: "f32[1536]", getitem_1095: "f32[512, 512]", getitem_1096: "f32[512]", getitem_1097: "f32[2048, 512]", getitem_1098: "f32[2048]", getitem_1099: "f32[512, 2048]", getitem_1100: "f32[512]", getitem_1101: "f32[512]", getitem_1102: "f32[512]", getitem_1103: "f32[512]", getitem_1104: "f32[512]", getitem_1105: "f32[1536, 512]", getitem_1106: "f32[1536]", getitem_1107: "f32[512, 512]", getitem_1108: "f32[512]", getitem_1109: "f32[2048, 512]", getitem_1110: "f32[2048]", getitem_1111: "f32[512, 2048]", getitem_1112: "f32[512]", getitem_1113: "f32[512]", getitem_1114: "f32[512]", getitem_1115: "f32[512]", getitem_1116: "f32[512]", getitem_1117: "f32[1536, 512]", getitem_1118: "f32[1536]", getitem_1119: "f32[512, 512]", getitem_1120: "f32[512]", getitem_1121: "f32[2048, 512]", getitem_1122: "f32[2048]", getitem_1123: "f32[512, 2048]", getitem_1124: "f32[512]", getitem_1125: "f32[512]", getitem_1126: "f32[512]", getitem_1127: "f32[512]", getitem_1128: "f32[512]", getitem_1129: "f32[1536, 512]", getitem_1130: "f32[1536]", getitem_1131: "f32[512, 512]", getitem_1132: "f32[512]", getitem_1133: "f32[2048, 512]", getitem_1134: "f32[2048]", getitem_1135: "f32[512, 2048]", getitem_1136: "f32[512]", getitem_1137: "f32[512]", getitem_1138: "f32[512]", getitem_1139: "f32[512]", getitem_1140: "f32[512]", getitem_1141: "f32[1536, 512]", getitem_1142: "f32[1536]", getitem_1143: "f32[512, 512]", getitem_1144: "f32[512]", getitem_1145: "f32[2048, 512]", getitem_1146: "f32[2048]", getitem_1147: "f32[512, 2048]", getitem_1148: "f32[512]", getitem_1149: "f32[512]", getitem_1150: "f32[512]", getitem_1151: "f32[512]", getitem_1152: "f32[512]", getitem_1153: "f32[1536, 512]", getitem_1154: "f32[1536]", getitem_1155: "f32[512, 512]", getitem_1156: "f32[512]", getitem_1157: "f32[2048, 512]", getitem_1158: "f32[2048]", getitem_1159: "f32[512, 2048]", getitem_1160: "f32[512]", getitem_1161: "f32[512]", getitem_1162: "f32[512]", getitem_1163: "f32[512]", getitem_1164: "f32[512]", getitem_1165: "f32[1536, 512]", getitem_1166: "f32[1536]", getitem_1167: "f32[512, 512]", getitem_1168: "f32[512]", getitem_1169: "f32[2048, 512]", getitem_1170: "f32[2048]", getitem_1171: "f32[512, 2048]", getitem_1172: "f32[512]", getitem_1173: "f32[512]", getitem_1174: "f32[512]", getitem_1175: "f32[512]", getitem_1176: "f32[512]", getitem_1177: "f32[1536, 512]", getitem_1178: "f32[1536]", getitem_1179: "f32[512, 512]", getitem_1180: "f32[512]", getitem_1181: "f32[2048, 512]", getitem_1182: "f32[2048]", getitem_1183: "f32[512, 2048]", getitem_1184: "f32[512]", getitem_1185: "f32[512]", getitem_1186: "f32[512]", getitem_1187: "f32[512]", getitem_1188: "f32[512]", getitem_1189: "f32[1536, 512]", getitem_1190: "f32[1536]", getitem_1191: "f32[512, 512]", getitem_1192: "f32[512]", getitem_1193: "f32[2048, 512]", getitem_1194: "f32[2048]", getitem_1195: "f32[512, 2048]", getitem_1196: "f32[512]", getitem_1197: "f32[512]", getitem_1198: "f32[512]", getitem_1199: "f32[512]", getitem_1200: "f32[512]", getitem_1201: "f32[512]", getitem_1202: "f32[512]", getitem_1203: "f32[512, 512]", arg1204_1: "f32[768]", arg1205_1: "f32[50, 768]", arg1206_1: "f32[768, 512]", arg1207_1: "f32[768, 3, 32, 32]", arg1208_1: "f32[768]", arg1209_1: "f32[768]", arg1210_1: "f32[2304, 768]", arg1211_1: "f32[2304]", arg1212_1: "f32[768, 768]", arg1213_1: "f32[768]", arg1214_1: "f32[3072, 768]", arg1215_1: "f32[3072]", arg1216_1: "f32[768, 3072]", arg1217_1: "f32[768]", arg1218_1: "f32[768]", arg1219_1: "f32[768]", arg1220_1: "f32[768]", arg1221_1: "f32[768]", arg1222_1: "f32[2304, 768]", arg1223_1: "f32[2304]", arg1224_1: "f32[768, 768]", arg1225_1: "f32[768]", arg1226_1: "f32[3072, 768]", arg1227_1: "f32[3072]", arg1228_1: "f32[768, 3072]", arg1229_1: "f32[768]", arg1230_1: "f32[768]", arg1231_1: "f32[768]", arg1232_1: "f32[768]", arg1233_1: "f32[768]", arg1234_1: "f32[2304, 768]", arg1235_1: "f32[2304]", arg1236_1: "f32[768, 768]", arg1237_1: "f32[768]", arg1238_1: "f32[3072, 768]", arg1239_1: "f32[3072]", arg1240_1: "f32[768, 3072]", arg1241_1: "f32[768]", arg1242_1: "f32[768]", arg1243_1: "f32[768]", arg1244_1: "f32[768]", arg1245_1: "f32[768]", arg1246_1: "f32[2304, 768]", arg1247_1: "f32[2304]", arg1248_1: "f32[768, 768]", arg1249_1: "f32[768]", arg1250_1: "f32[3072, 768]", arg1251_1: "f32[3072]", arg1252_1: "f32[768, 3072]", arg1253_1: "f32[768]", arg1254_1: "f32[768]", arg1255_1: "f32[768]", arg1256_1: "f32[768]", arg1257_1: "f32[768]", arg1258_1: "f32[2304, 768]", arg1259_1: "f32[2304]", arg1260_1: "f32[768, 768]", arg1261_1: "f32[768]", arg1262_1: "f32[3072, 768]", arg1263_1: "f32[3072]", arg1264_1: "f32[768, 3072]", arg1265_1: "f32[768]", arg1266_1: "f32[768]", arg1267_1: "f32[768]", arg1268_1: "f32[768]", arg1269_1: "f32[768]", arg1270_1: "f32[2304, 768]", arg1271_1: "f32[2304]", arg1272_1: "f32[768, 768]", arg1273_1: "f32[768]", arg1274_1: "f32[3072, 768]", arg1275_1: "f32[3072]", arg1276_1: "f32[768, 3072]", arg1277_1: "f32[768]", arg1278_1: "f32[768]", arg1279_1: "f32[768]", arg1280_1: "f32[768]", arg1281_1: "f32[768]", arg1282_1: "f32[2304, 768]", arg1283_1: "f32[2304]", arg1284_1: "f32[768, 768]", arg1285_1: "f32[768]", arg1286_1: "f32[3072, 768]", arg1287_1: "f32[3072]", arg1288_1: "f32[768, 3072]", arg1289_1: "f32[768]", arg1290_1: "f32[768]", arg1291_1: "f32[768]", arg1292_1: "f32[768]", arg1293_1: "f32[768]", arg1294_1: "f32[2304, 768]", arg1295_1: "f32[2304]", arg1296_1: "f32[768, 768]", arg1297_1: "f32[768]", arg1298_1: "f32[3072, 768]", arg1299_1: "f32[3072]", arg1300_1: "f32[768, 3072]", arg1301_1: "f32[768]", arg1302_1: "f32[768]", arg1303_1: "f32[768]", arg1304_1: "f32[768]", arg1305_1: "f32[768]", arg1306_1: "f32[2304, 768]", arg1307_1: "f32[2304]", arg1308_1: "f32[768, 768]", arg1309_1: "f32[768]", arg1310_1: "f32[3072, 768]", arg1311_1: "f32[3072]", arg1312_1: "f32[768, 3072]", arg1313_1: "f32[768]", arg1314_1: "f32[768]", arg1315_1: "f32[768]", arg1316_1: "f32[768]", arg1317_1: "f32[768]", arg1318_1: "f32[2304, 768]", arg1319_1: "f32[2304]", arg1320_1: "f32[768, 768]", arg1321_1: "f32[768]", arg1322_1: "f32[3072, 768]", arg1323_1: "f32[3072]", arg1324_1: "f32[768, 3072]", arg1325_1: "f32[768]", arg1326_1: "f32[768]", arg1327_1: "f32[768]", arg1328_1: "f32[768]", arg1329_1: "f32[768]", arg1330_1: "f32[2304, 768]", arg1331_1: "f32[2304]", arg1332_1: "f32[768, 768]", arg1333_1: "f32[768]", arg1334_1: "f32[3072, 768]", arg1335_1: "f32[3072]", arg1336_1: "f32[768, 3072]", arg1337_1: "f32[768]", arg1338_1: "f32[768]", arg1339_1: "f32[768]", arg1340_1: "f32[768]", arg1341_1: "f32[768]", arg1342_1: "f32[2304, 768]", arg1343_1: "f32[2304]", arg1344_1: "f32[768, 768]", arg1345_1: "f32[768]", arg1346_1: "f32[3072, 768]", arg1347_1: "f32[3072]", arg1348_1: "f32[768, 3072]", arg1349_1: "f32[768]", arg1350_1: "f32[768]", arg1351_1: "f32[768]", arg1352_1: "f32[768]", arg1353_1: "f32[768]", arg1354_1: "f32[768]", arg1355_1: "f32[768]", arg1356_1: "f32[77, 512]", arg1357_1: "f32[49408, 512]", arg1358_1: "f32[1536, 512]", arg1359_1: "f32[1536]", arg1360_1: "f32[512, 512]", arg1361_1: "f32[512]", arg1362_1: "f32[2048, 512]", arg1363_1: "f32[2048]", arg1364_1: "f32[512, 2048]", arg1365_1: "f32[512]", arg1366_1: "f32[512]", arg1367_1: "f32[512]", arg1368_1: "f32[512]", arg1369_1: "f32[512]", arg1370_1: "f32[1536, 512]", arg1371_1: "f32[1536]", arg1372_1: "f32[512, 512]", arg1373_1: "f32[512]", arg1374_1: "f32[2048, 512]", arg1375_1: "f32[2048]", arg1376_1: "f32[512, 2048]", arg1377_1: "f32[512]", arg1378_1: "f32[512]", arg1379_1: "f32[512]", arg1380_1: "f32[512]", arg1381_1: "f32[512]", arg1382_1: "f32[1536, 512]", arg1383_1: "f32[1536]", arg1384_1: "f32[512, 512]", arg1385_1: "f32[512]", arg1386_1: "f32[2048, 512]", arg1387_1: "f32[2048]", arg1388_1: "f32[512, 2048]", arg1389_1: "f32[512]", arg1390_1: "f32[512]", arg1391_1: "f32[512]", arg1392_1: "f32[512]", arg1393_1: "f32[512]", arg1394_1: "f32[1536, 512]", arg1395_1: "f32[1536]", arg1396_1: "f32[512, 512]", arg1397_1: "f32[512]", arg1398_1: "f32[2048, 512]", arg1399_1: "f32[2048]", arg1400_1: "f32[512, 2048]", arg1401_1: "f32[512]", arg1402_1: "f32[512]", arg1403_1: "f32[512]", arg1404_1: "f32[512]", arg1405_1: "f32[512]", arg1406_1: "f32[1536, 512]", arg1407_1: "f32[1536]", arg1408_1: "f32[512, 512]", arg1409_1: "f32[512]", arg1410_1: "f32[2048, 512]", arg1411_1: "f32[2048]", arg1412_1: "f32[512, 2048]", arg1413_1: "f32[512]", arg1414_1: "f32[512]", arg1415_1: "f32[512]", arg1416_1: "f32[512]", arg1417_1: "f32[512]", arg1418_1: "f32[1536, 512]", arg1419_1: "f32[1536]", arg1420_1: "f32[512, 512]", arg1421_1: "f32[512]", arg1422_1: "f32[2048, 512]", arg1423_1: "f32[2048]", arg1424_1: "f32[512, 2048]", arg1425_1: "f32[512]", arg1426_1: "f32[512]", arg1427_1: "f32[512]", arg1428_1: "f32[512]", arg1429_1: "f32[512]", arg1430_1: "f32[1536, 512]", arg1431_1: "f32[1536]", arg1432_1: "f32[512, 512]", arg1433_1: "f32[512]", arg1434_1: "f32[2048, 512]", arg1435_1: "f32[2048]", arg1436_1: "f32[512, 2048]", arg1437_1: "f32[512]", arg1438_1: "f32[512]", arg1439_1: "f32[512]", arg1440_1: "f32[512]", arg1441_1: "f32[512]", arg1442_1: "f32[1536, 512]", arg1443_1: "f32[1536]", arg1444_1: "f32[512, 512]", arg1445_1: "f32[512]", arg1446_1: "f32[2048, 512]", arg1447_1: "f32[2048]", arg1448_1: "f32[512, 2048]", arg1449_1: "f32[512]", arg1450_1: "f32[512]", arg1451_1: "f32[512]", arg1452_1: "f32[512]", arg1453_1: "f32[512]", arg1454_1: "f32[1536, 512]", arg1455_1: "f32[1536]", arg1456_1: "f32[512, 512]", arg1457_1: "f32[512]", arg1458_1: "f32[2048, 512]", arg1459_1: "f32[2048]", arg1460_1: "f32[512, 2048]", arg1461_1: "f32[512]", arg1462_1: "f32[512]", arg1463_1: "f32[512]", arg1464_1: "f32[512]", arg1465_1: "f32[512]", arg1466_1: "f32[1536, 512]", arg1467_1: "f32[1536]", arg1468_1: "f32[512, 512]", arg1469_1: "f32[512]", arg1470_1: "f32[2048, 512]", arg1471_1: "f32[2048]", arg1472_1: "f32[512, 2048]", arg1473_1: "f32[512]", arg1474_1: "f32[512]", arg1475_1: "f32[512]", arg1476_1: "f32[512]", arg1477_1: "f32[512]", arg1478_1: "f32[1536, 512]", arg1479_1: "f32[1536]", arg1480_1: "f32[512, 512]", arg1481_1: "f32[512]", arg1482_1: "f32[2048, 512]", arg1483_1: "f32[2048]", arg1484_1: "f32[512, 2048]", arg1485_1: "f32[512]", arg1486_1: "f32[512]", arg1487_1: "f32[512]", arg1488_1: "f32[512]", arg1489_1: "f32[512]", arg1490_1: "f32[1536, 512]", arg1491_1: "f32[1536]", arg1492_1: "f32[512, 512]", arg1493_1: "f32[512]", arg1494_1: "f32[2048, 512]", arg1495_1: "f32[2048]", arg1496_1: "f32[512, 2048]", arg1497_1: "f32[512]", arg1498_1: "f32[512]", arg1499_1: "f32[512]", arg1500_1: "f32[512]", arg1501_1: "f32[512]", arg1502_1: "f32[512]", arg1503_1: "f32[512]", arg1504_1: "f32[512, 512]", getitem: "f32[]", getitem_1: "f32[]", getitem_2: "f32[]", getitem_3: "f32[]", getitem_4: "f32[]", getitem_5: "f32[]", getitem_6: "f32[]", getitem_7: "f32[]", getitem_8: "f32[]", getitem_9: "f32[]", getitem_10: "f32[]", getitem_11: "f32[]", getitem_12: "f32[]", getitem_13: "f32[]", getitem_14: "f32[]", getitem_15: "f32[]", getitem_16: "f32[]", getitem_17: "f32[]", getitem_18: "f32[]", getitem_19: "f32[]", getitem_20: "f32[]", getitem_21: "f32[]", getitem_22: "f32[]", getitem_23: "f32[]", getitem_24: "f32[]", getitem_25: "f32[]", getitem_26: "f32[]", getitem_27: "f32[]", getitem_28: "f32[]", getitem_29: "f32[]", getitem_30: "f32[]", getitem_31: "f32[]", getitem_32: "f32[]", getitem_33: "f32[]", getitem_34: "f32[]", getitem_35: "f32[]", getitem_36: "f32[]", getitem_37: "f32[]", getitem_38: "f32[]", getitem_39: "f32[]", getitem_40: "f32[]", getitem_41: "f32[]", getitem_42: "f32[]", getitem_43: "f32[]", getitem_44: "f32[]", getitem_45: "f32[]", getitem_46: "f32[]", getitem_47: "f32[]", getitem_48: "f32[]", getitem_49: "f32[]", getitem_50: "f32[]", getitem_51: "f32[]", getitem_52: "f32[]", getitem_53: "f32[]", getitem_54: "f32[]", getitem_55: "f32[]", getitem_56: "f32[]", getitem_57: "f32[]", getitem_58: "f32[]", getitem_59: "f32[]", getitem_60: "f32[]", getitem_61: "f32[]", getitem_62: "f32[]", getitem_63: "f32[]", getitem_64: "f32[]", getitem_65: "f32[]", getitem_66: "f32[]", getitem_67: "f32[]", getitem_68: "f32[]", getitem_69: "f32[]", getitem_70: "f32[]", getitem_71: "f32[]", getitem_72: "f32[]", getitem_73: "f32[]", getitem_74: "f32[]", getitem_75: "f32[]", getitem_76: "f32[]", getitem_77: "f32[]", getitem_78: "f32[]", getitem_79: "f32[]", getitem_80: "f32[]", getitem_81: "f32[]", getitem_82: "f32[]", getitem_83: "f32[]", getitem_84: "f32[]", getitem_85: "f32[]", getitem_86: "f32[]", getitem_87: "f32[]", getitem_88: "f32[]", getitem_89: "f32[]", getitem_90: "f32[]", getitem_91: "f32[]", getitem_92: "f32[]", getitem_93: "f32[]", getitem_94: "f32[]", getitem_95: "f32[]", getitem_96: "f32[]", getitem_97: "f32[]", getitem_98: "f32[]", getitem_99: "f32[]", getitem_100: "f32[]", getitem_101: "f32[]", getitem_102: "f32[]", getitem_103: "f32[]", getitem_104: "f32[]", getitem_105: "f32[]", getitem_106: "f32[]", getitem_107: "f32[]", getitem_108: "f32[]", getitem_109: "f32[]", getitem_110: "f32[]", getitem_111: "f32[]", getitem_112: "f32[]", getitem_113: "f32[]", getitem_114: "f32[]", getitem_115: "f32[]", getitem_116: "f32[]", getitem_117: "f32[]", getitem_118: "f32[]", getitem_119: "f32[]", getitem_120: "f32[]", getitem_121: "f32[]", getitem_122: "f32[]", getitem_123: "f32[]", getitem_124: "f32[]", getitem_125: "f32[]", getitem_126: "f32[]", getitem_127: "f32[]", getitem_128: "f32[]", getitem_129: "f32[]", getitem_130: "f32[]", getitem_131: "f32[]", getitem_132: "f32[]", getitem_133: "f32[]", getitem_134: "f32[]", getitem_135: "f32[]", getitem_136: "f32[]", getitem_137: "f32[]", getitem_138: "f32[]", getitem_139: "f32[]", getitem_140: "f32[]", getitem_141: "f32[]", getitem_142: "f32[]", getitem_143: "f32[]", getitem_144: "f32[]", getitem_145: "f32[]", getitem_146: "f32[]", getitem_147: "f32[]", getitem_148: "f32[]", getitem_149: "f32[]", getitem_150: "f32[]", getitem_151: "f32[]", getitem_152: "f32[]", getitem_153: "f32[]", getitem_154: "f32[]", getitem_155: "f32[]", getitem_156: "f32[]", getitem_157: "f32[]", getitem_158: "f32[]", getitem_159: "f32[]", getitem_160: "f32[]", getitem_161: "f32[]", getitem_162: "f32[]", getitem_163: "f32[]", getitem_164: "f32[]", getitem_165: "f32[]", getitem_166: "f32[]", getitem_167: "f32[]", getitem_168: "f32[]", getitem_169: "f32[]", getitem_170: "f32[]", getitem_171: "f32[]", getitem_172: "f32[]", getitem_173: "f32[]", getitem_174: "f32[]", getitem_175: "f32[]", getitem_176: "f32[]", getitem_177: "f32[]", getitem_178: "f32[]", getitem_179: "f32[]", getitem_180: "f32[]", getitem_181: "f32[]", getitem_182: "f32[]", getitem_183: "f32[]", getitem_184: "f32[]", getitem_185: "f32[]", getitem_186: "f32[]", getitem_187: "f32[]", getitem_188: "f32[]", getitem_189: "f32[]", getitem_190: "f32[]", getitem_191: "f32[]", getitem_192: "f32[]", getitem_193: "f32[]", getitem_194: "f32[]", getitem_195: "f32[]", getitem_196: "f32[]", getitem_197: "f32[]", getitem_198: "f32[]", getitem_199: "f32[]", getitem_200: "f32[]", getitem_201: "f32[]", getitem_202: "f32[]", getitem_203: "f32[]", getitem_204: "f32[]", getitem_205: "f32[]", getitem_206: "f32[]", getitem_207: "f32[]", getitem_208: "f32[]", getitem_209: "f32[]", getitem_210: "f32[]", getitem_211: "f32[]", getitem_212: "f32[]", getitem_213: "f32[]", getitem_214: "f32[]", getitem_215: "f32[]", getitem_216: "f32[]", getitem_217: "f32[]", getitem_218: "f32[]", getitem_219: "f32[]", getitem_220: "f32[]", getitem_221: "f32[]", getitem_222: "f32[]", getitem_223: "f32[]", getitem_224: "f32[]", getitem_225: "f32[]", getitem_226: "f32[]", getitem_227: "f32[]", getitem_228: "f32[]", getitem_229: "f32[]", getitem_230: "f32[]", getitem_231: "f32[]", getitem_232: "f32[]", getitem_233: "f32[]", getitem_234: "f32[]", getitem_235: "f32[]", getitem_236: "f32[]", getitem_237: "f32[]", getitem_238: "f32[]", getitem_239: "f32[]", getitem_240: "f32[]", getitem_241: "f32[]", getitem_242: "f32[]", getitem_243: "f32[]", getitem_244: "f32[]", getitem_245: "f32[]", getitem_246: "f32[]", getitem_247: "f32[]", getitem_248: "f32[]", getitem_249: "f32[]", getitem_250: "f32[]", getitem_251: "f32[]", getitem_252: "f32[]", getitem_253: "f32[]", getitem_254: "f32[]", getitem_255: "f32[]", getitem_256: "f32[]", getitem_257: "f32[]", getitem_258: "f32[]", getitem_259: "f32[]", getitem_260: "f32[]", getitem_261: "f32[]", getitem_262: "f32[]", getitem_263: "f32[]", getitem_264: "f32[]", getitem_265: "f32[]", getitem_266: "f32[]", getitem_267: "f32[]", getitem_268: "f32[]", getitem_269: "f32[]", getitem_270: "f32[]", getitem_271: "f32[]", getitem_272: "f32[]", getitem_273: "f32[]", getitem_274: "f32[]", getitem_275: "f32[]", getitem_276: "f32[]", getitem_277: "f32[]", getitem_278: "f32[]", getitem_279: "f32[]", getitem_280: "f32[]", getitem_281: "f32[]", getitem_282: "f32[]", getitem_283: "f32[]", getitem_284: "f32[]", getitem_285: "f32[]", getitem_286: "f32[]", getitem_287: "f32[]", getitem_288: "f32[]", getitem_289: "f32[]", getitem_290: "f32[]", getitem_291: "f32[]", getitem_292: "f32[]", getitem_293: "f32[]", getitem_294: "f32[]", getitem_295: "f32[]", getitem_296: "f32[]", getitem_297: "f32[]", getitem_298: "f32[]", getitem_299: "f32[]", getitem_300: "f32[]", getitem_2408: "f32[]", getitem_2409: "f32[]", getitem_2410: "f32[]", getitem_2411: "f32[]", getitem_2412: "f32[]", getitem_2413: "f32[]", getitem_2414: "f32[]", getitem_2415: "f32[]", getitem_2416: "f32[]", getitem_2417: "f32[]", getitem_2418: "f32[]", getitem_2419: "f32[]", getitem_2420: "f32[]", getitem_2421: "f32[]", getitem_2422: "f32[]", getitem_2423: "f32[]", getitem_2424: "f32[]", getitem_2425: "f32[]", getitem_2426: "f32[]", getitem_2427: "f32[]", getitem_2428: "f32[]", getitem_2429: "f32[]", getitem_2430: "f32[]", getitem_2431: "f32[]", getitem_2432: "f32[]", getitem_2433: "f32[]", getitem_2434: "f32[]", getitem_2435: "f32[]", getitem_2436: "f32[]", getitem_2437: "f32[]", getitem_2438: "f32[]", getitem_2439: "f32[]", getitem_2440: "f32[]", getitem_2441: "f32[]", getitem_2442: "f32[]", getitem_2443: "f32[]", getitem_2444: "f32[]", getitem_2445: "f32[]", getitem_2446: "f32[]", getitem_2447: "f32[]", getitem_2448: "f32[]", getitem_2449: "f32[]", getitem_2450: "f32[]", getitem_2451: "f32[]", getitem_2452: "f32[]", getitem_2453: "f32[]", getitem_2454: "f32[]", getitem_2455: "f32[]", getitem_2456: "f32[]", getitem_2457: "f32[]", getitem_2458: "f32[]", getitem_2459: "f32[]", getitem_2460: "f32[]", getitem_2461: "f32[]", getitem_2462: "f32[]", getitem_2463: "f32[]", getitem_2464: "f32[]", getitem_2465: "f32[]", getitem_2466: "f32[]", getitem_2467: "f32[]", getitem_2468: "f32[]", getitem_2469: "f32[]", getitem_2470: "f32[]", getitem_2471: "f32[]", getitem_2472: "f32[]", getitem_2473: "f32[]", getitem_2474: "f32[]", getitem_2475: "f32[]", getitem_2476: "f32[]", getitem_2477: "f32[]", getitem_2478: "f32[]", getitem_2479: "f32[]", getitem_2480: "f32[]", getitem_2481: "f32[]", getitem_2482: "f32[]", getitem_2483: "f32[]", getitem_2484: "f32[]", getitem_2485: "f32[]", getitem_2486: "f32[]", getitem_2487: "f32[]", getitem_2488: "f32[]", getitem_2489: "f32[]", getitem_2490: "f32[]", getitem_2491: "f32[]", getitem_2492: "f32[]", getitem_2493: "f32[]", getitem_2494: "f32[]", getitem_2495: "f32[]", getitem_2496: "f32[]", getitem_2497: "f32[]", getitem_2498: "f32[]", getitem_2499: "f32[]", getitem_2500: "f32[]", getitem_2501: "f32[]", getitem_2502: "f32[]", getitem_2503: "f32[]", getitem_2504: "f32[]", getitem_2505: "f32[]", getitem_2506: "f32[]", getitem_2507: "f32[]", getitem_2508: "f32[]", getitem_2509: "f32[]", getitem_2510: "f32[]", getitem_2511: "f32[]", getitem_2512: "f32[]", getitem_2513: "f32[]", getitem_2514: "f32[]", getitem_2515: "f32[]", getitem_2516: "f32[]", getitem_2517: "f32[]", getitem_2518: "f32[]", getitem_2519: "f32[]", getitem_2520: "f32[]", getitem_2521: "f32[]", getitem_2522: "f32[]", getitem_2523: "f32[]", getitem_2524: "f32[]", getitem_2525: "f32[]", getitem_2526: "f32[]", getitem_2527: "f32[]", getitem_2528: "f32[]", getitem_2529: "f32[]", getitem_2530: "f32[]", getitem_2531: "f32[]", getitem_2532: "f32[]", getitem_2533: "f32[]", getitem_2534: "f32[]", getitem_2535: "f32[]", getitem_2536: "f32[]", getitem_2537: "f32[]", getitem_2538: "f32[]", getitem_2539: "f32[]", getitem_2540: "f32[]", getitem_2541: "f32[]", getitem_2542: "f32[]", getitem_2543: "f32[]", getitem_2544: "f32[]", getitem_2545: "f32[]", getitem_2546: "f32[]", getitem_2547: "f32[]", getitem_2548: "f32[]", getitem_2549: "f32[]", getitem_2550: "f32[]", getitem_2551: "f32[]", getitem_2552: "f32[]", getitem_2553: "f32[]", getitem_2554: "f32[]", getitem_2555: "f32[]", getitem_2556: "f32[]", getitem_2557: "f32[]", getitem_2558: "f32[]", getitem_2559: "f32[]", getitem_2560: "f32[]", getitem_2561: "f32[]", getitem_2562: "f32[]", getitem_2563: "f32[]", getitem_2564: "f32[]", getitem_2565: "f32[]", getitem_2566: "f32[]", getitem_2567: "f32[]", getitem_2568: "f32[]", getitem_2569: "f32[]", getitem_2570: "f32[]", getitem_2571: "f32[]", getitem_2572: "f32[]", getitem_2573: "f32[]", getitem_2574: "f32[]", getitem_2575: "f32[]", getitem_2576: "f32[]", getitem_2577: "f32[]", getitem_2578: "f32[]", getitem_2579: "f32[]", getitem_2580: "f32[]", getitem_2581: "f32[]", getitem_2582: "f32[]", getitem_2583: "f32[]", getitem_2584: "f32[]", getitem_2585: "f32[]", getitem_2586: "f32[]", getitem_2587: "f32[]", getitem_2588: "f32[]", getitem_2589: "f32[]", getitem_2590: "f32[]", getitem_2591: "f32[]", getitem_2592: "f32[]", getitem_2593: "f32[]", getitem_2594: "f32[]", getitem_2595: "f32[]", getitem_2596: "f32[]", getitem_2597: "f32[]", getitem_2598: "f32[]", getitem_2599: "f32[]", getitem_2600: "f32[]", getitem_2601: "f32[]", getitem_2602: "f32[]", getitem_2603: "f32[]", getitem_2604: "f32[]", getitem_2605: "f32[]", getitem_2606: "f32[]", getitem_2607: "f32[]", getitem_2608: "f32[]", getitem_2609: "f32[]", getitem_2610: "f32[]", getitem_2611: "f32[]", getitem_2612: "f32[]", getitem_2613: "f32[]", getitem_2614: "f32[]", getitem_2615: "f32[]", getitem_2616: "f32[]", getitem_2617: "f32[]", getitem_2618: "f32[]", getitem_2619: "f32[]", getitem_2620: "f32[]", getitem_2621: "f32[]", getitem_2622: "f32[]", getitem_2623: "f32[]", getitem_2624: "f32[]", getitem_2625: "f32[]", getitem_2626: "f32[]", getitem_2627: "f32[]", getitem_2628: "f32[]", getitem_2629: "f32[]", getitem_2630: "f32[]", getitem_2631: "f32[]", getitem_2632: "f32[]", getitem_2633: "f32[]", getitem_2634: "f32[]", getitem_2635: "f32[]", getitem_2636: "f32[]", getitem_2637: "f32[]", getitem_2638: "f32[]", getitem_2639: "f32[]", getitem_2640: "f32[]", getitem_2641: "f32[]", getitem_2642: "f32[]", getitem_2643: "f32[]", getitem_2644: "f32[]", getitem_2645: "f32[]", getitem_2646: "f32[]", getitem_2647: "f32[]", getitem_2648: "f32[]", getitem_2649: "f32[]", getitem_2650: "f32[]", getitem_2651: "f32[]", getitem_2652: "f32[]", getitem_2653: "f32[]", getitem_2654: "f32[]", getitem_2655: "f32[]", getitem_2656: "f32[]", getitem_2657: "f32[]", getitem_2658: "f32[]", getitem_2659: "f32[]", getitem_2660: "f32[]", getitem_2661: "f32[]", getitem_2662: "f32[]", getitem_2663: "f32[]", getitem_2664: "f32[]", getitem_2665: "f32[]", getitem_2666: "f32[]", getitem_2667: "f32[]", getitem_2668: "f32[]", getitem_2669: "f32[]", getitem_2670: "f32[]", getitem_2671: "f32[]", getitem_2672: "f32[]", getitem_2673: "f32[]", getitem_2674: "f32[]", getitem_2675: "f32[]", getitem_2676: "f32[]", getitem_2677: "f32[]", getitem_2678: "f32[]", getitem_2679: "f32[]", getitem_2680: "f32[]", getitem_2681: "f32[]", getitem_2682: "f32[]", getitem_2683: "f32[]", getitem_2684: "f32[]", getitem_2685: "f32[]", getitem_2686: "f32[]", getitem_2687: "f32[]", getitem_2688: "f32[]", getitem_2689: "f32[]", getitem_2690: "f32[]", getitem_2691: "f32[]", getitem_2692: "f32[]", getitem_2693: "f32[]", getitem_2694: "f32[]", getitem_2695: "f32[]", getitem_2696: "f32[]", getitem_2697: "f32[]", getitem_2698: "f32[]", getitem_2699: "f32[]", getitem_2700: "f32[]", getitem_2701: "f32[]", getitem_2702: "f32[]", getitem_2703: "f32[]", getitem_2704: "f32[]", getitem_2705: "f32[]", getitem_2706: "f32[]", getitem_2707: "f32[]", getitem_2708: "f32[]"):
        # No stacktrace found for following nodes
        _foreach_addcmul_scalar = torch.ops.aten._foreach_addcmul.Scalar([getitem_903, getitem_904, getitem_905, getitem_906, getitem_907, getitem_908, getitem_909, getitem_910, getitem_911, getitem_912, getitem_913, getitem_914, getitem_915, getitem_916, getitem_917, getitem_918, getitem_919, getitem_920, getitem_921, getitem_922, getitem_923, getitem_924, getitem_925, getitem_926, getitem_927, getitem_928, getitem_929, getitem_930, getitem_931, getitem_932, getitem_933, getitem_934, getitem_935, getitem_936, getitem_937, getitem_938, getitem_939, getitem_940, getitem_941, getitem_942, getitem_943, getitem_944, getitem_945, getitem_946, getitem_947, getitem_948, getitem_949, getitem_950, getitem_951, getitem_952, getitem_953, getitem_954, getitem_955, getitem_956, getitem_957, getitem_958, getitem_959, getitem_960, getitem_961, getitem_962, getitem_963, getitem_964, getitem_965, getitem_966, getitem_967, getitem_968, getitem_969, getitem_970, getitem_971, getitem_972, getitem_973, getitem_974, getitem_975, getitem_976, getitem_977, getitem_978, getitem_979, getitem_980, getitem_981, getitem_982, getitem_983, getitem_984, getitem_985, getitem_986, getitem_987, getitem_988, getitem_989, getitem_990, getitem_991, getitem_992, getitem_993, getitem_994, getitem_995, getitem_996, getitem_997, getitem_998, getitem_999, getitem_1000, getitem_1001, getitem_1002, getitem_1003, getitem_1004, getitem_1005, getitem_1006, getitem_1007, getitem_1008, getitem_1009, getitem_1010, getitem_1011, getitem_1012, getitem_1013, getitem_1014, getitem_1015, getitem_1016, getitem_1017, getitem_1018, getitem_1019, getitem_1020, getitem_1021, getitem_1022, getitem_1023, getitem_1024, getitem_1025, getitem_1026, getitem_1027, getitem_1028, getitem_1029, getitem_1030, getitem_1031, getitem_1032, getitem_1033, getitem_1034, getitem_1035, getitem_1036, getitem_1037, getitem_1038, getitem_1039, getitem_1040, getitem_1041, getitem_1042, getitem_1043, getitem_1044, getitem_1045, getitem_1046, getitem_1047, getitem_1048, getitem_1049, getitem_1050, getitem_1051, getitem_1052, getitem_1053, getitem_1054, getitem_1055, getitem_1056, getitem_1057, getitem_1058, getitem_1059, getitem_1060, getitem_1061, getitem_1062, getitem_1063, getitem_1064, getitem_1065, getitem_1066, getitem_1067, getitem_1068, getitem_1069, getitem_1070, getitem_1071, getitem_1072, getitem_1073, getitem_1074, getitem_1075, getitem_1076, getitem_1077, getitem_1078, getitem_1079, getitem_1080, getitem_1081, getitem_1082, getitem_1083, getitem_1084, getitem_1085, getitem_1086, getitem_1087, getitem_1088, getitem_1089, getitem_1090, getitem_1091, getitem_1092, getitem_1093, getitem_1094, getitem_1095, getitem_1096, getitem_1097, getitem_1098, getitem_1099, getitem_1100, getitem_1101, getitem_1102, getitem_1103, getitem_1104, getitem_1105, getitem_1106, getitem_1107, getitem_1108, getitem_1109, getitem_1110, getitem_1111, getitem_1112, getitem_1113, getitem_1114, getitem_1115, getitem_1116, getitem_1117, getitem_1118, getitem_1119, getitem_1120, getitem_1121, getitem_1122, getitem_1123, getitem_1124, getitem_1125, getitem_1126, getitem_1127, getitem_1128, getitem_1129, getitem_1130, getitem_1131, getitem_1132, getitem_1133, getitem_1134, getitem_1135, getitem_1136, getitem_1137, getitem_1138, getitem_1139, getitem_1140, getitem_1141, getitem_1142, getitem_1143, getitem_1144, getitem_1145, getitem_1146, getitem_1147, getitem_1148, getitem_1149, getitem_1150, getitem_1151, getitem_1152, getitem_1153, getitem_1154, getitem_1155, getitem_1156, getitem_1157, getitem_1158, getitem_1159, getitem_1160, getitem_1161, getitem_1162, getitem_1163, getitem_1164, getitem_1165, getitem_1166, getitem_1167, getitem_1168, getitem_1169, getitem_1170, getitem_1171, getitem_1172, getitem_1173, getitem_1174, getitem_1175, getitem_1176, getitem_1177, getitem_1178, getitem_1179, getitem_1180, getitem_1181, getitem_1182, getitem_1183, getitem_1184, getitem_1185, getitem_1186, getitem_1187, getitem_1188, getitem_1189, getitem_1190, getitem_1191, getitem_1192, getitem_1193, getitem_1194, getitem_1195, getitem_1196, getitem_1197, getitem_1198, getitem_1199, getitem_1200, getitem_1201, getitem_1202, getitem_1203], [arg1204_1, arg1205_1, arg1206_1, arg1207_1, arg1208_1, arg1209_1, arg1210_1, arg1211_1, arg1212_1, arg1213_1, arg1214_1, arg1215_1, arg1216_1, arg1217_1, arg1218_1, arg1219_1, arg1220_1, arg1221_1, arg1222_1, arg1223_1, arg1224_1, arg1225_1, arg1226_1, arg1227_1, arg1228_1, arg1229_1, arg1230_1, arg1231_1, arg1232_1, arg1233_1, arg1234_1, arg1235_1, arg1236_1, arg1237_1, arg1238_1, arg1239_1, arg1240_1, arg1241_1, arg1242_1, arg1243_1, arg1244_1, arg1245_1, arg1246_1, arg1247_1, arg1248_1, arg1249_1, arg1250_1, arg1251_1, arg1252_1, arg1253_1, arg1254_1, arg1255_1, arg1256_1, arg1257_1, arg1258_1, arg1259_1, arg1260_1, arg1261_1, arg1262_1, arg1263_1, arg1264_1, arg1265_1, arg1266_1, arg1267_1, arg1268_1, arg1269_1, arg1270_1, arg1271_1, arg1272_1, arg1273_1, arg1274_1, arg1275_1, arg1276_1, arg1277_1, arg1278_1, arg1279_1, arg1280_1, arg1281_1, arg1282_1, arg1283_1, arg1284_1, arg1285_1, arg1286_1, arg1287_1, arg1288_1, arg1289_1, arg1290_1, arg1291_1, arg1292_1, arg1293_1, arg1294_1, arg1295_1, arg1296_1, arg1297_1, arg1298_1, arg1299_1, arg1300_1, arg1301_1, arg1302_1, arg1303_1, arg1304_1, arg1305_1, arg1306_1, arg1307_1, arg1308_1, arg1309_1, arg1310_1, arg1311_1, arg1312_1, arg1313_1, arg1314_1, arg1315_1, arg1316_1, arg1317_1, arg1318_1, arg1319_1, arg1320_1, arg1321_1, arg1322_1, arg1323_1, arg1324_1, arg1325_1, arg1326_1, arg1327_1, arg1328_1, arg1329_1, arg1330_1, arg1331_1, arg1332_1, arg1333_1, arg1334_1, arg1335_1, arg1336_1, arg1337_1, arg1338_1, arg1339_1, arg1340_1, arg1341_1, arg1342_1, arg1343_1, arg1344_1, arg1345_1, arg1346_1, arg1347_1, arg1348_1, arg1349_1, arg1350_1, arg1351_1, arg1352_1, arg1353_1, arg1354_1, arg1355_1, arg1356_1, arg1357_1, arg1358_1, arg1359_1, arg1360_1, arg1361_1, arg1362_1, arg1363_1, arg1364_1, arg1365_1, arg1366_1, arg1367_1, arg1368_1, arg1369_1, arg1370_1, arg1371_1, arg1372_1, arg1373_1, arg1374_1, arg1375_1, arg1376_1, arg1377_1, arg1378_1, arg1379_1, arg1380_1, arg1381_1, arg1382_1, arg1383_1, arg1384_1, arg1385_1, arg1386_1, arg1387_1, arg1388_1, arg1389_1, arg1390_1, arg1391_1, arg1392_1, arg1393_1, arg1394_1, arg1395_1, arg1396_1, arg1397_1, arg1398_1, arg1399_1, arg1400_1, arg1401_1, arg1402_1, arg1403_1, arg1404_1, arg1405_1, arg1406_1, arg1407_1, arg1408_1, arg1409_1, arg1410_1, arg1411_1, arg1412_1, arg1413_1, arg1414_1, arg1415_1, arg1416_1, arg1417_1, arg1418_1, arg1419_1, arg1420_1, arg1421_1, arg1422_1, arg1423_1, arg1424_1, arg1425_1, arg1426_1, arg1427_1, arg1428_1, arg1429_1, arg1430_1, arg1431_1, arg1432_1, arg1433_1, arg1434_1, arg1435_1, arg1436_1, arg1437_1, arg1438_1, arg1439_1, arg1440_1, arg1441_1, arg1442_1, arg1443_1, arg1444_1, arg1445_1, arg1446_1, arg1447_1, arg1448_1, arg1449_1, arg1450_1, arg1451_1, arg1452_1, arg1453_1, arg1454_1, arg1455_1, arg1456_1, arg1457_1, arg1458_1, arg1459_1, arg1460_1, arg1461_1, arg1462_1, arg1463_1, arg1464_1, arg1465_1, arg1466_1, arg1467_1, arg1468_1, arg1469_1, arg1470_1, arg1471_1, arg1472_1, arg1473_1, arg1474_1, arg1475_1, arg1476_1, arg1477_1, arg1478_1, arg1479_1, arg1480_1, arg1481_1, arg1482_1, arg1483_1, arg1484_1, arg1485_1, arg1486_1, arg1487_1, arg1488_1, arg1489_1, arg1490_1, arg1491_1, arg1492_1, arg1493_1, arg1494_1, arg1495_1, arg1496_1, arg1497_1, arg1498_1, arg1499_1, arg1500_1, arg1501_1, arg1502_1, arg1503_1, arg1504_1], [arg1204_1, arg1205_1, arg1206_1, arg1207_1, arg1208_1, arg1209_1, arg1210_1, arg1211_1, arg1212_1, arg1213_1, arg1214_1, arg1215_1, arg1216_1, arg1217_1, arg1218_1, arg1219_1, arg1220_1, arg1221_1, arg1222_1, arg1223_1, arg1224_1, arg1225_1, arg1226_1, arg1227_1, arg1228_1, arg1229_1, arg1230_1, arg1231_1, arg1232_1, arg1233_1, arg1234_1, arg1235_1, arg1236_1, arg1237_1, arg1238_1, arg1239_1, arg1240_1, arg1241_1, arg1242_1, arg1243_1, arg1244_1, arg1245_1, arg1246_1, arg1247_1, arg1248_1, arg1249_1, arg1250_1, arg1251_1, arg1252_1, arg1253_1, arg1254_1, arg1255_1, arg1256_1, arg1257_1, arg1258_1, arg1259_1, arg1260_1, arg1261_1, arg1262_1, arg1263_1, arg1264_1, arg1265_1, arg1266_1, arg1267_1, arg1268_1, arg1269_1, arg1270_1, arg1271_1, arg1272_1, arg1273_1, arg1274_1, arg1275_1, arg1276_1, arg1277_1, arg1278_1, arg1279_1, arg1280_1, arg1281_1, arg1282_1, arg1283_1, arg1284_1, arg1285_1, arg1286_1, arg1287_1, arg1288_1, arg1289_1, arg1290_1, arg1291_1, arg1292_1, arg1293_1, arg1294_1, arg1295_1, arg1296_1, arg1297_1, arg1298_1, arg1299_1, arg1300_1, arg1301_1, arg1302_1, arg1303_1, arg1304_1, arg1305_1, arg1306_1, arg1307_1, arg1308_1, arg1309_1, arg1310_1, arg1311_1, arg1312_1, arg1313_1, arg1314_1, arg1315_1, arg1316_1, arg1317_1, arg1318_1, arg1319_1, arg1320_1, arg1321_1, arg1322_1, arg1323_1, arg1324_1, arg1325_1, arg1326_1, arg1327_1, arg1328_1, arg1329_1, arg1330_1, arg1331_1, arg1332_1, arg1333_1, arg1334_1, arg1335_1, arg1336_1, arg1337_1, arg1338_1, arg1339_1, arg1340_1, arg1341_1, arg1342_1, arg1343_1, arg1344_1, arg1345_1, arg1346_1, arg1347_1, arg1348_1, arg1349_1, arg1350_1, arg1351_1, arg1352_1, arg1353_1, arg1354_1, arg1355_1, arg1356_1, arg1357_1, arg1358_1, arg1359_1, arg1360_1, arg1361_1, arg1362_1, arg1363_1, arg1364_1, arg1365_1, arg1366_1, arg1367_1, arg1368_1, arg1369_1, arg1370_1, arg1371_1, arg1372_1, arg1373_1, arg1374_1, arg1375_1, arg1376_1, arg1377_1, arg1378_1, arg1379_1, arg1380_1, arg1381_1, arg1382_1, arg1383_1, arg1384_1, arg1385_1, arg1386_1, arg1387_1, arg1388_1, arg1389_1, arg1390_1, arg1391_1, arg1392_1, arg1393_1, arg1394_1, arg1395_1, arg1396_1, arg1397_1, arg1398_1, arg1399_1, arg1400_1, arg1401_1, arg1402_1, arg1403_1, arg1404_1, arg1405_1, arg1406_1, arg1407_1, arg1408_1, arg1409_1, arg1410_1, arg1411_1, arg1412_1, arg1413_1, arg1414_1, arg1415_1, arg1416_1, arg1417_1, arg1418_1, arg1419_1, arg1420_1, arg1421_1, arg1422_1, arg1423_1, arg1424_1, arg1425_1, arg1426_1, arg1427_1, arg1428_1, arg1429_1, arg1430_1, arg1431_1, arg1432_1, arg1433_1, arg1434_1, arg1435_1, arg1436_1, arg1437_1, arg1438_1, arg1439_1, arg1440_1, arg1441_1, arg1442_1, arg1443_1, arg1444_1, arg1445_1, arg1446_1, arg1447_1, arg1448_1, arg1449_1, arg1450_1, arg1451_1, arg1452_1, arg1453_1, arg1454_1, arg1455_1, arg1456_1, arg1457_1, arg1458_1, arg1459_1, arg1460_1, arg1461_1, arg1462_1, arg1463_1, arg1464_1, arg1465_1, arg1466_1, arg1467_1, arg1468_1, arg1469_1, arg1470_1, arg1471_1, arg1472_1, arg1473_1, arg1474_1, arg1475_1, arg1476_1, arg1477_1, arg1478_1, arg1479_1, arg1480_1, arg1481_1, arg1482_1, arg1483_1, arg1484_1, arg1485_1, arg1486_1, arg1487_1, arg1488_1, arg1489_1, arg1490_1, arg1491_1, arg1492_1, arg1493_1, arg1494_1, arg1495_1, arg1496_1, arg1497_1, arg1498_1, arg1499_1, arg1500_1, arg1501_1, arg1502_1, arg1503_1, arg1504_1], 0.0010000000000000009);  getitem_903 = getitem_904 = getitem_905 = getitem_906 = getitem_907 = getitem_908 = getitem_909 = getitem_910 = getitem_911 = getitem_912 = getitem_913 = getitem_914 = getitem_915 = getitem_916 = getitem_917 = getitem_918 = getitem_919 = getitem_920 = getitem_921 = getitem_922 = getitem_923 = getitem_924 = getitem_925 = getitem_926 = getitem_927 = getitem_928 = getitem_929 = getitem_930 = getitem_931 = getitem_932 = getitem_933 = getitem_934 = getitem_935 = getitem_936 = getitem_937 = getitem_938 = getitem_939 = getitem_940 = getitem_941 = getitem_942 = getitem_943 = getitem_944 = getitem_945 = getitem_946 = getitem_947 = getitem_948 = getitem_949 = getitem_950 = getitem_951 = getitem_952 = getitem_953 = getitem_954 = getitem_955 = getitem_956 = getitem_957 = getitem_958 = getitem_959 = getitem_960 = getitem_961 = getitem_962 = getitem_963 = getitem_964 = getitem_965 = getitem_966 = getitem_967 = getitem_968 = getitem_969 = getitem_970 = getitem_971 = getitem_972 = getitem_973 = getitem_974 = getitem_975 = getitem_976 = getitem_977 = getitem_978 = getitem_979 = getitem_980 = getitem_981 = getitem_982 = getitem_983 = getitem_984 = getitem_985 = getitem_986 = getitem_987 = getitem_988 = getitem_989 = getitem_990 = getitem_991 = getitem_992 = getitem_993 = getitem_994 = getitem_995 = getitem_996 = getitem_997 = getitem_998 = getitem_999 = getitem_1000 = getitem_1001 = getitem_1002 = getitem_1003 = getitem_1004 = getitem_1005 = getitem_1006 = getitem_1007 = getitem_1008 = getitem_1009 = getitem_1010 = getitem_1011 = getitem_1012 = getitem_1013 = getitem_1014 = getitem_1015 = getitem_1016 = getitem_1017 = getitem_1018 = getitem_1019 = getitem_1020 = getitem_1021 = getitem_1022 = getitem_1023 = getitem_1024 = getitem_1025 = getitem_1026 = getitem_1027 = getitem_1028 = getitem_1029 = getitem_1030 = getitem_1031 = getitem_1032 = getitem_1033 = getitem_1034 = getitem_1035 = getitem_1036 = getitem_1037 = getitem_1038 = getitem_1039 = getitem_1040 = getitem_1041 = getitem_1042 = getitem_1043 = getitem_1044 = getitem_1045 = getitem_1046 = getitem_1047 = getitem_1048 = getitem_1049 = getitem_1050 = getitem_1051 = getitem_1052 = getitem_1053 = getitem_1054 = getitem_1055 = getitem_1056 = getitem_1057 = getitem_1058 = getitem_1059 = getitem_1060 = getitem_1061 = getitem_1062 = getitem_1063 = getitem_1064 = getitem_1065 = getitem_1066 = getitem_1067 = getitem_1068 = getitem_1069 = getitem_1070 = getitem_1071 = getitem_1072 = getitem_1073 = getitem_1074 = getitem_1075 = getitem_1076 = getitem_1077 = getitem_1078 = getitem_1079 = getitem_1080 = getitem_1081 = getitem_1082 = getitem_1083 = getitem_1084 = getitem_1085 = getitem_1086 = getitem_1087 = getitem_1088 = getitem_1089 = getitem_1090 = getitem_1091 = getitem_1092 = getitem_1093 = getitem_1094 = getitem_1095 = getitem_1096 = getitem_1097 = getitem_1098 = getitem_1099 = getitem_1100 = getitem_1101 = getitem_1102 = getitem_1103 = getitem_1104 = getitem_1105 = getitem_1106 = getitem_1107 = getitem_1108 = getitem_1109 = getitem_1110 = getitem_1111 = getitem_1112 = getitem_1113 = getitem_1114 = getitem_1115 = getitem_1116 = getitem_1117 = getitem_1118 = getitem_1119 = getitem_1120 = getitem_1121 = getitem_1122 = getitem_1123 = getitem_1124 = getitem_1125 = getitem_1126 = getitem_1127 = getitem_1128 = getitem_1129 = getitem_1130 = getitem_1131 = getitem_1132 = getitem_1133 = getitem_1134 = getitem_1135 = getitem_1136 = getitem_1137 = getitem_1138 = getitem_1139 = getitem_1140 = getitem_1141 = getitem_1142 = getitem_1143 = getitem_1144 = getitem_1145 = getitem_1146 = getitem_1147 = getitem_1148 = getitem_1149 = getitem_1150 = getitem_1151 = getitem_1152 = getitem_1153 = getitem_1154 = getitem_1155 = getitem_1156 = getitem_1157 = getitem_1158 = getitem_1159 = getitem_1160 = getitem_1161 = getitem_1162 = getitem_1163 = getitem_1164 = getitem_1165 = getitem_1166 = getitem_1167 = getitem_1168 = getitem_1169 = getitem_1170 = getitem_1171 = getitem_1172 = getitem_1173 = getitem_1174 = getitem_1175 = getitem_1176 = getitem_1177 = getitem_1178 = getitem_1179 = getitem_1180 = getitem_1181 = getitem_1182 = getitem_1183 = getitem_1184 = getitem_1185 = getitem_1186 = getitem_1187 = getitem_1188 = getitem_1189 = getitem_1190 = getitem_1191 = getitem_1192 = getitem_1193 = getitem_1194 = getitem_1195 = getitem_1196 = getitem_1197 = getitem_1198 = getitem_1199 = getitem_1200 = getitem_1201 = getitem_1202 = getitem_1203 = arg1204_1 = arg1205_1 = arg1206_1 = arg1207_1 = arg1208_1 = arg1209_1 = arg1210_1 = arg1211_1 = arg1212_1 = arg1213_1 = arg1214_1 = arg1215_1 = arg1216_1 = arg1217_1 = arg1218_1 = arg1219_1 = arg1220_1 = arg1221_1 = arg1222_1 = arg1223_1 = arg1224_1 = arg1225_1 = arg1226_1 = arg1227_1 = arg1228_1 = arg1229_1 = arg1230_1 = arg1231_1 = arg1232_1 = arg1233_1 = arg1234_1 = arg1235_1 = arg1236_1 = arg1237_1 = arg1238_1 = arg1239_1 = arg1240_1 = arg1241_1 = arg1242_1 = arg1243_1 = arg1244_1 = arg1245_1 = arg1246_1 = arg1247_1 = arg1248_1 = arg1249_1 = arg1250_1 = arg1251_1 = arg1252_1 = arg1253_1 = arg1254_1 = arg1255_1 = arg1256_1 = arg1257_1 = arg1258_1 = arg1259_1 = arg1260_1 = arg1261_1 = arg1262_1 = arg1263_1 = arg1264_1 = arg1265_1 = arg1266_1 = arg1267_1 = arg1268_1 = arg1269_1 = arg1270_1 = arg1271_1 = arg1272_1 = arg1273_1 = arg1274_1 = arg1275_1 = arg1276_1 = arg1277_1 = arg1278_1 = arg1279_1 = arg1280_1 = arg1281_1 = arg1282_1 = arg1283_1 = arg1284_1 = arg1285_1 = arg1286_1 = arg1287_1 = arg1288_1 = arg1289_1 = arg1290_1 = arg1291_1 = arg1292_1 = arg1293_1 = arg1294_1 = arg1295_1 = arg1296_1 = arg1297_1 = arg1298_1 = arg1299_1 = arg1300_1 = arg1301_1 = arg1302_1 = arg1303_1 = arg1304_1 = arg1305_1 = arg1306_1 = arg1307_1 = arg1308_1 = arg1309_1 = arg1310_1 = arg1311_1 = arg1312_1 = arg1313_1 = arg1314_1 = arg1315_1 = arg1316_1 = arg1317_1 = arg1318_1 = arg1319_1 = arg1320_1 = arg1321_1 = arg1322_1 = arg1323_1 = arg1324_1 = arg1325_1 = arg1326_1 = arg1327_1 = arg1328_1 = arg1329_1 = arg1330_1 = arg1331_1 = arg1332_1 = arg1333_1 = arg1334_1 = arg1335_1 = arg1336_1 = arg1337_1 = arg1338_1 = arg1339_1 = arg1340_1 = arg1341_1 = arg1342_1 = arg1343_1 = arg1344_1 = arg1345_1 = arg1346_1 = arg1347_1 = arg1348_1 = arg1349_1 = arg1350_1 = arg1351_1 = arg1352_1 = arg1353_1 = arg1354_1 = arg1355_1 = arg1356_1 = arg1357_1 = arg1358_1 = arg1359_1 = arg1360_1 = arg1361_1 = arg1362_1 = arg1363_1 = arg1364_1 = arg1365_1 = arg1366_1 = arg1367_1 = arg1368_1 = arg1369_1 = arg1370_1 = arg1371_1 = arg1372_1 = arg1373_1 = arg1374_1 = arg1375_1 = arg1376_1 = arg1377_1 = arg1378_1 = arg1379_1 = arg1380_1 = arg1381_1 = arg1382_1 = arg1383_1 = arg1384_1 = arg1385_1 = arg1386_1 = arg1387_1 = arg1388_1 = arg1389_1 = arg1390_1 = arg1391_1 = arg1392_1 = arg1393_1 = arg1394_1 = arg1395_1 = arg1396_1 = arg1397_1 = arg1398_1 = arg1399_1 = arg1400_1 = arg1401_1 = arg1402_1 = arg1403_1 = arg1404_1 = arg1405_1 = arg1406_1 = arg1407_1 = arg1408_1 = arg1409_1 = arg1410_1 = arg1411_1 = arg1412_1 = arg1413_1 = arg1414_1 = arg1415_1 = arg1416_1 = arg1417_1 = arg1418_1 = arg1419_1 = arg1420_1 = arg1421_1 = arg1422_1 = arg1423_1 = arg1424_1 = arg1425_1 = arg1426_1 = arg1427_1 = arg1428_1 = arg1429_1 = arg1430_1 = arg1431_1 = arg1432_1 = arg1433_1 = arg1434_1 = arg1435_1 = arg1436_1 = arg1437_1 = arg1438_1 = arg1439_1 = arg1440_1 = arg1441_1 = arg1442_1 = arg1443_1 = arg1444_1 = arg1445_1 = arg1446_1 = arg1447_1 = arg1448_1 = arg1449_1 = arg1450_1 = arg1451_1 = arg1452_1 = arg1453_1 = arg1454_1 = arg1455_1 = arg1456_1 = arg1457_1 = arg1458_1 = arg1459_1 = arg1460_1 = arg1461_1 = arg1462_1 = arg1463_1 = arg1464_1 = arg1465_1 = arg1466_1 = arg1467_1 = arg1468_1 = arg1469_1 = arg1470_1 = arg1471_1 = arg1472_1 = arg1473_1 = arg1474_1 = arg1475_1 = arg1476_1 = arg1477_1 = arg1478_1 = arg1479_1 = arg1480_1 = arg1481_1 = arg1482_1 = arg1483_1 = arg1484_1 = arg1485_1 = arg1486_1 = arg1487_1 = arg1488_1 = arg1489_1 = arg1490_1 = arg1491_1 = arg1492_1 = arg1493_1 = arg1494_1 = arg1495_1 = arg1496_1 = arg1497_1 = arg1498_1 = arg1499_1 = arg1500_1 = arg1501_1 = arg1502_1 = arg1503_1 = arg1504_1 = None
        getitem: "f32[768]" = _foreach_addcmul_scalar[0]
        getitem_1204: "f32[50, 768]" = _foreach_addcmul_scalar[1]
        getitem_1205: "f32[768, 512]" = _foreach_addcmul_scalar[2]
        getitem_1206: "f32[768, 3, 32, 32]" = _foreach_addcmul_scalar[3]
        getitem_1207: "f32[768]" = _foreach_addcmul_scalar[4]
        getitem_1208: "f32[768]" = _foreach_addcmul_scalar[5]
        getitem_1209: "f32[2304, 768]" = _foreach_addcmul_scalar[6]
        getitem_1210: "f32[2304]" = _foreach_addcmul_scalar[7]
        getitem_1211: "f32[768, 768]" = _foreach_addcmul_scalar[8]
        getitem_1212: "f32[768]" = _foreach_addcmul_scalar[9]
        getitem_1213: "f32[3072, 768]" = _foreach_addcmul_scalar[10]
        getitem_1214: "f32[3072]" = _foreach_addcmul_scalar[11]
        getitem_1215: "f32[768, 3072]" = _foreach_addcmul_scalar[12]
        getitem_1216: "f32[768]" = _foreach_addcmul_scalar[13]
        getitem_1217: "f32[768]" = _foreach_addcmul_scalar[14]
        getitem_1218: "f32[768]" = _foreach_addcmul_scalar[15]
        getitem_1219: "f32[768]" = _foreach_addcmul_scalar[16]
        getitem_1220: "f32[768]" = _foreach_addcmul_scalar[17]
        getitem_1221: "f32[2304, 768]" = _foreach_addcmul_scalar[18]
        getitem_1222: "f32[2304]" = _foreach_addcmul_scalar[19]
        getitem_1223: "f32[768, 768]" = _foreach_addcmul_scalar[20]
        getitem_1224: "f32[768]" = _foreach_addcmul_scalar[21]
        getitem_1225: "f32[3072, 768]" = _foreach_addcmul_scalar[22]
        getitem_1226: "f32[3072]" = _foreach_addcmul_scalar[23]
        getitem_1227: "f32[768, 3072]" = _foreach_addcmul_scalar[24]
        getitem_1228: "f32[768]" = _foreach_addcmul_scalar[25]
        getitem_1229: "f32[768]" = _foreach_addcmul_scalar[26]
        getitem_1230: "f32[768]" = _foreach_addcmul_scalar[27]
        getitem_1231: "f32[768]" = _foreach_addcmul_scalar[28]
        getitem_1232: "f32[768]" = _foreach_addcmul_scalar[29]
        getitem_1233: "f32[2304, 768]" = _foreach_addcmul_scalar[30]
        getitem_1234: "f32[2304]" = _foreach_addcmul_scalar[31]
        getitem_1235: "f32[768, 768]" = _foreach_addcmul_scalar[32]
        getitem_1236: "f32[768]" = _foreach_addcmul_scalar[33]
        getitem_1237: "f32[3072, 768]" = _foreach_addcmul_scalar[34]
        getitem_1238: "f32[3072]" = _foreach_addcmul_scalar[35]
        getitem_1239: "f32[768, 3072]" = _foreach_addcmul_scalar[36]
        getitem_1240: "f32[768]" = _foreach_addcmul_scalar[37]
        getitem_1241: "f32[768]" = _foreach_addcmul_scalar[38]
        getitem_1242: "f32[768]" = _foreach_addcmul_scalar[39]
        getitem_1243: "f32[768]" = _foreach_addcmul_scalar[40]
        getitem_1244: "f32[768]" = _foreach_addcmul_scalar[41]
        getitem_1245: "f32[2304, 768]" = _foreach_addcmul_scalar[42]
        getitem_1246: "f32[2304]" = _foreach_addcmul_scalar[43]
        getitem_1247: "f32[768, 768]" = _foreach_addcmul_scalar[44]
        getitem_1248: "f32[768]" = _foreach_addcmul_scalar[45]
        getitem_1249: "f32[3072, 768]" = _foreach_addcmul_scalar[46]
        getitem_1250: "f32[3072]" = _foreach_addcmul_scalar[47]
        getitem_1251: "f32[768, 3072]" = _foreach_addcmul_scalar[48]
        getitem_1252: "f32[768]" = _foreach_addcmul_scalar[49]
        getitem_1253: "f32[768]" = _foreach_addcmul_scalar[50]
        getitem_1254: "f32[768]" = _foreach_addcmul_scalar[51]
        getitem_1255: "f32[768]" = _foreach_addcmul_scalar[52]
        getitem_1256: "f32[768]" = _foreach_addcmul_scalar[53]
        getitem_1257: "f32[2304, 768]" = _foreach_addcmul_scalar[54]
        getitem_1258: "f32[2304]" = _foreach_addcmul_scalar[55]
        getitem_1259: "f32[768, 768]" = _foreach_addcmul_scalar[56]
        getitem_1260: "f32[768]" = _foreach_addcmul_scalar[57]
        getitem_1261: "f32[3072, 768]" = _foreach_addcmul_scalar[58]
        getitem_1262: "f32[3072]" = _foreach_addcmul_scalar[59]
        getitem_1263: "f32[768, 3072]" = _foreach_addcmul_scalar[60]
        getitem_1264: "f32[768]" = _foreach_addcmul_scalar[61]
        getitem_1265: "f32[768]" = _foreach_addcmul_scalar[62]
        getitem_1266: "f32[768]" = _foreach_addcmul_scalar[63]
        getitem_1267: "f32[768]" = _foreach_addcmul_scalar[64]
        getitem_1268: "f32[768]" = _foreach_addcmul_scalar[65]
        getitem_1269: "f32[2304, 768]" = _foreach_addcmul_scalar[66]
        getitem_1270: "f32[2304]" = _foreach_addcmul_scalar[67]
        getitem_1271: "f32[768, 768]" = _foreach_addcmul_scalar[68]
        getitem_1272: "f32[768]" = _foreach_addcmul_scalar[69]
        getitem_1273: "f32[3072, 768]" = _foreach_addcmul_scalar[70]
        getitem_1274: "f32[3072]" = _foreach_addcmul_scalar[71]
        getitem_1275: "f32[768, 3072]" = _foreach_addcmul_scalar[72]
        getitem_1276: "f32[768]" = _foreach_addcmul_scalar[73]
        getitem_1277: "f32[768]" = _foreach_addcmul_scalar[74]
        getitem_1278: "f32[768]" = _foreach_addcmul_scalar[75]
        getitem_1279: "f32[768]" = _foreach_addcmul_scalar[76]
        getitem_1280: "f32[768]" = _foreach_addcmul_scalar[77]
        getitem_1281: "f32[2304, 768]" = _foreach_addcmul_scalar[78]
        getitem_1282: "f32[2304]" = _foreach_addcmul_scalar[79]
        getitem_1283: "f32[768, 768]" = _foreach_addcmul_scalar[80]
        getitem_1284: "f32[768]" = _foreach_addcmul_scalar[81]
        getitem_1285: "f32[3072, 768]" = _foreach_addcmul_scalar[82]
        getitem_1286: "f32[3072]" = _foreach_addcmul_scalar[83]
        getitem_1287: "f32[768, 3072]" = _foreach_addcmul_scalar[84]
        getitem_1288: "f32[768]" = _foreach_addcmul_scalar[85]
        getitem_1289: "f32[768]" = _foreach_addcmul_scalar[86]
        getitem_1290: "f32[768]" = _foreach_addcmul_scalar[87]
        getitem_1291: "f32[768]" = _foreach_addcmul_scalar[88]
        getitem_1292: "f32[768]" = _foreach_addcmul_scalar[89]
        getitem_1293: "f32[2304, 768]" = _foreach_addcmul_scalar[90]
        getitem_1294: "f32[2304]" = _foreach_addcmul_scalar[91]
        getitem_1295: "f32[768, 768]" = _foreach_addcmul_scalar[92]
        getitem_1296: "f32[768]" = _foreach_addcmul_scalar[93]
        getitem_1297: "f32[3072, 768]" = _foreach_addcmul_scalar[94]
        getitem_1298: "f32[3072]" = _foreach_addcmul_scalar[95]
        getitem_1299: "f32[768, 3072]" = _foreach_addcmul_scalar[96]
        getitem_1300: "f32[768]" = _foreach_addcmul_scalar[97]
        getitem_1301: "f32[768]" = _foreach_addcmul_scalar[98]
        getitem_1302: "f32[768]" = _foreach_addcmul_scalar[99]
        getitem_1303: "f32[768]" = _foreach_addcmul_scalar[100]
        getitem_1304: "f32[768]" = _foreach_addcmul_scalar[101]
        getitem_1305: "f32[2304, 768]" = _foreach_addcmul_scalar[102]
        getitem_1306: "f32[2304]" = _foreach_addcmul_scalar[103]
        getitem_1307: "f32[768, 768]" = _foreach_addcmul_scalar[104]
        getitem_1308: "f32[768]" = _foreach_addcmul_scalar[105]
        getitem_1309: "f32[3072, 768]" = _foreach_addcmul_scalar[106]
        getitem_1310: "f32[3072]" = _foreach_addcmul_scalar[107]
        getitem_1311: "f32[768, 3072]" = _foreach_addcmul_scalar[108]
        getitem_1312: "f32[768]" = _foreach_addcmul_scalar[109]
        getitem_1313: "f32[768]" = _foreach_addcmul_scalar[110]
        getitem_1314: "f32[768]" = _foreach_addcmul_scalar[111]
        getitem_1315: "f32[768]" = _foreach_addcmul_scalar[112]
        getitem_1316: "f32[768]" = _foreach_addcmul_scalar[113]
        getitem_1317: "f32[2304, 768]" = _foreach_addcmul_scalar[114]
        getitem_1318: "f32[2304]" = _foreach_addcmul_scalar[115]
        getitem_1319: "f32[768, 768]" = _foreach_addcmul_scalar[116]
        getitem_1320: "f32[768]" = _foreach_addcmul_scalar[117]
        getitem_1321: "f32[3072, 768]" = _foreach_addcmul_scalar[118]
        getitem_1322: "f32[3072]" = _foreach_addcmul_scalar[119]
        getitem_1323: "f32[768, 3072]" = _foreach_addcmul_scalar[120]
        getitem_1324: "f32[768]" = _foreach_addcmul_scalar[121]
        getitem_1325: "f32[768]" = _foreach_addcmul_scalar[122]
        getitem_1326: "f32[768]" = _foreach_addcmul_scalar[123]
        getitem_1327: "f32[768]" = _foreach_addcmul_scalar[124]
        getitem_1328: "f32[768]" = _foreach_addcmul_scalar[125]
        getitem_1329: "f32[2304, 768]" = _foreach_addcmul_scalar[126]
        getitem_1330: "f32[2304]" = _foreach_addcmul_scalar[127]
        getitem_1331: "f32[768, 768]" = _foreach_addcmul_scalar[128]
        getitem_1332: "f32[768]" = _foreach_addcmul_scalar[129]
        getitem_1333: "f32[3072, 768]" = _foreach_addcmul_scalar[130]
        getitem_1334: "f32[3072]" = _foreach_addcmul_scalar[131]
        getitem_1335: "f32[768, 3072]" = _foreach_addcmul_scalar[132]
        getitem_1336: "f32[768]" = _foreach_addcmul_scalar[133]
        getitem_1337: "f32[768]" = _foreach_addcmul_scalar[134]
        getitem_1338: "f32[768]" = _foreach_addcmul_scalar[135]
        getitem_1339: "f32[768]" = _foreach_addcmul_scalar[136]
        getitem_1340: "f32[768]" = _foreach_addcmul_scalar[137]
        getitem_1341: "f32[2304, 768]" = _foreach_addcmul_scalar[138]
        getitem_1342: "f32[2304]" = _foreach_addcmul_scalar[139]
        getitem_1343: "f32[768, 768]" = _foreach_addcmul_scalar[140]
        getitem_1344: "f32[768]" = _foreach_addcmul_scalar[141]
        getitem_1345: "f32[3072, 768]" = _foreach_addcmul_scalar[142]
        getitem_1346: "f32[3072]" = _foreach_addcmul_scalar[143]
        getitem_1347: "f32[768, 3072]" = _foreach_addcmul_scalar[144]
        getitem_1348: "f32[768]" = _foreach_addcmul_scalar[145]
        getitem_1349: "f32[768]" = _foreach_addcmul_scalar[146]
        getitem_1350: "f32[768]" = _foreach_addcmul_scalar[147]
        getitem_1351: "f32[768]" = _foreach_addcmul_scalar[148]
        getitem_1352: "f32[768]" = _foreach_addcmul_scalar[149]
        getitem_1353: "f32[768]" = _foreach_addcmul_scalar[150]
        getitem_1354: "f32[768]" = _foreach_addcmul_scalar[151]
        getitem_1355: "f32[77, 512]" = _foreach_addcmul_scalar[152]
        getitem_1356: "f32[49408, 512]" = _foreach_addcmul_scalar[153]
        getitem_1357: "f32[1536, 512]" = _foreach_addcmul_scalar[154]
        getitem_1358: "f32[1536]" = _foreach_addcmul_scalar[155]
        getitem_1359: "f32[512, 512]" = _foreach_addcmul_scalar[156]
        getitem_1360: "f32[512]" = _foreach_addcmul_scalar[157]
        getitem_1361: "f32[2048, 512]" = _foreach_addcmul_scalar[158]
        getitem_1362: "f32[2048]" = _foreach_addcmul_scalar[159]
        getitem_1363: "f32[512, 2048]" = _foreach_addcmul_scalar[160]
        getitem_1364: "f32[512]" = _foreach_addcmul_scalar[161]
        getitem_1365: "f32[512]" = _foreach_addcmul_scalar[162]
        getitem_1366: "f32[512]" = _foreach_addcmul_scalar[163]
        getitem_1367: "f32[512]" = _foreach_addcmul_scalar[164]
        getitem_1368: "f32[512]" = _foreach_addcmul_scalar[165]
        getitem_1369: "f32[1536, 512]" = _foreach_addcmul_scalar[166]
        getitem_1370: "f32[1536]" = _foreach_addcmul_scalar[167]
        getitem_1371: "f32[512, 512]" = _foreach_addcmul_scalar[168]
        getitem_1372: "f32[512]" = _foreach_addcmul_scalar[169]
        getitem_1373: "f32[2048, 512]" = _foreach_addcmul_scalar[170]
        getitem_1374: "f32[2048]" = _foreach_addcmul_scalar[171]
        getitem_1375: "f32[512, 2048]" = _foreach_addcmul_scalar[172]
        getitem_1376: "f32[512]" = _foreach_addcmul_scalar[173]
        getitem_1377: "f32[512]" = _foreach_addcmul_scalar[174]
        getitem_1378: "f32[512]" = _foreach_addcmul_scalar[175]
        getitem_1379: "f32[512]" = _foreach_addcmul_scalar[176]
        getitem_1380: "f32[512]" = _foreach_addcmul_scalar[177]
        getitem_1381: "f32[1536, 512]" = _foreach_addcmul_scalar[178]
        getitem_1382: "f32[1536]" = _foreach_addcmul_scalar[179]
        getitem_1383: "f32[512, 512]" = _foreach_addcmul_scalar[180]
        getitem_1384: "f32[512]" = _foreach_addcmul_scalar[181]
        getitem_1385: "f32[2048, 512]" = _foreach_addcmul_scalar[182]
        getitem_1386: "f32[2048]" = _foreach_addcmul_scalar[183]
        getitem_1387: "f32[512, 2048]" = _foreach_addcmul_scalar[184]
        getitem_1388: "f32[512]" = _foreach_addcmul_scalar[185]
        getitem_1389: "f32[512]" = _foreach_addcmul_scalar[186]
        getitem_1390: "f32[512]" = _foreach_addcmul_scalar[187]
        getitem_1391: "f32[512]" = _foreach_addcmul_scalar[188]
        getitem_1392: "f32[512]" = _foreach_addcmul_scalar[189]
        getitem_1393: "f32[1536, 512]" = _foreach_addcmul_scalar[190]
        getitem_1394: "f32[1536]" = _foreach_addcmul_scalar[191]
        getitem_1395: "f32[512, 512]" = _foreach_addcmul_scalar[192]
        getitem_1396: "f32[512]" = _foreach_addcmul_scalar[193]
        getitem_1397: "f32[2048, 512]" = _foreach_addcmul_scalar[194]
        getitem_1398: "f32[2048]" = _foreach_addcmul_scalar[195]
        getitem_1399: "f32[512, 2048]" = _foreach_addcmul_scalar[196]
        getitem_1400: "f32[512]" = _foreach_addcmul_scalar[197]
        getitem_1401: "f32[512]" = _foreach_addcmul_scalar[198]
        getitem_1402: "f32[512]" = _foreach_addcmul_scalar[199]
        getitem_1403: "f32[512]" = _foreach_addcmul_scalar[200]
        getitem_1404: "f32[512]" = _foreach_addcmul_scalar[201]
        getitem_1405: "f32[1536, 512]" = _foreach_addcmul_scalar[202]
        getitem_1406: "f32[1536]" = _foreach_addcmul_scalar[203]
        getitem_1407: "f32[512, 512]" = _foreach_addcmul_scalar[204]
        getitem_1408: "f32[512]" = _foreach_addcmul_scalar[205]
        getitem_1409: "f32[2048, 512]" = _foreach_addcmul_scalar[206]
        getitem_1410: "f32[2048]" = _foreach_addcmul_scalar[207]
        getitem_1411: "f32[512, 2048]" = _foreach_addcmul_scalar[208]
        getitem_1412: "f32[512]" = _foreach_addcmul_scalar[209]
        getitem_1413: "f32[512]" = _foreach_addcmul_scalar[210]
        getitem_1414: "f32[512]" = _foreach_addcmul_scalar[211]
        getitem_1415: "f32[512]" = _foreach_addcmul_scalar[212]
        getitem_1416: "f32[512]" = _foreach_addcmul_scalar[213]
        getitem_1417: "f32[1536, 512]" = _foreach_addcmul_scalar[214]
        getitem_1418: "f32[1536]" = _foreach_addcmul_scalar[215]
        getitem_1419: "f32[512, 512]" = _foreach_addcmul_scalar[216]
        getitem_1420: "f32[512]" = _foreach_addcmul_scalar[217]
        getitem_1421: "f32[2048, 512]" = _foreach_addcmul_scalar[218]
        getitem_1422: "f32[2048]" = _foreach_addcmul_scalar[219]
        getitem_1423: "f32[512, 2048]" = _foreach_addcmul_scalar[220]
        getitem_1424: "f32[512]" = _foreach_addcmul_scalar[221]
        getitem_1425: "f32[512]" = _foreach_addcmul_scalar[222]
        getitem_1426: "f32[512]" = _foreach_addcmul_scalar[223]
        getitem_1427: "f32[512]" = _foreach_addcmul_scalar[224]
        getitem_1428: "f32[512]" = _foreach_addcmul_scalar[225]
        getitem_1429: "f32[1536, 512]" = _foreach_addcmul_scalar[226]
        getitem_1430: "f32[1536]" = _foreach_addcmul_scalar[227]
        getitem_1431: "f32[512, 512]" = _foreach_addcmul_scalar[228]
        getitem_1432: "f32[512]" = _foreach_addcmul_scalar[229]
        getitem_1433: "f32[2048, 512]" = _foreach_addcmul_scalar[230]
        getitem_1434: "f32[2048]" = _foreach_addcmul_scalar[231]
        getitem_1435: "f32[512, 2048]" = _foreach_addcmul_scalar[232]
        getitem_1436: "f32[512]" = _foreach_addcmul_scalar[233]
        getitem_1437: "f32[512]" = _foreach_addcmul_scalar[234]
        getitem_1438: "f32[512]" = _foreach_addcmul_scalar[235]
        getitem_1439: "f32[512]" = _foreach_addcmul_scalar[236]
        getitem_1440: "f32[512]" = _foreach_addcmul_scalar[237]
        getitem_1441: "f32[1536, 512]" = _foreach_addcmul_scalar[238]
        getitem_1442: "f32[1536]" = _foreach_addcmul_scalar[239]
        getitem_1443: "f32[512, 512]" = _foreach_addcmul_scalar[240]
        getitem_1444: "f32[512]" = _foreach_addcmul_scalar[241]
        getitem_1445: "f32[2048, 512]" = _foreach_addcmul_scalar[242]
        getitem_1446: "f32[2048]" = _foreach_addcmul_scalar[243]
        getitem_1447: "f32[512, 2048]" = _foreach_addcmul_scalar[244]
        getitem_1448: "f32[512]" = _foreach_addcmul_scalar[245]
        getitem_1449: "f32[512]" = _foreach_addcmul_scalar[246]
        getitem_1450: "f32[512]" = _foreach_addcmul_scalar[247]
        getitem_1451: "f32[512]" = _foreach_addcmul_scalar[248]
        getitem_1452: "f32[512]" = _foreach_addcmul_scalar[249]
        getitem_1453: "f32[1536, 512]" = _foreach_addcmul_scalar[250]
        getitem_1454: "f32[1536]" = _foreach_addcmul_scalar[251]
        getitem_1455: "f32[512, 512]" = _foreach_addcmul_scalar[252]
        getitem_1456: "f32[512]" = _foreach_addcmul_scalar[253]
        getitem_1457: "f32[2048, 512]" = _foreach_addcmul_scalar[254]
        getitem_1458: "f32[2048]" = _foreach_addcmul_scalar[255]
        getitem_1459: "f32[512, 2048]" = _foreach_addcmul_scalar[256]
        getitem_1460: "f32[512]" = _foreach_addcmul_scalar[257]
        getitem_1461: "f32[512]" = _foreach_addcmul_scalar[258]
        getitem_1462: "f32[512]" = _foreach_addcmul_scalar[259]
        getitem_1463: "f32[512]" = _foreach_addcmul_scalar[260]
        getitem_1464: "f32[512]" = _foreach_addcmul_scalar[261]
        getitem_1465: "f32[1536, 512]" = _foreach_addcmul_scalar[262]
        getitem_1466: "f32[1536]" = _foreach_addcmul_scalar[263]
        getitem_1467: "f32[512, 512]" = _foreach_addcmul_scalar[264]
        getitem_1468: "f32[512]" = _foreach_addcmul_scalar[265]
        getitem_1469: "f32[2048, 512]" = _foreach_addcmul_scalar[266]
        getitem_1470: "f32[2048]" = _foreach_addcmul_scalar[267]
        getitem_1471: "f32[512, 2048]" = _foreach_addcmul_scalar[268]
        getitem_1472: "f32[512]" = _foreach_addcmul_scalar[269]
        getitem_1473: "f32[512]" = _foreach_addcmul_scalar[270]
        getitem_1474: "f32[512]" = _foreach_addcmul_scalar[271]
        getitem_1475: "f32[512]" = _foreach_addcmul_scalar[272]
        getitem_1476: "f32[512]" = _foreach_addcmul_scalar[273]
        getitem_1477: "f32[1536, 512]" = _foreach_addcmul_scalar[274]
        getitem_1478: "f32[1536]" = _foreach_addcmul_scalar[275]
        getitem_1479: "f32[512, 512]" = _foreach_addcmul_scalar[276]
        getitem_1480: "f32[512]" = _foreach_addcmul_scalar[277]
        getitem_1481: "f32[2048, 512]" = _foreach_addcmul_scalar[278]
        getitem_1482: "f32[2048]" = _foreach_addcmul_scalar[279]
        getitem_1483: "f32[512, 2048]" = _foreach_addcmul_scalar[280]
        getitem_1484: "f32[512]" = _foreach_addcmul_scalar[281]
        getitem_1485: "f32[512]" = _foreach_addcmul_scalar[282]
        getitem_1486: "f32[512]" = _foreach_addcmul_scalar[283]
        getitem_1487: "f32[512]" = _foreach_addcmul_scalar[284]
        getitem_1488: "f32[512]" = _foreach_addcmul_scalar[285]
        getitem_1489: "f32[1536, 512]" = _foreach_addcmul_scalar[286]
        getitem_1490: "f32[1536]" = _foreach_addcmul_scalar[287]
        getitem_1491: "f32[512, 512]" = _foreach_addcmul_scalar[288]
        getitem_1492: "f32[512]" = _foreach_addcmul_scalar[289]
        getitem_1493: "f32[2048, 512]" = _foreach_addcmul_scalar[290]
        getitem_1494: "f32[2048]" = _foreach_addcmul_scalar[291]
        getitem_1495: "f32[512, 2048]" = _foreach_addcmul_scalar[292]
        getitem_1496: "f32[512]" = _foreach_addcmul_scalar[293]
        getitem_1497: "f32[512]" = _foreach_addcmul_scalar[294]
        getitem_1498: "f32[512]" = _foreach_addcmul_scalar[295]
        getitem_1499: "f32[512]" = _foreach_addcmul_scalar[296]
        getitem_1500: "f32[512]" = _foreach_addcmul_scalar[297]
        getitem_1501: "f32[512]" = _foreach_addcmul_scalar[298]
        getitem_1502: "f32[512]" = _foreach_addcmul_scalar[299]
        getitem_1503: "f32[512, 512]" = _foreach_addcmul_scalar[300];  _foreach_addcmul_scalar = None
        getitem_1504 = getitem
        _foreach_pow_scalar_and_tensor = torch.ops.aten._foreach_pow.ScalarAndTensor(0.9, [getitem_1504, getitem_1, getitem_2, getitem_3, getitem_4, getitem_5, getitem_6, getitem_7, getitem_8, getitem_9, getitem_10, getitem_11, getitem_12, getitem_13, getitem_14, getitem_15, getitem_16, getitem_17, getitem_18, getitem_19, getitem_20, getitem_21, getitem_22, getitem_23, getitem_24, getitem_25, getitem_26, getitem_27, getitem_28, getitem_29, getitem_30, getitem_31, getitem_32, getitem_33, getitem_34, getitem_35, getitem_36, getitem_37, getitem_38, getitem_39, getitem_40, getitem_41, getitem_42, getitem_43, getitem_44, getitem_45, getitem_46, getitem_47, getitem_48, getitem_49, getitem_50, getitem_51, getitem_52, getitem_53, getitem_54, getitem_55, getitem_56, getitem_57, getitem_58, getitem_59, getitem_60, getitem_61, getitem_62, getitem_63, getitem_64, getitem_65, getitem_66, getitem_67, getitem_68, getitem_69, getitem_70, getitem_71, getitem_72, getitem_73, getitem_74, getitem_75, getitem_76, getitem_77, getitem_78, getitem_79, getitem_80, getitem_81, getitem_82, getitem_83, getitem_84, getitem_85, getitem_86, getitem_87, getitem_88, getitem_89, getitem_90, getitem_91, getitem_92, getitem_93, getitem_94, getitem_95, getitem_96, getitem_97, getitem_98, getitem_99, getitem_100, getitem_101, getitem_102, getitem_103, getitem_104, getitem_105, getitem_106, getitem_107, getitem_108, getitem_109, getitem_110, getitem_111, getitem_112, getitem_113, getitem_114, getitem_115, getitem_116, getitem_117, getitem_118, getitem_119, getitem_120, getitem_121, getitem_122, getitem_123, getitem_124, getitem_125, getitem_126, getitem_127, getitem_128, getitem_129, getitem_130, getitem_131, getitem_132, getitem_133, getitem_134, getitem_135, getitem_136, getitem_137, getitem_138, getitem_139, getitem_140, getitem_141, getitem_142, getitem_143, getitem_144, getitem_145, getitem_146, getitem_147, getitem_148, getitem_149, getitem_150, getitem_151, getitem_152, getitem_153, getitem_154, getitem_155, getitem_156, getitem_157, getitem_158, getitem_159, getitem_160, getitem_161, getitem_162, getitem_163, getitem_164, getitem_165, getitem_166, getitem_167, getitem_168, getitem_169, getitem_170, getitem_171, getitem_172, getitem_173, getitem_174, getitem_175, getitem_176, getitem_177, getitem_178, getitem_179, getitem_180, getitem_181, getitem_182, getitem_183, getitem_184, getitem_185, getitem_186, getitem_187, getitem_188, getitem_189, getitem_190, getitem_191, getitem_192, getitem_193, getitem_194, getitem_195, getitem_196, getitem_197, getitem_198, getitem_199, getitem_200, getitem_201, getitem_202, getitem_203, getitem_204, getitem_205, getitem_206, getitem_207, getitem_208, getitem_209, getitem_210, getitem_211, getitem_212, getitem_213, getitem_214, getitem_215, getitem_216, getitem_217, getitem_218, getitem_219, getitem_220, getitem_221, getitem_222, getitem_223, getitem_224, getitem_225, getitem_226, getitem_227, getitem_228, getitem_229, getitem_230, getitem_231, getitem_232, getitem_233, getitem_234, getitem_235, getitem_236, getitem_237, getitem_238, getitem_239, getitem_240, getitem_241, getitem_242, getitem_243, getitem_244, getitem_245, getitem_246, getitem_247, getitem_248, getitem_249, getitem_250, getitem_251, getitem_252, getitem_253, getitem_254, getitem_255, getitem_256, getitem_257, getitem_258, getitem_259, getitem_260, getitem_261, getitem_262, getitem_263, getitem_264, getitem_265, getitem_266, getitem_267, getitem_268, getitem_269, getitem_270, getitem_271, getitem_272, getitem_273, getitem_274, getitem_275, getitem_276, getitem_277, getitem_278, getitem_279, getitem_280, getitem_281, getitem_282, getitem_283, getitem_284, getitem_285, getitem_286, getitem_287, getitem_288, getitem_289, getitem_290, getitem_291, getitem_292, getitem_293, getitem_294, getitem_295, getitem_296, getitem_297, getitem_298, getitem_299, getitem_300]);  getitem_1504 = getitem_1 = getitem_2 = getitem_3 = getitem_4 = getitem_5 = getitem_6 = getitem_7 = getitem_8 = getitem_9 = getitem_10 = getitem_11 = getitem_12 = getitem_13 = getitem_14 = getitem_15 = getitem_16 = getitem_17 = getitem_18 = getitem_19 = getitem_20 = getitem_21 = getitem_22 = getitem_23 = getitem_24 = getitem_25 = getitem_26 = getitem_27 = getitem_28 = getitem_29 = getitem_30 = getitem_31 = getitem_32 = getitem_33 = getitem_34 = getitem_35 = getitem_36 = getitem_37 = getitem_38 = getitem_39 = getitem_40 = getitem_41 = getitem_42 = getitem_43 = getitem_44 = getitem_45 = getitem_46 = getitem_47 = getitem_48 = getitem_49 = getitem_50 = getitem_51 = getitem_52 = getitem_53 = getitem_54 = getitem_55 = getitem_56 = getitem_57 = getitem_58 = getitem_59 = getitem_60 = getitem_61 = getitem_62 = getitem_63 = getitem_64 = getitem_65 = getitem_66 = getitem_67 = getitem_68 = getitem_69 = getitem_70 = getitem_71 = getitem_72 = getitem_73 = getitem_74 = getitem_75 = getitem_76 = getitem_77 = getitem_78 = getitem_79 = getitem_80 = getitem_81 = getitem_82 = getitem_83 = getitem_84 = getitem_85 = getitem_86 = getitem_87 = getitem_88 = getitem_89 = getitem_90 = getitem_91 = getitem_92 = getitem_93 = getitem_94 = getitem_95 = getitem_96 = getitem_97 = getitem_98 = getitem_99 = getitem_100 = getitem_101 = getitem_102 = getitem_103 = getitem_104 = getitem_105 = getitem_106 = getitem_107 = getitem_108 = getitem_109 = getitem_110 = getitem_111 = getitem_112 = getitem_113 = getitem_114 = getitem_115 = getitem_116 = getitem_117 = getitem_118 = getitem_119 = getitem_120 = getitem_121 = getitem_122 = getitem_123 = getitem_124 = getitem_125 = getitem_126 = getitem_127 = getitem_128 = getitem_129 = getitem_130 = getitem_131 = getitem_132 = getitem_133 = getitem_134 = getitem_135 = getitem_136 = getitem_137 = getitem_138 = getitem_139 = getitem_140 = getitem_141 = getitem_142 = getitem_143 = getitem_144 = getitem_145 = getitem_146 = getitem_147 = getitem_148 = getitem_149 = getitem_150 = getitem_151 = getitem_152 = getitem_153 = getitem_154 = getitem_155 = getitem_156 = getitem_157 = getitem_158 = getitem_159 = getitem_160 = getitem_161 = getitem_162 = getitem_163 = getitem_164 = getitem_165 = getitem_166 = getitem_167 = getitem_168 = getitem_169 = getitem_170 = getitem_171 = getitem_172 = getitem_173 = getitem_174 = getitem_175 = getitem_176 = getitem_177 = getitem_178 = getitem_179 = getitem_180 = getitem_181 = getitem_182 = getitem_183 = getitem_184 = getitem_185 = getitem_186 = getitem_187 = getitem_188 = getitem_189 = getitem_190 = getitem_191 = getitem_192 = getitem_193 = getitem_194 = getitem_195 = getitem_196 = getitem_197 = getitem_198 = getitem_199 = getitem_200 = getitem_201 = getitem_202 = getitem_203 = getitem_204 = getitem_205 = getitem_206 = getitem_207 = getitem_208 = getitem_209 = getitem_210 = getitem_211 = getitem_212 = getitem_213 = getitem_214 = getitem_215 = getitem_216 = getitem_217 = getitem_218 = getitem_219 = getitem_220 = getitem_221 = getitem_222 = getitem_223 = getitem_224 = getitem_225 = getitem_226 = getitem_227 = getitem_228 = getitem_229 = getitem_230 = getitem_231 = getitem_232 = getitem_233 = getitem_234 = getitem_235 = getitem_236 = getitem_237 = getitem_238 = getitem_239 = getitem_240 = getitem_241 = getitem_242 = getitem_243 = getitem_244 = getitem_245 = getitem_246 = getitem_247 = getitem_248 = getitem_249 = getitem_250 = getitem_251 = getitem_252 = getitem_253 = getitem_254 = getitem_255 = getitem_256 = getitem_257 = getitem_258 = getitem_259 = getitem_260 = getitem_261 = getitem_262 = getitem_263 = getitem_264 = getitem_265 = getitem_266 = getitem_267 = getitem_268 = getitem_269 = getitem_270 = getitem_271 = getitem_272 = getitem_273 = getitem_274 = getitem_275 = getitem_276 = getitem_277 = getitem_278 = getitem_279 = getitem_280 = getitem_281 = getitem_282 = getitem_283 = getitem_284 = getitem_285 = getitem_286 = getitem_287 = getitem_288 = getitem_289 = getitem_290 = getitem_291 = getitem_292 = getitem_293 = getitem_294 = getitem_295 = getitem_296 = getitem_297 = getitem_298 = getitem_299 = getitem_300 = None
        getitem_301: "f32[]" = _foreach_pow_scalar_and_tensor[0]
        getitem_302: "f32[]" = _foreach_pow_scalar_and_tensor[1]
        getitem_303: "f32[]" = _foreach_pow_scalar_and_tensor[2]
        getitem_304: "f32[]" = _foreach_pow_scalar_and_tensor[3]
        getitem_305: "f32[]" = _foreach_pow_scalar_and_tensor[4]
        getitem_306: "f32[]" = _foreach_pow_scalar_and_tensor[5]
        getitem_307: "f32[]" = _foreach_pow_scalar_and_tensor[6]
        getitem_308: "f32[]" = _foreach_pow_scalar_and_tensor[7]
        getitem_309: "f32[]" = _foreach_pow_scalar_and_tensor[8]
        getitem_310: "f32[]" = _foreach_pow_scalar_and_tensor[9]
        getitem_311: "f32[]" = _foreach_pow_scalar_and_tensor[10]
        getitem_312: "f32[]" = _foreach_pow_scalar_and_tensor[11]
        getitem_313: "f32[]" = _foreach_pow_scalar_and_tensor[12]
        getitem_314: "f32[]" = _foreach_pow_scalar_and_tensor[13]
        getitem_315: "f32[]" = _foreach_pow_scalar_and_tensor[14]
        getitem_316: "f32[]" = _foreach_pow_scalar_and_tensor[15]
        getitem_317: "f32[]" = _foreach_pow_scalar_and_tensor[16]
        getitem_318: "f32[]" = _foreach_pow_scalar_and_tensor[17]
        getitem_319: "f32[]" = _foreach_pow_scalar_and_tensor[18]
        getitem_320: "f32[]" = _foreach_pow_scalar_and_tensor[19]
        getitem_321: "f32[]" = _foreach_pow_scalar_and_tensor[20]
        getitem_322: "f32[]" = _foreach_pow_scalar_and_tensor[21]
        getitem_323: "f32[]" = _foreach_pow_scalar_and_tensor[22]
        getitem_324: "f32[]" = _foreach_pow_scalar_and_tensor[23]
        getitem_325: "f32[]" = _foreach_pow_scalar_and_tensor[24]
        getitem_326: "f32[]" = _foreach_pow_scalar_and_tensor[25]
        getitem_327: "f32[]" = _foreach_pow_scalar_and_tensor[26]
        getitem_328: "f32[]" = _foreach_pow_scalar_and_tensor[27]
        getitem_329: "f32[]" = _foreach_pow_scalar_and_tensor[28]
        getitem_330: "f32[]" = _foreach_pow_scalar_and_tensor[29]
        getitem_331: "f32[]" = _foreach_pow_scalar_and_tensor[30]
        getitem_332: "f32[]" = _foreach_pow_scalar_and_tensor[31]
        getitem_333: "f32[]" = _foreach_pow_scalar_and_tensor[32]
        getitem_334: "f32[]" = _foreach_pow_scalar_and_tensor[33]
        getitem_335: "f32[]" = _foreach_pow_scalar_and_tensor[34]
        getitem_336: "f32[]" = _foreach_pow_scalar_and_tensor[35]
        getitem_337: "f32[]" = _foreach_pow_scalar_and_tensor[36]
        getitem_338: "f32[]" = _foreach_pow_scalar_and_tensor[37]
        getitem_339: "f32[]" = _foreach_pow_scalar_and_tensor[38]
        getitem_340: "f32[]" = _foreach_pow_scalar_and_tensor[39]
        getitem_341: "f32[]" = _foreach_pow_scalar_and_tensor[40]
        getitem_342: "f32[]" = _foreach_pow_scalar_and_tensor[41]
        getitem_343: "f32[]" = _foreach_pow_scalar_and_tensor[42]
        getitem_344: "f32[]" = _foreach_pow_scalar_and_tensor[43]
        getitem_345: "f32[]" = _foreach_pow_scalar_and_tensor[44]
        getitem_346: "f32[]" = _foreach_pow_scalar_and_tensor[45]
        getitem_347: "f32[]" = _foreach_pow_scalar_and_tensor[46]
        getitem_348: "f32[]" = _foreach_pow_scalar_and_tensor[47]
        getitem_349: "f32[]" = _foreach_pow_scalar_and_tensor[48]
        getitem_350: "f32[]" = _foreach_pow_scalar_and_tensor[49]
        getitem_351: "f32[]" = _foreach_pow_scalar_and_tensor[50]
        getitem_352: "f32[]" = _foreach_pow_scalar_and_tensor[51]
        getitem_353: "f32[]" = _foreach_pow_scalar_and_tensor[52]
        getitem_354: "f32[]" = _foreach_pow_scalar_and_tensor[53]
        getitem_355: "f32[]" = _foreach_pow_scalar_and_tensor[54]
        getitem_356: "f32[]" = _foreach_pow_scalar_and_tensor[55]
        getitem_357: "f32[]" = _foreach_pow_scalar_and_tensor[56]
        getitem_358: "f32[]" = _foreach_pow_scalar_and_tensor[57]
        getitem_359: "f32[]" = _foreach_pow_scalar_and_tensor[58]
        getitem_360: "f32[]" = _foreach_pow_scalar_and_tensor[59]
        getitem_361: "f32[]" = _foreach_pow_scalar_and_tensor[60]
        getitem_362: "f32[]" = _foreach_pow_scalar_and_tensor[61]
        getitem_363: "f32[]" = _foreach_pow_scalar_and_tensor[62]
        getitem_364: "f32[]" = _foreach_pow_scalar_and_tensor[63]
        getitem_365: "f32[]" = _foreach_pow_scalar_and_tensor[64]
        getitem_366: "f32[]" = _foreach_pow_scalar_and_tensor[65]
        getitem_367: "f32[]" = _foreach_pow_scalar_and_tensor[66]
        getitem_368: "f32[]" = _foreach_pow_scalar_and_tensor[67]
        getitem_369: "f32[]" = _foreach_pow_scalar_and_tensor[68]
        getitem_370: "f32[]" = _foreach_pow_scalar_and_tensor[69]
        getitem_371: "f32[]" = _foreach_pow_scalar_and_tensor[70]
        getitem_372: "f32[]" = _foreach_pow_scalar_and_tensor[71]
        getitem_373: "f32[]" = _foreach_pow_scalar_and_tensor[72]
        getitem_374: "f32[]" = _foreach_pow_scalar_and_tensor[73]
        getitem_375: "f32[]" = _foreach_pow_scalar_and_tensor[74]
        getitem_376: "f32[]" = _foreach_pow_scalar_and_tensor[75]
        getitem_377: "f32[]" = _foreach_pow_scalar_and_tensor[76]
        getitem_378: "f32[]" = _foreach_pow_scalar_and_tensor[77]
        getitem_379: "f32[]" = _foreach_pow_scalar_and_tensor[78]
        getitem_380: "f32[]" = _foreach_pow_scalar_and_tensor[79]
        getitem_381: "f32[]" = _foreach_pow_scalar_and_tensor[80]
        getitem_382: "f32[]" = _foreach_pow_scalar_and_tensor[81]
        getitem_383: "f32[]" = _foreach_pow_scalar_and_tensor[82]
        getitem_384: "f32[]" = _foreach_pow_scalar_and_tensor[83]
        getitem_385: "f32[]" = _foreach_pow_scalar_and_tensor[84]
        getitem_386: "f32[]" = _foreach_pow_scalar_and_tensor[85]
        getitem_387: "f32[]" = _foreach_pow_scalar_and_tensor[86]
        getitem_388: "f32[]" = _foreach_pow_scalar_and_tensor[87]
        getitem_389: "f32[]" = _foreach_pow_scalar_and_tensor[88]
        getitem_390: "f32[]" = _foreach_pow_scalar_and_tensor[89]
        getitem_391: "f32[]" = _foreach_pow_scalar_and_tensor[90]
        getitem_392: "f32[]" = _foreach_pow_scalar_and_tensor[91]
        getitem_393: "f32[]" = _foreach_pow_scalar_and_tensor[92]
        getitem_394: "f32[]" = _foreach_pow_scalar_and_tensor[93]
        getitem_395: "f32[]" = _foreach_pow_scalar_and_tensor[94]
        getitem_396: "f32[]" = _foreach_pow_scalar_and_tensor[95]
        getitem_397: "f32[]" = _foreach_pow_scalar_and_tensor[96]
        getitem_398: "f32[]" = _foreach_pow_scalar_and_tensor[97]
        getitem_399: "f32[]" = _foreach_pow_scalar_and_tensor[98]
        getitem_400: "f32[]" = _foreach_pow_scalar_and_tensor[99]
        getitem_401: "f32[]" = _foreach_pow_scalar_and_tensor[100]
        getitem_402: "f32[]" = _foreach_pow_scalar_and_tensor[101]
        getitem_403: "f32[]" = _foreach_pow_scalar_and_tensor[102]
        getitem_404: "f32[]" = _foreach_pow_scalar_and_tensor[103]
        getitem_405: "f32[]" = _foreach_pow_scalar_and_tensor[104]
        getitem_406: "f32[]" = _foreach_pow_scalar_and_tensor[105]
        getitem_407: "f32[]" = _foreach_pow_scalar_and_tensor[106]
        getitem_408: "f32[]" = _foreach_pow_scalar_and_tensor[107]
        getitem_409: "f32[]" = _foreach_pow_scalar_and_tensor[108]
        getitem_410: "f32[]" = _foreach_pow_scalar_and_tensor[109]
        getitem_411: "f32[]" = _foreach_pow_scalar_and_tensor[110]
        getitem_412: "f32[]" = _foreach_pow_scalar_and_tensor[111]
        getitem_413: "f32[]" = _foreach_pow_scalar_and_tensor[112]
        getitem_414: "f32[]" = _foreach_pow_scalar_and_tensor[113]
        getitem_415: "f32[]" = _foreach_pow_scalar_and_tensor[114]
        getitem_416: "f32[]" = _foreach_pow_scalar_and_tensor[115]
        getitem_417: "f32[]" = _foreach_pow_scalar_and_tensor[116]
        getitem_418: "f32[]" = _foreach_pow_scalar_and_tensor[117]
        getitem_419: "f32[]" = _foreach_pow_scalar_and_tensor[118]
        getitem_420: "f32[]" = _foreach_pow_scalar_and_tensor[119]
        getitem_421: "f32[]" = _foreach_pow_scalar_and_tensor[120]
        getitem_422: "f32[]" = _foreach_pow_scalar_and_tensor[121]
        getitem_423: "f32[]" = _foreach_pow_scalar_and_tensor[122]
        getitem_424: "f32[]" = _foreach_pow_scalar_and_tensor[123]
        getitem_425: "f32[]" = _foreach_pow_scalar_and_tensor[124]
        getitem_426: "f32[]" = _foreach_pow_scalar_and_tensor[125]
        getitem_427: "f32[]" = _foreach_pow_scalar_and_tensor[126]
        getitem_428: "f32[]" = _foreach_pow_scalar_and_tensor[127]
        getitem_429: "f32[]" = _foreach_pow_scalar_and_tensor[128]
        getitem_430: "f32[]" = _foreach_pow_scalar_and_tensor[129]
        getitem_431: "f32[]" = _foreach_pow_scalar_and_tensor[130]
        getitem_432: "f32[]" = _foreach_pow_scalar_and_tensor[131]
        getitem_433: "f32[]" = _foreach_pow_scalar_and_tensor[132]
        getitem_434: "f32[]" = _foreach_pow_scalar_and_tensor[133]
        getitem_435: "f32[]" = _foreach_pow_scalar_and_tensor[134]
        getitem_436: "f32[]" = _foreach_pow_scalar_and_tensor[135]
        getitem_437: "f32[]" = _foreach_pow_scalar_and_tensor[136]
        getitem_438: "f32[]" = _foreach_pow_scalar_and_tensor[137]
        getitem_439: "f32[]" = _foreach_pow_scalar_and_tensor[138]
        getitem_440: "f32[]" = _foreach_pow_scalar_and_tensor[139]
        getitem_441: "f32[]" = _foreach_pow_scalar_and_tensor[140]
        getitem_442: "f32[]" = _foreach_pow_scalar_and_tensor[141]
        getitem_443: "f32[]" = _foreach_pow_scalar_and_tensor[142]
        getitem_444: "f32[]" = _foreach_pow_scalar_and_tensor[143]
        getitem_445: "f32[]" = _foreach_pow_scalar_and_tensor[144]
        getitem_446: "f32[]" = _foreach_pow_scalar_and_tensor[145]
        getitem_447: "f32[]" = _foreach_pow_scalar_and_tensor[146]
        getitem_448: "f32[]" = _foreach_pow_scalar_and_tensor[147]
        getitem_449: "f32[]" = _foreach_pow_scalar_and_tensor[148]
        getitem_450: "f32[]" = _foreach_pow_scalar_and_tensor[149]
        getitem_451: "f32[]" = _foreach_pow_scalar_and_tensor[150]
        getitem_452: "f32[]" = _foreach_pow_scalar_and_tensor[151]
        getitem_453: "f32[]" = _foreach_pow_scalar_and_tensor[152]
        getitem_454: "f32[]" = _foreach_pow_scalar_and_tensor[153]
        getitem_455: "f32[]" = _foreach_pow_scalar_and_tensor[154]
        getitem_456: "f32[]" = _foreach_pow_scalar_and_tensor[155]
        getitem_457: "f32[]" = _foreach_pow_scalar_and_tensor[156]
        getitem_458: "f32[]" = _foreach_pow_scalar_and_tensor[157]
        getitem_459: "f32[]" = _foreach_pow_scalar_and_tensor[158]
        getitem_460: "f32[]" = _foreach_pow_scalar_and_tensor[159]
        getitem_461: "f32[]" = _foreach_pow_scalar_and_tensor[160]
        getitem_462: "f32[]" = _foreach_pow_scalar_and_tensor[161]
        getitem_463: "f32[]" = _foreach_pow_scalar_and_tensor[162]
        getitem_464: "f32[]" = _foreach_pow_scalar_and_tensor[163]
        getitem_465: "f32[]" = _foreach_pow_scalar_and_tensor[164]
        getitem_466: "f32[]" = _foreach_pow_scalar_and_tensor[165]
        getitem_467: "f32[]" = _foreach_pow_scalar_and_tensor[166]
        getitem_468: "f32[]" = _foreach_pow_scalar_and_tensor[167]
        getitem_469: "f32[]" = _foreach_pow_scalar_and_tensor[168]
        getitem_470: "f32[]" = _foreach_pow_scalar_and_tensor[169]
        getitem_471: "f32[]" = _foreach_pow_scalar_and_tensor[170]
        getitem_472: "f32[]" = _foreach_pow_scalar_and_tensor[171]
        getitem_473: "f32[]" = _foreach_pow_scalar_and_tensor[172]
        getitem_474: "f32[]" = _foreach_pow_scalar_and_tensor[173]
        getitem_475: "f32[]" = _foreach_pow_scalar_and_tensor[174]
        getitem_476: "f32[]" = _foreach_pow_scalar_and_tensor[175]
        getitem_477: "f32[]" = _foreach_pow_scalar_and_tensor[176]
        getitem_478: "f32[]" = _foreach_pow_scalar_and_tensor[177]
        getitem_479: "f32[]" = _foreach_pow_scalar_and_tensor[178]
        getitem_480: "f32[]" = _foreach_pow_scalar_and_tensor[179]
        getitem_481: "f32[]" = _foreach_pow_scalar_and_tensor[180]
        getitem_482: "f32[]" = _foreach_pow_scalar_and_tensor[181]
        getitem_483: "f32[]" = _foreach_pow_scalar_and_tensor[182]
        getitem_484: "f32[]" = _foreach_pow_scalar_and_tensor[183]
        getitem_485: "f32[]" = _foreach_pow_scalar_and_tensor[184]
        getitem_486: "f32[]" = _foreach_pow_scalar_and_tensor[185]
        getitem_487: "f32[]" = _foreach_pow_scalar_and_tensor[186]
        getitem_488: "f32[]" = _foreach_pow_scalar_and_tensor[187]
        getitem_489: "f32[]" = _foreach_pow_scalar_and_tensor[188]
        getitem_490: "f32[]" = _foreach_pow_scalar_and_tensor[189]
        getitem_491: "f32[]" = _foreach_pow_scalar_and_tensor[190]
        getitem_492: "f32[]" = _foreach_pow_scalar_and_tensor[191]
        getitem_493: "f32[]" = _foreach_pow_scalar_and_tensor[192]
        getitem_494: "f32[]" = _foreach_pow_scalar_and_tensor[193]
        getitem_495: "f32[]" = _foreach_pow_scalar_and_tensor[194]
        getitem_496: "f32[]" = _foreach_pow_scalar_and_tensor[195]
        getitem_497: "f32[]" = _foreach_pow_scalar_and_tensor[196]
        getitem_498: "f32[]" = _foreach_pow_scalar_and_tensor[197]
        getitem_499: "f32[]" = _foreach_pow_scalar_and_tensor[198]
        getitem_500: "f32[]" = _foreach_pow_scalar_and_tensor[199]
        getitem_501: "f32[]" = _foreach_pow_scalar_and_tensor[200]
        getitem_502: "f32[]" = _foreach_pow_scalar_and_tensor[201]
        getitem_503: "f32[]" = _foreach_pow_scalar_and_tensor[202]
        getitem_504: "f32[]" = _foreach_pow_scalar_and_tensor[203]
        getitem_505: "f32[]" = _foreach_pow_scalar_and_tensor[204]
        getitem_506: "f32[]" = _foreach_pow_scalar_and_tensor[205]
        getitem_507: "f32[]" = _foreach_pow_scalar_and_tensor[206]
        getitem_508: "f32[]" = _foreach_pow_scalar_and_tensor[207]
        getitem_509: "f32[]" = _foreach_pow_scalar_and_tensor[208]
        getitem_510: "f32[]" = _foreach_pow_scalar_and_tensor[209]
        getitem_511: "f32[]" = _foreach_pow_scalar_and_tensor[210]
        getitem_512: "f32[]" = _foreach_pow_scalar_and_tensor[211]
        getitem_513: "f32[]" = _foreach_pow_scalar_and_tensor[212]
        getitem_514: "f32[]" = _foreach_pow_scalar_and_tensor[213]
        getitem_515: "f32[]" = _foreach_pow_scalar_and_tensor[214]
        getitem_516: "f32[]" = _foreach_pow_scalar_and_tensor[215]
        getitem_517: "f32[]" = _foreach_pow_scalar_and_tensor[216]
        getitem_518: "f32[]" = _foreach_pow_scalar_and_tensor[217]
        getitem_519: "f32[]" = _foreach_pow_scalar_and_tensor[218]
        getitem_520: "f32[]" = _foreach_pow_scalar_and_tensor[219]
        getitem_521: "f32[]" = _foreach_pow_scalar_and_tensor[220]
        getitem_522: "f32[]" = _foreach_pow_scalar_and_tensor[221]
        getitem_523: "f32[]" = _foreach_pow_scalar_and_tensor[222]
        getitem_524: "f32[]" = _foreach_pow_scalar_and_tensor[223]
        getitem_525: "f32[]" = _foreach_pow_scalar_and_tensor[224]
        getitem_526: "f32[]" = _foreach_pow_scalar_and_tensor[225]
        getitem_527: "f32[]" = _foreach_pow_scalar_and_tensor[226]
        getitem_528: "f32[]" = _foreach_pow_scalar_and_tensor[227]
        getitem_529: "f32[]" = _foreach_pow_scalar_and_tensor[228]
        getitem_530: "f32[]" = _foreach_pow_scalar_and_tensor[229]
        getitem_531: "f32[]" = _foreach_pow_scalar_and_tensor[230]
        getitem_532: "f32[]" = _foreach_pow_scalar_and_tensor[231]
        getitem_533: "f32[]" = _foreach_pow_scalar_and_tensor[232]
        getitem_534: "f32[]" = _foreach_pow_scalar_and_tensor[233]
        getitem_535: "f32[]" = _foreach_pow_scalar_and_tensor[234]
        getitem_536: "f32[]" = _foreach_pow_scalar_and_tensor[235]
        getitem_537: "f32[]" = _foreach_pow_scalar_and_tensor[236]
        getitem_538: "f32[]" = _foreach_pow_scalar_and_tensor[237]
        getitem_539: "f32[]" = _foreach_pow_scalar_and_tensor[238]
        getitem_540: "f32[]" = _foreach_pow_scalar_and_tensor[239]
        getitem_541: "f32[]" = _foreach_pow_scalar_and_tensor[240]
        getitem_542: "f32[]" = _foreach_pow_scalar_and_tensor[241]
        getitem_543: "f32[]" = _foreach_pow_scalar_and_tensor[242]
        getitem_544: "f32[]" = _foreach_pow_scalar_and_tensor[243]
        getitem_545: "f32[]" = _foreach_pow_scalar_and_tensor[244]
        getitem_546: "f32[]" = _foreach_pow_scalar_and_tensor[245]
        getitem_547: "f32[]" = _foreach_pow_scalar_and_tensor[246]
        getitem_548: "f32[]" = _foreach_pow_scalar_and_tensor[247]
        getitem_549: "f32[]" = _foreach_pow_scalar_and_tensor[248]
        getitem_550: "f32[]" = _foreach_pow_scalar_and_tensor[249]
        getitem_551: "f32[]" = _foreach_pow_scalar_and_tensor[250]
        getitem_552: "f32[]" = _foreach_pow_scalar_and_tensor[251]
        getitem_553: "f32[]" = _foreach_pow_scalar_and_tensor[252]
        getitem_554: "f32[]" = _foreach_pow_scalar_and_tensor[253]
        getitem_555: "f32[]" = _foreach_pow_scalar_and_tensor[254]
        getitem_556: "f32[]" = _foreach_pow_scalar_and_tensor[255]
        getitem_557: "f32[]" = _foreach_pow_scalar_and_tensor[256]
        getitem_558: "f32[]" = _foreach_pow_scalar_and_tensor[257]
        getitem_559: "f32[]" = _foreach_pow_scalar_and_tensor[258]
        getitem_560: "f32[]" = _foreach_pow_scalar_and_tensor[259]
        getitem_561: "f32[]" = _foreach_pow_scalar_and_tensor[260]
        getitem_562: "f32[]" = _foreach_pow_scalar_and_tensor[261]
        getitem_563: "f32[]" = _foreach_pow_scalar_and_tensor[262]
        getitem_564: "f32[]" = _foreach_pow_scalar_and_tensor[263]
        getitem_565: "f32[]" = _foreach_pow_scalar_and_tensor[264]
        getitem_566: "f32[]" = _foreach_pow_scalar_and_tensor[265]
        getitem_567: "f32[]" = _foreach_pow_scalar_and_tensor[266]
        getitem_568: "f32[]" = _foreach_pow_scalar_and_tensor[267]
        getitem_569: "f32[]" = _foreach_pow_scalar_and_tensor[268]
        getitem_570: "f32[]" = _foreach_pow_scalar_and_tensor[269]
        getitem_571: "f32[]" = _foreach_pow_scalar_and_tensor[270]
        getitem_572: "f32[]" = _foreach_pow_scalar_and_tensor[271]
        getitem_573: "f32[]" = _foreach_pow_scalar_and_tensor[272]
        getitem_574: "f32[]" = _foreach_pow_scalar_and_tensor[273]
        getitem_575: "f32[]" = _foreach_pow_scalar_and_tensor[274]
        getitem_576: "f32[]" = _foreach_pow_scalar_and_tensor[275]
        getitem_577: "f32[]" = _foreach_pow_scalar_and_tensor[276]
        getitem_578: "f32[]" = _foreach_pow_scalar_and_tensor[277]
        getitem_579: "f32[]" = _foreach_pow_scalar_and_tensor[278]
        getitem_580: "f32[]" = _foreach_pow_scalar_and_tensor[279]
        getitem_581: "f32[]" = _foreach_pow_scalar_and_tensor[280]
        getitem_582: "f32[]" = _foreach_pow_scalar_and_tensor[281]
        getitem_583: "f32[]" = _foreach_pow_scalar_and_tensor[282]
        getitem_584: "f32[]" = _foreach_pow_scalar_and_tensor[283]
        getitem_585: "f32[]" = _foreach_pow_scalar_and_tensor[284]
        getitem_586: "f32[]" = _foreach_pow_scalar_and_tensor[285]
        getitem_587: "f32[]" = _foreach_pow_scalar_and_tensor[286]
        getitem_588: "f32[]" = _foreach_pow_scalar_and_tensor[287]
        getitem_589: "f32[]" = _foreach_pow_scalar_and_tensor[288]
        getitem_590: "f32[]" = _foreach_pow_scalar_and_tensor[289]
        getitem_591: "f32[]" = _foreach_pow_scalar_and_tensor[290]
        getitem_592: "f32[]" = _foreach_pow_scalar_and_tensor[291]
        getitem_593: "f32[]" = _foreach_pow_scalar_and_tensor[292]
        getitem_594: "f32[]" = _foreach_pow_scalar_and_tensor[293]
        getitem_595: "f32[]" = _foreach_pow_scalar_and_tensor[294]
        getitem_596: "f32[]" = _foreach_pow_scalar_and_tensor[295]
        getitem_597: "f32[]" = _foreach_pow_scalar_and_tensor[296]
        getitem_598: "f32[]" = _foreach_pow_scalar_and_tensor[297]
        getitem_599: "f32[]" = _foreach_pow_scalar_and_tensor[298]
        getitem_600: "f32[]" = _foreach_pow_scalar_and_tensor[299]
        getitem_601: "f32[]" = _foreach_pow_scalar_and_tensor[300];  _foreach_pow_scalar_and_tensor = None
        _foreach_neg_default = torch.ops.aten._foreach_neg.default([getitem_2408, getitem_2409, getitem_2410, getitem_2411, getitem_2412, getitem_2413, getitem_2414, getitem_2415, getitem_2416, getitem_2417, getitem_2418, getitem_2419, getitem_2420, getitem_2421, getitem_2422, getitem_2423, getitem_2424, getitem_2425, getitem_2426, getitem_2427, getitem_2428, getitem_2429, getitem_2430, getitem_2431, getitem_2432, getitem_2433, getitem_2434, getitem_2435, getitem_2436, getitem_2437, getitem_2438, getitem_2439, getitem_2440, getitem_2441, getitem_2442, getitem_2443, getitem_2444, getitem_2445, getitem_2446, getitem_2447, getitem_2448, getitem_2449, getitem_2450, getitem_2451, getitem_2452, getitem_2453, getitem_2454, getitem_2455, getitem_2456, getitem_2457, getitem_2458, getitem_2459, getitem_2460, getitem_2461, getitem_2462, getitem_2463, getitem_2464, getitem_2465, getitem_2466, getitem_2467, getitem_2468, getitem_2469, getitem_2470, getitem_2471, getitem_2472, getitem_2473, getitem_2474, getitem_2475, getitem_2476, getitem_2477, getitem_2478, getitem_2479, getitem_2480, getitem_2481, getitem_2482, getitem_2483, getitem_2484, getitem_2485, getitem_2486, getitem_2487, getitem_2488, getitem_2489, getitem_2490, getitem_2491, getitem_2492, getitem_2493, getitem_2494, getitem_2495, getitem_2496, getitem_2497, getitem_2498, getitem_2499, getitem_2500, getitem_2501, getitem_2502, getitem_2503, getitem_2504, getitem_2505, getitem_2506, getitem_2507, getitem_2508, getitem_2509, getitem_2510, getitem_2511, getitem_2512, getitem_2513, getitem_2514, getitem_2515, getitem_2516, getitem_2517, getitem_2518, getitem_2519, getitem_2520, getitem_2521, getitem_2522, getitem_2523, getitem_2524, getitem_2525, getitem_2526, getitem_2527, getitem_2528, getitem_2529, getitem_2530, getitem_2531, getitem_2532, getitem_2533, getitem_2534, getitem_2535, getitem_2536, getitem_2537, getitem_2538, getitem_2539, getitem_2540, getitem_2541, getitem_2542, getitem_2543, getitem_2544, getitem_2545, getitem_2546, getitem_2547, getitem_2548, getitem_2549, getitem_2550, getitem_2551, getitem_2552, getitem_2553, getitem_2554, getitem_2555, getitem_2556, getitem_2557, getitem_2558, getitem_2559, getitem_2560, getitem_2561, getitem_2562, getitem_2563, getitem_2564, getitem_2565, getitem_2566, getitem_2567, getitem_2568, getitem_2569, getitem_2570, getitem_2571, getitem_2572, getitem_2573, getitem_2574, getitem_2575, getitem_2576, getitem_2577, getitem_2578, getitem_2579, getitem_2580, getitem_2581, getitem_2582, getitem_2583, getitem_2584, getitem_2585, getitem_2586, getitem_2587, getitem_2588, getitem_2589, getitem_2590, getitem_2591, getitem_2592, getitem_2593, getitem_2594, getitem_2595, getitem_2596, getitem_2597, getitem_2598, getitem_2599, getitem_2600, getitem_2601, getitem_2602, getitem_2603, getitem_2604, getitem_2605, getitem_2606, getitem_2607, getitem_2608, getitem_2609, getitem_2610, getitem_2611, getitem_2612, getitem_2613, getitem_2614, getitem_2615, getitem_2616, getitem_2617, getitem_2618, getitem_2619, getitem_2620, getitem_2621, getitem_2622, getitem_2623, getitem_2624, getitem_2625, getitem_2626, getitem_2627, getitem_2628, getitem_2629, getitem_2630, getitem_2631, getitem_2632, getitem_2633, getitem_2634, getitem_2635, getitem_2636, getitem_2637, getitem_2638, getitem_2639, getitem_2640, getitem_2641, getitem_2642, getitem_2643, getitem_2644, getitem_2645, getitem_2646, getitem_2647, getitem_2648, getitem_2649, getitem_2650, getitem_2651, getitem_2652, getitem_2653, getitem_2654, getitem_2655, getitem_2656, getitem_2657, getitem_2658, getitem_2659, getitem_2660, getitem_2661, getitem_2662, getitem_2663, getitem_2664, getitem_2665, getitem_2666, getitem_2667, getitem_2668, getitem_2669, getitem_2670, getitem_2671, getitem_2672, getitem_2673, getitem_2674, getitem_2675, getitem_2676, getitem_2677, getitem_2678, getitem_2679, getitem_2680, getitem_2681, getitem_2682, getitem_2683, getitem_2684, getitem_2685, getitem_2686, getitem_2687, getitem_2688, getitem_2689, getitem_2690, getitem_2691, getitem_2692, getitem_2693, getitem_2694, getitem_2695, getitem_2696, getitem_2697, getitem_2698, getitem_2699, getitem_2700, getitem_2701, getitem_2702, getitem_2703, getitem_2704, getitem_2705, getitem_2706, getitem_2707, getitem_2708]);  getitem_2408 = getitem_2409 = getitem_2410 = getitem_2411 = getitem_2412 = getitem_2413 = getitem_2414 = getitem_2415 = getitem_2416 = getitem_2417 = getitem_2418 = getitem_2419 = getitem_2420 = getitem_2421 = getitem_2422 = getitem_2423 = getitem_2424 = getitem_2425 = getitem_2426 = getitem_2427 = getitem_2428 = getitem_2429 = getitem_2430 = getitem_2431 = getitem_2432 = getitem_2433 = getitem_2434 = getitem_2435 = getitem_2436 = getitem_2437 = getitem_2438 = getitem_2439 = getitem_2440 = getitem_2441 = getitem_2442 = getitem_2443 = getitem_2444 = getitem_2445 = getitem_2446 = getitem_2447 = getitem_2448 = getitem_2449 = getitem_2450 = getitem_2451 = getitem_2452 = getitem_2453 = getitem_2454 = getitem_2455 = getitem_2456 = getitem_2457 = getitem_2458 = getitem_2459 = getitem_2460 = getitem_2461 = getitem_2462 = getitem_2463 = getitem_2464 = getitem_2465 = getitem_2466 = getitem_2467 = getitem_2468 = getitem_2469 = getitem_2470 = getitem_2471 = getitem_2472 = getitem_2473 = getitem_2474 = getitem_2475 = getitem_2476 = getitem_2477 = getitem_2478 = getitem_2479 = getitem_2480 = getitem_2481 = getitem_2482 = getitem_2483 = getitem_2484 = getitem_2485 = getitem_2486 = getitem_2487 = getitem_2488 = getitem_2489 = getitem_2490 = getitem_2491 = getitem_2492 = getitem_2493 = getitem_2494 = getitem_2495 = getitem_2496 = getitem_2497 = getitem_2498 = getitem_2499 = getitem_2500 = getitem_2501 = getitem_2502 = getitem_2503 = getitem_2504 = getitem_2505 = getitem_2506 = getitem_2507 = getitem_2508 = getitem_2509 = getitem_2510 = getitem_2511 = getitem_2512 = getitem_2513 = getitem_2514 = getitem_2515 = getitem_2516 = getitem_2517 = getitem_2518 = getitem_2519 = getitem_2520 = getitem_2521 = getitem_2522 = getitem_2523 = getitem_2524 = getitem_2525 = getitem_2526 = getitem_2527 = getitem_2528 = getitem_2529 = getitem_2530 = getitem_2531 = getitem_2532 = getitem_2533 = getitem_2534 = getitem_2535 = getitem_2536 = getitem_2537 = getitem_2538 = getitem_2539 = getitem_2540 = getitem_2541 = getitem_2542 = getitem_2543 = getitem_2544 = getitem_2545 = getitem_2546 = getitem_2547 = getitem_2548 = getitem_2549 = getitem_2550 = getitem_2551 = getitem_2552 = getitem_2553 = getitem_2554 = getitem_2555 = getitem_2556 = getitem_2557 = getitem_2558 = getitem_2559 = getitem_2560 = getitem_2561 = getitem_2562 = getitem_2563 = getitem_2564 = getitem_2565 = getitem_2566 = getitem_2567 = getitem_2568 = getitem_2569 = getitem_2570 = getitem_2571 = getitem_2572 = getitem_2573 = getitem_2574 = getitem_2575 = getitem_2576 = getitem_2577 = getitem_2578 = getitem_2579 = getitem_2580 = getitem_2581 = getitem_2582 = getitem_2583 = getitem_2584 = getitem_2585 = getitem_2586 = getitem_2587 = getitem_2588 = getitem_2589 = getitem_2590 = getitem_2591 = getitem_2592 = getitem_2593 = getitem_2594 = getitem_2595 = getitem_2596 = getitem_2597 = getitem_2598 = getitem_2599 = getitem_2600 = getitem_2601 = getitem_2602 = getitem_2603 = getitem_2604 = getitem_2605 = getitem_2606 = getitem_2607 = getitem_2608 = getitem_2609 = getitem_2610 = getitem_2611 = getitem_2612 = getitem_2613 = getitem_2614 = getitem_2615 = getitem_2616 = getitem_2617 = getitem_2618 = getitem_2619 = getitem_2620 = getitem_2621 = getitem_2622 = getitem_2623 = getitem_2624 = getitem_2625 = getitem_2626 = getitem_2627 = getitem_2628 = getitem_2629 = getitem_2630 = getitem_2631 = getitem_2632 = getitem_2633 = getitem_2634 = getitem_2635 = getitem_2636 = getitem_2637 = getitem_2638 = getitem_2639 = getitem_2640 = getitem_2641 = getitem_2642 = getitem_2643 = getitem_2644 = getitem_2645 = getitem_2646 = getitem_2647 = getitem_2648 = getitem_2649 = getitem_2650 = getitem_2651 = getitem_2652 = getitem_2653 = getitem_2654 = getitem_2655 = getitem_2656 = getitem_2657 = getitem_2658 = getitem_2659 = getitem_2660 = getitem_2661 = getitem_2662 = getitem_2663 = getitem_2664 = getitem_2665 = getitem_2666 = getitem_2667 = getitem_2668 = getitem_2669 = getitem_2670 = getitem_2671 = getitem_2672 = getitem_2673 = getitem_2674 = getitem_2675 = getitem_2676 = getitem_2677 = getitem_2678 = getitem_2679 = getitem_2680 = getitem_2681 = getitem_2682 = getitem_2683 = getitem_2684 = getitem_2685 = getitem_2686 = getitem_2687 = getitem_2688 = getitem_2689 = getitem_2690 = getitem_2691 = getitem_2692 = getitem_2693 = getitem_2694 = getitem_2695 = getitem_2696 = getitem_2697 = getitem_2698 = getitem_2699 = getitem_2700 = getitem_2701 = getitem_2702 = getitem_2703 = getitem_2704 = getitem_2705 = getitem_2706 = getitem_2707 = getitem_2708 = None
        getitem_2709: "f32[]" = _foreach_neg_default[0]
        getitem_2710: "f32[]" = _foreach_neg_default[1]
        getitem_2711: "f32[]" = _foreach_neg_default[2]
        getitem_2712: "f32[]" = _foreach_neg_default[3]
        getitem_2713: "f32[]" = _foreach_neg_default[4]
        getitem_2714: "f32[]" = _foreach_neg_default[5]
        getitem_2715: "f32[]" = _foreach_neg_default[6]
        getitem_2716: "f32[]" = _foreach_neg_default[7]
        getitem_2717: "f32[]" = _foreach_neg_default[8]
        getitem_2718: "f32[]" = _foreach_neg_default[9]
        getitem_2719: "f32[]" = _foreach_neg_default[10]
        getitem_2720: "f32[]" = _foreach_neg_default[11]
        getitem_2721: "f32[]" = _foreach_neg_default[12]
        getitem_2722: "f32[]" = _foreach_neg_default[13]
        getitem_2723: "f32[]" = _foreach_neg_default[14]
        getitem_2724: "f32[]" = _foreach_neg_default[15]
        getitem_2725: "f32[]" = _foreach_neg_default[16]
        getitem_2726: "f32[]" = _foreach_neg_default[17]
        getitem_2727: "f32[]" = _foreach_neg_default[18]
        getitem_2728: "f32[]" = _foreach_neg_default[19]
        getitem_2729: "f32[]" = _foreach_neg_default[20]
        getitem_2730: "f32[]" = _foreach_neg_default[21]
        getitem_2731: "f32[]" = _foreach_neg_default[22]
        getitem_2732: "f32[]" = _foreach_neg_default[23]
        getitem_2733: "f32[]" = _foreach_neg_default[24]
        getitem_2734: "f32[]" = _foreach_neg_default[25]
        getitem_2735: "f32[]" = _foreach_neg_default[26]
        getitem_2736: "f32[]" = _foreach_neg_default[27]
        getitem_2737: "f32[]" = _foreach_neg_default[28]
        getitem_2738: "f32[]" = _foreach_neg_default[29]
        getitem_2739: "f32[]" = _foreach_neg_default[30]
        getitem_2740: "f32[]" = _foreach_neg_default[31]
        getitem_2741: "f32[]" = _foreach_neg_default[32]
        getitem_2742: "f32[]" = _foreach_neg_default[33]
        getitem_2743: "f32[]" = _foreach_neg_default[34]
        getitem_2744: "f32[]" = _foreach_neg_default[35]
        getitem_2745: "f32[]" = _foreach_neg_default[36]
        getitem_2746: "f32[]" = _foreach_neg_default[37]
        getitem_2747: "f32[]" = _foreach_neg_default[38]
        getitem_2748: "f32[]" = _foreach_neg_default[39]
        getitem_2749: "f32[]" = _foreach_neg_default[40]
        getitem_2750: "f32[]" = _foreach_neg_default[41]
        getitem_2751: "f32[]" = _foreach_neg_default[42]
        getitem_2752: "f32[]" = _foreach_neg_default[43]
        getitem_2753: "f32[]" = _foreach_neg_default[44]
        getitem_2754: "f32[]" = _foreach_neg_default[45]
        getitem_2755: "f32[]" = _foreach_neg_default[46]
        getitem_2756: "f32[]" = _foreach_neg_default[47]
        getitem_2757: "f32[]" = _foreach_neg_default[48]
        getitem_2758: "f32[]" = _foreach_neg_default[49]
        getitem_2759: "f32[]" = _foreach_neg_default[50]
        getitem_2760: "f32[]" = _foreach_neg_default[51]
        getitem_2761: "f32[]" = _foreach_neg_default[52]
        getitem_2762: "f32[]" = _foreach_neg_default[53]
        getitem_2763: "f32[]" = _foreach_neg_default[54]
        getitem_2764: "f32[]" = _foreach_neg_default[55]
        getitem_2765: "f32[]" = _foreach_neg_default[56]
        getitem_2766: "f32[]" = _foreach_neg_default[57]
        getitem_2767: "f32[]" = _foreach_neg_default[58]
        getitem_2768: "f32[]" = _foreach_neg_default[59]
        getitem_2769: "f32[]" = _foreach_neg_default[60]
        getitem_2770: "f32[]" = _foreach_neg_default[61]
        getitem_2771: "f32[]" = _foreach_neg_default[62]
        getitem_2772: "f32[]" = _foreach_neg_default[63]
        getitem_2773: "f32[]" = _foreach_neg_default[64]
        getitem_2774: "f32[]" = _foreach_neg_default[65]
        getitem_2775: "f32[]" = _foreach_neg_default[66]
        getitem_2776: "f32[]" = _foreach_neg_default[67]
        getitem_2777: "f32[]" = _foreach_neg_default[68]
        getitem_2778: "f32[]" = _foreach_neg_default[69]
        getitem_2779: "f32[]" = _foreach_neg_default[70]
        getitem_2780: "f32[]" = _foreach_neg_default[71]
        getitem_2781: "f32[]" = _foreach_neg_default[72]
        getitem_2782: "f32[]" = _foreach_neg_default[73]
        getitem_2783: "f32[]" = _foreach_neg_default[74]
        getitem_2784: "f32[]" = _foreach_neg_default[75]
        getitem_2785: "f32[]" = _foreach_neg_default[76]
        getitem_2786: "f32[]" = _foreach_neg_default[77]
        getitem_2787: "f32[]" = _foreach_neg_default[78]
        getitem_2788: "f32[]" = _foreach_neg_default[79]
        getitem_2789: "f32[]" = _foreach_neg_default[80]
        getitem_2790: "f32[]" = _foreach_neg_default[81]
        getitem_2791: "f32[]" = _foreach_neg_default[82]
        getitem_2792: "f32[]" = _foreach_neg_default[83]
        getitem_2793: "f32[]" = _foreach_neg_default[84]
        getitem_2794: "f32[]" = _foreach_neg_default[85]
        getitem_2795: "f32[]" = _foreach_neg_default[86]
        getitem_2796: "f32[]" = _foreach_neg_default[87]
        getitem_2797: "f32[]" = _foreach_neg_default[88]
        getitem_2798: "f32[]" = _foreach_neg_default[89]
        getitem_2799: "f32[]" = _foreach_neg_default[90]
        getitem_2800: "f32[]" = _foreach_neg_default[91]
        getitem_2801: "f32[]" = _foreach_neg_default[92]
        getitem_2802: "f32[]" = _foreach_neg_default[93]
        getitem_2803: "f32[]" = _foreach_neg_default[94]
        getitem_2804: "f32[]" = _foreach_neg_default[95]
        getitem_2805: "f32[]" = _foreach_neg_default[96]
        getitem_2806: "f32[]" = _foreach_neg_default[97]
        getitem_2807: "f32[]" = _foreach_neg_default[98]
        getitem_2808: "f32[]" = _foreach_neg_default[99]
        getitem_2809: "f32[]" = _foreach_neg_default[100]
        getitem_2810: "f32[]" = _foreach_neg_default[101]
        getitem_2811: "f32[]" = _foreach_neg_default[102]
        getitem_2812: "f32[]" = _foreach_neg_default[103]
        getitem_2813: "f32[]" = _foreach_neg_default[104]
        getitem_2814: "f32[]" = _foreach_neg_default[105]
        getitem_2815: "f32[]" = _foreach_neg_default[106]
        getitem_2816: "f32[]" = _foreach_neg_default[107]
        getitem_2817: "f32[]" = _foreach_neg_default[108]
        getitem_2818: "f32[]" = _foreach_neg_default[109]
        getitem_2819: "f32[]" = _foreach_neg_default[110]
        getitem_2820: "f32[]" = _foreach_neg_default[111]
        getitem_2821: "f32[]" = _foreach_neg_default[112]
        getitem_2822: "f32[]" = _foreach_neg_default[113]
        getitem_2823: "f32[]" = _foreach_neg_default[114]
        getitem_2824: "f32[]" = _foreach_neg_default[115]
        getitem_2825: "f32[]" = _foreach_neg_default[116]
        getitem_2826: "f32[]" = _foreach_neg_default[117]
        getitem_2827: "f32[]" = _foreach_neg_default[118]
        getitem_2828: "f32[]" = _foreach_neg_default[119]
        getitem_2829: "f32[]" = _foreach_neg_default[120]
        getitem_2830: "f32[]" = _foreach_neg_default[121]
        getitem_2831: "f32[]" = _foreach_neg_default[122]
        getitem_2832: "f32[]" = _foreach_neg_default[123]
        getitem_2833: "f32[]" = _foreach_neg_default[124]
        getitem_2834: "f32[]" = _foreach_neg_default[125]
        getitem_2835: "f32[]" = _foreach_neg_default[126]
        getitem_2836: "f32[]" = _foreach_neg_default[127]
        getitem_2837: "f32[]" = _foreach_neg_default[128]
        getitem_2838: "f32[]" = _foreach_neg_default[129]
        getitem_2839: "f32[]" = _foreach_neg_default[130]
        getitem_2840: "f32[]" = _foreach_neg_default[131]
        getitem_2841: "f32[]" = _foreach_neg_default[132]
        getitem_2842: "f32[]" = _foreach_neg_default[133]
        getitem_2843: "f32[]" = _foreach_neg_default[134]
        getitem_2844: "f32[]" = _foreach_neg_default[135]
        getitem_2845: "f32[]" = _foreach_neg_default[136]
        getitem_2846: "f32[]" = _foreach_neg_default[137]
        getitem_2847: "f32[]" = _foreach_neg_default[138]
        getitem_2848: "f32[]" = _foreach_neg_default[139]
        getitem_2849: "f32[]" = _foreach_neg_default[140]
        getitem_2850: "f32[]" = _foreach_neg_default[141]
        getitem_2851: "f32[]" = _foreach_neg_default[142]
        getitem_2852: "f32[]" = _foreach_neg_default[143]
        getitem_2853: "f32[]" = _foreach_neg_default[144]
        getitem_2854: "f32[]" = _foreach_neg_default[145]
        getitem_2855: "f32[]" = _foreach_neg_default[146]
        getitem_2856: "f32[]" = _foreach_neg_default[147]
        getitem_2857: "f32[]" = _foreach_neg_default[148]
        getitem_2858: "f32[]" = _foreach_neg_default[149]
        getitem_2859: "f32[]" = _foreach_neg_default[150]
        getitem_2860: "f32[]" = _foreach_neg_default[151]
        getitem_2861: "f32[]" = _foreach_neg_default[152]
        getitem_2862: "f32[]" = _foreach_neg_default[153]
        getitem_2863: "f32[]" = _foreach_neg_default[154]
        getitem_2864: "f32[]" = _foreach_neg_default[155]
        getitem_2865: "f32[]" = _foreach_neg_default[156]
        getitem_2866: "f32[]" = _foreach_neg_default[157]
        getitem_2867: "f32[]" = _foreach_neg_default[158]
        getitem_2868: "f32[]" = _foreach_neg_default[159]
        getitem_2869: "f32[]" = _foreach_neg_default[160]
        getitem_2870: "f32[]" = _foreach_neg_default[161]
        getitem_2871: "f32[]" = _foreach_neg_default[162]
        getitem_2872: "f32[]" = _foreach_neg_default[163]
        getitem_2873: "f32[]" = _foreach_neg_default[164]
        getitem_2874: "f32[]" = _foreach_neg_default[165]
        getitem_2875: "f32[]" = _foreach_neg_default[166]
        getitem_2876: "f32[]" = _foreach_neg_default[167]
        getitem_2877: "f32[]" = _foreach_neg_default[168]
        getitem_2878: "f32[]" = _foreach_neg_default[169]
        getitem_2879: "f32[]" = _foreach_neg_default[170]
        getitem_2880: "f32[]" = _foreach_neg_default[171]
        getitem_2881: "f32[]" = _foreach_neg_default[172]
        getitem_2882: "f32[]" = _foreach_neg_default[173]
        getitem_2883: "f32[]" = _foreach_neg_default[174]
        getitem_2884: "f32[]" = _foreach_neg_default[175]
        getitem_2885: "f32[]" = _foreach_neg_default[176]
        getitem_2886: "f32[]" = _foreach_neg_default[177]
        getitem_2887: "f32[]" = _foreach_neg_default[178]
        getitem_2888: "f32[]" = _foreach_neg_default[179]
        getitem_2889: "f32[]" = _foreach_neg_default[180]
        getitem_2890: "f32[]" = _foreach_neg_default[181]
        getitem_2891: "f32[]" = _foreach_neg_default[182]
        getitem_2892: "f32[]" = _foreach_neg_default[183]
        getitem_2893: "f32[]" = _foreach_neg_default[184]
        getitem_2894: "f32[]" = _foreach_neg_default[185]
        getitem_2895: "f32[]" = _foreach_neg_default[186]
        getitem_2896: "f32[]" = _foreach_neg_default[187]
        getitem_2897: "f32[]" = _foreach_neg_default[188]
        getitem_2898: "f32[]" = _foreach_neg_default[189]
        getitem_2899: "f32[]" = _foreach_neg_default[190]
        getitem_2900: "f32[]" = _foreach_neg_default[191]
        getitem_2901: "f32[]" = _foreach_neg_default[192]
        getitem_2902: "f32[]" = _foreach_neg_default[193]
        getitem_2903: "f32[]" = _foreach_neg_default[194]
        getitem_2904: "f32[]" = _foreach_neg_default[195]
        getitem_2905: "f32[]" = _foreach_neg_default[196]
        getitem_2906: "f32[]" = _foreach_neg_default[197]
        getitem_2907: "f32[]" = _foreach_neg_default[198]
        getitem_2908: "f32[]" = _foreach_neg_default[199]
        getitem_2909: "f32[]" = _foreach_neg_default[200]
        getitem_2910: "f32[]" = _foreach_neg_default[201]
        getitem_2911: "f32[]" = _foreach_neg_default[202]
        getitem_2912: "f32[]" = _foreach_neg_default[203]
        getitem_2913: "f32[]" = _foreach_neg_default[204]
        getitem_2914: "f32[]" = _foreach_neg_default[205]
        getitem_2915: "f32[]" = _foreach_neg_default[206]
        getitem_2916: "f32[]" = _foreach_neg_default[207]
        getitem_2917: "f32[]" = _foreach_neg_default[208]
        getitem_2918: "f32[]" = _foreach_neg_default[209]
        getitem_2919: "f32[]" = _foreach_neg_default[210]
        getitem_2920: "f32[]" = _foreach_neg_default[211]
        getitem_2921: "f32[]" = _foreach_neg_default[212]
        getitem_2922: "f32[]" = _foreach_neg_default[213]
        getitem_2923: "f32[]" = _foreach_neg_default[214]
        getitem_2924: "f32[]" = _foreach_neg_default[215]
        getitem_2925: "f32[]" = _foreach_neg_default[216]
        getitem_2926: "f32[]" = _foreach_neg_default[217]
        getitem_2927: "f32[]" = _foreach_neg_default[218]
        getitem_2928: "f32[]" = _foreach_neg_default[219]
        getitem_2929: "f32[]" = _foreach_neg_default[220]
        getitem_2930: "f32[]" = _foreach_neg_default[221]
        getitem_2931: "f32[]" = _foreach_neg_default[222]
        getitem_2932: "f32[]" = _foreach_neg_default[223]
        getitem_2933: "f32[]" = _foreach_neg_default[224]
        getitem_2934: "f32[]" = _foreach_neg_default[225]
        getitem_2935: "f32[]" = _foreach_neg_default[226]
        getitem_2936: "f32[]" = _foreach_neg_default[227]
        getitem_2937: "f32[]" = _foreach_neg_default[228]
        getitem_2938: "f32[]" = _foreach_neg_default[229]
        getitem_2939: "f32[]" = _foreach_neg_default[230]
        getitem_2940: "f32[]" = _foreach_neg_default[231]
        getitem_2941: "f32[]" = _foreach_neg_default[232]
        getitem_2942: "f32[]" = _foreach_neg_default[233]
        getitem_2943: "f32[]" = _foreach_neg_default[234]
        getitem_2944: "f32[]" = _foreach_neg_default[235]
        getitem_2945: "f32[]" = _foreach_neg_default[236]
        getitem_2946: "f32[]" = _foreach_neg_default[237]
        getitem_2947: "f32[]" = _foreach_neg_default[238]
        getitem_2948: "f32[]" = _foreach_neg_default[239]
        getitem_2949: "f32[]" = _foreach_neg_default[240]
        getitem_2950: "f32[]" = _foreach_neg_default[241]
        getitem_2951: "f32[]" = _foreach_neg_default[242]
        getitem_2952: "f32[]" = _foreach_neg_default[243]
        getitem_2953: "f32[]" = _foreach_neg_default[244]
        getitem_2954: "f32[]" = _foreach_neg_default[245]
        getitem_2955: "f32[]" = _foreach_neg_default[246]
        getitem_2956: "f32[]" = _foreach_neg_default[247]
        getitem_2957: "f32[]" = _foreach_neg_default[248]
        getitem_2958: "f32[]" = _foreach_neg_default[249]
        getitem_2959: "f32[]" = _foreach_neg_default[250]
        getitem_2960: "f32[]" = _foreach_neg_default[251]
        getitem_2961: "f32[]" = _foreach_neg_default[252]
        getitem_2962: "f32[]" = _foreach_neg_default[253]
        getitem_2963: "f32[]" = _foreach_neg_default[254]
        getitem_2964: "f32[]" = _foreach_neg_default[255]
        getitem_2965: "f32[]" = _foreach_neg_default[256]
        getitem_2966: "f32[]" = _foreach_neg_default[257]
        getitem_2967: "f32[]" = _foreach_neg_default[258]
        getitem_2968: "f32[]" = _foreach_neg_default[259]
        getitem_2969: "f32[]" = _foreach_neg_default[260]
        getitem_2970: "f32[]" = _foreach_neg_default[261]
        getitem_2971: "f32[]" = _foreach_neg_default[262]
        getitem_2972: "f32[]" = _foreach_neg_default[263]
        getitem_2973: "f32[]" = _foreach_neg_default[264]
        getitem_2974: "f32[]" = _foreach_neg_default[265]
        getitem_2975: "f32[]" = _foreach_neg_default[266]
        getitem_2976: "f32[]" = _foreach_neg_default[267]
        getitem_2977: "f32[]" = _foreach_neg_default[268]
        getitem_2978: "f32[]" = _foreach_neg_default[269]
        getitem_2979: "f32[]" = _foreach_neg_default[270]
        getitem_2980: "f32[]" = _foreach_neg_default[271]
        getitem_2981: "f32[]" = _foreach_neg_default[272]
        getitem_2982: "f32[]" = _foreach_neg_default[273]
        getitem_2983: "f32[]" = _foreach_neg_default[274]
        getitem_2984: "f32[]" = _foreach_neg_default[275]
        getitem_2985: "f32[]" = _foreach_neg_default[276]
        getitem_2986: "f32[]" = _foreach_neg_default[277]
        getitem_2987: "f32[]" = _foreach_neg_default[278]
        getitem_2988: "f32[]" = _foreach_neg_default[279]
        getitem_2989: "f32[]" = _foreach_neg_default[280]
        getitem_2990: "f32[]" = _foreach_neg_default[281]
        getitem_2991: "f32[]" = _foreach_neg_default[282]
        getitem_2992: "f32[]" = _foreach_neg_default[283]
        getitem_2993: "f32[]" = _foreach_neg_default[284]
        getitem_2994: "f32[]" = _foreach_neg_default[285]
        getitem_2995: "f32[]" = _foreach_neg_default[286]
        getitem_2996: "f32[]" = _foreach_neg_default[287]
        getitem_2997: "f32[]" = _foreach_neg_default[288]
        getitem_2998: "f32[]" = _foreach_neg_default[289]
        getitem_2999: "f32[]" = _foreach_neg_default[290]
        getitem_3000: "f32[]" = _foreach_neg_default[291]
        getitem_3001: "f32[]" = _foreach_neg_default[292]
        getitem_3002: "f32[]" = _foreach_neg_default[293]
        getitem_3003: "f32[]" = _foreach_neg_default[294]
        getitem_3004: "f32[]" = _foreach_neg_default[295]
        getitem_3005: "f32[]" = _foreach_neg_default[296]
        getitem_3006: "f32[]" = _foreach_neg_default[297]
        getitem_3007: "f32[]" = _foreach_neg_default[298]
        getitem_3008: "f32[]" = _foreach_neg_default[299]
        getitem_3009: "f32[]" = _foreach_neg_default[300];  _foreach_neg_default = None
        return (getitem, getitem_1204, getitem_1205, getitem_1206, getitem_1207, getitem_1208, getitem_1209, getitem_1210, getitem_1211, getitem_1212, getitem_1213, getitem_1214, getitem_1215, getitem_1216, getitem_1217, getitem_1218, getitem_1219, getitem_1220, getitem_1221, getitem_1222, getitem_1223, getitem_1224, getitem_1225, getitem_1226, getitem_1227, getitem_1228, getitem_1229, getitem_1230, getitem_1231, getitem_1232, getitem_1233, getitem_1234, getitem_1235, getitem_1236, getitem_1237, getitem_1238, getitem_1239, getitem_1240, getitem_1241, getitem_1242, getitem_1243, getitem_1244, getitem_1245, getitem_1246, getitem_1247, getitem_1248, getitem_1249, getitem_1250, getitem_1251, getitem_1252, getitem_1253, getitem_1254, getitem_1255, getitem_1256, getitem_1257, getitem_1258, getitem_1259, getitem_1260, getitem_1261, getitem_1262, getitem_1263, getitem_1264, getitem_1265, getitem_1266, getitem_1267, getitem_1268, getitem_1269, getitem_1270, getitem_1271, getitem_1272, getitem_1273, getitem_1274, getitem_1275, getitem_1276, getitem_1277, getitem_1278, getitem_1279, getitem_1280, getitem_1281, getitem_1282, getitem_1283, getitem_1284, getitem_1285, getitem_1286, getitem_1287, getitem_1288, getitem_1289, getitem_1290, getitem_1291, getitem_1292, getitem_1293, getitem_1294, getitem_1295, getitem_1296, getitem_1297, getitem_1298, getitem_1299, getitem_1300, getitem_1301, getitem_1302, getitem_1303, getitem_1304, getitem_1305, getitem_1306, getitem_1307, getitem_1308, getitem_1309, getitem_1310, getitem_1311, getitem_1312, getitem_1313, getitem_1314, getitem_1315, getitem_1316, getitem_1317, getitem_1318, getitem_1319, getitem_1320, getitem_1321, getitem_1322, getitem_1323, getitem_1324, getitem_1325, getitem_1326, getitem_1327, getitem_1328, getitem_1329, getitem_1330, getitem_1331, getitem_1332, getitem_1333, getitem_1334, getitem_1335, getitem_1336, getitem_1337, getitem_1338, getitem_1339, getitem_1340, getitem_1341, getitem_1342, getitem_1343, getitem_1344, getitem_1345, getitem_1346, getitem_1347, getitem_1348, getitem_1349, getitem_1350, getitem_1351, getitem_1352, getitem_1353, getitem_1354, getitem_1355, getitem_1356, getitem_1357, getitem_1358, getitem_1359, getitem_1360, getitem_1361, getitem_1362, getitem_1363, getitem_1364, getitem_1365, getitem_1366, getitem_1367, getitem_1368, getitem_1369, getitem_1370, getitem_1371, getitem_1372, getitem_1373, getitem_1374, getitem_1375, getitem_1376, getitem_1377, getitem_1378, getitem_1379, getitem_1380, getitem_1381, getitem_1382, getitem_1383, getitem_1384, getitem_1385, getitem_1386, getitem_1387, getitem_1388, getitem_1389, getitem_1390, getitem_1391, getitem_1392, getitem_1393, getitem_1394, getitem_1395, getitem_1396, getitem_1397, getitem_1398, getitem_1399, getitem_1400, getitem_1401, getitem_1402, getitem_1403, getitem_1404, getitem_1405, getitem_1406, getitem_1407, getitem_1408, getitem_1409, getitem_1410, getitem_1411, getitem_1412, getitem_1413, getitem_1414, getitem_1415, getitem_1416, getitem_1417, getitem_1418, getitem_1419, getitem_1420, getitem_1421, getitem_1422, getitem_1423, getitem_1424, getitem_1425, getitem_1426, getitem_1427, getitem_1428, getitem_1429, getitem_1430, getitem_1431, getitem_1432, getitem_1433, getitem_1434, getitem_1435, getitem_1436, getitem_1437, getitem_1438, getitem_1439, getitem_1440, getitem_1441, getitem_1442, getitem_1443, getitem_1444, getitem_1445, getitem_1446, getitem_1447, getitem_1448, getitem_1449, getitem_1450, getitem_1451, getitem_1452, getitem_1453, getitem_1454, getitem_1455, getitem_1456, getitem_1457, getitem_1458, getitem_1459, getitem_1460, getitem_1461, getitem_1462, getitem_1463, getitem_1464, getitem_1465, getitem_1466, getitem_1467, getitem_1468, getitem_1469, getitem_1470, getitem_1471, getitem_1472, getitem_1473, getitem_1474, getitem_1475, getitem_1476, getitem_1477, getitem_1478, getitem_1479, getitem_1480, getitem_1481, getitem_1482, getitem_1483, getitem_1484, getitem_1485, getitem_1486, getitem_1487, getitem_1488, getitem_1489, getitem_1490, getitem_1491, getitem_1492, getitem_1493, getitem_1494, getitem_1495, getitem_1496, getitem_1497, getitem_1498, getitem_1499, getitem_1500, getitem_1501, getitem_1502, getitem_1503, getitem_301, getitem_302, getitem_303, getitem_304, getitem_305, getitem_306, getitem_307, getitem_308, getitem_309, getitem_310, getitem_311, getitem_312, getitem_313, getitem_314, getitem_315, getitem_316, getitem_317, getitem_318, getitem_319, getitem_320, getitem_321, getitem_322, getitem_323, getitem_324, getitem_325, getitem_326, getitem_327, getitem_328, getitem_329, getitem_330, getitem_331, getitem_332, getitem_333, getitem_334, getitem_335, getitem_336, getitem_337, getitem_338, getitem_339, getitem_340, getitem_341, getitem_342, getitem_343, getitem_344, getitem_345, getitem_346, getitem_347, getitem_348, getitem_349, getitem_350, getitem_351, getitem_352, getitem_353, getitem_354, getitem_355, getitem_356, getitem_357, getitem_358, getitem_359, getitem_360, getitem_361, getitem_362, getitem_363, getitem_364, getitem_365, getitem_366, getitem_367, getitem_368, getitem_369, getitem_370, getitem_371, getitem_372, getitem_373, getitem_374, getitem_375, getitem_376, getitem_377, getitem_378, getitem_379, getitem_380, getitem_381, getitem_382, getitem_383, getitem_384, getitem_385, getitem_386, getitem_387, getitem_388, getitem_389, getitem_390, getitem_391, getitem_392, getitem_393, getitem_394, getitem_395, getitem_396, getitem_397, getitem_398, getitem_399, getitem_400, getitem_401, getitem_402, getitem_403, getitem_404, getitem_405, getitem_406, getitem_407, getitem_408, getitem_409, getitem_410, getitem_411, getitem_412, getitem_413, getitem_414, getitem_415, getitem_416, getitem_417, getitem_418, getitem_419, getitem_420, getitem_421, getitem_422, getitem_423, getitem_424, getitem_425, getitem_426, getitem_427, getitem_428, getitem_429, getitem_430, getitem_431, getitem_432, getitem_433, getitem_434, getitem_435, getitem_436, getitem_437, getitem_438, getitem_439, getitem_440, getitem_441, getitem_442, getitem_443, getitem_444, getitem_445, getitem_446, getitem_447, getitem_448, getitem_449, getitem_450, getitem_451, getitem_452, getitem_453, getitem_454, getitem_455, getitem_456, getitem_457, getitem_458, getitem_459, getitem_460, getitem_461, getitem_462, getitem_463, getitem_464, getitem_465, getitem_466, getitem_467, getitem_468, getitem_469, getitem_470, getitem_471, getitem_472, getitem_473, getitem_474, getitem_475, getitem_476, getitem_477, getitem_478, getitem_479, getitem_480, getitem_481, getitem_482, getitem_483, getitem_484, getitem_485, getitem_486, getitem_487, getitem_488, getitem_489, getitem_490, getitem_491, getitem_492, getitem_493, getitem_494, getitem_495, getitem_496, getitem_497, getitem_498, getitem_499, getitem_500, getitem_501, getitem_502, getitem_503, getitem_504, getitem_505, getitem_506, getitem_507, getitem_508, getitem_509, getitem_510, getitem_511, getitem_512, getitem_513, getitem_514, getitem_515, getitem_516, getitem_517, getitem_518, getitem_519, getitem_520, getitem_521, getitem_522, getitem_523, getitem_524, getitem_525, getitem_526, getitem_527, getitem_528, getitem_529, getitem_530, getitem_531, getitem_532, getitem_533, getitem_534, getitem_535, getitem_536, getitem_537, getitem_538, getitem_539, getitem_540, getitem_541, getitem_542, getitem_543, getitem_544, getitem_545, getitem_546, getitem_547, getitem_548, getitem_549, getitem_550, getitem_551, getitem_552, getitem_553, getitem_554, getitem_555, getitem_556, getitem_557, getitem_558, getitem_559, getitem_560, getitem_561, getitem_562, getitem_563, getitem_564, getitem_565, getitem_566, getitem_567, getitem_568, getitem_569, getitem_570, getitem_571, getitem_572, getitem_573, getitem_574, getitem_575, getitem_576, getitem_577, getitem_578, getitem_579, getitem_580, getitem_581, getitem_582, getitem_583, getitem_584, getitem_585, getitem_586, getitem_587, getitem_588, getitem_589, getitem_590, getitem_591, getitem_592, getitem_593, getitem_594, getitem_595, getitem_596, getitem_597, getitem_598, getitem_599, getitem_600, getitem_601, getitem_2709, getitem_2710, getitem_2711, getitem_2712, getitem_2713, getitem_2714, getitem_2715, getitem_2716, getitem_2717, getitem_2718, getitem_2719, getitem_2720, getitem_2721, getitem_2722, getitem_2723, getitem_2724, getitem_2725, getitem_2726, getitem_2727, getitem_2728, getitem_2729, getitem_2730, getitem_2731, getitem_2732, getitem_2733, getitem_2734, getitem_2735, getitem_2736, getitem_2737, getitem_2738, getitem_2739, getitem_2740, getitem_2741, getitem_2742, getitem_2743, getitem_2744, getitem_2745, getitem_2746, getitem_2747, getitem_2748, getitem_2749, getitem_2750, getitem_2751, getitem_2752, getitem_2753, getitem_2754, getitem_2755, getitem_2756, getitem_2757, getitem_2758, getitem_2759, getitem_2760, getitem_2761, getitem_2762, getitem_2763, getitem_2764, getitem_2765, getitem_2766, getitem_2767, getitem_2768, getitem_2769, getitem_2770, getitem_2771, getitem_2772, getitem_2773, getitem_2774, getitem_2775, getitem_2776, getitem_2777, getitem_2778, getitem_2779, getitem_2780, getitem_2781, getitem_2782, getitem_2783, getitem_2784, getitem_2785, getitem_2786, getitem_2787, getitem_2788, getitem_2789, getitem_2790, getitem_2791, getitem_2792, getitem_2793, getitem_2794, getitem_2795, getitem_2796, getitem_2797, getitem_2798, getitem_2799, getitem_2800, getitem_2801, getitem_2802, getitem_2803, getitem_2804, getitem_2805, getitem_2806, getitem_2807, getitem_2808, getitem_2809, getitem_2810, getitem_2811, getitem_2812, getitem_2813, getitem_2814, getitem_2815, getitem_2816, getitem_2817, getitem_2818, getitem_2819, getitem_2820, getitem_2821, getitem_2822, getitem_2823, getitem_2824, getitem_2825, getitem_2826, getitem_2827, getitem_2828, getitem_2829, getitem_2830, getitem_2831, getitem_2832, getitem_2833, getitem_2834, getitem_2835, getitem_2836, getitem_2837, getitem_2838, getitem_2839, getitem_2840, getitem_2841, getitem_2842, getitem_2843, getitem_2844, getitem_2845, getitem_2846, getitem_2847, getitem_2848, getitem_2849, getitem_2850, getitem_2851, getitem_2852, getitem_2853, getitem_2854, getitem_2855, getitem_2856, getitem_2857, getitem_2858, getitem_2859, getitem_2860, getitem_2861, getitem_2862, getitem_2863, getitem_2864, getitem_2865, getitem_2866, getitem_2867, getitem_2868, getitem_2869, getitem_2870, getitem_2871, getitem_2872, getitem_2873, getitem_2874, getitem_2875, getitem_2876, getitem_2877, getitem_2878, getitem_2879, getitem_2880, getitem_2881, getitem_2882, getitem_2883, getitem_2884, getitem_2885, getitem_2886, getitem_2887, getitem_2888, getitem_2889, getitem_2890, getitem_2891, getitem_2892, getitem_2893, getitem_2894, getitem_2895, getitem_2896, getitem_2897, getitem_2898, getitem_2899, getitem_2900, getitem_2901, getitem_2902, getitem_2903, getitem_2904, getitem_2905, getitem_2906, getitem_2907, getitem_2908, getitem_2909, getitem_2910, getitem_2911, getitem_2912, getitem_2913, getitem_2914, getitem_2915, getitem_2916, getitem_2917, getitem_2918, getitem_2919, getitem_2920, getitem_2921, getitem_2922, getitem_2923, getitem_2924, getitem_2925, getitem_2926, getitem_2927, getitem_2928, getitem_2929, getitem_2930, getitem_2931, getitem_2932, getitem_2933, getitem_2934, getitem_2935, getitem_2936, getitem_2937, getitem_2938, getitem_2939, getitem_2940, getitem_2941, getitem_2942, getitem_2943, getitem_2944, getitem_2945, getitem_2946, getitem_2947, getitem_2948, getitem_2949, getitem_2950, getitem_2951, getitem_2952, getitem_2953, getitem_2954, getitem_2955, getitem_2956, getitem_2957, getitem_2958, getitem_2959, getitem_2960, getitem_2961, getitem_2962, getitem_2963, getitem_2964, getitem_2965, getitem_2966, getitem_2967, getitem_2968, getitem_2969, getitem_2970, getitem_2971, getitem_2972, getitem_2973, getitem_2974, getitem_2975, getitem_2976, getitem_2977, getitem_2978, getitem_2979, getitem_2980, getitem_2981, getitem_2982, getitem_2983, getitem_2984, getitem_2985, getitem_2986, getitem_2987, getitem_2988, getitem_2989, getitem_2990, getitem_2991, getitem_2992, getitem_2993, getitem_2994, getitem_2995, getitem_2996, getitem_2997, getitem_2998, getitem_2999, getitem_3000, getitem_3001, getitem_3002, getitem_3003, getitem_3004, getitem_3005, getitem_3006, getitem_3007, getitem_3008, getitem_3009)


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
    torch.tensor(1),  # getitem_1504 (unknown shape)
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
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
