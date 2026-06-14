# Oracle Migration Worklog

Updated: 2026-06-13

## Current Worker Pool

- Row 1160 `pointwise_e8ae58ea3c65`: active worker `019ebd71-bd99-7ed3-9664-561ca57b630e`.
- Row 1161 `pointwise_e972081e0aab`: active worker `019ebd73-a5cc-7672-95d6-b8cd6a888e02`.
- Row 1163 `pointwise_ee5e035b433c`: active worker `019ebd74-7360-7202-89cd-ec88aae96af5`.
- Row 1162 `pointwise_ee5b7a9ccfd4`: active worker `019ebd75-6886-78e3-bf4e-af115c359931`.
- Row 1159 `pointwise_e65ebcfd5b2c`: active worker `019ebd79-2f9f-7013-9350-8b73673b29a7`.

Tail lane owned by this session: rows 1549-1654. Rows 1514-1548 are claimed by a midqueue worker and must not be touched by this tail lane. Rows 1549, 1554, 1647, 1648, 1659, 1673, 1700, 1701, 1711, 1716, and 1718 are marked needs_work after parent-confirmed bench failures. Rows 1550-1553, 1555-1558, 1560-1629, 1630-1646, 1649-1658, 1660-1672, 1674-1699, 1702-1710, 1712-1715, 1717, and 1719-1727 are parent-verified on H100 fallback, pending B200 verification. Rows 577-588 were released to avoid colliding with the forward-moving manager. Recheck origin before reusing any older historical batch listed in the queue.

Lower-gap refill buffer claimed by this session: rows 1159-1163 are active. Rows 1173, 1197, 1204, 1207, 1212, 1230, 1233, 1238, 1247, 1251, 1253, 1255, 1271, 1274, 1281, 1288, 1301, 1312, 1323, 1336, 1344, 1357, 1365, 1369-1370, 1373, 1382, 1385, and 1398 are marked needs_work after parent-confirmed bench failures; rows 1164-1172, 1174-1196, 1198-1203, 1205-1206, 1208-1211, 1213-1218, 1219-1229, 1231-1232, 1234-1237, 1239-1246, 1248-1250, 1252, 1254, 1256-1270, 1272-1273, 1275-1280, 1282-1287, 1289-1292, 1294-1300, 1302-1311, 1313-1322, 1324-1335, 1337-1343, 1345-1356, 1358-1364, 1366-1368, 1371-1372, 1374-1381, 1383-1384, 1386-1397, and 1399 are parent-verified on H100 fallback, pending B200 verification.

Additional lower-gap refill buffer claimed by this session: rows 1170-1179.

Additional lower-gap refill buffer claimed by this session: rows 1160-1169.

Additional lower-gap refill buffer claimed by this session: rows 1150-1159.

Additional lower-gap refill buffer claimed by this session: rows 1360-1369.

Additional lower-gap refill buffer claimed by this session: rows 1350-1359.

Additional lower-gap refill buffer claimed by this session: rows 1340-1349.

Additional lower-gap refill buffer claimed by this session: rows 1330-1339.

Additional lower-gap refill buffer claimed by this session: rows 1320-1329.

Additional lower-gap refill buffer claimed by this session: rows 1310-1319.

Additional lower-gap refill buffer claimed by this session: rows 1300-1309.

Additional lower-gap refill buffer claimed by this session: rows 1290-1299.

Additional lower-gap refill buffer claimed by this session: rows 1280-1289.

Additional lower-gap refill buffer claimed by this session: rows 1270-1279.

Additional lower-gap refill buffer claimed by this session: rows 1260-1269.

Additional lower-gap refill buffer claimed by this session: rows 1250-1259.

Additional lower-gap refill buffer claimed by this session: rows 1240-1249.

Additional lower-gap refill buffer claimed by this session: rows 1230-1239.

Additional lower-gap refill buffer claimed by this session: rows 1220-1229.

Additional lower-gap refill buffer claimed by this session: rows 1210-1219.

Additional lower-gap refill buffer claimed by this session: rows 1200-1209.

Additional lower-gap refill buffer claimed by this session: rows 1190-1199.

Additional lower-gap refill buffer claimed by this session: rows 1180-1189.

Additional lower-gap refill buffer claimed by this session: rows 1380-1389.

Additional lower-gap refill buffer claimed by this session: rows 1370-1379.

## Pending Parent Review

- Row 94 `var_mean_5a22dd21d88e`: worker checks passed; fallback bench was `23GOOD_1BAD_ORACLE`.
- Row 124 `var_mean_88858c55c3b4`: worker checks passed; fallback bench was `18GOOD_1BAD_ORACLE`.
- Row 172 `sum_abcd9bccce7d`: worker checks passed; fallback bench was `6GOOD_2AT_FLOOR`.
- Row 181 `pointwise_2c331ef4f17f`: worker checks passed; fallback bench failed before timing with `NUMERICS_WORSE_THAN_COMPILED` on all 23 points.
- Row 194 `var_mean_var_mean_5cc92f5d49c1`: worker checks passed; fallback bench was `23GOOD`.
- Row 203 `pointwise_c911fb4f9b47`: worker checks passed; fallback bench failed before timing with `NUMERICS_WORSE_THAN_COMPILED` on all 20 points.
- Row 246 `sum_sum_sum_ddcfccfb8340`: worker checks passed; fallback bench was `2GOOD_2BAD_ORACLE`.
- Row 275 `pointwise_ee22f47c826d`: worker checks passed; fallback bench was `1AT_FLOOR`.
- Row 207 `sum_sum_sum_fb3a1658dadb`: worker checks passed; fallback bench failed before timing with `NUMERICS_WORSE_THAN_COMPILED` on all 20 points.
- Row 232 `var_mean_mean_e790938418f4`: worker checks passed; fallback bench was `9GOOD_1BAD_ORACLE`.

## Pushed By This Session

- Rows 1164-1165: measured, 2/2 checks, H100 fallback bench `2AT_FLOOR`, no CUDAGraph warnings. Row 1164 has a stochastic output; no-skip check passed, normal bench skips stochastic comparison.
- Rows 1166-1169: measured, 4/4 checks, H100 fallback bench `4AT_FLOOR`, no CUDAGraph warnings. Row 1167 has a stochastic output; no-skip check passed, normal bench skips stochastic comparison.
- Rows 1170-1171: measured, 2/2 checks, H100 fallback bench `1GOOD_1AT_FLOOR`, no CUDAGraph warnings.
- Row 1173 `pointwise_faf725b387f9`: marked `needs_work`; checks pass but parent rerun locked bench is `1BAD_ORACLE`, oracle.py deleted locally.
- Rows 1172, 1174, and 1178: measured, 3/3 checks, H100 fallback bench `2GOOD_1AT_FLOOR`, no CUDAGraph warnings.
- Rows 1175-1176: measured, 2/2 checks, H100 fallback bench `1GOOD_1AT_FLOOR`, no CUDAGraph warnings.
- Rows 1177, 1179, and 1180: measured, 3/3 checks, H100 fallback bench `3AT_FLOOR`, no CUDAGraph warnings.
- Rows 1181-1183: measured, 3/3 checks, H100 fallback bench `2GOOD_1AT_FLOOR`, no CUDAGraph warnings.
- Rows 1184-1188: measured, 5/5 checks, H100 fallback bench `4GOOD_1AT_FLOOR`, no CUDAGraph warnings.
- Rows 1189-1193: measured, 5/5 checks, H100 fallback bench `3GOOD_2AT_FLOOR`, no CUDAGraph warnings.
- Rows 1194-1196: measured, 3/3 checks, H100 fallback bench `2GOOD_1AT_FLOOR`, no CUDAGraph warnings. Row 1195 had a non-CUDAGraph CUDA allocator OOM warning but completed with valid timing.
- Row 1197 `sum_4fd6e4019857`: marked `needs_work`; checks pass but parent rerun locked bench OOMs in the fp64 numerics gate before timing, oracle.py deleted locally.
- Rows 1198 and 1201: measured, 2/2 checks, H100 fallback bench `1GOOD_1AT_FLOOR`, no CUDAGraph warnings. Row 1198 had a non-CUDAGraph CUDA allocator OOM warning but completed with valid timing; row 1201 had a transient pre-bench OOM and passed on retry.
- Row 1207 `sum_75456ad2c2c7`: marked `needs_work`; checks pass but parent rerun locked bench is `1BAD_ORACLE`, oracle.py deleted locally.
- Rows 1199, 1200, and 1203: measured, 3/3 checks, H100 fallback bench `1GOOD_2AT_FLOOR`, no CUDAGraph warnings.
- Rows 1202 and 1205: measured, 2/2 checks, H100 fallback bench `2GOOD`, no CUDAGraph warnings.
- Row 1204 `sum_68fcffe5c7fb`: marked `needs_work`; checks pass but parent rerun locked bench is `1BAD_ORACLE`, oracle.py deleted locally.
- Rows 1206, 1208, and 1209: measured, 3/3 checks, H100 fallback bench `2GOOD_1AT_FLOOR`, no CUDAGraph warnings.
- Rows 1210-1211: measured, 2/2 checks, H100 fallback bench `1GOOD_1AT_FLOOR`, no CUDAGraph warnings.
- Row 1212 `sum_84a4b14c240d`: marked `needs_work`; checks pass but parent rerun locked bench is `1BAD_ORACLE`, oracle.py deleted locally.
- Rows 1213-1214 and 1218: measured, 3/3 checks, H100 fallback bench `3AT_FLOOR`, no CUDAGraph warnings.
- Rows 1215-1216: measured, 2/2 checks, H100 fallback bench `2AT_FLOOR`, no CUDAGraph warnings.
- Row 1217: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`, no CUDAGraph warnings.
- Row 1230 `sum_b011c33c656b`: marked `needs_work`; checks pass but parent rerun locked bench is `NUMERICS_WORSE_THAN_COMPILED`, oracle.py deleted locally.
- Rows 1219-1221: measured, 3/3 checks, H100 fallback bench `2GOOD_1AT_FLOOR`, no CUDAGraph warnings.
- Rows 1223-1225: measured, 3/3 checks, H100 fallback bench `1GOOD_2AT_FLOOR`, no CUDAGraph warnings.
- Row 1222: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`, no CUDAGraph warnings.
- Rows 1226-1228: measured, 3/3 checks, H100 fallback bench `2GOOD_1AT_FLOOR`, no CUDAGraph warnings.
- Rows 1229, 1231, and 1235: measured, 3/3 checks, H100 fallback bench `2GOOD_1AT_FLOOR`, no CUDAGraph warnings.
- Row 1233 `sum_b2873b4c9052`: marked `needs_work`; checks pass but parent rerun locked bench OOMs in the fp64 numerics gate before timing, oracle.py deleted locally.
- Rows 1232 and 1234: measured, 2/2 checks, H100 fallback bench `1GOOD_1AT_FLOOR`, no CUDAGraph warnings.
- Row 1249: measured, 1/1 checks, H100 fallback bench `1GOOD`, no CUDAGraph warnings.
- Rows 1236 and 1242: measured, 2/2 checks, H100 fallback bench `2AT_FLOOR`, no CUDAGraph warnings.
- Row 1238 `sum_c5765737e761`: marked `needs_work`; checks pass but parent rerun locked bench is `NUMERICS_WORSE_THAN_COMPILED`, oracle.py deleted locally.
- Row 1237: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`, no CUDAGraph warnings.
- Row 1239: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`, no CUDAGraph warnings.
- Rows 1241 and 1244: measured, 2/2 checks, H100 fallback bench `1GOOD_1AT_FLOOR`, no CUDAGraph warnings.
- Row 1240: measured, 1/1 checks, H100 fallback bench `1GOOD`, no CUDAGraph warnings.
- Row 1243: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`, no CUDAGraph warnings.
- Row 1245: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`, no CUDAGraph warnings.
- Rows 1246 and 1250: measured, 2/2 checks, H100 fallback bench `1GOOD_1AT_FLOOR`, no CUDAGraph warnings.
- Row 1247 `sum_sum_0189bf613c7e`: marked `needs_work`; checks pass but parent rerun locked bench is `NUMERICS_WORSE_THAN_COMPILED`, oracle.py deleted locally.
- Row 1255 `sum_sum_139d37b65771`: marked `needs_work`; checks pass but parent rerun locked bench is `1BAD_ORACLE`, oracle.py deleted locally.
- Row 1248: measured, 1/1 checks, H100 fallback bench `1GOOD`, no CUDAGraph warnings.
- Row 1251 `sum_sum_0751856ae3a8`: marked `needs_work`; checks pass but parent rerun locked bench is `NUMERICS_WORSE_THAN_COMPILED`, oracle.py deleted locally.
- Rows 1252 and 1254: measured, 2/2 checks, H100 fallback bench `1GOOD_1AT_FLOOR`, no CUDAGraph warnings.
- Row 1253 `sum_sum_0fbee3d7ac79`: marked `needs_work`; checks pass but parent rerun locked bench is `1BAD_ORACLE`, oracle.py deleted locally.
- Row 1260: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`, no CUDAGraph warnings.
- Row 1257: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`, no CUDAGraph warnings.
- Row 1256: measured, 1/1 checks, H100 fallback bench `1GOOD`, no CUDAGraph warnings.
- Row 1259: measured, 1/1 checks, H100 fallback bench `1GOOD`, no CUDAGraph warnings.
- Row 1261: measured, 1/1 checks, H100 fallback bench `1GOOD`, no CUDAGraph warnings.
- Row 1258: measured, 1/1 checks, H100 fallback bench `1GOOD`, no CUDAGraph warnings.
- Rows 1262 and 1263: measured, 2/2 checks, H100 fallback bench `1GOOD_1AT_FLOOR`, no CUDAGraph warnings.
- Rows 1264 and 1265: measured, 2/2 checks, H100 fallback bench `2GOOD`, no CUDAGraph warnings.
- Row 1280: measured from existing worker oracle, 1/1 checks, H100 fallback bench `1GOOD`, no CUDAGraph warnings.
- Rows 1266 and 1268: measured, 2/2 checks, H100 fallback bench `1GOOD_1AT_FLOOR`, no CUDAGraph warnings.
- Row 1267: measured, 1/1 checks, H100 fallback bench `1GOOD`, no CUDAGraph warnings.
- Row 1274 `sum_sum_370ed60792c7`: marked `needs_work`; checks pass but parent rerun locked bench is `1BAD_ORACLE`, oracle.py deleted locally.
- Row 1269: measured, 1/1 checks, H100 fallback bench `1GOOD`, no CUDAGraph warnings.
- Row 1271 `sum_sum_3104a9ce2ada`: marked `needs_work`; checks pass but parent rerun locked bench is `NUMERICS_WORSE_THAN_COMPILED`, oracle.py deleted locally.
- Row 1270: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`, no CUDAGraph warnings.
- Row 1275: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`, no CUDAGraph warnings.
- Row 1272: measured, 1/1 checks, H100 fallback bench `1GOOD`, no CUDAGraph warnings.
- Row 1273: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`, no CUDAGraph warnings.
- Row 1281 `sum_sum_4289edc98a8d`: marked `needs_work`; checks pass but parent rerun locked bench is `1BAD_ORACLE`, oracle.py deleted locally.
- Row 1288 `sum_sum_563676e3726c`: marked `needs_work`; checks pass but parent rerun locked bench is `1BAD_ORACLE`, oracle.py deleted locally.
- Row 1276: measured, 1/1 checks, H100 fallback bench `1GOOD`, no CUDAGraph warnings.
- Row 1277: measured, 1/1 checks, H100 fallback bench `1GOOD`, no CUDAGraph warnings.
- Row 1278: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`, no CUDAGraph warnings.
- Row 1279: measured, 1/1 checks, H100 fallback bench `1GOOD`, no CUDAGraph warnings.
- Row 1282: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`, allocator OOM warning during bench setup but no CUDAGraph warnings.
- Rows 1283 and 1285: measured, 2/2 checks, H100 fallback bench `2AT_FLOOR`, no CUDAGraph warnings.
- Row 1284: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`, no CUDAGraph warnings.
- Row 1301 `sum_sum_77f6be69be60`: marked `needs_work`; checks pass but parent rerun locked bench is `1BAD_ORACLE`, oracle.py deleted locally.
- Rows 1286-1287: measured, 2/2 checks, H100 fallback bench `1GOOD_1AT_FLOOR`, no CUDAGraph warnings.
- Rows 1289-1291 and 1293: measured, 4/4 checks, H100 fallback bench `2GOOD_2AT_FLOOR`, no CUDAGraph warnings.
- Rows 1292, 1294, and 1298: measured, 3/3 checks, H100 fallback bench `3GOOD`, no CUDAGraph warnings.
- Rows 1295-1297: measured, 3/3 checks, H100 fallback bench `1GOOD_2AT_FLOOR`, no CUDAGraph warnings.
- Rows 1299-1300: measured, 2/2 checks, H100 fallback bench `2GOOD`, no CUDAGraph warnings.
- Rows 1302-1303: measured, 2/2 checks, H100 fallback bench `2GOOD`, no CUDAGraph warnings.
- Rows 1304-1305: measured, 2/2 checks, H100 fallback bench `2GOOD`, no CUDAGraph warnings.
- Row 1312 `sum_sum_936e8304ff14`: marked `needs_work`; checks pass but parent rerun locked bench is `1BAD_ORACLE`, oracle.py deleted locally.
- Rows 1306 and 1308: measured, 2/2 checks, H100 fallback bench `2GOOD`, no CUDAGraph warnings.
- Rows 1307 and 1310: measured, 2/2 checks, H100 fallback bench `1GOOD_1AT_FLOOR`, no CUDAGraph warnings.
- Row 1309: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`, no CUDAGraph warnings.
- Rows 1311 and 1314: measured, 2/2 checks, H100 fallback bench `2AT_FLOOR`, no CUDAGraph warnings.
- Row 1313: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`, no CUDAGraph warnings.
- Row 1323 `sum_sum_b089f1ca0d13`: marked `needs_work`; checks pass but parent rerun locked bench is `1BAD_ORACLE`, oracle.py deleted locally.
- Row 1315: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`, no CUDAGraph warnings.
- Rows 1316-1318: measured, 3/3 checks, H100 fallback bench `1GOOD_2AT_FLOOR`, no CUDAGraph warnings.
- Rows 1319-1320 and 1324: measured, 3/3 checks, H100 fallback bench `3GOOD`, no CUDAGraph warnings.
- Row 1327: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`, no CUDAGraph warnings.
- Row 1321: measured, 1/1 checks, H100 fallback bench `1GOOD`, no CUDAGraph warnings.
- Row 1325: measured, 1/1 checks, H100 fallback bench `1GOOD`, no CUDAGraph warnings.
- Row 1322: measured, 1/1 checks, H100 fallback bench `1GOOD`, no CUDAGraph warnings.
- Row 1328: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`, no CUDAGraph warnings.
- Rows 1326 and 1329: measured, 2/2 checks, H100 fallback bench `2GOOD`, no CUDAGraph warnings.
- Rows 1330, 1332, and 1334: measured, 3/3 checks, H100 fallback bench `1GOOD_2AT_FLOOR`, no CUDAGraph warnings.
- Row 1336 `sum_sum_cc37fdc0b4ba`: marked `needs_work`; checks pass but parent rerun locked bench is `1BAD_ORACLE`.
- Row 1331: measured, 1/1 checks, H100 fallback bench `1GOOD`, no CUDAGraph warnings.
- Rows 1333 and 1335: measured, 2/2 checks, H100 fallback bench `2GOOD`, no CUDAGraph warnings.
- Rows 1337-1338 and 1340-1341: measured, 4/4 checks, H100 fallback bench `1GOOD_3AT_FLOOR`, no CUDAGraph warnings.
- Rows 1339 and 1343: measured, 2/2 checks, H100 fallback bench `1GOOD_1AT_FLOOR`, no CUDAGraph warnings.
- Row 1344 `sum_sum_e5ed56d5d094`: marked `needs_work`; checks pass but parent rerun locked bench is `1BAD_ORACLE`.
- Rows 1342, 1345, and 1353: measured, 3/3 checks, H100 fallback bench `1GOOD_2AT_FLOOR`, no CUDAGraph warnings.
- Rows 1346-1347 and 1349: measured, 3/3 checks, H100 fallback bench `3GOOD`, no CUDAGraph warnings.
- Row 1348: measured, 1/1 checks, H100 fallback bench `1GOOD`, no CUDAGraph warnings.
- Rows 1351-1352: measured, 2/2 checks, H100 fallback bench `2GOOD`, no CUDAGraph warnings.
- Rows 1350 and 1354: measured, 2/2 checks, H100 fallback bench `1GOOD_1AT_FLOOR`, no CUDAGraph warnings.
- Rows 1355-1356: measured, 2/2 checks, H100 fallback bench `2GOOD`, no CUDAGraph warnings.
- Row 1357 `sum_sum_sum_0a8714a3041e`: marked `needs_work`; checks pass but parent rerun locked bench fails `NUMERICS_WORSE_THAN_COMPILED`.
- Row 1361: measured, 1/1 checks, H100 fallback bench `1GOOD`, no CUDAGraph warnings.
- Rows 1358-1360: measured, 3/3 checks, H100 fallback bench `3GOOD`, no CUDAGraph warnings.
- Row 1363: measured, 1/1 checks, H100 fallback bench `1GOOD`, no CUDAGraph warnings.
- Rows 1362 and 1364: measured, 2/2 checks, H100 fallback bench `2GOOD`, no CUDAGraph warnings.
- Row 1369 `sum_sum_sum_23c53e2c6899`: marked `needs_work`; checks pass but parent rerun locked bench is `1BAD_ORACLE`.
- Row 1365 `sum_sum_sum_1a561863a1c6`: marked `needs_work`; checks pass but parent rerun locked bench fails `NUMERICS_WORSE_THAN_COMPILED`.
- Row 1368: measured, 1/1 checks, H100 fallback bench `1GOOD`, no CUDAGraph warnings.
- Row 1367: measured, 1/1 checks, H100 fallback bench `1GOOD`, no CUDAGraph warnings.
- Rows 1366 and 1372: measured, 2/2 checks, H100 fallback bench `2GOOD`, no CUDAGraph warnings.
- Rows 1371 and 1375: measured, 2/2 checks, H100 fallback bench `1GOOD_1AT_FLOOR`, no CUDAGraph warnings.
- Rows 1370 and 1373: marked `needs_work`; checks pass but parent rerun locked bench fails `NUMERICS_WORSE_THAN_COMPILED`.
- Rows 1374, 1376, and 1378: measured, 3/3 checks, H100 fallback bench `2GOOD_1AT_FLOOR`, no CUDAGraph warnings.
- Row 1385 `sum_sum_sum_603e69b709ae`: marked `needs_work`; 1/1 checks pass but parent rerun locked bench is `1BAD_ORACLE`.
- Rows 1695-1697: measured, 3/3 checks, H100 fallback bench `3AT_FLOOR`, no CUDAGraph warnings.
- Rows 1377 and 1379: measured, 2/2 checks, H100 fallback bench `2GOOD`, no CUDAGraph warnings.
- Row 1382 `sum_sum_sum_51b3bd5aa388`: marked `needs_work`; 1/1 checks pass but parent rerun locked bench is `1BAD_ORACLE`.
- Rows 1380-1381: measured, 2/2 checks, H100 fallback bench `2GOOD`, no CUDAGraph warnings.
- Row 1386 `sum_sum_sum_60a418792eff`: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`, no CUDAGraph warnings.
- Row 1384 `sum_sum_sum_59c5b1609d60`: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`, no CUDAGraph warnings.
- Rows 1383 and 1387: measured, 2/2 checks, H100 fallback bench `1GOOD_1AT_FLOOR`, no CUDAGraph warnings.
- Rows 1388-1389 and 1391: measured, 3/3 checks, H100 fallback bench `1GOOD_2AT_FLOOR`, no CUDAGraph warnings.
- Rows 1390, 1393, 1396, and 1399: measured, 4/4 checks, H100 fallback bench `3GOOD_1AT_FLOOR`, no CUDAGraph warnings.
- Rows 1392 and 1394: measured, 2/2 checks, H100 fallback bench `2GOOD`, no CUDAGraph warnings.
- Rows 1395 and 1397: measured, 2/2 checks, H100 fallback bench `2GOOD`, no CUDAGraph warnings.
- Row 1398 `sum_sum_sum_86e661f4be59`: marked `needs_work`; 1/1 checks pass but parent rerun locked bench fails `NUMERICS_WORSE_THAN_COMPILED`.
- Row 1556 `var_mean_7947f1107256`: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`, no CUDAGraph warnings.
- Row 1549 `var_mean_6cd69c8f3b06`: marked `needs_work`; 1/1 checks pass but parent rerun locked bench is `1BAD_ORACLE`.
- Rows 1550-1552 and 1559: measured, 4/4 checks, H100 fallback bench `1GOOD_3AT_FLOOR`, no CUDAGraph warnings.
- Rows 1553 and 1555: measured, 2/2 checks, H100 fallback bench `2AT_FLOOR`, no CUDAGraph warnings.
- Row 1554 `var_mean_771a0a39f4b9`: marked `needs_work`; 1/1 checks pass but parent rerun locked bench is `1BAD_ORACLE`.
- Rows 1557-1558 and 1560-1561: measured, 4/4 checks, H100 fallback bench `1GOOD_3AT_FLOOR`, no CUDAGraph warnings.
- Rows 1562-1566: measured, 5/5 checks, H100 fallback bench `2GOOD_3AT_FLOOR`, no CUDAGraph warnings.
- Rows 1567-1571: measured, 5/5 checks, H100 fallback bench `2GOOD_3AT_FLOOR`, no CUDAGraph warnings.
- Rows 1572-1576: measured, 5/5 checks, H100 fallback bench `5AT_FLOOR`, no CUDAGraph warnings.
- Rows 1577-1581: measured, 5/5 checks, H100 fallback bench `1GOOD_4AT_FLOOR`, no CUDAGraph warnings.
- Rows 1582-1586: measured, 5/5 checks, H100 fallback bench `2GOOD_3AT_FLOOR`, no CUDAGraph warnings.
- Rows 1587-1591: measured, 5/5 checks, H100 fallback bench `2GOOD_3AT_FLOOR`, no CUDAGraph warnings.
- Rows 1592-1596: measured, 5/5 checks, H100 fallback bench `5AT_FLOOR`, no CUDAGraph warnings.
- Rows 1597-1601: measured, 5/5 checks, H100 fallback bench `5AT_FLOOR`, no CUDAGraph warnings.
- Rows 1602-1606: measured, 5/5 checks, H100 fallback bench `3GOOD_2AT_FLOOR`, no CUDAGraph warnings.
- Rows 1607-1611: measured, 5/5 checks, H100 fallback bench `2GOOD_3AT_FLOOR`, no CUDAGraph warnings.
- Rows 1612-1616: measured, 5/5 checks, H100 fallback bench `1GOOD_4AT_FLOOR`, no CUDAGraph warnings.
- Rows 1617-1621: measured, 5/5 checks, H100 fallback bench `2GOOD_3AT_FLOOR`, no CUDAGraph warnings.
- Rows 1622-1625 and 1628: measured, 5/5 checks, H100 fallback bench `2GOOD_3AT_FLOOR`, no CUDAGraph warnings.
- Rows 1626 and 1629: measured, 2/2 checks, H100 fallback bench `1GOOD_1AT_FLOOR`, no CUDAGraph warnings.
- Rows 1627 and 1632: measured, 2/2 checks, H100 fallback bench `1GOOD_1AT_FLOOR`, no CUDAGraph warnings.
- Rows 1630-1631 and 1633-1634: measured, 4/4 checks, H100 fallback bench `4AT_FLOOR`, no CUDAGraph warnings.
- Rows 1635-1638: measured, 4/4 checks, H100 fallback bench `4AT_FLOOR`, no CUDAGraph warnings.
- Row 1648 `var_mean_d1f8c258a72a`: marked `needs_work`; checks pass but parent rerun locked bench is `1BAD_ORACLE`, oracle.py deleted locally.
- Row 1639 `var_mean_c7e6741fa8cd`: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`, no CUDAGraph warnings.
- Row 1647 `var_mean_d0ae6f6e894f`: marked `needs_work`; checks pass but parent rerun locked bench fails `NUMERICS_WORSE_THAN_COMPILED` on 4/4 outputs, oracle.py deleted locally.
- Rows 1640-1642: measured, 3/3 checks, H100 fallback bench `2GOOD_1AT_FLOOR`, no CUDAGraph warnings.
- Rows 1643-1644 and 1646: measured, 3/3 checks, H100 fallback bench `1GOOD_2AT_FLOOR`, no CUDAGraph warnings.
- Rows 1645, 1649, and 1651-1654: measured, 6/6 checks, H100 fallback bench `2GOOD_4AT_FLOOR`, no CUDAGraph warnings.
- Rows 1650 and 1655-1658: measured, 5/5 checks, H100 fallback bench `1GOOD_4AT_FLOOR`, no CUDAGraph warnings.
- Row 1659 `var_mean_db63028c4eb9`: marked needs_work after parent rerun reproduced `1BAD_ORACLE`; oracle.py deleted locally.
- Rows 1660-1669 and 1671: measured, 11/11 checks, H100 fallback bench `11AT_FLOOR`, no CUDAGraph warnings.
- Rows 1670, 1672, 1674-1676, and 1678: measured, 6/6 checks, H100 fallback bench `6AT_FLOOR`, no CUDAGraph warnings.
- Row 1673 `var_mean_e59fb82095be`: marked needs_work after parent rerun reproduced `1BAD_ORACLE`; oracle.py deleted locally.
- Rows 1677, 1679, 1681, and 1682: measured, 4/4 checks, H100 fallback bench `1GOOD_3AT_FLOOR`, no CUDAGraph warnings.
- Rows 1680, 1683, and 1687: measured, 3/3 checks, H100 fallback bench `2GOOD_1AT_FLOOR`, no CUDAGraph warnings.
- Rows 1684-1686: measured, 3/3 checks, H100 fallback bench `3AT_FLOOR`, no CUDAGraph warnings.
- Rows 1688-1691: measured, 4/4 checks, H100 fallback bench `1GOOD_3AT_FLOOR`, no CUDAGraph warnings.
- Rows 1692-1694: measured, 3/3 checks, H100 fallback bench `2GOOD_1AT_FLOOR`, no CUDAGraph warnings.
- Row 1700 `var_mean_fbc9e58f2e01`: marked needs_work after parent rerun reproduced `NUMERICS_WORSE_THAN_COMPILED`; oracle.py deleted locally.
- Row 1701 `var_mean_fbfc0104897d`: marked needs_work after parent rerun reproduced `1BAD_ORACLE`; oracle.py deleted locally.
- Rows 1698, 1699, and 1702: measured, 3/3 checks, H100 fallback bench `1GOOD_2AT_FLOOR`, no CUDAGraph warnings.
- Rows 1703, 1706, and 1707: measured, 3/3 checks, H100 fallback bench `3AT_FLOOR`, no CUDAGraph warnings.
- Rows 1704, 1705, and 1708: measured, 3/3 checks, H100 fallback bench `2GOOD_1AT_FLOOR`, no CUDAGraph warnings.
- Rows 1709, 1710, 1712, and 1715: measured, 4/4 checks, H100 fallback bench `2GOOD_2AT_FLOOR`, no CUDAGraph warnings.
- Row 1711 `var_mean_mean_1fc0030c74ea`: marked needs_work after parent rerun reproduced `1BAD_ORACLE`; oracle.py deleted locally.
- Rows 1713, 1714, and 1717: measured, 3/3 checks, H100 fallback bench `1GOOD_2AT_FLOOR`, no CUDAGraph warnings.
- Row 1716 `var_mean_mean_94d5b5ab5e6e`: marked needs_work after parent rerun reproduced `1BAD_ORACLE`; oracle.py deleted locally.
- Row 1718 `var_mean_mean_c2c0ad00da48`: marked needs_work after parent rerun reproduced `1BAD_ORACLE`; oracle.py deleted locally.
- Rows 1719-1722: measured, 4/4 checks, H100 fallback bench `1GOOD_3AT_FLOOR`, no CUDAGraph warnings.
- Rows 1723-1727: measured, 5/5 checks, H100 fallback bench `3GOOD_2AT_FLOOR`, no CUDAGraph warnings.
- Row 61 `var_mean_2e254a2827d8`: measured, 22/22 checks, H100 fallback bench `20GOOD_1AT_FLOOR_1BAD_ORACLE`.
- Row 70 `sum_785c25a716ed`: measured after rework, 3/3 checks, H100 fallback bench `3GOOD`; duplicate shape registration warning is expected for same-shape/different-stride ConvBERT points.
- Row 81 `sum_4a4493837e6e`: measured, 16/16 checks, H100 fallback bench `15GOOD_1AT_FLOOR`.
- Row 84 `pointwise_1c9e8dc48812`: marked `needs_work`; eager-matching oracle fails FP64 bench gate because compiled uses fp32 RoPE arithmetic.
- Row 85 `pointwise_a77badc5e988`: marked `needs_work`; checks pass but FP64 bench gate is NaN/invalid with no timing.
- Row 86 `pointwise_bd0149b22f68`: measured, 1/1 checks, H100 fallback bench `1GOOD`.
- Row 87 `pointwise_e52ac85e10fc`: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`.
- Row 88 `var_mean_eac408f45b9d`: measured, 1/1 checks, H100 fallback bench `1GOOD`.
- Row 89 `sum_7cfdb80c54c5`: measured, 23/23 checks, H100 fallback bench `21GOOD_2AT_FLOOR`.
- Row 90 `sum_6d68a671ec4a`: measured, 5/5 checks, H100 fallback bench `4GOOD_1AT_FLOOR`.
- Row 91 `sum_sum_sum_c5cdd9ab78b4`: measured, 1/1 checks, H100 fallback bench `1GOOD`.
- Row 92 `var_mean_ec0f56a425b2`: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`.
- Row 93 `pointwise_e6ddc8e897ec`: marked `needs_work`; 24/24 checks pass but locked bench fails the FP64 numerics gate with NaN errors and emits no timings.
- Row 95 `pointwise_c509446d4a84`: measured, 3/3 checks, H100 fallback bench `3GOOD`.
- Row 111 `sum_sum_4c6ae5dbcf21`: measured, 2/2 checks, H100 fallback bench `2GOOD`.
- Row 112 `var_mean_45f7dfd4a983`: measured, 5/5 checks, H100 fallback bench `4GOOD_1AT_FLOOR`.
- Row 113 `pointwise_35ecf6633bb0`: measured, 2/2 checks, H100 fallback bench `1GOOD_1AT_FLOOR`.
- Row 114 `pointwise_733dafce05a6`: measured, 2/2 checks, H100 fallback bench `2GOOD`.
- Row 115 `sum_e529e567d636`: measured, 2/2 checks, H100 fallback bench `1GOOD_1AT_FLOOR`.
- Row 121 `sum_sum_sum_e7781939b0a2`: measured, 2/2 checks, H100 fallback bench `1GOOD_1AT_FLOOR`.
- Row 122 `pointwise_88bffcefddc4`: marked `needs_work`; 19/19 checks pass but locked bench fails the FP64 numerics gate with NaN errors for every point and emits no timings.
- Row 123 `sum_sum_cd8694c00507`: measured, 19/19 checks, H100 fallback bench `19GOOD`.
- Row 125 `pointwise_182f6f9450b9`: measured, 10/10 checks, H100 fallback bench `9GOOD_1AT_FLOOR`.
- Row 141 `pointwise_435f6504efa7`: measured, 2/2 checks, H100 fallback bench `1GOOD_1AT_FLOOR`.
- Row 142 `sum_623a84402e27`: measured, 2/2 checks, H100 fallback bench `2GOOD`.
- Row 143 `sum_sum_sum_00516eacb000`: measured, 2/2 checks, H100 fallback bench `1GOOD_1BAD_ORACLE`.
- Row 144 `amax_sum_7f67e161bd21`: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`.
- Row 145 `any_amax_amax_af50781fc699`: measured, 1/1 checks, H100 fallback bench `1GOOD`.
- Row 146 `mean_f21cc667fe83`: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`.
- Row 147 `pointwise_000209e1748d`: marked `needs_work`; 1/1 checks pass but parent rerun locked bench is `1BAD_ORACLE`.
- Row 148 `pointwise_09973679af31`: measured, 24/24 checks, H100 fallback bench `11GOOD_11AT_FLOOR_2BAD_ORACLE`.
- Row 149 `pointwise_25cec8e73161`: marked `needs_work`; 1/1 checks pass but parent rerun locked bench is `1BAD_ORACLE`.
- Row 150 `pointwise_2c1752cb59b4`: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`.
- Row 151 `pointwise_2db21af13668`: measured, 4/4 checks, H100 fallback bench `4GOOD`.
- Row 152 `pointwise_39610fd5aba3`: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`.
- Row 153 `pointwise_3c92a46da990`: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`.
- Row 154 `pointwise_452ad66ee287`: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`.
- Row 155 `pointwise_4f45960cc89d`: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`.
- Row 173 `sum_sum_668480e6f63c`: measured, 1/1 checks, H100 fallback bench `1GOOD`.
- Row 174 `sum_sum_sum_11d45d703ba6`: measured, 1/1 checks, H100 fallback bench `1GOOD`.
- Row 175 `sum_sum_sum_51593d0552e5`: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`.
- Row 176 `sum_sum_sum_565b9b0299d1`: measured, 1/1 checks, H100 fallback bench `1GOOD`.
- Row 177 `var_mean_60f28772f7d2`: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`.
- Row 178 `amax_sum_69008a1fbe7e`: measured, 1/1 checks, H100 fallback bench `1GOOD`.
- Row 179 `amax_sum_sum_a184947064f0`: measured, 1/1 checks, H100 fallback bench `1GOOD`.
- Row 180 `pointwise_260db4f7087d`: measured, 1/1 checks, H100 fallback bench `1GOOD`.
- Row 195 `max_amax_sum_66e6dc6d2131`: measured, 1/1 checks, H100 fallback bench `1GOOD`.
- Row 196 `pointwise_0cd85fd63f82`: marked `needs_work`; 1/1 checks pass but parent rerun locked bench is `1BAD_ORACLE`.
- Row 197 `pointwise_544fac7c5583`: measured, 3/3 checks, H100 fallback bench `2AT_FLOOR_1BAD_ORACLE`.
- Row 198 `sum_be521af00034`: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`.
- Row 199 `sum_sum_sum_33df09c4b328`: measured, 1/1 checks, H100 fallback bench `1GOOD`.
- Row 200 `var_mean_787e1d544efe`: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`.
- Row 201 `pointwise_95ed2ef35da9`: measured, 2/2 checks, H100 fallback bench `2AT_FLOOR`.
- Row 202 `amax_sum_fa4cc85fe5ad`: measured, 2/2 checks, H100 fallback bench `1GOOD_1AT_FLOOR`.
- Row 204 `pointwise_f11bfb86b6b6`: marked `needs_work`; 3/3 checks pass but locked bench fails `NUMERICS_WORSE_THAN_COMPILED` on all points.
- Row 205 `sum_d5a292f49eef`: measured, 2/2 checks, H100 fallback bench `1GOOD_1AT_FLOOR`.
- Row 206 `sum_sum_e2c16e4d1008`: measured, 3/3 checks, H100 fallback bench `1GOOD_2AT_FLOOR`.
- Row 208 `var_mean_841c9dfd6146`: measured, 3/3 checks, H100 fallback bench `2GOOD_1BAD_ORACLE`.
- Row 209 `pointwise_15e408415993`: measured, 3/3 checks, H100 fallback bench `2GOOD_1AT_FLOOR`.
- Row 210 `pointwise_1611fb0c7845`: marked `needs_work`; 2/2 checks pass but locked bench emits empty CUDAGraph warnings.
- Row 211 `pointwise_47bdc8c7c0cd`: measured, 2/2 checks, H100 fallback bench `1GOOD_1AT_FLOOR`.
- Row 212 `pointwise_c8d23ac4414d`: measured, 11/11 checks, H100 fallback bench `8GOOD_2AT_FLOOR_1BAD_ORACLE`.
- Row 213 `pointwise_d5a413a2c233`: marked `needs_work`; 2/2 checks pass but locked bench emits empty CUDAGraph warnings.
- Row 226 `pointwise_ae2a7de08481`: measured, 6/6 checks, H100 fallback bench `4GOOD_2AT_FLOOR`.
- Row 227 `pointwise_ba870b00a0b3`: measured, 6/6 checks, H100 fallback bench `4GOOD_2AT_FLOOR`.
- Row 228 `pointwise_fb4ca2518d23`: measured, 6/6 checks, H100 fallback bench `4GOOD_2AT_FLOOR`.
- Row 229 `sum_sum_292e665668b6`: measured, 10/10 checks, H100 fallback bench `6GOOD_1AT_FLOOR_3BAD_ORACLE`.
- Row 230 `sum_sum_4df091034d07`: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`.
- Row 231 `sum_sum_c91da8ecf163`: measured, 10/10 checks, H100 fallback bench `9GOOD_1BAD_ORACLE`.
- Row 233 `sum_sum_sum_951d09ecbec1`: measured, 3/3 checks, H100 fallback bench `2GOOD_1BAD_ORACLE`.
- Row 234 `var_mean_0ada225c0a04`: measured, 13/13 checks, H100 fallback bench `11GOOD_2AT_FLOOR`.
- Row 235 `var_mean_547bcc63004b`: measured, 3/3 checks, H100 fallback bench `2GOOD_1BAD_ORACLE`.
- Row 236 `var_mean_eb058371f885`: measured, 4/4 checks, H100 fallback bench `3GOOD_1AT_FLOOR`.
- Row 237 `sum_2f173ee403d1`: measured, 2/2 checks, H100 fallback bench `1GOOD_1AT_FLOOR`.
- Row 238 `sum_mean_1db528935b36`: measured, 4/4 checks, H100 fallback bench `2GOOD_2AT_FLOOR`.
- Row 239 `sum_mean_6150664fe0bf`: measured, 4/4 checks, H100 fallback bench `3GOOD_1BAD_ORACLE`.
- Row 241 `sum_sum_sum_db03a2e026aa`: measured, 2/2 checks, H100 fallback bench `1GOOD_1AT_FLOOR`.
- Row 244 `pointwise_c707ca49a4aa`: measured, 1/1 checks, H100 fallback bench `1GOOD`.
- Row 245 `pointwise_eb3a50b8feaa`: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`.
- Row 243 `amax_sum_821fb95bd167`: measured, 4/4 checks, H100 fallback bench `4GOOD`.
- Row 242 `var_mean_e98d6d833b6e`: measured, 4/4 checks, H100 fallback bench `3GOOD_1AT_FLOOR`.
- Row 247 `amax_sum_528a3c274a41`: measured, 1/1 checks, H100 fallback bench `1GOOD`.
- Row 248 `amax_sum_7a65d2915044`: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`.
- Row 251 `pointwise_07c6ea330ab4`: measured, 1/1 checks, H100 fallback bench `1GOOD`.
- Row 249 `amax_sum_8002af197c08`: measured, 1/1 checks, H100 fallback bench `1GOOD`.
- Row 240 `sum_sum_sum_865e7fa30b8a`: measured, 4/4 checks, H100 fallback bench `3GOOD_1BAD_ORACLE`.
- Row 254 `pointwise_0c4549e63f48`: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`.
- Row 253 `pointwise_0c33f088df8a`: measured, 1/1 checks, H100 fallback bench `1GOOD`.
- Row 250 `mean_var_83ed19c04171`: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`.
- Row 252 `pointwise_07e2552977f5`: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`.
- Row 255 `pointwise_2b567494bf14`: measured, 1/1 checks, H100 fallback bench `1GOOD`.
- Row 256 `pointwise_331deef26ac5`: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`.
- Row 259 `pointwise_4d236bfe44e3`: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`.
- Row 258 `pointwise_400b9e2117e6`: measured, 1/1 checks, H100 fallback bench `1GOOD`.
- Row 260 `pointwise_5e648310bc29`: marked `needs_work`; 1/1 checks pass but parent rerun locked bench is `1BAD_ORACLE`.
- Row 264 `pointwise_642e9b69a215`: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`.
- Row 262 `pointwise_62b6a7509fae`: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`.
- Row 265 `pointwise_65717f3841e5`: measured, 1/1 checks, H100 fallback bench `1GOOD`.
- Row 263 `pointwise_62ec9910b2e0`: measured, 1/1 checks, H100 fallback bench `1GOOD`.
- Row 267 `pointwise_7939829e1a9f`: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`.
- Row 261 `pointwise_60c4eafecc9d`: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`.
- Row 266 `pointwise_732ce9dc68ae`: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`.
- Row 269 `pointwise_bae56b847d32`: measured, 1/1 checks, H100 fallback bench `1AT_FLOOR`.
- Row 268 `pointwise_8352994b2efb`: measured, 1/1 checks, H100 fallback bench `1GOOD`.
- Row 276 `sum_07fa37421ff1`: worker checks passed; fallback bench was `1GOOD`.
- Row 277 `sum_2ba0992ca00b`: worker checks passed; fallback bench was `1GOOD`.
- Row 257 `pointwise_3a0cd5d11499`: measured, 3/3 checks, H100 fallback bench `2GOOD_1AT_FLOOR`.

All H100 fallback rows still need native B200 measurement before treating timings as official B200 floors.

## Notes

- Parent verification is required before marking a worker row measured.
- Remote manager commits may convert H100 fallback successes to B200 `needs_work`; preserve those remote statuses during rebase.
- Do not commit active worker oracle files until their workers report completion and parent verification passes.
