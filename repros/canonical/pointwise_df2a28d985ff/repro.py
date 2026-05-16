"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s3_g60
Pattern hash: df2a28d985ff
Shape hash: 36649dda
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_1442: "f32[]", getitem_1443: "f32[]", getitem_1444: "f32[]", getitem_1445: "f32[]", getitem_1446: "f32[]", getitem_1447: "f32[]", getitem_1448: "f32[]", getitem_1449: "f32[]", getitem_1450: "f32[]", getitem_1451: "f32[]", getitem_1452: "f32[]", getitem_1453: "f32[]", getitem_1454: "f32[]", getitem_1455: "f32[]", getitem_1456: "f32[]", getitem_1457: "f32[]", getitem_1458: "f32[]", getitem_1459: "f32[]", getitem_1460: "f32[]", getitem_1461: "f32[]", getitem_1462: "f32[]", getitem_1463: "f32[]", getitem_1464: "f32[]", getitem_1465: "f32[]", getitem_1466: "f32[]", getitem_1467: "f32[]", getitem_1468: "f32[]", getitem_1469: "f32[]", getitem_1470: "f32[]", getitem_1471: "f32[]", getitem_1472: "f32[]", getitem_1473: "f32[]", getitem_1474: "f32[]", getitem_1475: "f32[]", getitem_1476: "f32[]", getitem_1477: "f32[]", getitem_1478: "f32[]", getitem_1479: "f32[]", getitem_1480: "f32[]", getitem_1481: "f32[]", getitem_1482: "f32[]", getitem_1483: "f32[]", getitem_1484: "f32[]", getitem_1485: "f32[]", getitem_1486: "f32[]", getitem_1487: "f32[]", getitem_1488: "f32[]", getitem_1489: "f32[]", getitem_1490: "f32[]", getitem_1491: "f32[]", getitem_1492: "f32[]", getitem_1493: "f32[]", getitem_1494: "f32[]", getitem_1495: "f32[]", getitem_1496: "f32[]", getitem_1497: "f32[]", getitem_1498: "f32[]", getitem_1499: "f32[]", getitem_1500: "f32[]", getitem_1501: "f32[]", getitem_1502: "f32[]", getitem_1503: "f32[]", getitem_1504: "f32[]", getitem_1505: "f32[]", getitem_1506: "f32[]", getitem_1507: "f32[]", getitem_1508: "f32[]", getitem_1509: "f32[]", getitem_1510: "f32[]", getitem_1511: "f32[]", getitem_1512: "f32[]", getitem_1513: "f32[]", getitem_1514: "f32[]", getitem_1515: "f32[]", getitem_1516: "f32[]", getitem_1517: "f32[]", getitem_1518: "f32[]", getitem_1519: "f32[]", getitem_1520: "f32[]", getitem_1521: "f32[]", getitem_1522: "f32[]", getitem_1523: "f32[]", getitem_1524: "f32[]", getitem_1525: "f32[]", getitem_1526: "f32[]", getitem_1527: "f32[]", getitem_1528: "f32[]", getitem_1529: "f32[]", getitem_1530: "f32[]", getitem_1531: "f32[]", getitem_1532: "f32[]", getitem_1533: "f32[]", getitem_1534: "f32[]", getitem_1535: "f32[]", getitem_1536: "f32[]", getitem_1537: "f32[]", getitem_1538: "f32[]", getitem_1539: "f32[]", getitem_1540: "f32[]", getitem_1541: "f32[]", getitem_1542: "f32[]", getitem_1543: "f32[]", getitem_1544: "f32[]", getitem_1545: "f32[]", getitem_1546: "f32[]", getitem_1547: "f32[]", getitem_1548: "f32[]", getitem_1549: "f32[]", getitem_1550: "f32[]", getitem_1551: "f32[]", getitem_1552: "f32[]", getitem_1553: "f32[]", getitem_1554: "f32[]", getitem_1555: "f32[]", getitem_1556: "f32[]", getitem_1557: "f32[]", getitem_1558: "f32[]", getitem_1559: "f32[]", getitem_1560: "f32[]", getitem_1561: "f32[]", getitem_1562: "f32[]", getitem_1563: "f32[]", getitem_1564: "f32[]", getitem_1565: "f32[]", getitem_1566: "f32[]", getitem_1567: "f32[]", getitem_1568: "f32[]", getitem_1569: "f32[]", getitem_1570: "f32[]", getitem_1571: "f32[]", getitem_1572: "f32[]", getitem_1573: "f32[]", getitem_1574: "f32[]", getitem_1575: "f32[]", getitem_1576: "f32[]", getitem_1577: "f32[]", getitem_1578: "f32[]", getitem_1579: "f32[]", getitem_1580: "f32[]", getitem_1581: "f32[]", getitem_1582: "f32[]", getitem_1583: "f32[]", getitem_1584: "f32[]", getitem_1585: "f32[]", getitem_1586: "f32[]", getitem_1587: "f32[]", getitem_1588: "f32[]", getitem_1589: "f32[]", getitem_1590: "f32[]", getitem_1591: "f32[]", getitem_1592: "f32[]", getitem_1593: "f32[]", getitem_1594: "f32[]", getitem_1595: "f32[]", getitem_1596: "f32[]", getitem_1597: "f32[]", getitem_1598: "f32[]", getitem_1599: "f32[]", getitem_1600: "f32[]", getitem_1601: "f32[]", getitem_1602: "f32[]", getitem_1603: "f32[]", getitem_1604: "f32[]", getitem_1605: "f32[]", getitem_1606: "f32[]", getitem_1607: "f32[]", getitem_1608: "f32[]", getitem_1609: "f32[]", getitem_1610: "f32[]", getitem_1611: "f32[]", getitem_1612: "f32[]", getitem_1613: "f32[]", getitem_1614: "f32[]", getitem_1615: "f32[]", getitem_1616: "f32[]", getitem_1617: "f32[]", getitem_1618: "f32[]", getitem_1619: "f32[]", getitem_1620: "f32[]", getitem_1621: "f32[]", getitem_1622: "f32[]", getitem_1623: "f32[]", getitem_1624: "f32[]", getitem_1625: "f32[]", getitem_1626: "f32[]", getitem_1627: "f32[]", getitem_1628: "f32[]", getitem_1629: "f32[]", getitem_1630: "f32[]", getitem_1631: "f32[]", getitem_1632: "f32[]", getitem_1633: "f32[]", getitem_1634: "f32[]", getitem_1635: "f32[]", getitem_1636: "f32[]", getitem_1637: "f32[]", getitem_1638: "f32[]", getitem_1639: "f32[]", getitem_1640: "f32[]", getitem_1641: "f32[]", getitem_1642: "f32[]", getitem_1643: "f32[]", getitem_1644: "f32[]", getitem_1645: "f32[]", getitem_1646: "f32[]", getitem_1647: "f32[]", getitem_2678: "f32[30522, 768]", getitem_2679: "f32[512, 768]", getitem_2680: "f32[1024, 768]", getitem_2681: "f32[1024, 768]", getitem_2682: "f32[1024, 768]", getitem_2683: "f32[1024, 768]", getitem_2684: "f32[2, 768]", getitem_2685: "f32[768]", getitem_2686: "f32[768]", getitem_2687: "f32[768, 768]", getitem_2688: "f32[768]", getitem_2689: "f32[768, 768]", getitem_2690: "f32[768]", getitem_2691: "f32[768, 768]", getitem_2692: "f32[768]", getitem_2693: "f32[768, 768]", getitem_2694: "f32[768]", getitem_2695: "f32[768]", getitem_2696: "f32[768]", getitem_2697: "f32[3072, 768]", getitem_2698: "f32[3072]", getitem_2699: "f32[768, 3072]", getitem_2700: "f32[768]", getitem_2701: "f32[768]", getitem_2702: "f32[768]", getitem_2703: "f32[768, 768]", getitem_2704: "f32[768]", getitem_2705: "f32[768, 768]", getitem_2706: "f32[768]", getitem_2707: "f32[768, 768]", getitem_2708: "f32[768]", getitem_2709: "f32[768, 768]", getitem_2710: "f32[768]", getitem_2711: "f32[768]", getitem_2712: "f32[768]", getitem_2713: "f32[3072, 768]", getitem_2714: "f32[3072]", getitem_2715: "f32[768, 3072]", getitem_2716: "f32[768]", getitem_2717: "f32[768]", getitem_2718: "f32[768]", getitem_2719: "f32[768, 768]", getitem_2720: "f32[768]", getitem_2721: "f32[768, 768]", getitem_2722: "f32[768]", getitem_2723: "f32[768, 768]", getitem_2724: "f32[768]", getitem_2725: "f32[768, 768]", getitem_2726: "f32[768]", getitem_2727: "f32[768]", getitem_2728: "f32[768]", getitem_2729: "f32[3072, 768]", getitem_2730: "f32[3072]", getitem_2731: "f32[768, 3072]", getitem_2732: "f32[768]", getitem_2733: "f32[768]", getitem_2734: "f32[768]", getitem_2735: "f32[768, 768]", getitem_2736: "f32[768]", getitem_2737: "f32[768, 768]", getitem_2738: "f32[768]", getitem_2739: "f32[768, 768]", getitem_2740: "f32[768]", getitem_2741: "f32[768, 768]", getitem_2742: "f32[768]", getitem_2743: "f32[768]", getitem_2744: "f32[768]", getitem_2745: "f32[3072, 768]", getitem_2746: "f32[3072]", getitem_2747: "f32[768, 3072]", getitem_2748: "f32[768]", getitem_2749: "f32[768]", getitem_2750: "f32[768]", getitem_2751: "f32[768, 768]", getitem_2752: "f32[768]", getitem_2753: "f32[768, 768]", getitem_2754: "f32[768]", getitem_2755: "f32[768, 768]", getitem_2756: "f32[768]", getitem_2757: "f32[768, 768]", getitem_2758: "f32[768]", getitem_2759: "f32[768]", getitem_2760: "f32[768]", getitem_2761: "f32[3072, 768]", getitem_2762: "f32[3072]", getitem_2763: "f32[768, 3072]", getitem_2764: "f32[768]", getitem_2765: "f32[768]", getitem_2766: "f32[768]", getitem_2767: "f32[768, 768]", getitem_2768: "f32[768]", getitem_2769: "f32[768, 768]", getitem_2770: "f32[768]", getitem_2771: "f32[768, 768]", getitem_2772: "f32[768]", getitem_2773: "f32[768, 768]", getitem_2774: "f32[768]", getitem_2775: "f32[768]", getitem_2776: "f32[768]", getitem_2777: "f32[3072, 768]", getitem_2778: "f32[3072]", getitem_2779: "f32[768, 3072]", getitem_2780: "f32[768]", getitem_2781: "f32[768]", getitem_2782: "f32[768]", getitem_2783: "f32[768, 768]", getitem_2784: "f32[768]", getitem_2785: "f32[768, 768]", getitem_2786: "f32[768]", getitem_2787: "f32[768, 768]", getitem_2788: "f32[768]", getitem_2789: "f32[768, 768]", getitem_2790: "f32[768]", getitem_2791: "f32[768]", getitem_2792: "f32[768]", getitem_2793: "f32[3072, 768]", getitem_2794: "f32[3072]", getitem_2795: "f32[768, 3072]", getitem_2796: "f32[768]", getitem_2797: "f32[768]", getitem_2798: "f32[768]", getitem_2799: "f32[768, 768]", getitem_2800: "f32[768]", getitem_2801: "f32[768, 768]", getitem_2802: "f32[768]", getitem_2803: "f32[768, 768]", getitem_2804: "f32[768]", getitem_2805: "f32[768, 768]", getitem_2806: "f32[768]", getitem_2807: "f32[768]", getitem_2808: "f32[768]", getitem_2809: "f32[3072, 768]", getitem_2810: "f32[3072]", getitem_2811: "f32[768, 3072]", getitem_2812: "f32[768]", getitem_2813: "f32[768]", getitem_2814: "f32[768]", getitem_2815: "f32[768, 768]", getitem_2816: "f32[768]", getitem_2817: "f32[768, 768]", getitem_2818: "f32[768]", getitem_2819: "f32[768, 768]", getitem_2820: "f32[768]", getitem_2821: "f32[768, 768]", getitem_2822: "f32[768]", getitem_2823: "f32[768]", getitem_2824: "f32[768]", getitem_2825: "f32[3072, 768]", getitem_2826: "f32[3072]", getitem_2827: "f32[768, 3072]", getitem_2828: "f32[768]", getitem_2829: "f32[768]", getitem_2830: "f32[768]", getitem_2831: "f32[768, 768]", getitem_2832: "f32[768]", getitem_2833: "f32[768, 768]", getitem_2834: "f32[768]", getitem_2835: "f32[768, 768]", getitem_2836: "f32[768]", getitem_2837: "f32[768, 768]", getitem_2838: "f32[768]", getitem_2839: "f32[768]", getitem_2840: "f32[768]", getitem_2841: "f32[3072, 768]", getitem_2842: "f32[3072]", getitem_2843: "f32[768, 3072]", getitem_2844: "f32[768]", getitem_2845: "f32[768]", getitem_2846: "f32[768]", getitem_2847: "f32[768, 768]", getitem_2848: "f32[768]", getitem_2849: "f32[768, 768]", getitem_2850: "f32[768]", getitem_2851: "f32[768, 768]", getitem_2852: "f32[768]", getitem_2853: "f32[768, 768]", getitem_2854: "f32[768]", getitem_2855: "f32[768]", getitem_2856: "f32[768]", getitem_2857: "f32[3072, 768]", getitem_2858: "f32[3072]", getitem_2859: "f32[768, 3072]", getitem_2860: "f32[768]", getitem_2861: "f32[768]", getitem_2862: "f32[768]", getitem_2863: "f32[768, 768]", getitem_2864: "f32[768]", getitem_2865: "f32[768, 768]", getitem_2866: "f32[768]", getitem_2867: "f32[768, 768]", getitem_2868: "f32[768]", getitem_2869: "f32[768, 768]", getitem_2870: "f32[768]", getitem_2871: "f32[768]", getitem_2872: "f32[768]", getitem_2873: "f32[3072, 768]", getitem_2874: "f32[3072]", getitem_2875: "f32[768, 3072]", getitem_2876: "f32[768]", getitem_2877: "f32[768]", getitem_2878: "f32[768]", getitem_2879: "f32[30522]", getitem_2880: "f32[768, 768]", getitem_2881: "f32[768]", getitem_2882: "f32[768]", getitem_2883: "f32[768]", getitem_2472: "f32[]", getitem_2473: "f32[]", getitem_2474: "f32[]", getitem_2475: "f32[]", getitem_2476: "f32[]", getitem_2477: "f32[]", getitem_2478: "f32[]", getitem_2479: "f32[]", getitem_2480: "f32[]", getitem_2481: "f32[]", getitem_2482: "f32[]", getitem_2483: "f32[]", getitem_2484: "f32[]", getitem_2485: "f32[]", getitem_2486: "f32[]", getitem_2487: "f32[]", getitem_2488: "f32[]", getitem_2489: "f32[]", getitem_2490: "f32[]", getitem_2491: "f32[]", getitem_2492: "f32[]", getitem_2493: "f32[]", getitem_2494: "f32[]", getitem_2495: "f32[]", getitem_2496: "f32[]", getitem_2497: "f32[]", getitem_2498: "f32[]", getitem_2499: "f32[]", getitem_2500: "f32[]", getitem_2501: "f32[]", getitem_2502: "f32[]", getitem_2503: "f32[]", getitem_2504: "f32[]", getitem_2505: "f32[]", getitem_2506: "f32[]", getitem_2507: "f32[]", getitem_2508: "f32[]", getitem_2509: "f32[]", getitem_2510: "f32[]", getitem_2511: "f32[]", getitem_2512: "f32[]", getitem_2513: "f32[]", getitem_2514: "f32[]", getitem_2515: "f32[]", getitem_2516: "f32[]", getitem_2517: "f32[]", getitem_2518: "f32[]", getitem_2519: "f32[]", getitem_2520: "f32[]", getitem_2521: "f32[]", getitem_2522: "f32[]", getitem_2523: "f32[]", getitem_2524: "f32[]", getitem_2525: "f32[]", getitem_2526: "f32[]", getitem_2527: "f32[]", getitem_2528: "f32[]", getitem_2529: "f32[]", getitem_2530: "f32[]", getitem_2531: "f32[]", getitem_2532: "f32[]", getitem_2533: "f32[]", getitem_2534: "f32[]", getitem_2535: "f32[]", getitem_2536: "f32[]", getitem_2537: "f32[]", getitem_2538: "f32[]", getitem_2539: "f32[]", getitem_2540: "f32[]", getitem_2541: "f32[]", getitem_2542: "f32[]", getitem_2543: "f32[]", getitem_2544: "f32[]", getitem_2545: "f32[]", getitem_2546: "f32[]", getitem_2547: "f32[]", getitem_2548: "f32[]", getitem_2549: "f32[]", getitem_2550: "f32[]", getitem_2551: "f32[]", getitem_2552: "f32[]", getitem_2553: "f32[]", getitem_2554: "f32[]", getitem_2555: "f32[]", getitem_2556: "f32[]", getitem_2557: "f32[]", getitem_2558: "f32[]", getitem_2559: "f32[]", getitem_2560: "f32[]", getitem_2561: "f32[]", getitem_2562: "f32[]", getitem_2563: "f32[]", getitem_2564: "f32[]", getitem_2565: "f32[]", getitem_2566: "f32[]", getitem_2567: "f32[]", getitem_2568: "f32[]", getitem_2569: "f32[]", getitem_2570: "f32[]", getitem_2571: "f32[]", getitem_2572: "f32[]", getitem_2573: "f32[]", getitem_2574: "f32[]", getitem_2575: "f32[]", getitem_2576: "f32[]", getitem_2577: "f32[]", getitem_2578: "f32[]", getitem_2579: "f32[]", getitem_2580: "f32[]", getitem_2581: "f32[]", getitem_2582: "f32[]", getitem_2583: "f32[]", getitem_2584: "f32[]", getitem_2585: "f32[]", getitem_2586: "f32[]", getitem_2587: "f32[]", getitem_2588: "f32[]", getitem_2589: "f32[]", getitem_2590: "f32[]", getitem_2591: "f32[]", getitem_2592: "f32[]", getitem_2593: "f32[]", getitem_2594: "f32[]", getitem_2595: "f32[]", getitem_2596: "f32[]", getitem_2597: "f32[]", getitem_2598: "f32[]", getitem_2599: "f32[]", getitem_2600: "f32[]", getitem_2601: "f32[]", getitem_2602: "f32[]", getitem_2603: "f32[]", getitem_2604: "f32[]", getitem_2605: "f32[]", getitem_2606: "f32[]", getitem_2607: "f32[]", getitem_2608: "f32[]", getitem_2609: "f32[]", getitem_2610: "f32[]", getitem_2611: "f32[]", getitem_2612: "f32[]", getitem_2613: "f32[]", getitem_2614: "f32[]", getitem_2615: "f32[]", getitem_2616: "f32[]", getitem_2617: "f32[]", getitem_2618: "f32[]", getitem_2619: "f32[]", getitem_2620: "f32[]", getitem_2621: "f32[]", getitem_2622: "f32[]", getitem_2623: "f32[]", getitem_2624: "f32[]", getitem_2625: "f32[]", getitem_2626: "f32[]", getitem_2627: "f32[]", getitem_2628: "f32[]", getitem_2629: "f32[]", getitem_2630: "f32[]", getitem_2631: "f32[]", getitem_2632: "f32[]", getitem_2633: "f32[]", getitem_2634: "f32[]", getitem_2635: "f32[]", getitem_2636: "f32[]", getitem_2637: "f32[]", getitem_2638: "f32[]", getitem_2639: "f32[]", getitem_2640: "f32[]", getitem_2641: "f32[]", getitem_2642: "f32[]", getitem_2643: "f32[]", getitem_2644: "f32[]", getitem_2645: "f32[]", getitem_2646: "f32[]", getitem_2647: "f32[]", getitem_2648: "f32[]", getitem_2649: "f32[]", getitem_2650: "f32[]", getitem_2651: "f32[]", getitem_2652: "f32[]", getitem_2653: "f32[]", getitem_2654: "f32[]", getitem_2655: "f32[]", getitem_2656: "f32[]", getitem_2657: "f32[]", getitem_2658: "f32[]", getitem_2659: "f32[]", getitem_2660: "f32[]", getitem_2661: "f32[]", getitem_2662: "f32[]", getitem_2663: "f32[]", getitem_2664: "f32[]", getitem_2665: "f32[]", getitem_2666: "f32[]", getitem_2667: "f32[]", getitem_2668: "f32[]", getitem_2669: "f32[]", getitem_2670: "f32[]", getitem_2671: "f32[]", getitem_2672: "f32[]", getitem_2673: "f32[]", getitem_2674: "f32[]", getitem_2675: "f32[]", getitem_2676: "f32[]", getitem_2677: "f32[]"):
        # No stacktrace found for following nodes
        _foreach_div_scalar = torch.ops.aten._foreach_div.Scalar([getitem_1442, getitem_1443, getitem_1444, getitem_1445, getitem_1446, getitem_1447, getitem_1448, getitem_1449, getitem_1450, getitem_1451, getitem_1452, getitem_1453, getitem_1454, getitem_1455, getitem_1456, getitem_1457, getitem_1458, getitem_1459, getitem_1460, getitem_1461, getitem_1462, getitem_1463, getitem_1464, getitem_1465, getitem_1466, getitem_1467, getitem_1468, getitem_1469, getitem_1470, getitem_1471, getitem_1472, getitem_1473, getitem_1474, getitem_1475, getitem_1476, getitem_1477, getitem_1478, getitem_1479, getitem_1480, getitem_1481, getitem_1482, getitem_1483, getitem_1484, getitem_1485, getitem_1486, getitem_1487, getitem_1488, getitem_1489, getitem_1490, getitem_1491, getitem_1492, getitem_1493, getitem_1494, getitem_1495, getitem_1496, getitem_1497, getitem_1498, getitem_1499, getitem_1500, getitem_1501, getitem_1502, getitem_1503, getitem_1504, getitem_1505, getitem_1506, getitem_1507, getitem_1508, getitem_1509, getitem_1510, getitem_1511, getitem_1512, getitem_1513, getitem_1514, getitem_1515, getitem_1516, getitem_1517, getitem_1518, getitem_1519, getitem_1520, getitem_1521, getitem_1522, getitem_1523, getitem_1524, getitem_1525, getitem_1526, getitem_1527, getitem_1528, getitem_1529, getitem_1530, getitem_1531, getitem_1532, getitem_1533, getitem_1534, getitem_1535, getitem_1536, getitem_1537, getitem_1538, getitem_1539, getitem_1540, getitem_1541, getitem_1542, getitem_1543, getitem_1544, getitem_1545, getitem_1546, getitem_1547, getitem_1548, getitem_1549, getitem_1550, getitem_1551, getitem_1552, getitem_1553, getitem_1554, getitem_1555, getitem_1556, getitem_1557, getitem_1558, getitem_1559, getitem_1560, getitem_1561, getitem_1562, getitem_1563, getitem_1564, getitem_1565, getitem_1566, getitem_1567, getitem_1568, getitem_1569, getitem_1570, getitem_1571, getitem_1572, getitem_1573, getitem_1574, getitem_1575, getitem_1576, getitem_1577, getitem_1578, getitem_1579, getitem_1580, getitem_1581, getitem_1582, getitem_1583, getitem_1584, getitem_1585, getitem_1586, getitem_1587, getitem_1588, getitem_1589, getitem_1590, getitem_1591, getitem_1592, getitem_1593, getitem_1594, getitem_1595, getitem_1596, getitem_1597, getitem_1598, getitem_1599, getitem_1600, getitem_1601, getitem_1602, getitem_1603, getitem_1604, getitem_1605, getitem_1606, getitem_1607, getitem_1608, getitem_1609, getitem_1610, getitem_1611, getitem_1612, getitem_1613, getitem_1614, getitem_1615, getitem_1616, getitem_1617, getitem_1618, getitem_1619, getitem_1620, getitem_1621, getitem_1622, getitem_1623, getitem_1624, getitem_1625, getitem_1626, getitem_1627, getitem_1628, getitem_1629, getitem_1630, getitem_1631, getitem_1632, getitem_1633, getitem_1634, getitem_1635, getitem_1636, getitem_1637, getitem_1638, getitem_1639, getitem_1640, getitem_1641, getitem_1642, getitem_1643, getitem_1644, getitem_1645, getitem_1646, getitem_1647], 0.01);  getitem_1442 = getitem_1443 = getitem_1444 = getitem_1445 = getitem_1446 = getitem_1447 = getitem_1448 = getitem_1449 = getitem_1450 = getitem_1451 = getitem_1452 = getitem_1453 = getitem_1454 = getitem_1455 = getitem_1456 = getitem_1457 = getitem_1458 = getitem_1459 = getitem_1460 = getitem_1461 = getitem_1462 = getitem_1463 = getitem_1464 = getitem_1465 = getitem_1466 = getitem_1467 = getitem_1468 = getitem_1469 = getitem_1470 = getitem_1471 = getitem_1472 = getitem_1473 = getitem_1474 = getitem_1475 = getitem_1476 = getitem_1477 = getitem_1478 = getitem_1479 = getitem_1480 = getitem_1481 = getitem_1482 = getitem_1483 = getitem_1484 = getitem_1485 = getitem_1486 = getitem_1487 = getitem_1488 = getitem_1489 = getitem_1490 = getitem_1491 = getitem_1492 = getitem_1493 = getitem_1494 = getitem_1495 = getitem_1496 = getitem_1497 = getitem_1498 = getitem_1499 = getitem_1500 = getitem_1501 = getitem_1502 = getitem_1503 = getitem_1504 = getitem_1505 = getitem_1506 = getitem_1507 = getitem_1508 = getitem_1509 = getitem_1510 = getitem_1511 = getitem_1512 = getitem_1513 = getitem_1514 = getitem_1515 = getitem_1516 = getitem_1517 = getitem_1518 = getitem_1519 = getitem_1520 = getitem_1521 = getitem_1522 = getitem_1523 = getitem_1524 = getitem_1525 = getitem_1526 = getitem_1527 = getitem_1528 = getitem_1529 = getitem_1530 = getitem_1531 = getitem_1532 = getitem_1533 = getitem_1534 = getitem_1535 = getitem_1536 = getitem_1537 = getitem_1538 = getitem_1539 = getitem_1540 = getitem_1541 = getitem_1542 = getitem_1543 = getitem_1544 = getitem_1545 = getitem_1546 = getitem_1547 = getitem_1548 = getitem_1549 = getitem_1550 = getitem_1551 = getitem_1552 = getitem_1553 = getitem_1554 = getitem_1555 = getitem_1556 = getitem_1557 = getitem_1558 = getitem_1559 = getitem_1560 = getitem_1561 = getitem_1562 = getitem_1563 = getitem_1564 = getitem_1565 = getitem_1566 = getitem_1567 = getitem_1568 = getitem_1569 = getitem_1570 = getitem_1571 = getitem_1572 = getitem_1573 = getitem_1574 = getitem_1575 = getitem_1576 = getitem_1577 = getitem_1578 = getitem_1579 = getitem_1580 = getitem_1581 = getitem_1582 = getitem_1583 = getitem_1584 = getitem_1585 = getitem_1586 = getitem_1587 = getitem_1588 = getitem_1589 = getitem_1590 = getitem_1591 = getitem_1592 = getitem_1593 = getitem_1594 = getitem_1595 = getitem_1596 = getitem_1597 = getitem_1598 = getitem_1599 = getitem_1600 = getitem_1601 = getitem_1602 = getitem_1603 = getitem_1604 = getitem_1605 = getitem_1606 = getitem_1607 = getitem_1608 = getitem_1609 = getitem_1610 = getitem_1611 = getitem_1612 = getitem_1613 = getitem_1614 = getitem_1615 = getitem_1616 = getitem_1617 = getitem_1618 = getitem_1619 = getitem_1620 = getitem_1621 = getitem_1622 = getitem_1623 = getitem_1624 = getitem_1625 = getitem_1626 = getitem_1627 = getitem_1628 = getitem_1629 = getitem_1630 = getitem_1631 = getitem_1632 = getitem_1633 = getitem_1634 = getitem_1635 = getitem_1636 = getitem_1637 = getitem_1638 = getitem_1639 = getitem_1640 = getitem_1641 = getitem_1642 = getitem_1643 = getitem_1644 = getitem_1645 = getitem_1646 = getitem_1647 = None
        getitem: "f32[]" = _foreach_div_scalar[0]
        getitem_1648: "f32[]" = _foreach_div_scalar[1]
        getitem_1649: "f32[]" = _foreach_div_scalar[2]
        getitem_1650: "f32[]" = _foreach_div_scalar[3]
        getitem_1651: "f32[]" = _foreach_div_scalar[4]
        getitem_1652: "f32[]" = _foreach_div_scalar[5]
        getitem_1653: "f32[]" = _foreach_div_scalar[6]
        getitem_1654: "f32[]" = _foreach_div_scalar[7]
        getitem_1655: "f32[]" = _foreach_div_scalar[8]
        getitem_1656: "f32[]" = _foreach_div_scalar[9]
        getitem_1657: "f32[]" = _foreach_div_scalar[10]
        getitem_1658: "f32[]" = _foreach_div_scalar[11]
        getitem_1659: "f32[]" = _foreach_div_scalar[12]
        getitem_1660: "f32[]" = _foreach_div_scalar[13]
        getitem_1661: "f32[]" = _foreach_div_scalar[14]
        getitem_1662: "f32[]" = _foreach_div_scalar[15]
        getitem_1663: "f32[]" = _foreach_div_scalar[16]
        getitem_1664: "f32[]" = _foreach_div_scalar[17]
        getitem_1665: "f32[]" = _foreach_div_scalar[18]
        getitem_1666: "f32[]" = _foreach_div_scalar[19]
        getitem_1667: "f32[]" = _foreach_div_scalar[20]
        getitem_1668: "f32[]" = _foreach_div_scalar[21]
        getitem_1669: "f32[]" = _foreach_div_scalar[22]
        getitem_1670: "f32[]" = _foreach_div_scalar[23]
        getitem_1671: "f32[]" = _foreach_div_scalar[24]
        getitem_1672: "f32[]" = _foreach_div_scalar[25]
        getitem_1673: "f32[]" = _foreach_div_scalar[26]
        getitem_1674: "f32[]" = _foreach_div_scalar[27]
        getitem_1675: "f32[]" = _foreach_div_scalar[28]
        getitem_1676: "f32[]" = _foreach_div_scalar[29]
        getitem_1677: "f32[]" = _foreach_div_scalar[30]
        getitem_1678: "f32[]" = _foreach_div_scalar[31]
        getitem_1679: "f32[]" = _foreach_div_scalar[32]
        getitem_1680: "f32[]" = _foreach_div_scalar[33]
        getitem_1681: "f32[]" = _foreach_div_scalar[34]
        getitem_1682: "f32[]" = _foreach_div_scalar[35]
        getitem_1683: "f32[]" = _foreach_div_scalar[36]
        getitem_1684: "f32[]" = _foreach_div_scalar[37]
        getitem_1685: "f32[]" = _foreach_div_scalar[38]
        getitem_1686: "f32[]" = _foreach_div_scalar[39]
        getitem_1687: "f32[]" = _foreach_div_scalar[40]
        getitem_1688: "f32[]" = _foreach_div_scalar[41]
        getitem_1689: "f32[]" = _foreach_div_scalar[42]
        getitem_1690: "f32[]" = _foreach_div_scalar[43]
        getitem_1691: "f32[]" = _foreach_div_scalar[44]
        getitem_1692: "f32[]" = _foreach_div_scalar[45]
        getitem_1693: "f32[]" = _foreach_div_scalar[46]
        getitem_1694: "f32[]" = _foreach_div_scalar[47]
        getitem_1695: "f32[]" = _foreach_div_scalar[48]
        getitem_1696: "f32[]" = _foreach_div_scalar[49]
        getitem_1697: "f32[]" = _foreach_div_scalar[50]
        getitem_1698: "f32[]" = _foreach_div_scalar[51]
        getitem_1699: "f32[]" = _foreach_div_scalar[52]
        getitem_1700: "f32[]" = _foreach_div_scalar[53]
        getitem_1701: "f32[]" = _foreach_div_scalar[54]
        getitem_1702: "f32[]" = _foreach_div_scalar[55]
        getitem_1703: "f32[]" = _foreach_div_scalar[56]
        getitem_1704: "f32[]" = _foreach_div_scalar[57]
        getitem_1705: "f32[]" = _foreach_div_scalar[58]
        getitem_1706: "f32[]" = _foreach_div_scalar[59]
        getitem_1707: "f32[]" = _foreach_div_scalar[60]
        getitem_1708: "f32[]" = _foreach_div_scalar[61]
        getitem_1709: "f32[]" = _foreach_div_scalar[62]
        getitem_1710: "f32[]" = _foreach_div_scalar[63]
        getitem_1711: "f32[]" = _foreach_div_scalar[64]
        getitem_1712: "f32[]" = _foreach_div_scalar[65]
        getitem_1713: "f32[]" = _foreach_div_scalar[66]
        getitem_1714: "f32[]" = _foreach_div_scalar[67]
        getitem_1715: "f32[]" = _foreach_div_scalar[68]
        getitem_1716: "f32[]" = _foreach_div_scalar[69]
        getitem_1717: "f32[]" = _foreach_div_scalar[70]
        getitem_1718: "f32[]" = _foreach_div_scalar[71]
        getitem_1719: "f32[]" = _foreach_div_scalar[72]
        getitem_1720: "f32[]" = _foreach_div_scalar[73]
        getitem_1721: "f32[]" = _foreach_div_scalar[74]
        getitem_1722: "f32[]" = _foreach_div_scalar[75]
        getitem_1723: "f32[]" = _foreach_div_scalar[76]
        getitem_1724: "f32[]" = _foreach_div_scalar[77]
        getitem_1725: "f32[]" = _foreach_div_scalar[78]
        getitem_1726: "f32[]" = _foreach_div_scalar[79]
        getitem_1727: "f32[]" = _foreach_div_scalar[80]
        getitem_1728: "f32[]" = _foreach_div_scalar[81]
        getitem_1729: "f32[]" = _foreach_div_scalar[82]
        getitem_1730: "f32[]" = _foreach_div_scalar[83]
        getitem_1731: "f32[]" = _foreach_div_scalar[84]
        getitem_1732: "f32[]" = _foreach_div_scalar[85]
        getitem_1733: "f32[]" = _foreach_div_scalar[86]
        getitem_1734: "f32[]" = _foreach_div_scalar[87]
        getitem_1735: "f32[]" = _foreach_div_scalar[88]
        getitem_1736: "f32[]" = _foreach_div_scalar[89]
        getitem_1737: "f32[]" = _foreach_div_scalar[90]
        getitem_1738: "f32[]" = _foreach_div_scalar[91]
        getitem_1739: "f32[]" = _foreach_div_scalar[92]
        getitem_1740: "f32[]" = _foreach_div_scalar[93]
        getitem_1741: "f32[]" = _foreach_div_scalar[94]
        getitem_1742: "f32[]" = _foreach_div_scalar[95]
        getitem_1743: "f32[]" = _foreach_div_scalar[96]
        getitem_1744: "f32[]" = _foreach_div_scalar[97]
        getitem_1745: "f32[]" = _foreach_div_scalar[98]
        getitem_1746: "f32[]" = _foreach_div_scalar[99]
        getitem_1747: "f32[]" = _foreach_div_scalar[100]
        getitem_1748: "f32[]" = _foreach_div_scalar[101]
        getitem_1749: "f32[]" = _foreach_div_scalar[102]
        getitem_1750: "f32[]" = _foreach_div_scalar[103]
        getitem_1751: "f32[]" = _foreach_div_scalar[104]
        getitem_1752: "f32[]" = _foreach_div_scalar[105]
        getitem_1753: "f32[]" = _foreach_div_scalar[106]
        getitem_1754: "f32[]" = _foreach_div_scalar[107]
        getitem_1755: "f32[]" = _foreach_div_scalar[108]
        getitem_1756: "f32[]" = _foreach_div_scalar[109]
        getitem_1757: "f32[]" = _foreach_div_scalar[110]
        getitem_1758: "f32[]" = _foreach_div_scalar[111]
        getitem_1759: "f32[]" = _foreach_div_scalar[112]
        getitem_1760: "f32[]" = _foreach_div_scalar[113]
        getitem_1761: "f32[]" = _foreach_div_scalar[114]
        getitem_1762: "f32[]" = _foreach_div_scalar[115]
        getitem_1763: "f32[]" = _foreach_div_scalar[116]
        getitem_1764: "f32[]" = _foreach_div_scalar[117]
        getitem_1765: "f32[]" = _foreach_div_scalar[118]
        getitem_1766: "f32[]" = _foreach_div_scalar[119]
        getitem_1767: "f32[]" = _foreach_div_scalar[120]
        getitem_1768: "f32[]" = _foreach_div_scalar[121]
        getitem_1769: "f32[]" = _foreach_div_scalar[122]
        getitem_1770: "f32[]" = _foreach_div_scalar[123]
        getitem_1771: "f32[]" = _foreach_div_scalar[124]
        getitem_1772: "f32[]" = _foreach_div_scalar[125]
        getitem_1773: "f32[]" = _foreach_div_scalar[126]
        getitem_1774: "f32[]" = _foreach_div_scalar[127]
        getitem_1775: "f32[]" = _foreach_div_scalar[128]
        getitem_1776: "f32[]" = _foreach_div_scalar[129]
        getitem_1777: "f32[]" = _foreach_div_scalar[130]
        getitem_1778: "f32[]" = _foreach_div_scalar[131]
        getitem_1779: "f32[]" = _foreach_div_scalar[132]
        getitem_1780: "f32[]" = _foreach_div_scalar[133]
        getitem_1781: "f32[]" = _foreach_div_scalar[134]
        getitem_1782: "f32[]" = _foreach_div_scalar[135]
        getitem_1783: "f32[]" = _foreach_div_scalar[136]
        getitem_1784: "f32[]" = _foreach_div_scalar[137]
        getitem_1785: "f32[]" = _foreach_div_scalar[138]
        getitem_1786: "f32[]" = _foreach_div_scalar[139]
        getitem_1787: "f32[]" = _foreach_div_scalar[140]
        getitem_1788: "f32[]" = _foreach_div_scalar[141]
        getitem_1789: "f32[]" = _foreach_div_scalar[142]
        getitem_1790: "f32[]" = _foreach_div_scalar[143]
        getitem_1791: "f32[]" = _foreach_div_scalar[144]
        getitem_1792: "f32[]" = _foreach_div_scalar[145]
        getitem_1793: "f32[]" = _foreach_div_scalar[146]
        getitem_1794: "f32[]" = _foreach_div_scalar[147]
        getitem_1795: "f32[]" = _foreach_div_scalar[148]
        getitem_1796: "f32[]" = _foreach_div_scalar[149]
        getitem_1797: "f32[]" = _foreach_div_scalar[150]
        getitem_1798: "f32[]" = _foreach_div_scalar[151]
        getitem_1799: "f32[]" = _foreach_div_scalar[152]
        getitem_1800: "f32[]" = _foreach_div_scalar[153]
        getitem_1801: "f32[]" = _foreach_div_scalar[154]
        getitem_1802: "f32[]" = _foreach_div_scalar[155]
        getitem_1803: "f32[]" = _foreach_div_scalar[156]
        getitem_1804: "f32[]" = _foreach_div_scalar[157]
        getitem_1805: "f32[]" = _foreach_div_scalar[158]
        getitem_1806: "f32[]" = _foreach_div_scalar[159]
        getitem_1807: "f32[]" = _foreach_div_scalar[160]
        getitem_1808: "f32[]" = _foreach_div_scalar[161]
        getitem_1809: "f32[]" = _foreach_div_scalar[162]
        getitem_1810: "f32[]" = _foreach_div_scalar[163]
        getitem_1811: "f32[]" = _foreach_div_scalar[164]
        getitem_1812: "f32[]" = _foreach_div_scalar[165]
        getitem_1813: "f32[]" = _foreach_div_scalar[166]
        getitem_1814: "f32[]" = _foreach_div_scalar[167]
        getitem_1815: "f32[]" = _foreach_div_scalar[168]
        getitem_1816: "f32[]" = _foreach_div_scalar[169]
        getitem_1817: "f32[]" = _foreach_div_scalar[170]
        getitem_1818: "f32[]" = _foreach_div_scalar[171]
        getitem_1819: "f32[]" = _foreach_div_scalar[172]
        getitem_1820: "f32[]" = _foreach_div_scalar[173]
        getitem_1821: "f32[]" = _foreach_div_scalar[174]
        getitem_1822: "f32[]" = _foreach_div_scalar[175]
        getitem_1823: "f32[]" = _foreach_div_scalar[176]
        getitem_1824: "f32[]" = _foreach_div_scalar[177]
        getitem_1825: "f32[]" = _foreach_div_scalar[178]
        getitem_1826: "f32[]" = _foreach_div_scalar[179]
        getitem_1827: "f32[]" = _foreach_div_scalar[180]
        getitem_1828: "f32[]" = _foreach_div_scalar[181]
        getitem_1829: "f32[]" = _foreach_div_scalar[182]
        getitem_1830: "f32[]" = _foreach_div_scalar[183]
        getitem_1831: "f32[]" = _foreach_div_scalar[184]
        getitem_1832: "f32[]" = _foreach_div_scalar[185]
        getitem_1833: "f32[]" = _foreach_div_scalar[186]
        getitem_1834: "f32[]" = _foreach_div_scalar[187]
        getitem_1835: "f32[]" = _foreach_div_scalar[188]
        getitem_1836: "f32[]" = _foreach_div_scalar[189]
        getitem_1837: "f32[]" = _foreach_div_scalar[190]
        getitem_1838: "f32[]" = _foreach_div_scalar[191]
        getitem_1839: "f32[]" = _foreach_div_scalar[192]
        getitem_1840: "f32[]" = _foreach_div_scalar[193]
        getitem_1841: "f32[]" = _foreach_div_scalar[194]
        getitem_1842: "f32[]" = _foreach_div_scalar[195]
        getitem_1843: "f32[]" = _foreach_div_scalar[196]
        getitem_1844: "f32[]" = _foreach_div_scalar[197]
        getitem_1845: "f32[]" = _foreach_div_scalar[198]
        getitem_1846: "f32[]" = _foreach_div_scalar[199]
        getitem_1847: "f32[]" = _foreach_div_scalar[200]
        getitem_1848: "f32[]" = _foreach_div_scalar[201]
        getitem_1849: "f32[]" = _foreach_div_scalar[202]
        getitem_1850: "f32[]" = _foreach_div_scalar[203]
        getitem_1851: "f32[]" = _foreach_div_scalar[204]
        getitem_1852: "f32[]" = _foreach_div_scalar[205];  _foreach_div_scalar = None
        _foreach_div_list = torch.ops.aten._foreach_div.List([getitem_2678, getitem_2679, getitem_2680, getitem_2681, getitem_2682, getitem_2683, getitem_2684, getitem_2685, getitem_2686, getitem_2687, getitem_2688, getitem_2689, getitem_2690, getitem_2691, getitem_2692, getitem_2693, getitem_2694, getitem_2695, getitem_2696, getitem_2697, getitem_2698, getitem_2699, getitem_2700, getitem_2701, getitem_2702, getitem_2703, getitem_2704, getitem_2705, getitem_2706, getitem_2707, getitem_2708, getitem_2709, getitem_2710, getitem_2711, getitem_2712, getitem_2713, getitem_2714, getitem_2715, getitem_2716, getitem_2717, getitem_2718, getitem_2719, getitem_2720, getitem_2721, getitem_2722, getitem_2723, getitem_2724, getitem_2725, getitem_2726, getitem_2727, getitem_2728, getitem_2729, getitem_2730, getitem_2731, getitem_2732, getitem_2733, getitem_2734, getitem_2735, getitem_2736, getitem_2737, getitem_2738, getitem_2739, getitem_2740, getitem_2741, getitem_2742, getitem_2743, getitem_2744, getitem_2745, getitem_2746, getitem_2747, getitem_2748, getitem_2749, getitem_2750, getitem_2751, getitem_2752, getitem_2753, getitem_2754, getitem_2755, getitem_2756, getitem_2757, getitem_2758, getitem_2759, getitem_2760, getitem_2761, getitem_2762, getitem_2763, getitem_2764, getitem_2765, getitem_2766, getitem_2767, getitem_2768, getitem_2769, getitem_2770, getitem_2771, getitem_2772, getitem_2773, getitem_2774, getitem_2775, getitem_2776, getitem_2777, getitem_2778, getitem_2779, getitem_2780, getitem_2781, getitem_2782, getitem_2783, getitem_2784, getitem_2785, getitem_2786, getitem_2787, getitem_2788, getitem_2789, getitem_2790, getitem_2791, getitem_2792, getitem_2793, getitem_2794, getitem_2795, getitem_2796, getitem_2797, getitem_2798, getitem_2799, getitem_2800, getitem_2801, getitem_2802, getitem_2803, getitem_2804, getitem_2805, getitem_2806, getitem_2807, getitem_2808, getitem_2809, getitem_2810, getitem_2811, getitem_2812, getitem_2813, getitem_2814, getitem_2815, getitem_2816, getitem_2817, getitem_2818, getitem_2819, getitem_2820, getitem_2821, getitem_2822, getitem_2823, getitem_2824, getitem_2825, getitem_2826, getitem_2827, getitem_2828, getitem_2829, getitem_2830, getitem_2831, getitem_2832, getitem_2833, getitem_2834, getitem_2835, getitem_2836, getitem_2837, getitem_2838, getitem_2839, getitem_2840, getitem_2841, getitem_2842, getitem_2843, getitem_2844, getitem_2845, getitem_2846, getitem_2847, getitem_2848, getitem_2849, getitem_2850, getitem_2851, getitem_2852, getitem_2853, getitem_2854, getitem_2855, getitem_2856, getitem_2857, getitem_2858, getitem_2859, getitem_2860, getitem_2861, getitem_2862, getitem_2863, getitem_2864, getitem_2865, getitem_2866, getitem_2867, getitem_2868, getitem_2869, getitem_2870, getitem_2871, getitem_2872, getitem_2873, getitem_2874, getitem_2875, getitem_2876, getitem_2877, getitem_2878, getitem_2879, getitem_2880, getitem_2881, getitem_2882, getitem_2883], [getitem_2472, getitem_2473, getitem_2474, getitem_2475, getitem_2476, getitem_2477, getitem_2478, getitem_2479, getitem_2480, getitem_2481, getitem_2482, getitem_2483, getitem_2484, getitem_2485, getitem_2486, getitem_2487, getitem_2488, getitem_2489, getitem_2490, getitem_2491, getitem_2492, getitem_2493, getitem_2494, getitem_2495, getitem_2496, getitem_2497, getitem_2498, getitem_2499, getitem_2500, getitem_2501, getitem_2502, getitem_2503, getitem_2504, getitem_2505, getitem_2506, getitem_2507, getitem_2508, getitem_2509, getitem_2510, getitem_2511, getitem_2512, getitem_2513, getitem_2514, getitem_2515, getitem_2516, getitem_2517, getitem_2518, getitem_2519, getitem_2520, getitem_2521, getitem_2522, getitem_2523, getitem_2524, getitem_2525, getitem_2526, getitem_2527, getitem_2528, getitem_2529, getitem_2530, getitem_2531, getitem_2532, getitem_2533, getitem_2534, getitem_2535, getitem_2536, getitem_2537, getitem_2538, getitem_2539, getitem_2540, getitem_2541, getitem_2542, getitem_2543, getitem_2544, getitem_2545, getitem_2546, getitem_2547, getitem_2548, getitem_2549, getitem_2550, getitem_2551, getitem_2552, getitem_2553, getitem_2554, getitem_2555, getitem_2556, getitem_2557, getitem_2558, getitem_2559, getitem_2560, getitem_2561, getitem_2562, getitem_2563, getitem_2564, getitem_2565, getitem_2566, getitem_2567, getitem_2568, getitem_2569, getitem_2570, getitem_2571, getitem_2572, getitem_2573, getitem_2574, getitem_2575, getitem_2576, getitem_2577, getitem_2578, getitem_2579, getitem_2580, getitem_2581, getitem_2582, getitem_2583, getitem_2584, getitem_2585, getitem_2586, getitem_2587, getitem_2588, getitem_2589, getitem_2590, getitem_2591, getitem_2592, getitem_2593, getitem_2594, getitem_2595, getitem_2596, getitem_2597, getitem_2598, getitem_2599, getitem_2600, getitem_2601, getitem_2602, getitem_2603, getitem_2604, getitem_2605, getitem_2606, getitem_2607, getitem_2608, getitem_2609, getitem_2610, getitem_2611, getitem_2612, getitem_2613, getitem_2614, getitem_2615, getitem_2616, getitem_2617, getitem_2618, getitem_2619, getitem_2620, getitem_2621, getitem_2622, getitem_2623, getitem_2624, getitem_2625, getitem_2626, getitem_2627, getitem_2628, getitem_2629, getitem_2630, getitem_2631, getitem_2632, getitem_2633, getitem_2634, getitem_2635, getitem_2636, getitem_2637, getitem_2638, getitem_2639, getitem_2640, getitem_2641, getitem_2642, getitem_2643, getitem_2644, getitem_2645, getitem_2646, getitem_2647, getitem_2648, getitem_2649, getitem_2650, getitem_2651, getitem_2652, getitem_2653, getitem_2654, getitem_2655, getitem_2656, getitem_2657, getitem_2658, getitem_2659, getitem_2660, getitem_2661, getitem_2662, getitem_2663, getitem_2664, getitem_2665, getitem_2666, getitem_2667, getitem_2668, getitem_2669, getitem_2670, getitem_2671, getitem_2672, getitem_2673, getitem_2674, getitem_2675, getitem_2676, getitem_2677]);  getitem_2678 = getitem_2679 = getitem_2680 = getitem_2681 = getitem_2682 = getitem_2683 = getitem_2684 = getitem_2685 = getitem_2686 = getitem_2687 = getitem_2688 = getitem_2689 = getitem_2690 = getitem_2691 = getitem_2692 = getitem_2693 = getitem_2694 = getitem_2695 = getitem_2696 = getitem_2697 = getitem_2698 = getitem_2699 = getitem_2700 = getitem_2701 = getitem_2702 = getitem_2703 = getitem_2704 = getitem_2705 = getitem_2706 = getitem_2707 = getitem_2708 = getitem_2709 = getitem_2710 = getitem_2711 = getitem_2712 = getitem_2713 = getitem_2714 = getitem_2715 = getitem_2716 = getitem_2717 = getitem_2718 = getitem_2719 = getitem_2720 = getitem_2721 = getitem_2722 = getitem_2723 = getitem_2724 = getitem_2725 = getitem_2726 = getitem_2727 = getitem_2728 = getitem_2729 = getitem_2730 = getitem_2731 = getitem_2732 = getitem_2733 = getitem_2734 = getitem_2735 = getitem_2736 = getitem_2737 = getitem_2738 = getitem_2739 = getitem_2740 = getitem_2741 = getitem_2742 = getitem_2743 = getitem_2744 = getitem_2745 = getitem_2746 = getitem_2747 = getitem_2748 = getitem_2749 = getitem_2750 = getitem_2751 = getitem_2752 = getitem_2753 = getitem_2754 = getitem_2755 = getitem_2756 = getitem_2757 = getitem_2758 = getitem_2759 = getitem_2760 = getitem_2761 = getitem_2762 = getitem_2763 = getitem_2764 = getitem_2765 = getitem_2766 = getitem_2767 = getitem_2768 = getitem_2769 = getitem_2770 = getitem_2771 = getitem_2772 = getitem_2773 = getitem_2774 = getitem_2775 = getitem_2776 = getitem_2777 = getitem_2778 = getitem_2779 = getitem_2780 = getitem_2781 = getitem_2782 = getitem_2783 = getitem_2784 = getitem_2785 = getitem_2786 = getitem_2787 = getitem_2788 = getitem_2789 = getitem_2790 = getitem_2791 = getitem_2792 = getitem_2793 = getitem_2794 = getitem_2795 = getitem_2796 = getitem_2797 = getitem_2798 = getitem_2799 = getitem_2800 = getitem_2801 = getitem_2802 = getitem_2803 = getitem_2804 = getitem_2805 = getitem_2806 = getitem_2807 = getitem_2808 = getitem_2809 = getitem_2810 = getitem_2811 = getitem_2812 = getitem_2813 = getitem_2814 = getitem_2815 = getitem_2816 = getitem_2817 = getitem_2818 = getitem_2819 = getitem_2820 = getitem_2821 = getitem_2822 = getitem_2823 = getitem_2824 = getitem_2825 = getitem_2826 = getitem_2827 = getitem_2828 = getitem_2829 = getitem_2830 = getitem_2831 = getitem_2832 = getitem_2833 = getitem_2834 = getitem_2835 = getitem_2836 = getitem_2837 = getitem_2838 = getitem_2839 = getitem_2840 = getitem_2841 = getitem_2842 = getitem_2843 = getitem_2844 = getitem_2845 = getitem_2846 = getitem_2847 = getitem_2848 = getitem_2849 = getitem_2850 = getitem_2851 = getitem_2852 = getitem_2853 = getitem_2854 = getitem_2855 = getitem_2856 = getitem_2857 = getitem_2858 = getitem_2859 = getitem_2860 = getitem_2861 = getitem_2862 = getitem_2863 = getitem_2864 = getitem_2865 = getitem_2866 = getitem_2867 = getitem_2868 = getitem_2869 = getitem_2870 = getitem_2871 = getitem_2872 = getitem_2873 = getitem_2874 = getitem_2875 = getitem_2876 = getitem_2877 = getitem_2878 = getitem_2879 = getitem_2880 = getitem_2881 = getitem_2882 = getitem_2883 = getitem_2472 = getitem_2473 = getitem_2474 = getitem_2475 = getitem_2476 = getitem_2477 = getitem_2478 = getitem_2479 = getitem_2480 = getitem_2481 = getitem_2482 = getitem_2483 = getitem_2484 = getitem_2485 = getitem_2486 = getitem_2487 = getitem_2488 = getitem_2489 = getitem_2490 = getitem_2491 = getitem_2492 = getitem_2493 = getitem_2494 = getitem_2495 = getitem_2496 = getitem_2497 = getitem_2498 = getitem_2499 = getitem_2500 = getitem_2501 = getitem_2502 = getitem_2503 = getitem_2504 = getitem_2505 = getitem_2506 = getitem_2507 = getitem_2508 = getitem_2509 = getitem_2510 = getitem_2511 = getitem_2512 = getitem_2513 = getitem_2514 = getitem_2515 = getitem_2516 = getitem_2517 = getitem_2518 = getitem_2519 = getitem_2520 = getitem_2521 = getitem_2522 = getitem_2523 = getitem_2524 = getitem_2525 = getitem_2526 = getitem_2527 = getitem_2528 = getitem_2529 = getitem_2530 = getitem_2531 = getitem_2532 = getitem_2533 = getitem_2534 = getitem_2535 = getitem_2536 = getitem_2537 = getitem_2538 = getitem_2539 = getitem_2540 = getitem_2541 = getitem_2542 = getitem_2543 = getitem_2544 = getitem_2545 = getitem_2546 = getitem_2547 = getitem_2548 = getitem_2549 = getitem_2550 = getitem_2551 = getitem_2552 = getitem_2553 = getitem_2554 = getitem_2555 = getitem_2556 = getitem_2557 = getitem_2558 = getitem_2559 = getitem_2560 = getitem_2561 = getitem_2562 = getitem_2563 = getitem_2564 = getitem_2565 = getitem_2566 = getitem_2567 = getitem_2568 = getitem_2569 = getitem_2570 = getitem_2571 = getitem_2572 = getitem_2573 = getitem_2574 = getitem_2575 = getitem_2576 = getitem_2577 = getitem_2578 = getitem_2579 = getitem_2580 = getitem_2581 = getitem_2582 = getitem_2583 = getitem_2584 = getitem_2585 = getitem_2586 = getitem_2587 = getitem_2588 = getitem_2589 = getitem_2590 = getitem_2591 = getitem_2592 = getitem_2593 = getitem_2594 = getitem_2595 = getitem_2596 = getitem_2597 = getitem_2598 = getitem_2599 = getitem_2600 = getitem_2601 = getitem_2602 = getitem_2603 = getitem_2604 = getitem_2605 = getitem_2606 = getitem_2607 = getitem_2608 = getitem_2609 = getitem_2610 = getitem_2611 = getitem_2612 = getitem_2613 = getitem_2614 = getitem_2615 = getitem_2616 = getitem_2617 = getitem_2618 = getitem_2619 = getitem_2620 = getitem_2621 = getitem_2622 = getitem_2623 = getitem_2624 = getitem_2625 = getitem_2626 = getitem_2627 = getitem_2628 = getitem_2629 = getitem_2630 = getitem_2631 = getitem_2632 = getitem_2633 = getitem_2634 = getitem_2635 = getitem_2636 = getitem_2637 = getitem_2638 = getitem_2639 = getitem_2640 = getitem_2641 = getitem_2642 = getitem_2643 = getitem_2644 = getitem_2645 = getitem_2646 = getitem_2647 = getitem_2648 = getitem_2649 = getitem_2650 = getitem_2651 = getitem_2652 = getitem_2653 = getitem_2654 = getitem_2655 = getitem_2656 = getitem_2657 = getitem_2658 = getitem_2659 = getitem_2660 = getitem_2661 = getitem_2662 = getitem_2663 = getitem_2664 = getitem_2665 = getitem_2666 = getitem_2667 = getitem_2668 = getitem_2669 = getitem_2670 = getitem_2671 = getitem_2672 = getitem_2673 = getitem_2674 = getitem_2675 = getitem_2676 = getitem_2677 = None
        getitem_2884: "f32[30522, 768]" = _foreach_div_list[0]
        getitem_2885: "f32[512, 768]" = _foreach_div_list[1]
        getitem_2886: "f32[1024, 768]" = _foreach_div_list[2]
        getitem_2887: "f32[1024, 768]" = _foreach_div_list[3]
        getitem_2888: "f32[1024, 768]" = _foreach_div_list[4]
        getitem_2889: "f32[1024, 768]" = _foreach_div_list[5]
        getitem_2890: "f32[2, 768]" = _foreach_div_list[6]
        getitem_2891: "f32[768]" = _foreach_div_list[7]
        getitem_2892: "f32[768]" = _foreach_div_list[8]
        getitem_2893: "f32[768, 768]" = _foreach_div_list[9]
        getitem_2894: "f32[768]" = _foreach_div_list[10]
        getitem_2895: "f32[768, 768]" = _foreach_div_list[11]
        getitem_2896: "f32[768]" = _foreach_div_list[12]
        getitem_2897: "f32[768, 768]" = _foreach_div_list[13]
        getitem_2898: "f32[768]" = _foreach_div_list[14]
        getitem_2899: "f32[768, 768]" = _foreach_div_list[15]
        getitem_2900: "f32[768]" = _foreach_div_list[16]
        getitem_2901: "f32[768]" = _foreach_div_list[17]
        getitem_2902: "f32[768]" = _foreach_div_list[18]
        getitem_2903: "f32[3072, 768]" = _foreach_div_list[19]
        getitem_2904: "f32[3072]" = _foreach_div_list[20]
        getitem_2905: "f32[768, 3072]" = _foreach_div_list[21]
        getitem_2906: "f32[768]" = _foreach_div_list[22]
        getitem_2907: "f32[768]" = _foreach_div_list[23]
        getitem_2908: "f32[768]" = _foreach_div_list[24]
        getitem_2909: "f32[768, 768]" = _foreach_div_list[25]
        getitem_2910: "f32[768]" = _foreach_div_list[26]
        getitem_2911: "f32[768, 768]" = _foreach_div_list[27]
        getitem_2912: "f32[768]" = _foreach_div_list[28]
        getitem_2913: "f32[768, 768]" = _foreach_div_list[29]
        getitem_2914: "f32[768]" = _foreach_div_list[30]
        getitem_2915: "f32[768, 768]" = _foreach_div_list[31]
        getitem_2916: "f32[768]" = _foreach_div_list[32]
        getitem_2917: "f32[768]" = _foreach_div_list[33]
        getitem_2918: "f32[768]" = _foreach_div_list[34]
        getitem_2919: "f32[3072, 768]" = _foreach_div_list[35]
        getitem_2920: "f32[3072]" = _foreach_div_list[36]
        getitem_2921: "f32[768, 3072]" = _foreach_div_list[37]
        getitem_2922: "f32[768]" = _foreach_div_list[38]
        getitem_2923: "f32[768]" = _foreach_div_list[39]
        getitem_2924: "f32[768]" = _foreach_div_list[40]
        getitem_2925: "f32[768, 768]" = _foreach_div_list[41]
        getitem_2926: "f32[768]" = _foreach_div_list[42]
        getitem_2927: "f32[768, 768]" = _foreach_div_list[43]
        getitem_2928: "f32[768]" = _foreach_div_list[44]
        getitem_2929: "f32[768, 768]" = _foreach_div_list[45]
        getitem_2930: "f32[768]" = _foreach_div_list[46]
        getitem_2931: "f32[768, 768]" = _foreach_div_list[47]
        getitem_2932: "f32[768]" = _foreach_div_list[48]
        getitem_2933: "f32[768]" = _foreach_div_list[49]
        getitem_2934: "f32[768]" = _foreach_div_list[50]
        getitem_2935: "f32[3072, 768]" = _foreach_div_list[51]
        getitem_2936: "f32[3072]" = _foreach_div_list[52]
        getitem_2937: "f32[768, 3072]" = _foreach_div_list[53]
        getitem_2938: "f32[768]" = _foreach_div_list[54]
        getitem_2939: "f32[768]" = _foreach_div_list[55]
        getitem_2940: "f32[768]" = _foreach_div_list[56]
        getitem_2941: "f32[768, 768]" = _foreach_div_list[57]
        getitem_2942: "f32[768]" = _foreach_div_list[58]
        getitem_2943: "f32[768, 768]" = _foreach_div_list[59]
        getitem_2944: "f32[768]" = _foreach_div_list[60]
        getitem_2945: "f32[768, 768]" = _foreach_div_list[61]
        getitem_2946: "f32[768]" = _foreach_div_list[62]
        getitem_2947: "f32[768, 768]" = _foreach_div_list[63]
        getitem_2948: "f32[768]" = _foreach_div_list[64]
        getitem_2949: "f32[768]" = _foreach_div_list[65]
        getitem_2950: "f32[768]" = _foreach_div_list[66]
        getitem_2951: "f32[3072, 768]" = _foreach_div_list[67]
        getitem_2952: "f32[3072]" = _foreach_div_list[68]
        getitem_2953: "f32[768, 3072]" = _foreach_div_list[69]
        getitem_2954: "f32[768]" = _foreach_div_list[70]
        getitem_2955: "f32[768]" = _foreach_div_list[71]
        getitem_2956: "f32[768]" = _foreach_div_list[72]
        getitem_2957: "f32[768, 768]" = _foreach_div_list[73]
        getitem_2958: "f32[768]" = _foreach_div_list[74]
        getitem_2959: "f32[768, 768]" = _foreach_div_list[75]
        getitem_2960: "f32[768]" = _foreach_div_list[76]
        getitem_2961: "f32[768, 768]" = _foreach_div_list[77]
        getitem_2962: "f32[768]" = _foreach_div_list[78]
        getitem_2963: "f32[768, 768]" = _foreach_div_list[79]
        getitem_2964: "f32[768]" = _foreach_div_list[80]
        getitem_2965: "f32[768]" = _foreach_div_list[81]
        getitem_2966: "f32[768]" = _foreach_div_list[82]
        getitem_2967: "f32[3072, 768]" = _foreach_div_list[83]
        getitem_2968: "f32[3072]" = _foreach_div_list[84]
        getitem_2969: "f32[768, 3072]" = _foreach_div_list[85]
        getitem_2970: "f32[768]" = _foreach_div_list[86]
        getitem_2971: "f32[768]" = _foreach_div_list[87]
        getitem_2972: "f32[768]" = _foreach_div_list[88]
        getitem_2973: "f32[768, 768]" = _foreach_div_list[89]
        getitem_2974: "f32[768]" = _foreach_div_list[90]
        getitem_2975: "f32[768, 768]" = _foreach_div_list[91]
        getitem_2976: "f32[768]" = _foreach_div_list[92]
        getitem_2977: "f32[768, 768]" = _foreach_div_list[93]
        getitem_2978: "f32[768]" = _foreach_div_list[94]
        getitem_2979: "f32[768, 768]" = _foreach_div_list[95]
        getitem_2980: "f32[768]" = _foreach_div_list[96]
        getitem_2981: "f32[768]" = _foreach_div_list[97]
        getitem_2982: "f32[768]" = _foreach_div_list[98]
        getitem_2983: "f32[3072, 768]" = _foreach_div_list[99]
        getitem_2984: "f32[3072]" = _foreach_div_list[100]
        getitem_2985: "f32[768, 3072]" = _foreach_div_list[101]
        getitem_2986: "f32[768]" = _foreach_div_list[102]
        getitem_2987: "f32[768]" = _foreach_div_list[103]
        getitem_2988: "f32[768]" = _foreach_div_list[104]
        getitem_2989: "f32[768, 768]" = _foreach_div_list[105]
        getitem_2990: "f32[768]" = _foreach_div_list[106]
        getitem_2991: "f32[768, 768]" = _foreach_div_list[107]
        getitem_2992: "f32[768]" = _foreach_div_list[108]
        getitem_2993: "f32[768, 768]" = _foreach_div_list[109]
        getitem_2994: "f32[768]" = _foreach_div_list[110]
        getitem_2995: "f32[768, 768]" = _foreach_div_list[111]
        getitem_2996: "f32[768]" = _foreach_div_list[112]
        getitem_2997: "f32[768]" = _foreach_div_list[113]
        getitem_2998: "f32[768]" = _foreach_div_list[114]
        getitem_2999: "f32[3072, 768]" = _foreach_div_list[115]
        getitem_3000: "f32[3072]" = _foreach_div_list[116]
        getitem_3001: "f32[768, 3072]" = _foreach_div_list[117]
        getitem_3002: "f32[768]" = _foreach_div_list[118]
        getitem_3003: "f32[768]" = _foreach_div_list[119]
        getitem_3004: "f32[768]" = _foreach_div_list[120]
        getitem_3005: "f32[768, 768]" = _foreach_div_list[121]
        getitem_3006: "f32[768]" = _foreach_div_list[122]
        getitem_3007: "f32[768, 768]" = _foreach_div_list[123]
        getitem_3008: "f32[768]" = _foreach_div_list[124]
        getitem_3009: "f32[768, 768]" = _foreach_div_list[125]
        getitem_3010: "f32[768]" = _foreach_div_list[126]
        getitem_3011: "f32[768, 768]" = _foreach_div_list[127]
        getitem_3012: "f32[768]" = _foreach_div_list[128]
        getitem_3013: "f32[768]" = _foreach_div_list[129]
        getitem_3014: "f32[768]" = _foreach_div_list[130]
        getitem_3015: "f32[3072, 768]" = _foreach_div_list[131]
        getitem_3016: "f32[3072]" = _foreach_div_list[132]
        getitem_3017: "f32[768, 3072]" = _foreach_div_list[133]
        getitem_3018: "f32[768]" = _foreach_div_list[134]
        getitem_3019: "f32[768]" = _foreach_div_list[135]
        getitem_3020: "f32[768]" = _foreach_div_list[136]
        getitem_3021: "f32[768, 768]" = _foreach_div_list[137]
        getitem_3022: "f32[768]" = _foreach_div_list[138]
        getitem_3023: "f32[768, 768]" = _foreach_div_list[139]
        getitem_3024: "f32[768]" = _foreach_div_list[140]
        getitem_3025: "f32[768, 768]" = _foreach_div_list[141]
        getitem_3026: "f32[768]" = _foreach_div_list[142]
        getitem_3027: "f32[768, 768]" = _foreach_div_list[143]
        getitem_3028: "f32[768]" = _foreach_div_list[144]
        getitem_3029: "f32[768]" = _foreach_div_list[145]
        getitem_3030: "f32[768]" = _foreach_div_list[146]
        getitem_3031: "f32[3072, 768]" = _foreach_div_list[147]
        getitem_3032: "f32[3072]" = _foreach_div_list[148]
        getitem_3033: "f32[768, 3072]" = _foreach_div_list[149]
        getitem_3034: "f32[768]" = _foreach_div_list[150]
        getitem_3035: "f32[768]" = _foreach_div_list[151]
        getitem_3036: "f32[768]" = _foreach_div_list[152]
        getitem_3037: "f32[768, 768]" = _foreach_div_list[153]
        getitem_3038: "f32[768]" = _foreach_div_list[154]
        getitem_3039: "f32[768, 768]" = _foreach_div_list[155]
        getitem_3040: "f32[768]" = _foreach_div_list[156]
        getitem_3041: "f32[768, 768]" = _foreach_div_list[157]
        getitem_3042: "f32[768]" = _foreach_div_list[158]
        getitem_3043: "f32[768, 768]" = _foreach_div_list[159]
        getitem_3044: "f32[768]" = _foreach_div_list[160]
        getitem_3045: "f32[768]" = _foreach_div_list[161]
        getitem_3046: "f32[768]" = _foreach_div_list[162]
        getitem_3047: "f32[3072, 768]" = _foreach_div_list[163]
        getitem_3048: "f32[3072]" = _foreach_div_list[164]
        getitem_3049: "f32[768, 3072]" = _foreach_div_list[165]
        getitem_3050: "f32[768]" = _foreach_div_list[166]
        getitem_3051: "f32[768]" = _foreach_div_list[167]
        getitem_3052: "f32[768]" = _foreach_div_list[168]
        getitem_3053: "f32[768, 768]" = _foreach_div_list[169]
        getitem_3054: "f32[768]" = _foreach_div_list[170]
        getitem_3055: "f32[768, 768]" = _foreach_div_list[171]
        getitem_3056: "f32[768]" = _foreach_div_list[172]
        getitem_3057: "f32[768, 768]" = _foreach_div_list[173]
        getitem_3058: "f32[768]" = _foreach_div_list[174]
        getitem_3059: "f32[768, 768]" = _foreach_div_list[175]
        getitem_3060: "f32[768]" = _foreach_div_list[176]
        getitem_3061: "f32[768]" = _foreach_div_list[177]
        getitem_3062: "f32[768]" = _foreach_div_list[178]
        getitem_3063: "f32[3072, 768]" = _foreach_div_list[179]
        getitem_3064: "f32[3072]" = _foreach_div_list[180]
        getitem_3065: "f32[768, 3072]" = _foreach_div_list[181]
        getitem_3066: "f32[768]" = _foreach_div_list[182]
        getitem_3067: "f32[768]" = _foreach_div_list[183]
        getitem_3068: "f32[768]" = _foreach_div_list[184]
        getitem_3069: "f32[768, 768]" = _foreach_div_list[185]
        getitem_3070: "f32[768]" = _foreach_div_list[186]
        getitem_3071: "f32[768, 768]" = _foreach_div_list[187]
        getitem_3072: "f32[768]" = _foreach_div_list[188]
        getitem_3073: "f32[768, 768]" = _foreach_div_list[189]
        getitem_3074: "f32[768]" = _foreach_div_list[190]
        getitem_3075: "f32[768, 768]" = _foreach_div_list[191]
        getitem_3076: "f32[768]" = _foreach_div_list[192]
        getitem_3077: "f32[768]" = _foreach_div_list[193]
        getitem_3078: "f32[768]" = _foreach_div_list[194]
        getitem_3079: "f32[3072, 768]" = _foreach_div_list[195]
        getitem_3080: "f32[3072]" = _foreach_div_list[196]
        getitem_3081: "f32[768, 3072]" = _foreach_div_list[197]
        getitem_3082: "f32[768]" = _foreach_div_list[198]
        getitem_3083: "f32[768]" = _foreach_div_list[199]
        getitem_3084: "f32[768]" = _foreach_div_list[200]
        getitem_3085: "f32[30522]" = _foreach_div_list[201]
        getitem_3086: "f32[768, 768]" = _foreach_div_list[202]
        getitem_3087: "f32[768]" = _foreach_div_list[203]
        getitem_3088: "f32[768]" = _foreach_div_list[204]
        getitem_3089: "f32[768]" = _foreach_div_list[205];  _foreach_div_list = None
        return (getitem, getitem_1648, getitem_1649, getitem_1650, getitem_1651, getitem_1652, getitem_1653, getitem_1654, getitem_1655, getitem_1656, getitem_1657, getitem_1658, getitem_1659, getitem_1660, getitem_1661, getitem_1662, getitem_1663, getitem_1664, getitem_1665, getitem_1666, getitem_1667, getitem_1668, getitem_1669, getitem_1670, getitem_1671, getitem_1672, getitem_1673, getitem_1674, getitem_1675, getitem_1676, getitem_1677, getitem_1678, getitem_1679, getitem_1680, getitem_1681, getitem_1682, getitem_1683, getitem_1684, getitem_1685, getitem_1686, getitem_1687, getitem_1688, getitem_1689, getitem_1690, getitem_1691, getitem_1692, getitem_1693, getitem_1694, getitem_1695, getitem_1696, getitem_1697, getitem_1698, getitem_1699, getitem_1700, getitem_1701, getitem_1702, getitem_1703, getitem_1704, getitem_1705, getitem_1706, getitem_1707, getitem_1708, getitem_1709, getitem_1710, getitem_1711, getitem_1712, getitem_1713, getitem_1714, getitem_1715, getitem_1716, getitem_1717, getitem_1718, getitem_1719, getitem_1720, getitem_1721, getitem_1722, getitem_1723, getitem_1724, getitem_1725, getitem_1726, getitem_1727, getitem_1728, getitem_1729, getitem_1730, getitem_1731, getitem_1732, getitem_1733, getitem_1734, getitem_1735, getitem_1736, getitem_1737, getitem_1738, getitem_1739, getitem_1740, getitem_1741, getitem_1742, getitem_1743, getitem_1744, getitem_1745, getitem_1746, getitem_1747, getitem_1748, getitem_1749, getitem_1750, getitem_1751, getitem_1752, getitem_1753, getitem_1754, getitem_1755, getitem_1756, getitem_1757, getitem_1758, getitem_1759, getitem_1760, getitem_1761, getitem_1762, getitem_1763, getitem_1764, getitem_1765, getitem_1766, getitem_1767, getitem_1768, getitem_1769, getitem_1770, getitem_1771, getitem_1772, getitem_1773, getitem_1774, getitem_1775, getitem_1776, getitem_1777, getitem_1778, getitem_1779, getitem_1780, getitem_1781, getitem_1782, getitem_1783, getitem_1784, getitem_1785, getitem_1786, getitem_1787, getitem_1788, getitem_1789, getitem_1790, getitem_1791, getitem_1792, getitem_1793, getitem_1794, getitem_1795, getitem_1796, getitem_1797, getitem_1798, getitem_1799, getitem_1800, getitem_1801, getitem_1802, getitem_1803, getitem_1804, getitem_1805, getitem_1806, getitem_1807, getitem_1808, getitem_1809, getitem_1810, getitem_1811, getitem_1812, getitem_1813, getitem_1814, getitem_1815, getitem_1816, getitem_1817, getitem_1818, getitem_1819, getitem_1820, getitem_1821, getitem_1822, getitem_1823, getitem_1824, getitem_1825, getitem_1826, getitem_1827, getitem_1828, getitem_1829, getitem_1830, getitem_1831, getitem_1832, getitem_1833, getitem_1834, getitem_1835, getitem_1836, getitem_1837, getitem_1838, getitem_1839, getitem_1840, getitem_1841, getitem_1842, getitem_1843, getitem_1844, getitem_1845, getitem_1846, getitem_1847, getitem_1848, getitem_1849, getitem_1850, getitem_1851, getitem_1852, getitem_2884, getitem_2885, getitem_2886, getitem_2887, getitem_2888, getitem_2889, getitem_2890, getitem_2891, getitem_2892, getitem_2893, getitem_2894, getitem_2895, getitem_2896, getitem_2897, getitem_2898, getitem_2899, getitem_2900, getitem_2901, getitem_2902, getitem_2903, getitem_2904, getitem_2905, getitem_2906, getitem_2907, getitem_2908, getitem_2909, getitem_2910, getitem_2911, getitem_2912, getitem_2913, getitem_2914, getitem_2915, getitem_2916, getitem_2917, getitem_2918, getitem_2919, getitem_2920, getitem_2921, getitem_2922, getitem_2923, getitem_2924, getitem_2925, getitem_2926, getitem_2927, getitem_2928, getitem_2929, getitem_2930, getitem_2931, getitem_2932, getitem_2933, getitem_2934, getitem_2935, getitem_2936, getitem_2937, getitem_2938, getitem_2939, getitem_2940, getitem_2941, getitem_2942, getitem_2943, getitem_2944, getitem_2945, getitem_2946, getitem_2947, getitem_2948, getitem_2949, getitem_2950, getitem_2951, getitem_2952, getitem_2953, getitem_2954, getitem_2955, getitem_2956, getitem_2957, getitem_2958, getitem_2959, getitem_2960, getitem_2961, getitem_2962, getitem_2963, getitem_2964, getitem_2965, getitem_2966, getitem_2967, getitem_2968, getitem_2969, getitem_2970, getitem_2971, getitem_2972, getitem_2973, getitem_2974, getitem_2975, getitem_2976, getitem_2977, getitem_2978, getitem_2979, getitem_2980, getitem_2981, getitem_2982, getitem_2983, getitem_2984, getitem_2985, getitem_2986, getitem_2987, getitem_2988, getitem_2989, getitem_2990, getitem_2991, getitem_2992, getitem_2993, getitem_2994, getitem_2995, getitem_2996, getitem_2997, getitem_2998, getitem_2999, getitem_3000, getitem_3001, getitem_3002, getitem_3003, getitem_3004, getitem_3005, getitem_3006, getitem_3007, getitem_3008, getitem_3009, getitem_3010, getitem_3011, getitem_3012, getitem_3013, getitem_3014, getitem_3015, getitem_3016, getitem_3017, getitem_3018, getitem_3019, getitem_3020, getitem_3021, getitem_3022, getitem_3023, getitem_3024, getitem_3025, getitem_3026, getitem_3027, getitem_3028, getitem_3029, getitem_3030, getitem_3031, getitem_3032, getitem_3033, getitem_3034, getitem_3035, getitem_3036, getitem_3037, getitem_3038, getitem_3039, getitem_3040, getitem_3041, getitem_3042, getitem_3043, getitem_3044, getitem_3045, getitem_3046, getitem_3047, getitem_3048, getitem_3049, getitem_3050, getitem_3051, getitem_3052, getitem_3053, getitem_3054, getitem_3055, getitem_3056, getitem_3057, getitem_3058, getitem_3059, getitem_3060, getitem_3061, getitem_3062, getitem_3063, getitem_3064, getitem_3065, getitem_3066, getitem_3067, getitem_3068, getitem_3069, getitem_3070, getitem_3071, getitem_3072, getitem_3073, getitem_3074, getitem_3075, getitem_3076, getitem_3077, getitem_3078, getitem_3079, getitem_3080, getitem_3081, getitem_3082, getitem_3083, getitem_3084, getitem_3085, getitem_3086, getitem_3087, getitem_3088, getitem_3089)


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
