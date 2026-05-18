"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s4_g59
Pattern hash: d9e89bfc82f1
Shape hash: cc382c50
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
    def forward(self, arg0_1: "f32[32128, 512]", arg1_1: "f32[512, 512]", arg2_1: "f32[512, 512]", arg3_1: "f32[512, 512]", arg4_1: "f32[512, 512]", arg5_1: "f32[32, 8]", arg6_1: "f32[512]", arg7_1: "f32[2048, 512]", arg8_1: "f32[512, 2048]", arg9_1: "f32[512]", arg10_1: "f32[512, 512]", arg11_1: "f32[512, 512]", arg12_1: "f32[512, 512]", arg13_1: "f32[512, 512]", arg14_1: "f32[512]", arg15_1: "f32[2048, 512]", arg16_1: "f32[512, 2048]", arg17_1: "f32[512]", arg18_1: "f32[512, 512]", arg19_1: "f32[512, 512]", arg20_1: "f32[512, 512]", arg21_1: "f32[512, 512]", arg22_1: "f32[512]", arg23_1: "f32[2048, 512]", arg24_1: "f32[512, 2048]", arg25_1: "f32[512]", arg26_1: "f32[512, 512]", arg27_1: "f32[512, 512]", arg28_1: "f32[512, 512]", arg29_1: "f32[512, 512]", arg30_1: "f32[512]", arg31_1: "f32[2048, 512]", arg32_1: "f32[512, 2048]", arg33_1: "f32[512]", arg34_1: "f32[512, 512]", arg35_1: "f32[512, 512]", arg36_1: "f32[512, 512]", arg37_1: "f32[512, 512]", arg38_1: "f32[512]", arg39_1: "f32[2048, 512]", arg40_1: "f32[512, 2048]", arg41_1: "f32[512]", arg42_1: "f32[512, 512]", arg43_1: "f32[512, 512]", arg44_1: "f32[512, 512]", arg45_1: "f32[512, 512]", arg46_1: "f32[512]", arg47_1: "f32[2048, 512]", arg48_1: "f32[512, 2048]", arg49_1: "f32[512]", arg50_1: "f32[512]", arg51_1: "f32[512, 512]", arg52_1: "f32[512, 512]", arg53_1: "f32[512, 512]", arg54_1: "f32[512, 512]", arg55_1: "f32[32, 8]", arg56_1: "f32[512]", arg57_1: "f32[512, 512]", arg58_1: "f32[512, 512]", arg59_1: "f32[512, 512]", arg60_1: "f32[512, 512]", arg61_1: "f32[512]", arg62_1: "f32[2048, 512]", arg63_1: "f32[512, 2048]", arg64_1: "f32[512]", arg65_1: "f32[512, 512]", arg66_1: "f32[512, 512]", arg67_1: "f32[512, 512]", arg68_1: "f32[512, 512]", arg69_1: "f32[512]", arg70_1: "f32[512, 512]", arg71_1: "f32[512, 512]", arg72_1: "f32[512, 512]", arg73_1: "f32[512, 512]", arg74_1: "f32[512]", arg75_1: "f32[2048, 512]", arg76_1: "f32[512, 2048]", arg77_1: "f32[512]", arg78_1: "f32[512, 512]", arg79_1: "f32[512, 512]", arg80_1: "f32[512, 512]", arg81_1: "f32[512, 512]", arg82_1: "f32[512]", arg83_1: "f32[512, 512]", arg84_1: "f32[512, 512]", arg85_1: "f32[512, 512]", arg86_1: "f32[512, 512]", arg87_1: "f32[512]", arg88_1: "f32[2048, 512]", arg89_1: "f32[512, 2048]", arg90_1: "f32[512]", arg91_1: "f32[512, 512]", arg92_1: "f32[512, 512]", arg93_1: "f32[512, 512]", arg94_1: "f32[512, 512]", arg95_1: "f32[512]", arg96_1: "f32[512, 512]", arg97_1: "f32[512, 512]", arg98_1: "f32[512, 512]", arg99_1: "f32[512, 512]", arg100_1: "f32[512]", arg101_1: "f32[2048, 512]", arg102_1: "f32[512, 2048]", arg103_1: "f32[512]", arg104_1: "f32[512, 512]", arg105_1: "f32[512, 512]", arg106_1: "f32[512, 512]", arg107_1: "f32[512, 512]", arg108_1: "f32[512]", arg109_1: "f32[512, 512]", arg110_1: "f32[512, 512]", arg111_1: "f32[512, 512]", arg112_1: "f32[512, 512]", arg113_1: "f32[512]", arg114_1: "f32[2048, 512]", arg115_1: "f32[512, 2048]", arg116_1: "f32[512]", arg117_1: "f32[512, 512]", arg118_1: "f32[512, 512]", arg119_1: "f32[512, 512]", arg120_1: "f32[512, 512]", arg121_1: "f32[512]", arg122_1: "f32[512, 512]", arg123_1: "f32[512, 512]", arg124_1: "f32[512, 512]", arg125_1: "f32[512, 512]", arg126_1: "f32[512]", arg127_1: "f32[2048, 512]", arg128_1: "f32[512, 2048]", arg129_1: "f32[512]", arg130_1: "f32[512]", getitem_262: "f32[32128, 512]", getitem_263: "f32[512, 512]", getitem_264: "f32[512, 512]", getitem_265: "f32[512, 512]", getitem_266: "f32[512, 512]", getitem_267: "f32[32, 8]", getitem_268: "f32[512]", getitem_269: "f32[2048, 512]", getitem_270: "f32[512, 2048]", getitem_271: "f32[512]", getitem_272: "f32[512, 512]", getitem_273: "f32[512, 512]", getitem_274: "f32[512, 512]", getitem_275: "f32[512, 512]", getitem_276: "f32[512]", getitem_277: "f32[2048, 512]", getitem_278: "f32[512, 2048]", getitem_279: "f32[512]", getitem_280: "f32[512, 512]", getitem_281: "f32[512, 512]", getitem_282: "f32[512, 512]", getitem_283: "f32[512, 512]", getitem_284: "f32[512]", getitem_285: "f32[2048, 512]", getitem_286: "f32[512, 2048]", getitem_287: "f32[512]", getitem_288: "f32[512, 512]", getitem_289: "f32[512, 512]", getitem_290: "f32[512, 512]", getitem_291: "f32[512, 512]", getitem_292: "f32[512]", getitem_293: "f32[2048, 512]", getitem_294: "f32[512, 2048]", getitem_295: "f32[512]", getitem_296: "f32[512, 512]", getitem_297: "f32[512, 512]", getitem_298: "f32[512, 512]", getitem_299: "f32[512, 512]", getitem_300: "f32[512]", getitem_301: "f32[2048, 512]", getitem_302: "f32[512, 2048]", getitem_303: "f32[512]", getitem_304: "f32[512, 512]", getitem_305: "f32[512, 512]", getitem_306: "f32[512, 512]", getitem_307: "f32[512, 512]", getitem_308: "f32[512]", getitem_309: "f32[2048, 512]", getitem_310: "f32[512, 2048]", getitem_311: "f32[512]", getitem_312: "f32[512]", getitem_313: "f32[512, 512]", getitem_314: "f32[512, 512]", getitem_315: "f32[512, 512]", getitem_316: "f32[512, 512]", getitem_317: "f32[32, 8]", getitem_318: "f32[512]", getitem_319: "f32[512, 512]", getitem_320: "f32[512, 512]", getitem_321: "f32[512, 512]", getitem_322: "f32[512, 512]", getitem_323: "f32[512]", getitem_324: "f32[2048, 512]", getitem_325: "f32[512, 2048]", getitem_326: "f32[512]", getitem_327: "f32[512, 512]", getitem_328: "f32[512, 512]", getitem_329: "f32[512, 512]", getitem_330: "f32[512, 512]", getitem_331: "f32[512]", getitem_332: "f32[512, 512]", getitem_333: "f32[512, 512]", getitem_334: "f32[512, 512]", getitem_335: "f32[512, 512]", getitem_336: "f32[512]", getitem_337: "f32[2048, 512]", getitem_338: "f32[512, 2048]", getitem_339: "f32[512]", getitem_340: "f32[512, 512]", getitem_341: "f32[512, 512]", getitem_342: "f32[512, 512]", getitem_343: "f32[512, 512]", getitem_344: "f32[512]", getitem_345: "f32[512, 512]", getitem_346: "f32[512, 512]", getitem_347: "f32[512, 512]", getitem_348: "f32[512, 512]", getitem_349: "f32[512]", getitem_350: "f32[2048, 512]", getitem_351: "f32[512, 2048]", getitem_352: "f32[512]", getitem_353: "f32[512, 512]", getitem_354: "f32[512, 512]", getitem_355: "f32[512, 512]", getitem_356: "f32[512, 512]", getitem_357: "f32[512]", getitem_358: "f32[512, 512]", getitem_359: "f32[512, 512]", getitem_360: "f32[512, 512]", getitem_361: "f32[512, 512]", getitem_362: "f32[512]", getitem_363: "f32[2048, 512]", getitem_364: "f32[512, 2048]", getitem_365: "f32[512]", getitem_366: "f32[512, 512]", getitem_367: "f32[512, 512]", getitem_368: "f32[512, 512]", getitem_369: "f32[512, 512]", getitem_370: "f32[512]", getitem_371: "f32[512, 512]", getitem_372: "f32[512, 512]", getitem_373: "f32[512, 512]", getitem_374: "f32[512, 512]", getitem_375: "f32[512]", getitem_376: "f32[2048, 512]", getitem_377: "f32[512, 2048]", getitem_378: "f32[512]", getitem_379: "f32[512, 512]", getitem_380: "f32[512, 512]", getitem_381: "f32[512, 512]", getitem_382: "f32[512, 512]", getitem_383: "f32[512]", getitem_384: "f32[512, 512]", getitem_385: "f32[512, 512]", getitem_386: "f32[512, 512]", getitem_387: "f32[512, 512]", getitem_388: "f32[512]", getitem_389: "f32[2048, 512]", getitem_390: "f32[512, 2048]", getitem_391: "f32[512]", getitem_392: "f32[512]", getitem_2096: "f32[32128, 512]", getitem_2097: "f32[512, 512]", getitem_2098: "f32[512, 512]", getitem_2099: "f32[512, 512]", getitem_2100: "f32[512, 512]", getitem_2101: "f32[32, 8]", getitem_2102: "f32[512]", getitem_2103: "f32[2048, 512]", getitem_2104: "f32[512, 2048]", getitem_2105: "f32[512]", getitem_2106: "f32[512, 512]", getitem_2107: "f32[512, 512]", getitem_2108: "f32[512, 512]", getitem_2109: "f32[512, 512]", getitem_2110: "f32[512]", getitem_2111: "f32[2048, 512]", getitem_2112: "f32[512, 2048]", getitem_2113: "f32[512]", getitem_2114: "f32[512, 512]", getitem_2115: "f32[512, 512]", getitem_2116: "f32[512, 512]", getitem_2117: "f32[512, 512]", getitem_2118: "f32[512]", getitem_2119: "f32[2048, 512]", getitem_2120: "f32[512, 2048]", getitem_2121: "f32[512]", getitem_2122: "f32[512, 512]", getitem_2123: "f32[512, 512]", getitem_2124: "f32[512, 512]", getitem_2125: "f32[512, 512]", getitem_2126: "f32[512]", getitem_2127: "f32[2048, 512]", getitem_2128: "f32[512, 2048]", getitem_2129: "f32[512]", getitem_2130: "f32[512, 512]", getitem_2131: "f32[512, 512]", getitem_2132: "f32[512, 512]", getitem_2133: "f32[512, 512]", getitem_2134: "f32[512]", getitem_2135: "f32[2048, 512]", getitem_2136: "f32[512, 2048]", getitem_2137: "f32[512]", getitem_2138: "f32[512, 512]", getitem_2139: "f32[512, 512]", getitem_2140: "f32[512, 512]", getitem_2141: "f32[512, 512]", getitem_2142: "f32[512]", getitem_2143: "f32[2048, 512]", getitem_2144: "f32[512, 2048]", getitem_2145: "f32[512]", getitem_2146: "f32[512]", getitem_2147: "f32[512, 512]", getitem_2148: "f32[512, 512]", getitem_2149: "f32[512, 512]", getitem_2150: "f32[512, 512]", getitem_2151: "f32[32, 8]", getitem_2152: "f32[512]", getitem_2153: "f32[512, 512]", getitem_2154: "f32[512, 512]", getitem_2155: "f32[512, 512]", getitem_2156: "f32[512, 512]", getitem_2157: "f32[512]", getitem_2158: "f32[2048, 512]", getitem_2159: "f32[512, 2048]", getitem_2160: "f32[512]", getitem_2161: "f32[512, 512]", getitem_2162: "f32[512, 512]", getitem_2163: "f32[512, 512]", getitem_2164: "f32[512, 512]", getitem_2165: "f32[512]", getitem_2166: "f32[512, 512]", getitem_2167: "f32[512, 512]", getitem_2168: "f32[512, 512]", getitem_2169: "f32[512, 512]", getitem_2170: "f32[512]", getitem_2171: "f32[2048, 512]", getitem_2172: "f32[512, 2048]", getitem_2173: "f32[512]", getitem_2174: "f32[512, 512]", getitem_2175: "f32[512, 512]", getitem_2176: "f32[512, 512]", getitem_2177: "f32[512, 512]", getitem_2178: "f32[512]", getitem_2179: "f32[512, 512]", getitem_2180: "f32[512, 512]", getitem_2181: "f32[512, 512]", getitem_2182: "f32[512, 512]", getitem_2183: "f32[512]", getitem_2184: "f32[2048, 512]", getitem_2185: "f32[512, 2048]", getitem_2186: "f32[512]", getitem_2187: "f32[512, 512]", getitem_2188: "f32[512, 512]", getitem_2189: "f32[512, 512]", getitem_2190: "f32[512, 512]", getitem_2191: "f32[512]", getitem_2192: "f32[512, 512]", getitem_2193: "f32[512, 512]", getitem_2194: "f32[512, 512]", getitem_2195: "f32[512, 512]", getitem_2196: "f32[512]", getitem_2197: "f32[2048, 512]", getitem_2198: "f32[512, 2048]", getitem_2199: "f32[512]", getitem_2200: "f32[512, 512]", getitem_2201: "f32[512, 512]", getitem_2202: "f32[512, 512]", getitem_2203: "f32[512, 512]", getitem_2204: "f32[512]", getitem_2205: "f32[512, 512]", getitem_2206: "f32[512, 512]", getitem_2207: "f32[512, 512]", getitem_2208: "f32[512, 512]", getitem_2209: "f32[512]", getitem_2210: "f32[2048, 512]", getitem_2211: "f32[512, 2048]", getitem_2212: "f32[512]", getitem_2213: "f32[512, 512]", getitem_2214: "f32[512, 512]", getitem_2215: "f32[512, 512]", getitem_2216: "f32[512, 512]", getitem_2217: "f32[512]", getitem_2218: "f32[512, 512]", getitem_2219: "f32[512, 512]", getitem_2220: "f32[512, 512]", getitem_2221: "f32[512, 512]", getitem_2222: "f32[512]", getitem_2223: "f32[2048, 512]", getitem_2224: "f32[512, 2048]", getitem_2225: "f32[512]", getitem_2226: "f32[512]"):
        # No stacktrace found for following nodes
        _foreach_addcdiv_scalar = torch.ops.aten._foreach_addcdiv.Scalar([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1], [getitem_262, getitem_263, getitem_264, getitem_265, getitem_266, getitem_267, getitem_268, getitem_269, getitem_270, getitem_271, getitem_272, getitem_273, getitem_274, getitem_275, getitem_276, getitem_277, getitem_278, getitem_279, getitem_280, getitem_281, getitem_282, getitem_283, getitem_284, getitem_285, getitem_286, getitem_287, getitem_288, getitem_289, getitem_290, getitem_291, getitem_292, getitem_293, getitem_294, getitem_295, getitem_296, getitem_297, getitem_298, getitem_299, getitem_300, getitem_301, getitem_302, getitem_303, getitem_304, getitem_305, getitem_306, getitem_307, getitem_308, getitem_309, getitem_310, getitem_311, getitem_312, getitem_313, getitem_314, getitem_315, getitem_316, getitem_317, getitem_318, getitem_319, getitem_320, getitem_321, getitem_322, getitem_323, getitem_324, getitem_325, getitem_326, getitem_327, getitem_328, getitem_329, getitem_330, getitem_331, getitem_332, getitem_333, getitem_334, getitem_335, getitem_336, getitem_337, getitem_338, getitem_339, getitem_340, getitem_341, getitem_342, getitem_343, getitem_344, getitem_345, getitem_346, getitem_347, getitem_348, getitem_349, getitem_350, getitem_351, getitem_352, getitem_353, getitem_354, getitem_355, getitem_356, getitem_357, getitem_358, getitem_359, getitem_360, getitem_361, getitem_362, getitem_363, getitem_364, getitem_365, getitem_366, getitem_367, getitem_368, getitem_369, getitem_370, getitem_371, getitem_372, getitem_373, getitem_374, getitem_375, getitem_376, getitem_377, getitem_378, getitem_379, getitem_380, getitem_381, getitem_382, getitem_383, getitem_384, getitem_385, getitem_386, getitem_387, getitem_388, getitem_389, getitem_390, getitem_391, getitem_392], [getitem_2096, getitem_2097, getitem_2098, getitem_2099, getitem_2100, getitem_2101, getitem_2102, getitem_2103, getitem_2104, getitem_2105, getitem_2106, getitem_2107, getitem_2108, getitem_2109, getitem_2110, getitem_2111, getitem_2112, getitem_2113, getitem_2114, getitem_2115, getitem_2116, getitem_2117, getitem_2118, getitem_2119, getitem_2120, getitem_2121, getitem_2122, getitem_2123, getitem_2124, getitem_2125, getitem_2126, getitem_2127, getitem_2128, getitem_2129, getitem_2130, getitem_2131, getitem_2132, getitem_2133, getitem_2134, getitem_2135, getitem_2136, getitem_2137, getitem_2138, getitem_2139, getitem_2140, getitem_2141, getitem_2142, getitem_2143, getitem_2144, getitem_2145, getitem_2146, getitem_2147, getitem_2148, getitem_2149, getitem_2150, getitem_2151, getitem_2152, getitem_2153, getitem_2154, getitem_2155, getitem_2156, getitem_2157, getitem_2158, getitem_2159, getitem_2160, getitem_2161, getitem_2162, getitem_2163, getitem_2164, getitem_2165, getitem_2166, getitem_2167, getitem_2168, getitem_2169, getitem_2170, getitem_2171, getitem_2172, getitem_2173, getitem_2174, getitem_2175, getitem_2176, getitem_2177, getitem_2178, getitem_2179, getitem_2180, getitem_2181, getitem_2182, getitem_2183, getitem_2184, getitem_2185, getitem_2186, getitem_2187, getitem_2188, getitem_2189, getitem_2190, getitem_2191, getitem_2192, getitem_2193, getitem_2194, getitem_2195, getitem_2196, getitem_2197, getitem_2198, getitem_2199, getitem_2200, getitem_2201, getitem_2202, getitem_2203, getitem_2204, getitem_2205, getitem_2206, getitem_2207, getitem_2208, getitem_2209, getitem_2210, getitem_2211, getitem_2212, getitem_2213, getitem_2214, getitem_2215, getitem_2216, getitem_2217, getitem_2218, getitem_2219, getitem_2220, getitem_2221, getitem_2222, getitem_2223, getitem_2224, getitem_2225, getitem_2226]);  arg0_1 = arg1_1 = arg2_1 = arg3_1 = arg4_1 = arg5_1 = arg6_1 = arg7_1 = arg8_1 = arg9_1 = arg10_1 = arg11_1 = arg12_1 = arg13_1 = arg14_1 = arg15_1 = arg16_1 = arg17_1 = arg18_1 = arg19_1 = arg20_1 = arg21_1 = arg22_1 = arg23_1 = arg24_1 = arg25_1 = arg26_1 = arg27_1 = arg28_1 = arg29_1 = arg30_1 = arg31_1 = arg32_1 = arg33_1 = arg34_1 = arg35_1 = arg36_1 = arg37_1 = arg38_1 = arg39_1 = arg40_1 = arg41_1 = arg42_1 = arg43_1 = arg44_1 = arg45_1 = arg46_1 = arg47_1 = arg48_1 = arg49_1 = arg50_1 = arg51_1 = arg52_1 = arg53_1 = arg54_1 = arg55_1 = arg56_1 = arg57_1 = arg58_1 = arg59_1 = arg60_1 = arg61_1 = arg62_1 = arg63_1 = arg64_1 = arg65_1 = arg66_1 = arg67_1 = arg68_1 = arg69_1 = arg70_1 = arg71_1 = arg72_1 = arg73_1 = arg74_1 = arg75_1 = arg76_1 = arg77_1 = arg78_1 = arg79_1 = arg80_1 = arg81_1 = arg82_1 = arg83_1 = arg84_1 = arg85_1 = arg86_1 = arg87_1 = arg88_1 = arg89_1 = arg90_1 = arg91_1 = arg92_1 = arg93_1 = arg94_1 = arg95_1 = arg96_1 = arg97_1 = arg98_1 = arg99_1 = arg100_1 = arg101_1 = arg102_1 = arg103_1 = arg104_1 = arg105_1 = arg106_1 = arg107_1 = arg108_1 = arg109_1 = arg110_1 = arg111_1 = arg112_1 = arg113_1 = arg114_1 = arg115_1 = arg116_1 = arg117_1 = arg118_1 = arg119_1 = arg120_1 = arg121_1 = arg122_1 = arg123_1 = arg124_1 = arg125_1 = arg126_1 = arg127_1 = arg128_1 = arg129_1 = arg130_1 = getitem_262 = getitem_263 = getitem_264 = getitem_265 = getitem_266 = getitem_267 = getitem_268 = getitem_269 = getitem_270 = getitem_271 = getitem_272 = getitem_273 = getitem_274 = getitem_275 = getitem_276 = getitem_277 = getitem_278 = getitem_279 = getitem_280 = getitem_281 = getitem_282 = getitem_283 = getitem_284 = getitem_285 = getitem_286 = getitem_287 = getitem_288 = getitem_289 = getitem_290 = getitem_291 = getitem_292 = getitem_293 = getitem_294 = getitem_295 = getitem_296 = getitem_297 = getitem_298 = getitem_299 = getitem_300 = getitem_301 = getitem_302 = getitem_303 = getitem_304 = getitem_305 = getitem_306 = getitem_307 = getitem_308 = getitem_309 = getitem_310 = getitem_311 = getitem_312 = getitem_313 = getitem_314 = getitem_315 = getitem_316 = getitem_317 = getitem_318 = getitem_319 = getitem_320 = getitem_321 = getitem_322 = getitem_323 = getitem_324 = getitem_325 = getitem_326 = getitem_327 = getitem_328 = getitem_329 = getitem_330 = getitem_331 = getitem_332 = getitem_333 = getitem_334 = getitem_335 = getitem_336 = getitem_337 = getitem_338 = getitem_339 = getitem_340 = getitem_341 = getitem_342 = getitem_343 = getitem_344 = getitem_345 = getitem_346 = getitem_347 = getitem_348 = getitem_349 = getitem_350 = getitem_351 = getitem_352 = getitem_353 = getitem_354 = getitem_355 = getitem_356 = getitem_357 = getitem_358 = getitem_359 = getitem_360 = getitem_361 = getitem_362 = getitem_363 = getitem_364 = getitem_365 = getitem_366 = getitem_367 = getitem_368 = getitem_369 = getitem_370 = getitem_371 = getitem_372 = getitem_373 = getitem_374 = getitem_375 = getitem_376 = getitem_377 = getitem_378 = getitem_379 = getitem_380 = getitem_381 = getitem_382 = getitem_383 = getitem_384 = getitem_385 = getitem_386 = getitem_387 = getitem_388 = getitem_389 = getitem_390 = getitem_391 = getitem_392 = getitem_2096 = getitem_2097 = getitem_2098 = getitem_2099 = getitem_2100 = getitem_2101 = getitem_2102 = getitem_2103 = getitem_2104 = getitem_2105 = getitem_2106 = getitem_2107 = getitem_2108 = getitem_2109 = getitem_2110 = getitem_2111 = getitem_2112 = getitem_2113 = getitem_2114 = getitem_2115 = getitem_2116 = getitem_2117 = getitem_2118 = getitem_2119 = getitem_2120 = getitem_2121 = getitem_2122 = getitem_2123 = getitem_2124 = getitem_2125 = getitem_2126 = getitem_2127 = getitem_2128 = getitem_2129 = getitem_2130 = getitem_2131 = getitem_2132 = getitem_2133 = getitem_2134 = getitem_2135 = getitem_2136 = getitem_2137 = getitem_2138 = getitem_2139 = getitem_2140 = getitem_2141 = getitem_2142 = getitem_2143 = getitem_2144 = getitem_2145 = getitem_2146 = getitem_2147 = getitem_2148 = getitem_2149 = getitem_2150 = getitem_2151 = getitem_2152 = getitem_2153 = getitem_2154 = getitem_2155 = getitem_2156 = getitem_2157 = getitem_2158 = getitem_2159 = getitem_2160 = getitem_2161 = getitem_2162 = getitem_2163 = getitem_2164 = getitem_2165 = getitem_2166 = getitem_2167 = getitem_2168 = getitem_2169 = getitem_2170 = getitem_2171 = getitem_2172 = getitem_2173 = getitem_2174 = getitem_2175 = getitem_2176 = getitem_2177 = getitem_2178 = getitem_2179 = getitem_2180 = getitem_2181 = getitem_2182 = getitem_2183 = getitem_2184 = getitem_2185 = getitem_2186 = getitem_2187 = getitem_2188 = getitem_2189 = getitem_2190 = getitem_2191 = getitem_2192 = getitem_2193 = getitem_2194 = getitem_2195 = getitem_2196 = getitem_2197 = getitem_2198 = getitem_2199 = getitem_2200 = getitem_2201 = getitem_2202 = getitem_2203 = getitem_2204 = getitem_2205 = getitem_2206 = getitem_2207 = getitem_2208 = getitem_2209 = getitem_2210 = getitem_2211 = getitem_2212 = getitem_2213 = getitem_2214 = getitem_2215 = getitem_2216 = getitem_2217 = getitem_2218 = getitem_2219 = getitem_2220 = getitem_2221 = getitem_2222 = getitem_2223 = getitem_2224 = getitem_2225 = getitem_2226 = None
        getitem: "f32[32128, 512]" = _foreach_addcdiv_scalar[0]
        getitem_2227: "f32[512, 512]" = _foreach_addcdiv_scalar[1]
        getitem_2228: "f32[512, 512]" = _foreach_addcdiv_scalar[2]
        getitem_2229: "f32[512, 512]" = _foreach_addcdiv_scalar[3]
        getitem_2230: "f32[512, 512]" = _foreach_addcdiv_scalar[4]
        getitem_2231: "f32[32, 8]" = _foreach_addcdiv_scalar[5]
        getitem_2232: "f32[512]" = _foreach_addcdiv_scalar[6]
        getitem_2233: "f32[2048, 512]" = _foreach_addcdiv_scalar[7]
        getitem_2234: "f32[512, 2048]" = _foreach_addcdiv_scalar[8]
        getitem_2235: "f32[512]" = _foreach_addcdiv_scalar[9]
        getitem_2236: "f32[512, 512]" = _foreach_addcdiv_scalar[10]
        getitem_2237: "f32[512, 512]" = _foreach_addcdiv_scalar[11]
        getitem_2238: "f32[512, 512]" = _foreach_addcdiv_scalar[12]
        getitem_2239: "f32[512, 512]" = _foreach_addcdiv_scalar[13]
        getitem_2240: "f32[512]" = _foreach_addcdiv_scalar[14]
        getitem_2241: "f32[2048, 512]" = _foreach_addcdiv_scalar[15]
        getitem_2242: "f32[512, 2048]" = _foreach_addcdiv_scalar[16]
        getitem_2243: "f32[512]" = _foreach_addcdiv_scalar[17]
        getitem_2244: "f32[512, 512]" = _foreach_addcdiv_scalar[18]
        getitem_2245: "f32[512, 512]" = _foreach_addcdiv_scalar[19]
        getitem_2246: "f32[512, 512]" = _foreach_addcdiv_scalar[20]
        getitem_2247: "f32[512, 512]" = _foreach_addcdiv_scalar[21]
        getitem_2248: "f32[512]" = _foreach_addcdiv_scalar[22]
        getitem_2249: "f32[2048, 512]" = _foreach_addcdiv_scalar[23]
        getitem_2250: "f32[512, 2048]" = _foreach_addcdiv_scalar[24]
        getitem_2251: "f32[512]" = _foreach_addcdiv_scalar[25]
        getitem_2252: "f32[512, 512]" = _foreach_addcdiv_scalar[26]
        getitem_2253: "f32[512, 512]" = _foreach_addcdiv_scalar[27]
        getitem_2254: "f32[512, 512]" = _foreach_addcdiv_scalar[28]
        getitem_2255: "f32[512, 512]" = _foreach_addcdiv_scalar[29]
        getitem_2256: "f32[512]" = _foreach_addcdiv_scalar[30]
        getitem_2257: "f32[2048, 512]" = _foreach_addcdiv_scalar[31]
        getitem_2258: "f32[512, 2048]" = _foreach_addcdiv_scalar[32]
        getitem_2259: "f32[512]" = _foreach_addcdiv_scalar[33]
        getitem_2260: "f32[512, 512]" = _foreach_addcdiv_scalar[34]
        getitem_2261: "f32[512, 512]" = _foreach_addcdiv_scalar[35]
        getitem_2262: "f32[512, 512]" = _foreach_addcdiv_scalar[36]
        getitem_2263: "f32[512, 512]" = _foreach_addcdiv_scalar[37]
        getitem_2264: "f32[512]" = _foreach_addcdiv_scalar[38]
        getitem_2265: "f32[2048, 512]" = _foreach_addcdiv_scalar[39]
        getitem_2266: "f32[512, 2048]" = _foreach_addcdiv_scalar[40]
        getitem_2267: "f32[512]" = _foreach_addcdiv_scalar[41]
        getitem_2268: "f32[512, 512]" = _foreach_addcdiv_scalar[42]
        getitem_2269: "f32[512, 512]" = _foreach_addcdiv_scalar[43]
        getitem_2270: "f32[512, 512]" = _foreach_addcdiv_scalar[44]
        getitem_2271: "f32[512, 512]" = _foreach_addcdiv_scalar[45]
        getitem_2272: "f32[512]" = _foreach_addcdiv_scalar[46]
        getitem_2273: "f32[2048, 512]" = _foreach_addcdiv_scalar[47]
        getitem_2274: "f32[512, 2048]" = _foreach_addcdiv_scalar[48]
        getitem_2275: "f32[512]" = _foreach_addcdiv_scalar[49]
        getitem_2276: "f32[512]" = _foreach_addcdiv_scalar[50]
        getitem_2277: "f32[512, 512]" = _foreach_addcdiv_scalar[51]
        getitem_2278: "f32[512, 512]" = _foreach_addcdiv_scalar[52]
        getitem_2279: "f32[512, 512]" = _foreach_addcdiv_scalar[53]
        getitem_2280: "f32[512, 512]" = _foreach_addcdiv_scalar[54]
        getitem_2281: "f32[32, 8]" = _foreach_addcdiv_scalar[55]
        getitem_2282: "f32[512]" = _foreach_addcdiv_scalar[56]
        getitem_2283: "f32[512, 512]" = _foreach_addcdiv_scalar[57]
        getitem_2284: "f32[512, 512]" = _foreach_addcdiv_scalar[58]
        getitem_2285: "f32[512, 512]" = _foreach_addcdiv_scalar[59]
        getitem_2286: "f32[512, 512]" = _foreach_addcdiv_scalar[60]
        getitem_2287: "f32[512]" = _foreach_addcdiv_scalar[61]
        getitem_2288: "f32[2048, 512]" = _foreach_addcdiv_scalar[62]
        getitem_2289: "f32[512, 2048]" = _foreach_addcdiv_scalar[63]
        getitem_2290: "f32[512]" = _foreach_addcdiv_scalar[64]
        getitem_2291: "f32[512, 512]" = _foreach_addcdiv_scalar[65]
        getitem_2292: "f32[512, 512]" = _foreach_addcdiv_scalar[66]
        getitem_2293: "f32[512, 512]" = _foreach_addcdiv_scalar[67]
        getitem_2294: "f32[512, 512]" = _foreach_addcdiv_scalar[68]
        getitem_2295: "f32[512]" = _foreach_addcdiv_scalar[69]
        getitem_2296: "f32[512, 512]" = _foreach_addcdiv_scalar[70]
        getitem_2297: "f32[512, 512]" = _foreach_addcdiv_scalar[71]
        getitem_2298: "f32[512, 512]" = _foreach_addcdiv_scalar[72]
        getitem_2299: "f32[512, 512]" = _foreach_addcdiv_scalar[73]
        getitem_2300: "f32[512]" = _foreach_addcdiv_scalar[74]
        getitem_2301: "f32[2048, 512]" = _foreach_addcdiv_scalar[75]
        getitem_2302: "f32[512, 2048]" = _foreach_addcdiv_scalar[76]
        getitem_2303: "f32[512]" = _foreach_addcdiv_scalar[77]
        getitem_2304: "f32[512, 512]" = _foreach_addcdiv_scalar[78]
        getitem_2305: "f32[512, 512]" = _foreach_addcdiv_scalar[79]
        getitem_2306: "f32[512, 512]" = _foreach_addcdiv_scalar[80]
        getitem_2307: "f32[512, 512]" = _foreach_addcdiv_scalar[81]
        getitem_2308: "f32[512]" = _foreach_addcdiv_scalar[82]
        getitem_2309: "f32[512, 512]" = _foreach_addcdiv_scalar[83]
        getitem_2310: "f32[512, 512]" = _foreach_addcdiv_scalar[84]
        getitem_2311: "f32[512, 512]" = _foreach_addcdiv_scalar[85]
        getitem_2312: "f32[512, 512]" = _foreach_addcdiv_scalar[86]
        getitem_2313: "f32[512]" = _foreach_addcdiv_scalar[87]
        getitem_2314: "f32[2048, 512]" = _foreach_addcdiv_scalar[88]
        getitem_2315: "f32[512, 2048]" = _foreach_addcdiv_scalar[89]
        getitem_2316: "f32[512]" = _foreach_addcdiv_scalar[90]
        getitem_2317: "f32[512, 512]" = _foreach_addcdiv_scalar[91]
        getitem_2318: "f32[512, 512]" = _foreach_addcdiv_scalar[92]
        getitem_2319: "f32[512, 512]" = _foreach_addcdiv_scalar[93]
        getitem_2320: "f32[512, 512]" = _foreach_addcdiv_scalar[94]
        getitem_2321: "f32[512]" = _foreach_addcdiv_scalar[95]
        getitem_2322: "f32[512, 512]" = _foreach_addcdiv_scalar[96]
        getitem_2323: "f32[512, 512]" = _foreach_addcdiv_scalar[97]
        getitem_2324: "f32[512, 512]" = _foreach_addcdiv_scalar[98]
        getitem_2325: "f32[512, 512]" = _foreach_addcdiv_scalar[99]
        getitem_2326: "f32[512]" = _foreach_addcdiv_scalar[100]
        getitem_2327: "f32[2048, 512]" = _foreach_addcdiv_scalar[101]
        getitem_2328: "f32[512, 2048]" = _foreach_addcdiv_scalar[102]
        getitem_2329: "f32[512]" = _foreach_addcdiv_scalar[103]
        getitem_2330: "f32[512, 512]" = _foreach_addcdiv_scalar[104]
        getitem_2331: "f32[512, 512]" = _foreach_addcdiv_scalar[105]
        getitem_2332: "f32[512, 512]" = _foreach_addcdiv_scalar[106]
        getitem_2333: "f32[512, 512]" = _foreach_addcdiv_scalar[107]
        getitem_2334: "f32[512]" = _foreach_addcdiv_scalar[108]
        getitem_2335: "f32[512, 512]" = _foreach_addcdiv_scalar[109]
        getitem_2336: "f32[512, 512]" = _foreach_addcdiv_scalar[110]
        getitem_2337: "f32[512, 512]" = _foreach_addcdiv_scalar[111]
        getitem_2338: "f32[512, 512]" = _foreach_addcdiv_scalar[112]
        getitem_2339: "f32[512]" = _foreach_addcdiv_scalar[113]
        getitem_2340: "f32[2048, 512]" = _foreach_addcdiv_scalar[114]
        getitem_2341: "f32[512, 2048]" = _foreach_addcdiv_scalar[115]
        getitem_2342: "f32[512]" = _foreach_addcdiv_scalar[116]
        getitem_2343: "f32[512, 512]" = _foreach_addcdiv_scalar[117]
        getitem_2344: "f32[512, 512]" = _foreach_addcdiv_scalar[118]
        getitem_2345: "f32[512, 512]" = _foreach_addcdiv_scalar[119]
        getitem_2346: "f32[512, 512]" = _foreach_addcdiv_scalar[120]
        getitem_2347: "f32[512]" = _foreach_addcdiv_scalar[121]
        getitem_2348: "f32[512, 512]" = _foreach_addcdiv_scalar[122]
        getitem_2349: "f32[512, 512]" = _foreach_addcdiv_scalar[123]
        getitem_2350: "f32[512, 512]" = _foreach_addcdiv_scalar[124]
        getitem_2351: "f32[512, 512]" = _foreach_addcdiv_scalar[125]
        getitem_2352: "f32[512]" = _foreach_addcdiv_scalar[126]
        getitem_2353: "f32[2048, 512]" = _foreach_addcdiv_scalar[127]
        getitem_2354: "f32[512, 2048]" = _foreach_addcdiv_scalar[128]
        getitem_2355: "f32[512]" = _foreach_addcdiv_scalar[129]
        getitem_2356: "f32[512]" = _foreach_addcdiv_scalar[130];  _foreach_addcdiv_scalar = None
        return (getitem, getitem_2227, getitem_2228, getitem_2229, getitem_2230, getitem_2231, getitem_2232, getitem_2233, getitem_2234, getitem_2235, getitem_2236, getitem_2237, getitem_2238, getitem_2239, getitem_2240, getitem_2241, getitem_2242, getitem_2243, getitem_2244, getitem_2245, getitem_2246, getitem_2247, getitem_2248, getitem_2249, getitem_2250, getitem_2251, getitem_2252, getitem_2253, getitem_2254, getitem_2255, getitem_2256, getitem_2257, getitem_2258, getitem_2259, getitem_2260, getitem_2261, getitem_2262, getitem_2263, getitem_2264, getitem_2265, getitem_2266, getitem_2267, getitem_2268, getitem_2269, getitem_2270, getitem_2271, getitem_2272, getitem_2273, getitem_2274, getitem_2275, getitem_2276, getitem_2277, getitem_2278, getitem_2279, getitem_2280, getitem_2281, getitem_2282, getitem_2283, getitem_2284, getitem_2285, getitem_2286, getitem_2287, getitem_2288, getitem_2289, getitem_2290, getitem_2291, getitem_2292, getitem_2293, getitem_2294, getitem_2295, getitem_2296, getitem_2297, getitem_2298, getitem_2299, getitem_2300, getitem_2301, getitem_2302, getitem_2303, getitem_2304, getitem_2305, getitem_2306, getitem_2307, getitem_2308, getitem_2309, getitem_2310, getitem_2311, getitem_2312, getitem_2313, getitem_2314, getitem_2315, getitem_2316, getitem_2317, getitem_2318, getitem_2319, getitem_2320, getitem_2321, getitem_2322, getitem_2323, getitem_2324, getitem_2325, getitem_2326, getitem_2327, getitem_2328, getitem_2329, getitem_2330, getitem_2331, getitem_2332, getitem_2333, getitem_2334, getitem_2335, getitem_2336, getitem_2337, getitem_2338, getitem_2339, getitem_2340, getitem_2341, getitem_2342, getitem_2343, getitem_2344, getitem_2345, getitem_2346, getitem_2347, getitem_2348, getitem_2349, getitem_2350, getitem_2351, getitem_2352, getitem_2353, getitem_2354, getitem_2355, getitem_2356)


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
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
