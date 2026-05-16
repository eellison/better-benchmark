"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s9_g77
Pattern hash: 97e5561d8ec6
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
    def forward(self, getitem_2107: "f32[]", getitem_2108: "f32[]", getitem_2109: "f32[]", getitem_2110: "f32[]", getitem_2111: "f32[]", getitem_2112: "f32[]", getitem_2113: "f32[]", getitem_2114: "f32[]", getitem_2115: "f32[]", getitem_2116: "f32[]", getitem_2117: "f32[]", getitem_2118: "f32[]", getitem_2119: "f32[]", getitem_2120: "f32[]", getitem_2121: "f32[]", getitem_2122: "f32[]", getitem_2123: "f32[]", getitem_2124: "f32[]", getitem_2125: "f32[]", getitem_2126: "f32[]", getitem_2127: "f32[]", getitem_2128: "f32[]", getitem_2129: "f32[]", getitem_2130: "f32[]", getitem_2131: "f32[]", getitem_2132: "f32[]", getitem_2133: "f32[]", getitem_2134: "f32[]", getitem_2135: "f32[]", getitem_2136: "f32[]", getitem_2137: "f32[]", getitem_2138: "f32[]", getitem_2139: "f32[]", getitem_2140: "f32[]", getitem_2141: "f32[]", getitem_2142: "f32[]", getitem_2143: "f32[]", getitem_2144: "f32[]", getitem_2145: "f32[]", getitem_2146: "f32[]", getitem_2147: "f32[]", getitem_2148: "f32[]", getitem_2149: "f32[]", getitem_2150: "f32[]", getitem_2151: "f32[]", getitem_2152: "f32[]", getitem_2153: "f32[]", getitem_2154: "f32[]", getitem_2155: "f32[]", getitem_2156: "f32[]", getitem_2157: "f32[]", getitem_2158: "f32[]", getitem_2159: "f32[]", getitem_2160: "f32[]", getitem_2161: "f32[]", getitem_2162: "f32[]", getitem_2163: "f32[]", getitem_2164: "f32[]", getitem_2165: "f32[]", getitem_2166: "f32[]", getitem_2167: "f32[]", getitem_2168: "f32[]", getitem_2169: "f32[]", getitem_2170: "f32[]", getitem_2171: "f32[]", getitem_2172: "f32[]", getitem_2173: "f32[]", getitem_2174: "f32[]", getitem_2175: "f32[]", getitem_2176: "f32[]", getitem_2177: "f32[]", getitem_2178: "f32[]", getitem_2179: "f32[]", getitem_2180: "f32[]", getitem_2181: "f32[]", getitem_2182: "f32[]", getitem_2183: "f32[]", getitem_2184: "f32[]", getitem_2185: "f32[]", getitem_2186: "f32[]", getitem_2187: "f32[]", getitem_2188: "f32[]", getitem_2189: "f32[]", getitem_2190: "f32[]", getitem_2191: "f32[]", getitem_2192: "f32[]", getitem_2193: "f32[]", getitem_2194: "f32[]", getitem_2195: "f32[]", getitem_2196: "f32[]", getitem_2197: "f32[]", getitem_2198: "f32[]", getitem_2199: "f32[]", getitem_2200: "f32[]", getitem_2201: "f32[]", getitem_2202: "f32[]", getitem_2203: "f32[]", getitem_2204: "f32[]", getitem_2205: "f32[]", getitem_2206: "f32[]", getitem_2207: "f32[]", getitem_2208: "f32[]", getitem_2209: "f32[]", getitem_2210: "f32[]", getitem_2211: "f32[]", getitem_2212: "f32[]", getitem_2213: "f32[]", getitem_2214: "f32[]", getitem_2215: "f32[]", getitem_2216: "f32[]", getitem_2217: "f32[]", getitem_2218: "f32[]", getitem_2219: "f32[]", getitem_2220: "f32[]", getitem_2221: "f32[]", getitem_2222: "f32[]", getitem_2223: "f32[]", getitem_2224: "f32[]", getitem_2225: "f32[]", getitem_2226: "f32[]", getitem_2227: "f32[]", getitem_2228: "f32[]", getitem_2229: "f32[]", getitem_2230: "f32[]", getitem_2231: "f32[]", getitem_2232: "f32[]", getitem_2233: "f32[]", getitem_2234: "f32[]", getitem_2235: "f32[]", getitem_2236: "f32[]", getitem_2237: "f32[]", getitem_2238: "f32[]", getitem_2239: "f32[]", getitem_2240: "f32[]", getitem_2241: "f32[]", getitem_2242: "f32[]", getitem_2243: "f32[]", getitem_2244: "f32[]", getitem_2245: "f32[]", getitem_2246: "f32[]", getitem_2247: "f32[]", getitem_2248: "f32[]", getitem_2249: "f32[]", getitem_2250: "f32[]", getitem_2251: "f32[]", getitem_2252: "f32[]", getitem_2253: "f32[]", getitem_2254: "f32[]", getitem_2255: "f32[]", getitem_2256: "f32[]", getitem_2257: "f32[]", getitem_2258: "f32[]", getitem_2259: "f32[]", getitem_2260: "f32[]", getitem_2261: "f32[]", getitem_2262: "f32[]", getitem_2263: "f32[]", getitem_2264: "f32[]", getitem_2265: "f32[]", getitem_2266: "f32[]", getitem_2267: "f32[]", getitem_2268: "f32[]", getitem_2269: "f32[]", getitem_2270: "f32[]", getitem_2271: "f32[]", getitem_2272: "f32[]", getitem_2273: "f32[]", getitem_2274: "f32[]", getitem_2275: "f32[]", getitem_2276: "f32[]", getitem_2277: "f32[]", getitem_2278: "f32[]", getitem_2279: "f32[]", getitem_2280: "f32[]", getitem_2281: "f32[]", getitem_2282: "f32[]", getitem_2283: "f32[]", getitem_2284: "f32[]", getitem_2285: "f32[]", getitem_2286: "f32[]", getitem_2287: "f32[]", getitem_2288: "f32[]", getitem_2289: "f32[]", getitem_2290: "f32[]", getitem_2291: "f32[]", getitem_2292: "f32[]", getitem_2293: "f32[]", getitem_2294: "f32[]", getitem_2295: "f32[]", getitem_2296: "f32[]", getitem_2297: "f32[]", getitem_2298: "f32[]", getitem_2299: "f32[]", getitem_2300: "f32[]", getitem_2301: "f32[]", getitem_2302: "f32[]", getitem_2303: "f32[]", getitem_2304: "f32[]", getitem_2305: "f32[]", getitem_2306: "f32[]", getitem_2307: "f32[]", getitem_2308: "f32[]", getitem_2309: "f32[]", getitem_2310: "f32[]", getitem_2311: "f32[]", getitem_2312: "f32[]", getitem_2313: "f32[]", getitem_2314: "f32[]", getitem_2315: "f32[]", getitem_2316: "f32[]", getitem_2317: "f32[]", getitem_2318: "f32[]", getitem_2319: "f32[]", getitem_2320: "f32[]", getitem_2321: "f32[]", getitem_2322: "f32[]", getitem_2323: "f32[]", getitem_2324: "f32[]", getitem_2325: "f32[]", getitem_2326: "f32[]", getitem_2327: "f32[]", getitem_2328: "f32[]", getitem_2329: "f32[]", getitem_2330: "f32[]", getitem_2331: "f32[]", getitem_2332: "f32[]", getitem_2333: "f32[]", getitem_2334: "f32[]", getitem_2335: "f32[]", getitem_2336: "f32[]", getitem_2337: "f32[]", getitem_2338: "f32[]", getitem_2339: "f32[]", getitem_2340: "f32[]", getitem_2341: "f32[]", getitem_2342: "f32[]", getitem_2343: "f32[]", getitem_2344: "f32[]", getitem_2345: "f32[]", getitem_2346: "f32[]", getitem_2347: "f32[]", getitem_2348: "f32[]", getitem_2349: "f32[]", getitem_2350: "f32[]", getitem_2351: "f32[]", getitem_2352: "f32[]", getitem_2353: "f32[]", getitem_2354: "f32[]", getitem_2355: "f32[]", getitem_2356: "f32[]", getitem_2357: "f32[]", getitem_2358: "f32[]", getitem_2359: "f32[]", getitem_2360: "f32[]", getitem_2361: "f32[]", getitem_2362: "f32[]", getitem_2363: "f32[]", getitem_2364: "f32[]", getitem_2365: "f32[]", getitem_2366: "f32[]", getitem_2367: "f32[]", getitem_2368: "f32[]", getitem_2369: "f32[]", getitem_2370: "f32[]", getitem_2371: "f32[]", getitem_2372: "f32[]", getitem_2373: "f32[]", getitem_2374: "f32[]", getitem_2375: "f32[]", getitem_2376: "f32[]", getitem_2377: "f32[]", getitem_2378: "f32[]", getitem_2379: "f32[]", getitem_2380: "f32[]", getitem_2381: "f32[]", getitem_2382: "f32[]", getitem_2383: "f32[]", getitem_2384: "f32[]", getitem_2385: "f32[]", getitem_2386: "f32[]", getitem_2387: "f32[]", getitem_2388: "f32[]", getitem_2389: "f32[]", getitem_2390: "f32[]", getitem_2391: "f32[]", getitem_2392: "f32[]", getitem_2393: "f32[]", getitem_2394: "f32[]", getitem_2395: "f32[]", getitem_2396: "f32[]", getitem_2397: "f32[]", getitem_2398: "f32[]", getitem_2399: "f32[]", getitem_2400: "f32[]", getitem_2401: "f32[]", getitem_2402: "f32[]", getitem_2403: "f32[]", getitem_2404: "f32[]", getitem_2405: "f32[]", getitem_2406: "f32[]", getitem_2407: "f32[]", getitem_3913: "f32[768]", getitem_3914: "f32[50, 768]", getitem_3915: "f32[768, 512]", getitem_3916: "f32[768, 3, 32, 32]", getitem_3917: "f32[768]", getitem_3918: "f32[768]", getitem_3919: "f32[2304, 768]", getitem_3920: "f32[2304]", getitem_3921: "f32[768, 768]", getitem_3922: "f32[768]", getitem_3923: "f32[3072, 768]", getitem_3924: "f32[3072]", getitem_3925: "f32[768, 3072]", getitem_3926: "f32[768]", getitem_3927: "f32[768]", getitem_3928: "f32[768]", getitem_3929: "f32[768]", getitem_3930: "f32[768]", getitem_3931: "f32[2304, 768]", getitem_3932: "f32[2304]", getitem_3933: "f32[768, 768]", getitem_3934: "f32[768]", getitem_3935: "f32[3072, 768]", getitem_3936: "f32[3072]", getitem_3937: "f32[768, 3072]", getitem_3938: "f32[768]", getitem_3939: "f32[768]", getitem_3940: "f32[768]", getitem_3941: "f32[768]", getitem_3942: "f32[768]", getitem_3943: "f32[2304, 768]", getitem_3944: "f32[2304]", getitem_3945: "f32[768, 768]", getitem_3946: "f32[768]", getitem_3947: "f32[3072, 768]", getitem_3948: "f32[3072]", getitem_3949: "f32[768, 3072]", getitem_3950: "f32[768]", getitem_3951: "f32[768]", getitem_3952: "f32[768]", getitem_3953: "f32[768]", getitem_3954: "f32[768]", getitem_3955: "f32[2304, 768]", getitem_3956: "f32[2304]", getitem_3957: "f32[768, 768]", getitem_3958: "f32[768]", getitem_3959: "f32[3072, 768]", getitem_3960: "f32[3072]", getitem_3961: "f32[768, 3072]", getitem_3962: "f32[768]", getitem_3963: "f32[768]", getitem_3964: "f32[768]", getitem_3965: "f32[768]", getitem_3966: "f32[768]", getitem_3967: "f32[2304, 768]", getitem_3968: "f32[2304]", getitem_3969: "f32[768, 768]", getitem_3970: "f32[768]", getitem_3971: "f32[3072, 768]", getitem_3972: "f32[3072]", getitem_3973: "f32[768, 3072]", getitem_3974: "f32[768]", getitem_3975: "f32[768]", getitem_3976: "f32[768]", getitem_3977: "f32[768]", getitem_3978: "f32[768]", getitem_3979: "f32[2304, 768]", getitem_3980: "f32[2304]", getitem_3981: "f32[768, 768]", getitem_3982: "f32[768]", getitem_3983: "f32[3072, 768]", getitem_3984: "f32[3072]", getitem_3985: "f32[768, 3072]", getitem_3986: "f32[768]", getitem_3987: "f32[768]", getitem_3988: "f32[768]", getitem_3989: "f32[768]", getitem_3990: "f32[768]", getitem_3991: "f32[2304, 768]", getitem_3992: "f32[2304]", getitem_3993: "f32[768, 768]", getitem_3994: "f32[768]", getitem_3995: "f32[3072, 768]", getitem_3996: "f32[3072]", getitem_3997: "f32[768, 3072]", getitem_3998: "f32[768]", getitem_3999: "f32[768]", getitem_4000: "f32[768]", getitem_4001: "f32[768]", getitem_4002: "f32[768]", getitem_4003: "f32[2304, 768]", getitem_4004: "f32[2304]", getitem_4005: "f32[768, 768]", getitem_4006: "f32[768]", getitem_4007: "f32[3072, 768]", getitem_4008: "f32[3072]", getitem_4009: "f32[768, 3072]", getitem_4010: "f32[768]", getitem_4011: "f32[768]", getitem_4012: "f32[768]", getitem_4013: "f32[768]", getitem_4014: "f32[768]", getitem_4015: "f32[2304, 768]", getitem_4016: "f32[2304]", getitem_4017: "f32[768, 768]", getitem_4018: "f32[768]", getitem_4019: "f32[3072, 768]", getitem_4020: "f32[3072]", getitem_4021: "f32[768, 3072]", getitem_4022: "f32[768]", getitem_4023: "f32[768]", getitem_4024: "f32[768]", getitem_4025: "f32[768]", getitem_4026: "f32[768]", getitem_4027: "f32[2304, 768]", getitem_4028: "f32[2304]", getitem_4029: "f32[768, 768]", getitem_4030: "f32[768]", getitem_4031: "f32[3072, 768]", getitem_4032: "f32[3072]", getitem_4033: "f32[768, 3072]", getitem_4034: "f32[768]", getitem_4035: "f32[768]", getitem_4036: "f32[768]", getitem_4037: "f32[768]", getitem_4038: "f32[768]", getitem_4039: "f32[2304, 768]", getitem_4040: "f32[2304]", getitem_4041: "f32[768, 768]", getitem_4042: "f32[768]", getitem_4043: "f32[3072, 768]", getitem_4044: "f32[3072]", getitem_4045: "f32[768, 3072]", getitem_4046: "f32[768]", getitem_4047: "f32[768]", getitem_4048: "f32[768]", getitem_4049: "f32[768]", getitem_4050: "f32[768]", getitem_4051: "f32[2304, 768]", getitem_4052: "f32[2304]", getitem_4053: "f32[768, 768]", getitem_4054: "f32[768]", getitem_4055: "f32[3072, 768]", getitem_4056: "f32[3072]", getitem_4057: "f32[768, 3072]", getitem_4058: "f32[768]", getitem_4059: "f32[768]", getitem_4060: "f32[768]", getitem_4061: "f32[768]", getitem_4062: "f32[768]", getitem_4063: "f32[768]", getitem_4064: "f32[768]", getitem_4065: "f32[77, 512]", getitem_4066: "f32[49408, 512]", getitem_4067: "f32[1536, 512]", getitem_4068: "f32[1536]", getitem_4069: "f32[512, 512]", getitem_4070: "f32[512]", getitem_4071: "f32[2048, 512]", getitem_4072: "f32[2048]", getitem_4073: "f32[512, 2048]", getitem_4074: "f32[512]", getitem_4075: "f32[512]", getitem_4076: "f32[512]", getitem_4077: "f32[512]", getitem_4078: "f32[512]", getitem_4079: "f32[1536, 512]", getitem_4080: "f32[1536]", getitem_4081: "f32[512, 512]", getitem_4082: "f32[512]", getitem_4083: "f32[2048, 512]", getitem_4084: "f32[2048]", getitem_4085: "f32[512, 2048]", getitem_4086: "f32[512]", getitem_4087: "f32[512]", getitem_4088: "f32[512]", getitem_4089: "f32[512]", getitem_4090: "f32[512]", getitem_4091: "f32[1536, 512]", getitem_4092: "f32[1536]", getitem_4093: "f32[512, 512]", getitem_4094: "f32[512]", getitem_4095: "f32[2048, 512]", getitem_4096: "f32[2048]", getitem_4097: "f32[512, 2048]", getitem_4098: "f32[512]", getitem_4099: "f32[512]", getitem_4100: "f32[512]", getitem_4101: "f32[512]", getitem_4102: "f32[512]", getitem_4103: "f32[1536, 512]", getitem_4104: "f32[1536]", getitem_4105: "f32[512, 512]", getitem_4106: "f32[512]", getitem_4107: "f32[2048, 512]", getitem_4108: "f32[2048]", getitem_4109: "f32[512, 2048]", getitem_4110: "f32[512]", getitem_4111: "f32[512]", getitem_4112: "f32[512]", getitem_4113: "f32[512]", getitem_4114: "f32[512]", getitem_4115: "f32[1536, 512]", getitem_4116: "f32[1536]", getitem_4117: "f32[512, 512]", getitem_4118: "f32[512]", getitem_4119: "f32[2048, 512]", getitem_4120: "f32[2048]", getitem_4121: "f32[512, 2048]", getitem_4122: "f32[512]", getitem_4123: "f32[512]", getitem_4124: "f32[512]", getitem_4125: "f32[512]", getitem_4126: "f32[512]", getitem_4127: "f32[1536, 512]", getitem_4128: "f32[1536]", getitem_4129: "f32[512, 512]", getitem_4130: "f32[512]", getitem_4131: "f32[2048, 512]", getitem_4132: "f32[2048]", getitem_4133: "f32[512, 2048]", getitem_4134: "f32[512]", getitem_4135: "f32[512]", getitem_4136: "f32[512]", getitem_4137: "f32[512]", getitem_4138: "f32[512]", getitem_4139: "f32[1536, 512]", getitem_4140: "f32[1536]", getitem_4141: "f32[512, 512]", getitem_4142: "f32[512]", getitem_4143: "f32[2048, 512]", getitem_4144: "f32[2048]", getitem_4145: "f32[512, 2048]", getitem_4146: "f32[512]", getitem_4147: "f32[512]", getitem_4148: "f32[512]", getitem_4149: "f32[512]", getitem_4150: "f32[512]", getitem_4151: "f32[1536, 512]", getitem_4152: "f32[1536]", getitem_4153: "f32[512, 512]", getitem_4154: "f32[512]", getitem_4155: "f32[2048, 512]", getitem_4156: "f32[2048]", getitem_4157: "f32[512, 2048]", getitem_4158: "f32[512]", getitem_4159: "f32[512]", getitem_4160: "f32[512]", getitem_4161: "f32[512]", getitem_4162: "f32[512]", getitem_4163: "f32[1536, 512]", getitem_4164: "f32[1536]", getitem_4165: "f32[512, 512]", getitem_4166: "f32[512]", getitem_4167: "f32[2048, 512]", getitem_4168: "f32[2048]", getitem_4169: "f32[512, 2048]", getitem_4170: "f32[512]", getitem_4171: "f32[512]", getitem_4172: "f32[512]", getitem_4173: "f32[512]", getitem_4174: "f32[512]", getitem_4175: "f32[1536, 512]", getitem_4176: "f32[1536]", getitem_4177: "f32[512, 512]", getitem_4178: "f32[512]", getitem_4179: "f32[2048, 512]", getitem_4180: "f32[2048]", getitem_4181: "f32[512, 2048]", getitem_4182: "f32[512]", getitem_4183: "f32[512]", getitem_4184: "f32[512]", getitem_4185: "f32[512]", getitem_4186: "f32[512]", getitem_4187: "f32[1536, 512]", getitem_4188: "f32[1536]", getitem_4189: "f32[512, 512]", getitem_4190: "f32[512]", getitem_4191: "f32[2048, 512]", getitem_4192: "f32[2048]", getitem_4193: "f32[512, 2048]", getitem_4194: "f32[512]", getitem_4195: "f32[512]", getitem_4196: "f32[512]", getitem_4197: "f32[512]", getitem_4198: "f32[512]", getitem_4199: "f32[1536, 512]", getitem_4200: "f32[1536]", getitem_4201: "f32[512, 512]", getitem_4202: "f32[512]", getitem_4203: "f32[2048, 512]", getitem_4204: "f32[2048]", getitem_4205: "f32[512, 2048]", getitem_4206: "f32[512]", getitem_4207: "f32[512]", getitem_4208: "f32[512]", getitem_4209: "f32[512]", getitem_4210: "f32[512]", getitem_4211: "f32[512]", getitem_4212: "f32[512]", getitem_4213: "f32[512, 512]", getitem_3612: "f32[]", getitem_3613: "f32[]", getitem_3614: "f32[]", getitem_3615: "f32[]", getitem_3616: "f32[]", getitem_3617: "f32[]", getitem_3618: "f32[]", getitem_3619: "f32[]", getitem_3620: "f32[]", getitem_3621: "f32[]", getitem_3622: "f32[]", getitem_3623: "f32[]", getitem_3624: "f32[]", getitem_3625: "f32[]", getitem_3626: "f32[]", getitem_3627: "f32[]", getitem_3628: "f32[]", getitem_3629: "f32[]", getitem_3630: "f32[]", getitem_3631: "f32[]", getitem_3632: "f32[]", getitem_3633: "f32[]", getitem_3634: "f32[]", getitem_3635: "f32[]", getitem_3636: "f32[]", getitem_3637: "f32[]", getitem_3638: "f32[]", getitem_3639: "f32[]", getitem_3640: "f32[]", getitem_3641: "f32[]", getitem_3642: "f32[]", getitem_3643: "f32[]", getitem_3644: "f32[]", getitem_3645: "f32[]", getitem_3646: "f32[]", getitem_3647: "f32[]", getitem_3648: "f32[]", getitem_3649: "f32[]", getitem_3650: "f32[]", getitem_3651: "f32[]", getitem_3652: "f32[]", getitem_3653: "f32[]", getitem_3654: "f32[]", getitem_3655: "f32[]", getitem_3656: "f32[]", getitem_3657: "f32[]", getitem_3658: "f32[]", getitem_3659: "f32[]", getitem_3660: "f32[]", getitem_3661: "f32[]", getitem_3662: "f32[]", getitem_3663: "f32[]", getitem_3664: "f32[]", getitem_3665: "f32[]", getitem_3666: "f32[]", getitem_3667: "f32[]", getitem_3668: "f32[]", getitem_3669: "f32[]", getitem_3670: "f32[]", getitem_3671: "f32[]", getitem_3672: "f32[]", getitem_3673: "f32[]", getitem_3674: "f32[]", getitem_3675: "f32[]", getitem_3676: "f32[]", getitem_3677: "f32[]", getitem_3678: "f32[]", getitem_3679: "f32[]", getitem_3680: "f32[]", getitem_3681: "f32[]", getitem_3682: "f32[]", getitem_3683: "f32[]", getitem_3684: "f32[]", getitem_3685: "f32[]", getitem_3686: "f32[]", getitem_3687: "f32[]", getitem_3688: "f32[]", getitem_3689: "f32[]", getitem_3690: "f32[]", getitem_3691: "f32[]", getitem_3692: "f32[]", getitem_3693: "f32[]", getitem_3694: "f32[]", getitem_3695: "f32[]", getitem_3696: "f32[]", getitem_3697: "f32[]", getitem_3698: "f32[]", getitem_3699: "f32[]", getitem_3700: "f32[]", getitem_3701: "f32[]", getitem_3702: "f32[]", getitem_3703: "f32[]", getitem_3704: "f32[]", getitem_3705: "f32[]", getitem_3706: "f32[]", getitem_3707: "f32[]", getitem_3708: "f32[]", getitem_3709: "f32[]", getitem_3710: "f32[]", getitem_3711: "f32[]", getitem_3712: "f32[]", getitem_3713: "f32[]", getitem_3714: "f32[]", getitem_3715: "f32[]", getitem_3716: "f32[]", getitem_3717: "f32[]", getitem_3718: "f32[]", getitem_3719: "f32[]", getitem_3720: "f32[]", getitem_3721: "f32[]", getitem_3722: "f32[]", getitem_3723: "f32[]", getitem_3724: "f32[]", getitem_3725: "f32[]", getitem_3726: "f32[]", getitem_3727: "f32[]", getitem_3728: "f32[]", getitem_3729: "f32[]", getitem_3730: "f32[]", getitem_3731: "f32[]", getitem_3732: "f32[]", getitem_3733: "f32[]", getitem_3734: "f32[]", getitem_3735: "f32[]", getitem_3736: "f32[]", getitem_3737: "f32[]", getitem_3738: "f32[]", getitem_3739: "f32[]", getitem_3740: "f32[]", getitem_3741: "f32[]", getitem_3742: "f32[]", getitem_3743: "f32[]", getitem_3744: "f32[]", getitem_3745: "f32[]", getitem_3746: "f32[]", getitem_3747: "f32[]", getitem_3748: "f32[]", getitem_3749: "f32[]", getitem_3750: "f32[]", getitem_3751: "f32[]", getitem_3752: "f32[]", getitem_3753: "f32[]", getitem_3754: "f32[]", getitem_3755: "f32[]", getitem_3756: "f32[]", getitem_3757: "f32[]", getitem_3758: "f32[]", getitem_3759: "f32[]", getitem_3760: "f32[]", getitem_3761: "f32[]", getitem_3762: "f32[]", getitem_3763: "f32[]", getitem_3764: "f32[]", getitem_3765: "f32[]", getitem_3766: "f32[]", getitem_3767: "f32[]", getitem_3768: "f32[]", getitem_3769: "f32[]", getitem_3770: "f32[]", getitem_3771: "f32[]", getitem_3772: "f32[]", getitem_3773: "f32[]", getitem_3774: "f32[]", getitem_3775: "f32[]", getitem_3776: "f32[]", getitem_3777: "f32[]", getitem_3778: "f32[]", getitem_3779: "f32[]", getitem_3780: "f32[]", getitem_3781: "f32[]", getitem_3782: "f32[]", getitem_3783: "f32[]", getitem_3784: "f32[]", getitem_3785: "f32[]", getitem_3786: "f32[]", getitem_3787: "f32[]", getitem_3788: "f32[]", getitem_3789: "f32[]", getitem_3790: "f32[]", getitem_3791: "f32[]", getitem_3792: "f32[]", getitem_3793: "f32[]", getitem_3794: "f32[]", getitem_3795: "f32[]", getitem_3796: "f32[]", getitem_3797: "f32[]", getitem_3798: "f32[]", getitem_3799: "f32[]", getitem_3800: "f32[]", getitem_3801: "f32[]", getitem_3802: "f32[]", getitem_3803: "f32[]", getitem_3804: "f32[]", getitem_3805: "f32[]", getitem_3806: "f32[]", getitem_3807: "f32[]", getitem_3808: "f32[]", getitem_3809: "f32[]", getitem_3810: "f32[]", getitem_3811: "f32[]", getitem_3812: "f32[]", getitem_3813: "f32[]", getitem_3814: "f32[]", getitem_3815: "f32[]", getitem_3816: "f32[]", getitem_3817: "f32[]", getitem_3818: "f32[]", getitem_3819: "f32[]", getitem_3820: "f32[]", getitem_3821: "f32[]", getitem_3822: "f32[]", getitem_3823: "f32[]", getitem_3824: "f32[]", getitem_3825: "f32[]", getitem_3826: "f32[]", getitem_3827: "f32[]", getitem_3828: "f32[]", getitem_3829: "f32[]", getitem_3830: "f32[]", getitem_3831: "f32[]", getitem_3832: "f32[]", getitem_3833: "f32[]", getitem_3834: "f32[]", getitem_3835: "f32[]", getitem_3836: "f32[]", getitem_3837: "f32[]", getitem_3838: "f32[]", getitem_3839: "f32[]", getitem_3840: "f32[]", getitem_3841: "f32[]", getitem_3842: "f32[]", getitem_3843: "f32[]", getitem_3844: "f32[]", getitem_3845: "f32[]", getitem_3846: "f32[]", getitem_3847: "f32[]", getitem_3848: "f32[]", getitem_3849: "f32[]", getitem_3850: "f32[]", getitem_3851: "f32[]", getitem_3852: "f32[]", getitem_3853: "f32[]", getitem_3854: "f32[]", getitem_3855: "f32[]", getitem_3856: "f32[]", getitem_3857: "f32[]", getitem_3858: "f32[]", getitem_3859: "f32[]", getitem_3860: "f32[]", getitem_3861: "f32[]", getitem_3862: "f32[]", getitem_3863: "f32[]", getitem_3864: "f32[]", getitem_3865: "f32[]", getitem_3866: "f32[]", getitem_3867: "f32[]", getitem_3868: "f32[]", getitem_3869: "f32[]", getitem_3870: "f32[]", getitem_3871: "f32[]", getitem_3872: "f32[]", getitem_3873: "f32[]", getitem_3874: "f32[]", getitem_3875: "f32[]", getitem_3876: "f32[]", getitem_3877: "f32[]", getitem_3878: "f32[]", getitem_3879: "f32[]", getitem_3880: "f32[]", getitem_3881: "f32[]", getitem_3882: "f32[]", getitem_3883: "f32[]", getitem_3884: "f32[]", getitem_3885: "f32[]", getitem_3886: "f32[]", getitem_3887: "f32[]", getitem_3888: "f32[]", getitem_3889: "f32[]", getitem_3890: "f32[]", getitem_3891: "f32[]", getitem_3892: "f32[]", getitem_3893: "f32[]", getitem_3894: "f32[]", getitem_3895: "f32[]", getitem_3896: "f32[]", getitem_3897: "f32[]", getitem_3898: "f32[]", getitem_3899: "f32[]", getitem_3900: "f32[]", getitem_3901: "f32[]", getitem_3902: "f32[]", getitem_3903: "f32[]", getitem_3904: "f32[]", getitem_3905: "f32[]", getitem_3906: "f32[]", getitem_3907: "f32[]", getitem_3908: "f32[]", getitem_3909: "f32[]", getitem_3910: "f32[]", getitem_3911: "f32[]", getitem_3912: "f32[]"):
        # No stacktrace found for following nodes
        _foreach_div_scalar = torch.ops.aten._foreach_div.Scalar([getitem_2107, getitem_2108, getitem_2109, getitem_2110, getitem_2111, getitem_2112, getitem_2113, getitem_2114, getitem_2115, getitem_2116, getitem_2117, getitem_2118, getitem_2119, getitem_2120, getitem_2121, getitem_2122, getitem_2123, getitem_2124, getitem_2125, getitem_2126, getitem_2127, getitem_2128, getitem_2129, getitem_2130, getitem_2131, getitem_2132, getitem_2133, getitem_2134, getitem_2135, getitem_2136, getitem_2137, getitem_2138, getitem_2139, getitem_2140, getitem_2141, getitem_2142, getitem_2143, getitem_2144, getitem_2145, getitem_2146, getitem_2147, getitem_2148, getitem_2149, getitem_2150, getitem_2151, getitem_2152, getitem_2153, getitem_2154, getitem_2155, getitem_2156, getitem_2157, getitem_2158, getitem_2159, getitem_2160, getitem_2161, getitem_2162, getitem_2163, getitem_2164, getitem_2165, getitem_2166, getitem_2167, getitem_2168, getitem_2169, getitem_2170, getitem_2171, getitem_2172, getitem_2173, getitem_2174, getitem_2175, getitem_2176, getitem_2177, getitem_2178, getitem_2179, getitem_2180, getitem_2181, getitem_2182, getitem_2183, getitem_2184, getitem_2185, getitem_2186, getitem_2187, getitem_2188, getitem_2189, getitem_2190, getitem_2191, getitem_2192, getitem_2193, getitem_2194, getitem_2195, getitem_2196, getitem_2197, getitem_2198, getitem_2199, getitem_2200, getitem_2201, getitem_2202, getitem_2203, getitem_2204, getitem_2205, getitem_2206, getitem_2207, getitem_2208, getitem_2209, getitem_2210, getitem_2211, getitem_2212, getitem_2213, getitem_2214, getitem_2215, getitem_2216, getitem_2217, getitem_2218, getitem_2219, getitem_2220, getitem_2221, getitem_2222, getitem_2223, getitem_2224, getitem_2225, getitem_2226, getitem_2227, getitem_2228, getitem_2229, getitem_2230, getitem_2231, getitem_2232, getitem_2233, getitem_2234, getitem_2235, getitem_2236, getitem_2237, getitem_2238, getitem_2239, getitem_2240, getitem_2241, getitem_2242, getitem_2243, getitem_2244, getitem_2245, getitem_2246, getitem_2247, getitem_2248, getitem_2249, getitem_2250, getitem_2251, getitem_2252, getitem_2253, getitem_2254, getitem_2255, getitem_2256, getitem_2257, getitem_2258, getitem_2259, getitem_2260, getitem_2261, getitem_2262, getitem_2263, getitem_2264, getitem_2265, getitem_2266, getitem_2267, getitem_2268, getitem_2269, getitem_2270, getitem_2271, getitem_2272, getitem_2273, getitem_2274, getitem_2275, getitem_2276, getitem_2277, getitem_2278, getitem_2279, getitem_2280, getitem_2281, getitem_2282, getitem_2283, getitem_2284, getitem_2285, getitem_2286, getitem_2287, getitem_2288, getitem_2289, getitem_2290, getitem_2291, getitem_2292, getitem_2293, getitem_2294, getitem_2295, getitem_2296, getitem_2297, getitem_2298, getitem_2299, getitem_2300, getitem_2301, getitem_2302, getitem_2303, getitem_2304, getitem_2305, getitem_2306, getitem_2307, getitem_2308, getitem_2309, getitem_2310, getitem_2311, getitem_2312, getitem_2313, getitem_2314, getitem_2315, getitem_2316, getitem_2317, getitem_2318, getitem_2319, getitem_2320, getitem_2321, getitem_2322, getitem_2323, getitem_2324, getitem_2325, getitem_2326, getitem_2327, getitem_2328, getitem_2329, getitem_2330, getitem_2331, getitem_2332, getitem_2333, getitem_2334, getitem_2335, getitem_2336, getitem_2337, getitem_2338, getitem_2339, getitem_2340, getitem_2341, getitem_2342, getitem_2343, getitem_2344, getitem_2345, getitem_2346, getitem_2347, getitem_2348, getitem_2349, getitem_2350, getitem_2351, getitem_2352, getitem_2353, getitem_2354, getitem_2355, getitem_2356, getitem_2357, getitem_2358, getitem_2359, getitem_2360, getitem_2361, getitem_2362, getitem_2363, getitem_2364, getitem_2365, getitem_2366, getitem_2367, getitem_2368, getitem_2369, getitem_2370, getitem_2371, getitem_2372, getitem_2373, getitem_2374, getitem_2375, getitem_2376, getitem_2377, getitem_2378, getitem_2379, getitem_2380, getitem_2381, getitem_2382, getitem_2383, getitem_2384, getitem_2385, getitem_2386, getitem_2387, getitem_2388, getitem_2389, getitem_2390, getitem_2391, getitem_2392, getitem_2393, getitem_2394, getitem_2395, getitem_2396, getitem_2397, getitem_2398, getitem_2399, getitem_2400, getitem_2401, getitem_2402, getitem_2403, getitem_2404, getitem_2405, getitem_2406, getitem_2407], 0.01);  getitem_2107 = getitem_2108 = getitem_2109 = getitem_2110 = getitem_2111 = getitem_2112 = getitem_2113 = getitem_2114 = getitem_2115 = getitem_2116 = getitem_2117 = getitem_2118 = getitem_2119 = getitem_2120 = getitem_2121 = getitem_2122 = getitem_2123 = getitem_2124 = getitem_2125 = getitem_2126 = getitem_2127 = getitem_2128 = getitem_2129 = getitem_2130 = getitem_2131 = getitem_2132 = getitem_2133 = getitem_2134 = getitem_2135 = getitem_2136 = getitem_2137 = getitem_2138 = getitem_2139 = getitem_2140 = getitem_2141 = getitem_2142 = getitem_2143 = getitem_2144 = getitem_2145 = getitem_2146 = getitem_2147 = getitem_2148 = getitem_2149 = getitem_2150 = getitem_2151 = getitem_2152 = getitem_2153 = getitem_2154 = getitem_2155 = getitem_2156 = getitem_2157 = getitem_2158 = getitem_2159 = getitem_2160 = getitem_2161 = getitem_2162 = getitem_2163 = getitem_2164 = getitem_2165 = getitem_2166 = getitem_2167 = getitem_2168 = getitem_2169 = getitem_2170 = getitem_2171 = getitem_2172 = getitem_2173 = getitem_2174 = getitem_2175 = getitem_2176 = getitem_2177 = getitem_2178 = getitem_2179 = getitem_2180 = getitem_2181 = getitem_2182 = getitem_2183 = getitem_2184 = getitem_2185 = getitem_2186 = getitem_2187 = getitem_2188 = getitem_2189 = getitem_2190 = getitem_2191 = getitem_2192 = getitem_2193 = getitem_2194 = getitem_2195 = getitem_2196 = getitem_2197 = getitem_2198 = getitem_2199 = getitem_2200 = getitem_2201 = getitem_2202 = getitem_2203 = getitem_2204 = getitem_2205 = getitem_2206 = getitem_2207 = getitem_2208 = getitem_2209 = getitem_2210 = getitem_2211 = getitem_2212 = getitem_2213 = getitem_2214 = getitem_2215 = getitem_2216 = getitem_2217 = getitem_2218 = getitem_2219 = getitem_2220 = getitem_2221 = getitem_2222 = getitem_2223 = getitem_2224 = getitem_2225 = getitem_2226 = getitem_2227 = getitem_2228 = getitem_2229 = getitem_2230 = getitem_2231 = getitem_2232 = getitem_2233 = getitem_2234 = getitem_2235 = getitem_2236 = getitem_2237 = getitem_2238 = getitem_2239 = getitem_2240 = getitem_2241 = getitem_2242 = getitem_2243 = getitem_2244 = getitem_2245 = getitem_2246 = getitem_2247 = getitem_2248 = getitem_2249 = getitem_2250 = getitem_2251 = getitem_2252 = getitem_2253 = getitem_2254 = getitem_2255 = getitem_2256 = getitem_2257 = getitem_2258 = getitem_2259 = getitem_2260 = getitem_2261 = getitem_2262 = getitem_2263 = getitem_2264 = getitem_2265 = getitem_2266 = getitem_2267 = getitem_2268 = getitem_2269 = getitem_2270 = getitem_2271 = getitem_2272 = getitem_2273 = getitem_2274 = getitem_2275 = getitem_2276 = getitem_2277 = getitem_2278 = getitem_2279 = getitem_2280 = getitem_2281 = getitem_2282 = getitem_2283 = getitem_2284 = getitem_2285 = getitem_2286 = getitem_2287 = getitem_2288 = getitem_2289 = getitem_2290 = getitem_2291 = getitem_2292 = getitem_2293 = getitem_2294 = getitem_2295 = getitem_2296 = getitem_2297 = getitem_2298 = getitem_2299 = getitem_2300 = getitem_2301 = getitem_2302 = getitem_2303 = getitem_2304 = getitem_2305 = getitem_2306 = getitem_2307 = getitem_2308 = getitem_2309 = getitem_2310 = getitem_2311 = getitem_2312 = getitem_2313 = getitem_2314 = getitem_2315 = getitem_2316 = getitem_2317 = getitem_2318 = getitem_2319 = getitem_2320 = getitem_2321 = getitem_2322 = getitem_2323 = getitem_2324 = getitem_2325 = getitem_2326 = getitem_2327 = getitem_2328 = getitem_2329 = getitem_2330 = getitem_2331 = getitem_2332 = getitem_2333 = getitem_2334 = getitem_2335 = getitem_2336 = getitem_2337 = getitem_2338 = getitem_2339 = getitem_2340 = getitem_2341 = getitem_2342 = getitem_2343 = getitem_2344 = getitem_2345 = getitem_2346 = getitem_2347 = getitem_2348 = getitem_2349 = getitem_2350 = getitem_2351 = getitem_2352 = getitem_2353 = getitem_2354 = getitem_2355 = getitem_2356 = getitem_2357 = getitem_2358 = getitem_2359 = getitem_2360 = getitem_2361 = getitem_2362 = getitem_2363 = getitem_2364 = getitem_2365 = getitem_2366 = getitem_2367 = getitem_2368 = getitem_2369 = getitem_2370 = getitem_2371 = getitem_2372 = getitem_2373 = getitem_2374 = getitem_2375 = getitem_2376 = getitem_2377 = getitem_2378 = getitem_2379 = getitem_2380 = getitem_2381 = getitem_2382 = getitem_2383 = getitem_2384 = getitem_2385 = getitem_2386 = getitem_2387 = getitem_2388 = getitem_2389 = getitem_2390 = getitem_2391 = getitem_2392 = getitem_2393 = getitem_2394 = getitem_2395 = getitem_2396 = getitem_2397 = getitem_2398 = getitem_2399 = getitem_2400 = getitem_2401 = getitem_2402 = getitem_2403 = getitem_2404 = getitem_2405 = getitem_2406 = getitem_2407 = None
        getitem: "f32[]" = _foreach_div_scalar[0]
        getitem_2408: "f32[]" = _foreach_div_scalar[1]
        getitem_2409: "f32[]" = _foreach_div_scalar[2]
        getitem_2410: "f32[]" = _foreach_div_scalar[3]
        getitem_2411: "f32[]" = _foreach_div_scalar[4]
        getitem_2412: "f32[]" = _foreach_div_scalar[5]
        getitem_2413: "f32[]" = _foreach_div_scalar[6]
        getitem_2414: "f32[]" = _foreach_div_scalar[7]
        getitem_2415: "f32[]" = _foreach_div_scalar[8]
        getitem_2416: "f32[]" = _foreach_div_scalar[9]
        getitem_2417: "f32[]" = _foreach_div_scalar[10]
        getitem_2418: "f32[]" = _foreach_div_scalar[11]
        getitem_2419: "f32[]" = _foreach_div_scalar[12]
        getitem_2420: "f32[]" = _foreach_div_scalar[13]
        getitem_2421: "f32[]" = _foreach_div_scalar[14]
        getitem_2422: "f32[]" = _foreach_div_scalar[15]
        getitem_2423: "f32[]" = _foreach_div_scalar[16]
        getitem_2424: "f32[]" = _foreach_div_scalar[17]
        getitem_2425: "f32[]" = _foreach_div_scalar[18]
        getitem_2426: "f32[]" = _foreach_div_scalar[19]
        getitem_2427: "f32[]" = _foreach_div_scalar[20]
        getitem_2428: "f32[]" = _foreach_div_scalar[21]
        getitem_2429: "f32[]" = _foreach_div_scalar[22]
        getitem_2430: "f32[]" = _foreach_div_scalar[23]
        getitem_2431: "f32[]" = _foreach_div_scalar[24]
        getitem_2432: "f32[]" = _foreach_div_scalar[25]
        getitem_2433: "f32[]" = _foreach_div_scalar[26]
        getitem_2434: "f32[]" = _foreach_div_scalar[27]
        getitem_2435: "f32[]" = _foreach_div_scalar[28]
        getitem_2436: "f32[]" = _foreach_div_scalar[29]
        getitem_2437: "f32[]" = _foreach_div_scalar[30]
        getitem_2438: "f32[]" = _foreach_div_scalar[31]
        getitem_2439: "f32[]" = _foreach_div_scalar[32]
        getitem_2440: "f32[]" = _foreach_div_scalar[33]
        getitem_2441: "f32[]" = _foreach_div_scalar[34]
        getitem_2442: "f32[]" = _foreach_div_scalar[35]
        getitem_2443: "f32[]" = _foreach_div_scalar[36]
        getitem_2444: "f32[]" = _foreach_div_scalar[37]
        getitem_2445: "f32[]" = _foreach_div_scalar[38]
        getitem_2446: "f32[]" = _foreach_div_scalar[39]
        getitem_2447: "f32[]" = _foreach_div_scalar[40]
        getitem_2448: "f32[]" = _foreach_div_scalar[41]
        getitem_2449: "f32[]" = _foreach_div_scalar[42]
        getitem_2450: "f32[]" = _foreach_div_scalar[43]
        getitem_2451: "f32[]" = _foreach_div_scalar[44]
        getitem_2452: "f32[]" = _foreach_div_scalar[45]
        getitem_2453: "f32[]" = _foreach_div_scalar[46]
        getitem_2454: "f32[]" = _foreach_div_scalar[47]
        getitem_2455: "f32[]" = _foreach_div_scalar[48]
        getitem_2456: "f32[]" = _foreach_div_scalar[49]
        getitem_2457: "f32[]" = _foreach_div_scalar[50]
        getitem_2458: "f32[]" = _foreach_div_scalar[51]
        getitem_2459: "f32[]" = _foreach_div_scalar[52]
        getitem_2460: "f32[]" = _foreach_div_scalar[53]
        getitem_2461: "f32[]" = _foreach_div_scalar[54]
        getitem_2462: "f32[]" = _foreach_div_scalar[55]
        getitem_2463: "f32[]" = _foreach_div_scalar[56]
        getitem_2464: "f32[]" = _foreach_div_scalar[57]
        getitem_2465: "f32[]" = _foreach_div_scalar[58]
        getitem_2466: "f32[]" = _foreach_div_scalar[59]
        getitem_2467: "f32[]" = _foreach_div_scalar[60]
        getitem_2468: "f32[]" = _foreach_div_scalar[61]
        getitem_2469: "f32[]" = _foreach_div_scalar[62]
        getitem_2470: "f32[]" = _foreach_div_scalar[63]
        getitem_2471: "f32[]" = _foreach_div_scalar[64]
        getitem_2472: "f32[]" = _foreach_div_scalar[65]
        getitem_2473: "f32[]" = _foreach_div_scalar[66]
        getitem_2474: "f32[]" = _foreach_div_scalar[67]
        getitem_2475: "f32[]" = _foreach_div_scalar[68]
        getitem_2476: "f32[]" = _foreach_div_scalar[69]
        getitem_2477: "f32[]" = _foreach_div_scalar[70]
        getitem_2478: "f32[]" = _foreach_div_scalar[71]
        getitem_2479: "f32[]" = _foreach_div_scalar[72]
        getitem_2480: "f32[]" = _foreach_div_scalar[73]
        getitem_2481: "f32[]" = _foreach_div_scalar[74]
        getitem_2482: "f32[]" = _foreach_div_scalar[75]
        getitem_2483: "f32[]" = _foreach_div_scalar[76]
        getitem_2484: "f32[]" = _foreach_div_scalar[77]
        getitem_2485: "f32[]" = _foreach_div_scalar[78]
        getitem_2486: "f32[]" = _foreach_div_scalar[79]
        getitem_2487: "f32[]" = _foreach_div_scalar[80]
        getitem_2488: "f32[]" = _foreach_div_scalar[81]
        getitem_2489: "f32[]" = _foreach_div_scalar[82]
        getitem_2490: "f32[]" = _foreach_div_scalar[83]
        getitem_2491: "f32[]" = _foreach_div_scalar[84]
        getitem_2492: "f32[]" = _foreach_div_scalar[85]
        getitem_2493: "f32[]" = _foreach_div_scalar[86]
        getitem_2494: "f32[]" = _foreach_div_scalar[87]
        getitem_2495: "f32[]" = _foreach_div_scalar[88]
        getitem_2496: "f32[]" = _foreach_div_scalar[89]
        getitem_2497: "f32[]" = _foreach_div_scalar[90]
        getitem_2498: "f32[]" = _foreach_div_scalar[91]
        getitem_2499: "f32[]" = _foreach_div_scalar[92]
        getitem_2500: "f32[]" = _foreach_div_scalar[93]
        getitem_2501: "f32[]" = _foreach_div_scalar[94]
        getitem_2502: "f32[]" = _foreach_div_scalar[95]
        getitem_2503: "f32[]" = _foreach_div_scalar[96]
        getitem_2504: "f32[]" = _foreach_div_scalar[97]
        getitem_2505: "f32[]" = _foreach_div_scalar[98]
        getitem_2506: "f32[]" = _foreach_div_scalar[99]
        getitem_2507: "f32[]" = _foreach_div_scalar[100]
        getitem_2508: "f32[]" = _foreach_div_scalar[101]
        getitem_2509: "f32[]" = _foreach_div_scalar[102]
        getitem_2510: "f32[]" = _foreach_div_scalar[103]
        getitem_2511: "f32[]" = _foreach_div_scalar[104]
        getitem_2512: "f32[]" = _foreach_div_scalar[105]
        getitem_2513: "f32[]" = _foreach_div_scalar[106]
        getitem_2514: "f32[]" = _foreach_div_scalar[107]
        getitem_2515: "f32[]" = _foreach_div_scalar[108]
        getitem_2516: "f32[]" = _foreach_div_scalar[109]
        getitem_2517: "f32[]" = _foreach_div_scalar[110]
        getitem_2518: "f32[]" = _foreach_div_scalar[111]
        getitem_2519: "f32[]" = _foreach_div_scalar[112]
        getitem_2520: "f32[]" = _foreach_div_scalar[113]
        getitem_2521: "f32[]" = _foreach_div_scalar[114]
        getitem_2522: "f32[]" = _foreach_div_scalar[115]
        getitem_2523: "f32[]" = _foreach_div_scalar[116]
        getitem_2524: "f32[]" = _foreach_div_scalar[117]
        getitem_2525: "f32[]" = _foreach_div_scalar[118]
        getitem_2526: "f32[]" = _foreach_div_scalar[119]
        getitem_2527: "f32[]" = _foreach_div_scalar[120]
        getitem_2528: "f32[]" = _foreach_div_scalar[121]
        getitem_2529: "f32[]" = _foreach_div_scalar[122]
        getitem_2530: "f32[]" = _foreach_div_scalar[123]
        getitem_2531: "f32[]" = _foreach_div_scalar[124]
        getitem_2532: "f32[]" = _foreach_div_scalar[125]
        getitem_2533: "f32[]" = _foreach_div_scalar[126]
        getitem_2534: "f32[]" = _foreach_div_scalar[127]
        getitem_2535: "f32[]" = _foreach_div_scalar[128]
        getitem_2536: "f32[]" = _foreach_div_scalar[129]
        getitem_2537: "f32[]" = _foreach_div_scalar[130]
        getitem_2538: "f32[]" = _foreach_div_scalar[131]
        getitem_2539: "f32[]" = _foreach_div_scalar[132]
        getitem_2540: "f32[]" = _foreach_div_scalar[133]
        getitem_2541: "f32[]" = _foreach_div_scalar[134]
        getitem_2542: "f32[]" = _foreach_div_scalar[135]
        getitem_2543: "f32[]" = _foreach_div_scalar[136]
        getitem_2544: "f32[]" = _foreach_div_scalar[137]
        getitem_2545: "f32[]" = _foreach_div_scalar[138]
        getitem_2546: "f32[]" = _foreach_div_scalar[139]
        getitem_2547: "f32[]" = _foreach_div_scalar[140]
        getitem_2548: "f32[]" = _foreach_div_scalar[141]
        getitem_2549: "f32[]" = _foreach_div_scalar[142]
        getitem_2550: "f32[]" = _foreach_div_scalar[143]
        getitem_2551: "f32[]" = _foreach_div_scalar[144]
        getitem_2552: "f32[]" = _foreach_div_scalar[145]
        getitem_2553: "f32[]" = _foreach_div_scalar[146]
        getitem_2554: "f32[]" = _foreach_div_scalar[147]
        getitem_2555: "f32[]" = _foreach_div_scalar[148]
        getitem_2556: "f32[]" = _foreach_div_scalar[149]
        getitem_2557: "f32[]" = _foreach_div_scalar[150]
        getitem_2558: "f32[]" = _foreach_div_scalar[151]
        getitem_2559: "f32[]" = _foreach_div_scalar[152]
        getitem_2560: "f32[]" = _foreach_div_scalar[153]
        getitem_2561: "f32[]" = _foreach_div_scalar[154]
        getitem_2562: "f32[]" = _foreach_div_scalar[155]
        getitem_2563: "f32[]" = _foreach_div_scalar[156]
        getitem_2564: "f32[]" = _foreach_div_scalar[157]
        getitem_2565: "f32[]" = _foreach_div_scalar[158]
        getitem_2566: "f32[]" = _foreach_div_scalar[159]
        getitem_2567: "f32[]" = _foreach_div_scalar[160]
        getitem_2568: "f32[]" = _foreach_div_scalar[161]
        getitem_2569: "f32[]" = _foreach_div_scalar[162]
        getitem_2570: "f32[]" = _foreach_div_scalar[163]
        getitem_2571: "f32[]" = _foreach_div_scalar[164]
        getitem_2572: "f32[]" = _foreach_div_scalar[165]
        getitem_2573: "f32[]" = _foreach_div_scalar[166]
        getitem_2574: "f32[]" = _foreach_div_scalar[167]
        getitem_2575: "f32[]" = _foreach_div_scalar[168]
        getitem_2576: "f32[]" = _foreach_div_scalar[169]
        getitem_2577: "f32[]" = _foreach_div_scalar[170]
        getitem_2578: "f32[]" = _foreach_div_scalar[171]
        getitem_2579: "f32[]" = _foreach_div_scalar[172]
        getitem_2580: "f32[]" = _foreach_div_scalar[173]
        getitem_2581: "f32[]" = _foreach_div_scalar[174]
        getitem_2582: "f32[]" = _foreach_div_scalar[175]
        getitem_2583: "f32[]" = _foreach_div_scalar[176]
        getitem_2584: "f32[]" = _foreach_div_scalar[177]
        getitem_2585: "f32[]" = _foreach_div_scalar[178]
        getitem_2586: "f32[]" = _foreach_div_scalar[179]
        getitem_2587: "f32[]" = _foreach_div_scalar[180]
        getitem_2588: "f32[]" = _foreach_div_scalar[181]
        getitem_2589: "f32[]" = _foreach_div_scalar[182]
        getitem_2590: "f32[]" = _foreach_div_scalar[183]
        getitem_2591: "f32[]" = _foreach_div_scalar[184]
        getitem_2592: "f32[]" = _foreach_div_scalar[185]
        getitem_2593: "f32[]" = _foreach_div_scalar[186]
        getitem_2594: "f32[]" = _foreach_div_scalar[187]
        getitem_2595: "f32[]" = _foreach_div_scalar[188]
        getitem_2596: "f32[]" = _foreach_div_scalar[189]
        getitem_2597: "f32[]" = _foreach_div_scalar[190]
        getitem_2598: "f32[]" = _foreach_div_scalar[191]
        getitem_2599: "f32[]" = _foreach_div_scalar[192]
        getitem_2600: "f32[]" = _foreach_div_scalar[193]
        getitem_2601: "f32[]" = _foreach_div_scalar[194]
        getitem_2602: "f32[]" = _foreach_div_scalar[195]
        getitem_2603: "f32[]" = _foreach_div_scalar[196]
        getitem_2604: "f32[]" = _foreach_div_scalar[197]
        getitem_2605: "f32[]" = _foreach_div_scalar[198]
        getitem_2606: "f32[]" = _foreach_div_scalar[199]
        getitem_2607: "f32[]" = _foreach_div_scalar[200]
        getitem_2608: "f32[]" = _foreach_div_scalar[201]
        getitem_2609: "f32[]" = _foreach_div_scalar[202]
        getitem_2610: "f32[]" = _foreach_div_scalar[203]
        getitem_2611: "f32[]" = _foreach_div_scalar[204]
        getitem_2612: "f32[]" = _foreach_div_scalar[205]
        getitem_2613: "f32[]" = _foreach_div_scalar[206]
        getitem_2614: "f32[]" = _foreach_div_scalar[207]
        getitem_2615: "f32[]" = _foreach_div_scalar[208]
        getitem_2616: "f32[]" = _foreach_div_scalar[209]
        getitem_2617: "f32[]" = _foreach_div_scalar[210]
        getitem_2618: "f32[]" = _foreach_div_scalar[211]
        getitem_2619: "f32[]" = _foreach_div_scalar[212]
        getitem_2620: "f32[]" = _foreach_div_scalar[213]
        getitem_2621: "f32[]" = _foreach_div_scalar[214]
        getitem_2622: "f32[]" = _foreach_div_scalar[215]
        getitem_2623: "f32[]" = _foreach_div_scalar[216]
        getitem_2624: "f32[]" = _foreach_div_scalar[217]
        getitem_2625: "f32[]" = _foreach_div_scalar[218]
        getitem_2626: "f32[]" = _foreach_div_scalar[219]
        getitem_2627: "f32[]" = _foreach_div_scalar[220]
        getitem_2628: "f32[]" = _foreach_div_scalar[221]
        getitem_2629: "f32[]" = _foreach_div_scalar[222]
        getitem_2630: "f32[]" = _foreach_div_scalar[223]
        getitem_2631: "f32[]" = _foreach_div_scalar[224]
        getitem_2632: "f32[]" = _foreach_div_scalar[225]
        getitem_2633: "f32[]" = _foreach_div_scalar[226]
        getitem_2634: "f32[]" = _foreach_div_scalar[227]
        getitem_2635: "f32[]" = _foreach_div_scalar[228]
        getitem_2636: "f32[]" = _foreach_div_scalar[229]
        getitem_2637: "f32[]" = _foreach_div_scalar[230]
        getitem_2638: "f32[]" = _foreach_div_scalar[231]
        getitem_2639: "f32[]" = _foreach_div_scalar[232]
        getitem_2640: "f32[]" = _foreach_div_scalar[233]
        getitem_2641: "f32[]" = _foreach_div_scalar[234]
        getitem_2642: "f32[]" = _foreach_div_scalar[235]
        getitem_2643: "f32[]" = _foreach_div_scalar[236]
        getitem_2644: "f32[]" = _foreach_div_scalar[237]
        getitem_2645: "f32[]" = _foreach_div_scalar[238]
        getitem_2646: "f32[]" = _foreach_div_scalar[239]
        getitem_2647: "f32[]" = _foreach_div_scalar[240]
        getitem_2648: "f32[]" = _foreach_div_scalar[241]
        getitem_2649: "f32[]" = _foreach_div_scalar[242]
        getitem_2650: "f32[]" = _foreach_div_scalar[243]
        getitem_2651: "f32[]" = _foreach_div_scalar[244]
        getitem_2652: "f32[]" = _foreach_div_scalar[245]
        getitem_2653: "f32[]" = _foreach_div_scalar[246]
        getitem_2654: "f32[]" = _foreach_div_scalar[247]
        getitem_2655: "f32[]" = _foreach_div_scalar[248]
        getitem_2656: "f32[]" = _foreach_div_scalar[249]
        getitem_2657: "f32[]" = _foreach_div_scalar[250]
        getitem_2658: "f32[]" = _foreach_div_scalar[251]
        getitem_2659: "f32[]" = _foreach_div_scalar[252]
        getitem_2660: "f32[]" = _foreach_div_scalar[253]
        getitem_2661: "f32[]" = _foreach_div_scalar[254]
        getitem_2662: "f32[]" = _foreach_div_scalar[255]
        getitem_2663: "f32[]" = _foreach_div_scalar[256]
        getitem_2664: "f32[]" = _foreach_div_scalar[257]
        getitem_2665: "f32[]" = _foreach_div_scalar[258]
        getitem_2666: "f32[]" = _foreach_div_scalar[259]
        getitem_2667: "f32[]" = _foreach_div_scalar[260]
        getitem_2668: "f32[]" = _foreach_div_scalar[261]
        getitem_2669: "f32[]" = _foreach_div_scalar[262]
        getitem_2670: "f32[]" = _foreach_div_scalar[263]
        getitem_2671: "f32[]" = _foreach_div_scalar[264]
        getitem_2672: "f32[]" = _foreach_div_scalar[265]
        getitem_2673: "f32[]" = _foreach_div_scalar[266]
        getitem_2674: "f32[]" = _foreach_div_scalar[267]
        getitem_2675: "f32[]" = _foreach_div_scalar[268]
        getitem_2676: "f32[]" = _foreach_div_scalar[269]
        getitem_2677: "f32[]" = _foreach_div_scalar[270]
        getitem_2678: "f32[]" = _foreach_div_scalar[271]
        getitem_2679: "f32[]" = _foreach_div_scalar[272]
        getitem_2680: "f32[]" = _foreach_div_scalar[273]
        getitem_2681: "f32[]" = _foreach_div_scalar[274]
        getitem_2682: "f32[]" = _foreach_div_scalar[275]
        getitem_2683: "f32[]" = _foreach_div_scalar[276]
        getitem_2684: "f32[]" = _foreach_div_scalar[277]
        getitem_2685: "f32[]" = _foreach_div_scalar[278]
        getitem_2686: "f32[]" = _foreach_div_scalar[279]
        getitem_2687: "f32[]" = _foreach_div_scalar[280]
        getitem_2688: "f32[]" = _foreach_div_scalar[281]
        getitem_2689: "f32[]" = _foreach_div_scalar[282]
        getitem_2690: "f32[]" = _foreach_div_scalar[283]
        getitem_2691: "f32[]" = _foreach_div_scalar[284]
        getitem_2692: "f32[]" = _foreach_div_scalar[285]
        getitem_2693: "f32[]" = _foreach_div_scalar[286]
        getitem_2694: "f32[]" = _foreach_div_scalar[287]
        getitem_2695: "f32[]" = _foreach_div_scalar[288]
        getitem_2696: "f32[]" = _foreach_div_scalar[289]
        getitem_2697: "f32[]" = _foreach_div_scalar[290]
        getitem_2698: "f32[]" = _foreach_div_scalar[291]
        getitem_2699: "f32[]" = _foreach_div_scalar[292]
        getitem_2700: "f32[]" = _foreach_div_scalar[293]
        getitem_2701: "f32[]" = _foreach_div_scalar[294]
        getitem_2702: "f32[]" = _foreach_div_scalar[295]
        getitem_2703: "f32[]" = _foreach_div_scalar[296]
        getitem_2704: "f32[]" = _foreach_div_scalar[297]
        getitem_2705: "f32[]" = _foreach_div_scalar[298]
        getitem_2706: "f32[]" = _foreach_div_scalar[299]
        getitem_2707: "f32[]" = _foreach_div_scalar[300];  _foreach_div_scalar = None
        _foreach_div_list = torch.ops.aten._foreach_div.List([getitem_3913, getitem_3914, getitem_3915, getitem_3916, getitem_3917, getitem_3918, getitem_3919, getitem_3920, getitem_3921, getitem_3922, getitem_3923, getitem_3924, getitem_3925, getitem_3926, getitem_3927, getitem_3928, getitem_3929, getitem_3930, getitem_3931, getitem_3932, getitem_3933, getitem_3934, getitem_3935, getitem_3936, getitem_3937, getitem_3938, getitem_3939, getitem_3940, getitem_3941, getitem_3942, getitem_3943, getitem_3944, getitem_3945, getitem_3946, getitem_3947, getitem_3948, getitem_3949, getitem_3950, getitem_3951, getitem_3952, getitem_3953, getitem_3954, getitem_3955, getitem_3956, getitem_3957, getitem_3958, getitem_3959, getitem_3960, getitem_3961, getitem_3962, getitem_3963, getitem_3964, getitem_3965, getitem_3966, getitem_3967, getitem_3968, getitem_3969, getitem_3970, getitem_3971, getitem_3972, getitem_3973, getitem_3974, getitem_3975, getitem_3976, getitem_3977, getitem_3978, getitem_3979, getitem_3980, getitem_3981, getitem_3982, getitem_3983, getitem_3984, getitem_3985, getitem_3986, getitem_3987, getitem_3988, getitem_3989, getitem_3990, getitem_3991, getitem_3992, getitem_3993, getitem_3994, getitem_3995, getitem_3996, getitem_3997, getitem_3998, getitem_3999, getitem_4000, getitem_4001, getitem_4002, getitem_4003, getitem_4004, getitem_4005, getitem_4006, getitem_4007, getitem_4008, getitem_4009, getitem_4010, getitem_4011, getitem_4012, getitem_4013, getitem_4014, getitem_4015, getitem_4016, getitem_4017, getitem_4018, getitem_4019, getitem_4020, getitem_4021, getitem_4022, getitem_4023, getitem_4024, getitem_4025, getitem_4026, getitem_4027, getitem_4028, getitem_4029, getitem_4030, getitem_4031, getitem_4032, getitem_4033, getitem_4034, getitem_4035, getitem_4036, getitem_4037, getitem_4038, getitem_4039, getitem_4040, getitem_4041, getitem_4042, getitem_4043, getitem_4044, getitem_4045, getitem_4046, getitem_4047, getitem_4048, getitem_4049, getitem_4050, getitem_4051, getitem_4052, getitem_4053, getitem_4054, getitem_4055, getitem_4056, getitem_4057, getitem_4058, getitem_4059, getitem_4060, getitem_4061, getitem_4062, getitem_4063, getitem_4064, getitem_4065, getitem_4066, getitem_4067, getitem_4068, getitem_4069, getitem_4070, getitem_4071, getitem_4072, getitem_4073, getitem_4074, getitem_4075, getitem_4076, getitem_4077, getitem_4078, getitem_4079, getitem_4080, getitem_4081, getitem_4082, getitem_4083, getitem_4084, getitem_4085, getitem_4086, getitem_4087, getitem_4088, getitem_4089, getitem_4090, getitem_4091, getitem_4092, getitem_4093, getitem_4094, getitem_4095, getitem_4096, getitem_4097, getitem_4098, getitem_4099, getitem_4100, getitem_4101, getitem_4102, getitem_4103, getitem_4104, getitem_4105, getitem_4106, getitem_4107, getitem_4108, getitem_4109, getitem_4110, getitem_4111, getitem_4112, getitem_4113, getitem_4114, getitem_4115, getitem_4116, getitem_4117, getitem_4118, getitem_4119, getitem_4120, getitem_4121, getitem_4122, getitem_4123, getitem_4124, getitem_4125, getitem_4126, getitem_4127, getitem_4128, getitem_4129, getitem_4130, getitem_4131, getitem_4132, getitem_4133, getitem_4134, getitem_4135, getitem_4136, getitem_4137, getitem_4138, getitem_4139, getitem_4140, getitem_4141, getitem_4142, getitem_4143, getitem_4144, getitem_4145, getitem_4146, getitem_4147, getitem_4148, getitem_4149, getitem_4150, getitem_4151, getitem_4152, getitem_4153, getitem_4154, getitem_4155, getitem_4156, getitem_4157, getitem_4158, getitem_4159, getitem_4160, getitem_4161, getitem_4162, getitem_4163, getitem_4164, getitem_4165, getitem_4166, getitem_4167, getitem_4168, getitem_4169, getitem_4170, getitem_4171, getitem_4172, getitem_4173, getitem_4174, getitem_4175, getitem_4176, getitem_4177, getitem_4178, getitem_4179, getitem_4180, getitem_4181, getitem_4182, getitem_4183, getitem_4184, getitem_4185, getitem_4186, getitem_4187, getitem_4188, getitem_4189, getitem_4190, getitem_4191, getitem_4192, getitem_4193, getitem_4194, getitem_4195, getitem_4196, getitem_4197, getitem_4198, getitem_4199, getitem_4200, getitem_4201, getitem_4202, getitem_4203, getitem_4204, getitem_4205, getitem_4206, getitem_4207, getitem_4208, getitem_4209, getitem_4210, getitem_4211, getitem_4212, getitem_4213], [getitem_3612, getitem_3613, getitem_3614, getitem_3615, getitem_3616, getitem_3617, getitem_3618, getitem_3619, getitem_3620, getitem_3621, getitem_3622, getitem_3623, getitem_3624, getitem_3625, getitem_3626, getitem_3627, getitem_3628, getitem_3629, getitem_3630, getitem_3631, getitem_3632, getitem_3633, getitem_3634, getitem_3635, getitem_3636, getitem_3637, getitem_3638, getitem_3639, getitem_3640, getitem_3641, getitem_3642, getitem_3643, getitem_3644, getitem_3645, getitem_3646, getitem_3647, getitem_3648, getitem_3649, getitem_3650, getitem_3651, getitem_3652, getitem_3653, getitem_3654, getitem_3655, getitem_3656, getitem_3657, getitem_3658, getitem_3659, getitem_3660, getitem_3661, getitem_3662, getitem_3663, getitem_3664, getitem_3665, getitem_3666, getitem_3667, getitem_3668, getitem_3669, getitem_3670, getitem_3671, getitem_3672, getitem_3673, getitem_3674, getitem_3675, getitem_3676, getitem_3677, getitem_3678, getitem_3679, getitem_3680, getitem_3681, getitem_3682, getitem_3683, getitem_3684, getitem_3685, getitem_3686, getitem_3687, getitem_3688, getitem_3689, getitem_3690, getitem_3691, getitem_3692, getitem_3693, getitem_3694, getitem_3695, getitem_3696, getitem_3697, getitem_3698, getitem_3699, getitem_3700, getitem_3701, getitem_3702, getitem_3703, getitem_3704, getitem_3705, getitem_3706, getitem_3707, getitem_3708, getitem_3709, getitem_3710, getitem_3711, getitem_3712, getitem_3713, getitem_3714, getitem_3715, getitem_3716, getitem_3717, getitem_3718, getitem_3719, getitem_3720, getitem_3721, getitem_3722, getitem_3723, getitem_3724, getitem_3725, getitem_3726, getitem_3727, getitem_3728, getitem_3729, getitem_3730, getitem_3731, getitem_3732, getitem_3733, getitem_3734, getitem_3735, getitem_3736, getitem_3737, getitem_3738, getitem_3739, getitem_3740, getitem_3741, getitem_3742, getitem_3743, getitem_3744, getitem_3745, getitem_3746, getitem_3747, getitem_3748, getitem_3749, getitem_3750, getitem_3751, getitem_3752, getitem_3753, getitem_3754, getitem_3755, getitem_3756, getitem_3757, getitem_3758, getitem_3759, getitem_3760, getitem_3761, getitem_3762, getitem_3763, getitem_3764, getitem_3765, getitem_3766, getitem_3767, getitem_3768, getitem_3769, getitem_3770, getitem_3771, getitem_3772, getitem_3773, getitem_3774, getitem_3775, getitem_3776, getitem_3777, getitem_3778, getitem_3779, getitem_3780, getitem_3781, getitem_3782, getitem_3783, getitem_3784, getitem_3785, getitem_3786, getitem_3787, getitem_3788, getitem_3789, getitem_3790, getitem_3791, getitem_3792, getitem_3793, getitem_3794, getitem_3795, getitem_3796, getitem_3797, getitem_3798, getitem_3799, getitem_3800, getitem_3801, getitem_3802, getitem_3803, getitem_3804, getitem_3805, getitem_3806, getitem_3807, getitem_3808, getitem_3809, getitem_3810, getitem_3811, getitem_3812, getitem_3813, getitem_3814, getitem_3815, getitem_3816, getitem_3817, getitem_3818, getitem_3819, getitem_3820, getitem_3821, getitem_3822, getitem_3823, getitem_3824, getitem_3825, getitem_3826, getitem_3827, getitem_3828, getitem_3829, getitem_3830, getitem_3831, getitem_3832, getitem_3833, getitem_3834, getitem_3835, getitem_3836, getitem_3837, getitem_3838, getitem_3839, getitem_3840, getitem_3841, getitem_3842, getitem_3843, getitem_3844, getitem_3845, getitem_3846, getitem_3847, getitem_3848, getitem_3849, getitem_3850, getitem_3851, getitem_3852, getitem_3853, getitem_3854, getitem_3855, getitem_3856, getitem_3857, getitem_3858, getitem_3859, getitem_3860, getitem_3861, getitem_3862, getitem_3863, getitem_3864, getitem_3865, getitem_3866, getitem_3867, getitem_3868, getitem_3869, getitem_3870, getitem_3871, getitem_3872, getitem_3873, getitem_3874, getitem_3875, getitem_3876, getitem_3877, getitem_3878, getitem_3879, getitem_3880, getitem_3881, getitem_3882, getitem_3883, getitem_3884, getitem_3885, getitem_3886, getitem_3887, getitem_3888, getitem_3889, getitem_3890, getitem_3891, getitem_3892, getitem_3893, getitem_3894, getitem_3895, getitem_3896, getitem_3897, getitem_3898, getitem_3899, getitem_3900, getitem_3901, getitem_3902, getitem_3903, getitem_3904, getitem_3905, getitem_3906, getitem_3907, getitem_3908, getitem_3909, getitem_3910, getitem_3911, getitem_3912]);  getitem_3913 = getitem_3914 = getitem_3915 = getitem_3916 = getitem_3917 = getitem_3918 = getitem_3919 = getitem_3920 = getitem_3921 = getitem_3922 = getitem_3923 = getitem_3924 = getitem_3925 = getitem_3926 = getitem_3927 = getitem_3928 = getitem_3929 = getitem_3930 = getitem_3931 = getitem_3932 = getitem_3933 = getitem_3934 = getitem_3935 = getitem_3936 = getitem_3937 = getitem_3938 = getitem_3939 = getitem_3940 = getitem_3941 = getitem_3942 = getitem_3943 = getitem_3944 = getitem_3945 = getitem_3946 = getitem_3947 = getitem_3948 = getitem_3949 = getitem_3950 = getitem_3951 = getitem_3952 = getitem_3953 = getitem_3954 = getitem_3955 = getitem_3956 = getitem_3957 = getitem_3958 = getitem_3959 = getitem_3960 = getitem_3961 = getitem_3962 = getitem_3963 = getitem_3964 = getitem_3965 = getitem_3966 = getitem_3967 = getitem_3968 = getitem_3969 = getitem_3970 = getitem_3971 = getitem_3972 = getitem_3973 = getitem_3974 = getitem_3975 = getitem_3976 = getitem_3977 = getitem_3978 = getitem_3979 = getitem_3980 = getitem_3981 = getitem_3982 = getitem_3983 = getitem_3984 = getitem_3985 = getitem_3986 = getitem_3987 = getitem_3988 = getitem_3989 = getitem_3990 = getitem_3991 = getitem_3992 = getitem_3993 = getitem_3994 = getitem_3995 = getitem_3996 = getitem_3997 = getitem_3998 = getitem_3999 = getitem_4000 = getitem_4001 = getitem_4002 = getitem_4003 = getitem_4004 = getitem_4005 = getitem_4006 = getitem_4007 = getitem_4008 = getitem_4009 = getitem_4010 = getitem_4011 = getitem_4012 = getitem_4013 = getitem_4014 = getitem_4015 = getitem_4016 = getitem_4017 = getitem_4018 = getitem_4019 = getitem_4020 = getitem_4021 = getitem_4022 = getitem_4023 = getitem_4024 = getitem_4025 = getitem_4026 = getitem_4027 = getitem_4028 = getitem_4029 = getitem_4030 = getitem_4031 = getitem_4032 = getitem_4033 = getitem_4034 = getitem_4035 = getitem_4036 = getitem_4037 = getitem_4038 = getitem_4039 = getitem_4040 = getitem_4041 = getitem_4042 = getitem_4043 = getitem_4044 = getitem_4045 = getitem_4046 = getitem_4047 = getitem_4048 = getitem_4049 = getitem_4050 = getitem_4051 = getitem_4052 = getitem_4053 = getitem_4054 = getitem_4055 = getitem_4056 = getitem_4057 = getitem_4058 = getitem_4059 = getitem_4060 = getitem_4061 = getitem_4062 = getitem_4063 = getitem_4064 = getitem_4065 = getitem_4066 = getitem_4067 = getitem_4068 = getitem_4069 = getitem_4070 = getitem_4071 = getitem_4072 = getitem_4073 = getitem_4074 = getitem_4075 = getitem_4076 = getitem_4077 = getitem_4078 = getitem_4079 = getitem_4080 = getitem_4081 = getitem_4082 = getitem_4083 = getitem_4084 = getitem_4085 = getitem_4086 = getitem_4087 = getitem_4088 = getitem_4089 = getitem_4090 = getitem_4091 = getitem_4092 = getitem_4093 = getitem_4094 = getitem_4095 = getitem_4096 = getitem_4097 = getitem_4098 = getitem_4099 = getitem_4100 = getitem_4101 = getitem_4102 = getitem_4103 = getitem_4104 = getitem_4105 = getitem_4106 = getitem_4107 = getitem_4108 = getitem_4109 = getitem_4110 = getitem_4111 = getitem_4112 = getitem_4113 = getitem_4114 = getitem_4115 = getitem_4116 = getitem_4117 = getitem_4118 = getitem_4119 = getitem_4120 = getitem_4121 = getitem_4122 = getitem_4123 = getitem_4124 = getitem_4125 = getitem_4126 = getitem_4127 = getitem_4128 = getitem_4129 = getitem_4130 = getitem_4131 = getitem_4132 = getitem_4133 = getitem_4134 = getitem_4135 = getitem_4136 = getitem_4137 = getitem_4138 = getitem_4139 = getitem_4140 = getitem_4141 = getitem_4142 = getitem_4143 = getitem_4144 = getitem_4145 = getitem_4146 = getitem_4147 = getitem_4148 = getitem_4149 = getitem_4150 = getitem_4151 = getitem_4152 = getitem_4153 = getitem_4154 = getitem_4155 = getitem_4156 = getitem_4157 = getitem_4158 = getitem_4159 = getitem_4160 = getitem_4161 = getitem_4162 = getitem_4163 = getitem_4164 = getitem_4165 = getitem_4166 = getitem_4167 = getitem_4168 = getitem_4169 = getitem_4170 = getitem_4171 = getitem_4172 = getitem_4173 = getitem_4174 = getitem_4175 = getitem_4176 = getitem_4177 = getitem_4178 = getitem_4179 = getitem_4180 = getitem_4181 = getitem_4182 = getitem_4183 = getitem_4184 = getitem_4185 = getitem_4186 = getitem_4187 = getitem_4188 = getitem_4189 = getitem_4190 = getitem_4191 = getitem_4192 = getitem_4193 = getitem_4194 = getitem_4195 = getitem_4196 = getitem_4197 = getitem_4198 = getitem_4199 = getitem_4200 = getitem_4201 = getitem_4202 = getitem_4203 = getitem_4204 = getitem_4205 = getitem_4206 = getitem_4207 = getitem_4208 = getitem_4209 = getitem_4210 = getitem_4211 = getitem_4212 = getitem_4213 = getitem_3612 = getitem_3613 = getitem_3614 = getitem_3615 = getitem_3616 = getitem_3617 = getitem_3618 = getitem_3619 = getitem_3620 = getitem_3621 = getitem_3622 = getitem_3623 = getitem_3624 = getitem_3625 = getitem_3626 = getitem_3627 = getitem_3628 = getitem_3629 = getitem_3630 = getitem_3631 = getitem_3632 = getitem_3633 = getitem_3634 = getitem_3635 = getitem_3636 = getitem_3637 = getitem_3638 = getitem_3639 = getitem_3640 = getitem_3641 = getitem_3642 = getitem_3643 = getitem_3644 = getitem_3645 = getitem_3646 = getitem_3647 = getitem_3648 = getitem_3649 = getitem_3650 = getitem_3651 = getitem_3652 = getitem_3653 = getitem_3654 = getitem_3655 = getitem_3656 = getitem_3657 = getitem_3658 = getitem_3659 = getitem_3660 = getitem_3661 = getitem_3662 = getitem_3663 = getitem_3664 = getitem_3665 = getitem_3666 = getitem_3667 = getitem_3668 = getitem_3669 = getitem_3670 = getitem_3671 = getitem_3672 = getitem_3673 = getitem_3674 = getitem_3675 = getitem_3676 = getitem_3677 = getitem_3678 = getitem_3679 = getitem_3680 = getitem_3681 = getitem_3682 = getitem_3683 = getitem_3684 = getitem_3685 = getitem_3686 = getitem_3687 = getitem_3688 = getitem_3689 = getitem_3690 = getitem_3691 = getitem_3692 = getitem_3693 = getitem_3694 = getitem_3695 = getitem_3696 = getitem_3697 = getitem_3698 = getitem_3699 = getitem_3700 = getitem_3701 = getitem_3702 = getitem_3703 = getitem_3704 = getitem_3705 = getitem_3706 = getitem_3707 = getitem_3708 = getitem_3709 = getitem_3710 = getitem_3711 = getitem_3712 = getitem_3713 = getitem_3714 = getitem_3715 = getitem_3716 = getitem_3717 = getitem_3718 = getitem_3719 = getitem_3720 = getitem_3721 = getitem_3722 = getitem_3723 = getitem_3724 = getitem_3725 = getitem_3726 = getitem_3727 = getitem_3728 = getitem_3729 = getitem_3730 = getitem_3731 = getitem_3732 = getitem_3733 = getitem_3734 = getitem_3735 = getitem_3736 = getitem_3737 = getitem_3738 = getitem_3739 = getitem_3740 = getitem_3741 = getitem_3742 = getitem_3743 = getitem_3744 = getitem_3745 = getitem_3746 = getitem_3747 = getitem_3748 = getitem_3749 = getitem_3750 = getitem_3751 = getitem_3752 = getitem_3753 = getitem_3754 = getitem_3755 = getitem_3756 = getitem_3757 = getitem_3758 = getitem_3759 = getitem_3760 = getitem_3761 = getitem_3762 = getitem_3763 = getitem_3764 = getitem_3765 = getitem_3766 = getitem_3767 = getitem_3768 = getitem_3769 = getitem_3770 = getitem_3771 = getitem_3772 = getitem_3773 = getitem_3774 = getitem_3775 = getitem_3776 = getitem_3777 = getitem_3778 = getitem_3779 = getitem_3780 = getitem_3781 = getitem_3782 = getitem_3783 = getitem_3784 = getitem_3785 = getitem_3786 = getitem_3787 = getitem_3788 = getitem_3789 = getitem_3790 = getitem_3791 = getitem_3792 = getitem_3793 = getitem_3794 = getitem_3795 = getitem_3796 = getitem_3797 = getitem_3798 = getitem_3799 = getitem_3800 = getitem_3801 = getitem_3802 = getitem_3803 = getitem_3804 = getitem_3805 = getitem_3806 = getitem_3807 = getitem_3808 = getitem_3809 = getitem_3810 = getitem_3811 = getitem_3812 = getitem_3813 = getitem_3814 = getitem_3815 = getitem_3816 = getitem_3817 = getitem_3818 = getitem_3819 = getitem_3820 = getitem_3821 = getitem_3822 = getitem_3823 = getitem_3824 = getitem_3825 = getitem_3826 = getitem_3827 = getitem_3828 = getitem_3829 = getitem_3830 = getitem_3831 = getitem_3832 = getitem_3833 = getitem_3834 = getitem_3835 = getitem_3836 = getitem_3837 = getitem_3838 = getitem_3839 = getitem_3840 = getitem_3841 = getitem_3842 = getitem_3843 = getitem_3844 = getitem_3845 = getitem_3846 = getitem_3847 = getitem_3848 = getitem_3849 = getitem_3850 = getitem_3851 = getitem_3852 = getitem_3853 = getitem_3854 = getitem_3855 = getitem_3856 = getitem_3857 = getitem_3858 = getitem_3859 = getitem_3860 = getitem_3861 = getitem_3862 = getitem_3863 = getitem_3864 = getitem_3865 = getitem_3866 = getitem_3867 = getitem_3868 = getitem_3869 = getitem_3870 = getitem_3871 = getitem_3872 = getitem_3873 = getitem_3874 = getitem_3875 = getitem_3876 = getitem_3877 = getitem_3878 = getitem_3879 = getitem_3880 = getitem_3881 = getitem_3882 = getitem_3883 = getitem_3884 = getitem_3885 = getitem_3886 = getitem_3887 = getitem_3888 = getitem_3889 = getitem_3890 = getitem_3891 = getitem_3892 = getitem_3893 = getitem_3894 = getitem_3895 = getitem_3896 = getitem_3897 = getitem_3898 = getitem_3899 = getitem_3900 = getitem_3901 = getitem_3902 = getitem_3903 = getitem_3904 = getitem_3905 = getitem_3906 = getitem_3907 = getitem_3908 = getitem_3909 = getitem_3910 = getitem_3911 = getitem_3912 = None
        getitem_4214: "f32[768]" = _foreach_div_list[0]
        getitem_4215: "f32[50, 768]" = _foreach_div_list[1]
        getitem_4216: "f32[768, 512]" = _foreach_div_list[2]
        getitem_4217: "f32[768, 3, 32, 32]" = _foreach_div_list[3]
        getitem_4218: "f32[768]" = _foreach_div_list[4]
        getitem_4219: "f32[768]" = _foreach_div_list[5]
        getitem_4220: "f32[2304, 768]" = _foreach_div_list[6]
        getitem_4221: "f32[2304]" = _foreach_div_list[7]
        getitem_4222: "f32[768, 768]" = _foreach_div_list[8]
        getitem_4223: "f32[768]" = _foreach_div_list[9]
        getitem_4224: "f32[3072, 768]" = _foreach_div_list[10]
        getitem_4225: "f32[3072]" = _foreach_div_list[11]
        getitem_4226: "f32[768, 3072]" = _foreach_div_list[12]
        getitem_4227: "f32[768]" = _foreach_div_list[13]
        getitem_4228: "f32[768]" = _foreach_div_list[14]
        getitem_4229: "f32[768]" = _foreach_div_list[15]
        getitem_4230: "f32[768]" = _foreach_div_list[16]
        getitem_4231: "f32[768]" = _foreach_div_list[17]
        getitem_4232: "f32[2304, 768]" = _foreach_div_list[18]
        getitem_4233: "f32[2304]" = _foreach_div_list[19]
        getitem_4234: "f32[768, 768]" = _foreach_div_list[20]
        getitem_4235: "f32[768]" = _foreach_div_list[21]
        getitem_4236: "f32[3072, 768]" = _foreach_div_list[22]
        getitem_4237: "f32[3072]" = _foreach_div_list[23]
        getitem_4238: "f32[768, 3072]" = _foreach_div_list[24]
        getitem_4239: "f32[768]" = _foreach_div_list[25]
        getitem_4240: "f32[768]" = _foreach_div_list[26]
        getitem_4241: "f32[768]" = _foreach_div_list[27]
        getitem_4242: "f32[768]" = _foreach_div_list[28]
        getitem_4243: "f32[768]" = _foreach_div_list[29]
        getitem_4244: "f32[2304, 768]" = _foreach_div_list[30]
        getitem_4245: "f32[2304]" = _foreach_div_list[31]
        getitem_4246: "f32[768, 768]" = _foreach_div_list[32]
        getitem_4247: "f32[768]" = _foreach_div_list[33]
        getitem_4248: "f32[3072, 768]" = _foreach_div_list[34]
        getitem_4249: "f32[3072]" = _foreach_div_list[35]
        getitem_4250: "f32[768, 3072]" = _foreach_div_list[36]
        getitem_4251: "f32[768]" = _foreach_div_list[37]
        getitem_4252: "f32[768]" = _foreach_div_list[38]
        getitem_4253: "f32[768]" = _foreach_div_list[39]
        getitem_4254: "f32[768]" = _foreach_div_list[40]
        getitem_4255: "f32[768]" = _foreach_div_list[41]
        getitem_4256: "f32[2304, 768]" = _foreach_div_list[42]
        getitem_4257: "f32[2304]" = _foreach_div_list[43]
        getitem_4258: "f32[768, 768]" = _foreach_div_list[44]
        getitem_4259: "f32[768]" = _foreach_div_list[45]
        getitem_4260: "f32[3072, 768]" = _foreach_div_list[46]
        getitem_4261: "f32[3072]" = _foreach_div_list[47]
        getitem_4262: "f32[768, 3072]" = _foreach_div_list[48]
        getitem_4263: "f32[768]" = _foreach_div_list[49]
        getitem_4264: "f32[768]" = _foreach_div_list[50]
        getitem_4265: "f32[768]" = _foreach_div_list[51]
        getitem_4266: "f32[768]" = _foreach_div_list[52]
        getitem_4267: "f32[768]" = _foreach_div_list[53]
        getitem_4268: "f32[2304, 768]" = _foreach_div_list[54]
        getitem_4269: "f32[2304]" = _foreach_div_list[55]
        getitem_4270: "f32[768, 768]" = _foreach_div_list[56]
        getitem_4271: "f32[768]" = _foreach_div_list[57]
        getitem_4272: "f32[3072, 768]" = _foreach_div_list[58]
        getitem_4273: "f32[3072]" = _foreach_div_list[59]
        getitem_4274: "f32[768, 3072]" = _foreach_div_list[60]
        getitem_4275: "f32[768]" = _foreach_div_list[61]
        getitem_4276: "f32[768]" = _foreach_div_list[62]
        getitem_4277: "f32[768]" = _foreach_div_list[63]
        getitem_4278: "f32[768]" = _foreach_div_list[64]
        getitem_4279: "f32[768]" = _foreach_div_list[65]
        getitem_4280: "f32[2304, 768]" = _foreach_div_list[66]
        getitem_4281: "f32[2304]" = _foreach_div_list[67]
        getitem_4282: "f32[768, 768]" = _foreach_div_list[68]
        getitem_4283: "f32[768]" = _foreach_div_list[69]
        getitem_4284: "f32[3072, 768]" = _foreach_div_list[70]
        getitem_4285: "f32[3072]" = _foreach_div_list[71]
        getitem_4286: "f32[768, 3072]" = _foreach_div_list[72]
        getitem_4287: "f32[768]" = _foreach_div_list[73]
        getitem_4288: "f32[768]" = _foreach_div_list[74]
        getitem_4289: "f32[768]" = _foreach_div_list[75]
        getitem_4290: "f32[768]" = _foreach_div_list[76]
        getitem_4291: "f32[768]" = _foreach_div_list[77]
        getitem_4292: "f32[2304, 768]" = _foreach_div_list[78]
        getitem_4293: "f32[2304]" = _foreach_div_list[79]
        getitem_4294: "f32[768, 768]" = _foreach_div_list[80]
        getitem_4295: "f32[768]" = _foreach_div_list[81]
        getitem_4296: "f32[3072, 768]" = _foreach_div_list[82]
        getitem_4297: "f32[3072]" = _foreach_div_list[83]
        getitem_4298: "f32[768, 3072]" = _foreach_div_list[84]
        getitem_4299: "f32[768]" = _foreach_div_list[85]
        getitem_4300: "f32[768]" = _foreach_div_list[86]
        getitem_4301: "f32[768]" = _foreach_div_list[87]
        getitem_4302: "f32[768]" = _foreach_div_list[88]
        getitem_4303: "f32[768]" = _foreach_div_list[89]
        getitem_4304: "f32[2304, 768]" = _foreach_div_list[90]
        getitem_4305: "f32[2304]" = _foreach_div_list[91]
        getitem_4306: "f32[768, 768]" = _foreach_div_list[92]
        getitem_4307: "f32[768]" = _foreach_div_list[93]
        getitem_4308: "f32[3072, 768]" = _foreach_div_list[94]
        getitem_4309: "f32[3072]" = _foreach_div_list[95]
        getitem_4310: "f32[768, 3072]" = _foreach_div_list[96]
        getitem_4311: "f32[768]" = _foreach_div_list[97]
        getitem_4312: "f32[768]" = _foreach_div_list[98]
        getitem_4313: "f32[768]" = _foreach_div_list[99]
        getitem_4314: "f32[768]" = _foreach_div_list[100]
        getitem_4315: "f32[768]" = _foreach_div_list[101]
        getitem_4316: "f32[2304, 768]" = _foreach_div_list[102]
        getitem_4317: "f32[2304]" = _foreach_div_list[103]
        getitem_4318: "f32[768, 768]" = _foreach_div_list[104]
        getitem_4319: "f32[768]" = _foreach_div_list[105]
        getitem_4320: "f32[3072, 768]" = _foreach_div_list[106]
        getitem_4321: "f32[3072]" = _foreach_div_list[107]
        getitem_4322: "f32[768, 3072]" = _foreach_div_list[108]
        getitem_4323: "f32[768]" = _foreach_div_list[109]
        getitem_4324: "f32[768]" = _foreach_div_list[110]
        getitem_4325: "f32[768]" = _foreach_div_list[111]
        getitem_4326: "f32[768]" = _foreach_div_list[112]
        getitem_4327: "f32[768]" = _foreach_div_list[113]
        getitem_4328: "f32[2304, 768]" = _foreach_div_list[114]
        getitem_4329: "f32[2304]" = _foreach_div_list[115]
        getitem_4330: "f32[768, 768]" = _foreach_div_list[116]
        getitem_4331: "f32[768]" = _foreach_div_list[117]
        getitem_4332: "f32[3072, 768]" = _foreach_div_list[118]
        getitem_4333: "f32[3072]" = _foreach_div_list[119]
        getitem_4334: "f32[768, 3072]" = _foreach_div_list[120]
        getitem_4335: "f32[768]" = _foreach_div_list[121]
        getitem_4336: "f32[768]" = _foreach_div_list[122]
        getitem_4337: "f32[768]" = _foreach_div_list[123]
        getitem_4338: "f32[768]" = _foreach_div_list[124]
        getitem_4339: "f32[768]" = _foreach_div_list[125]
        getitem_4340: "f32[2304, 768]" = _foreach_div_list[126]
        getitem_4341: "f32[2304]" = _foreach_div_list[127]
        getitem_4342: "f32[768, 768]" = _foreach_div_list[128]
        getitem_4343: "f32[768]" = _foreach_div_list[129]
        getitem_4344: "f32[3072, 768]" = _foreach_div_list[130]
        getitem_4345: "f32[3072]" = _foreach_div_list[131]
        getitem_4346: "f32[768, 3072]" = _foreach_div_list[132]
        getitem_4347: "f32[768]" = _foreach_div_list[133]
        getitem_4348: "f32[768]" = _foreach_div_list[134]
        getitem_4349: "f32[768]" = _foreach_div_list[135]
        getitem_4350: "f32[768]" = _foreach_div_list[136]
        getitem_4351: "f32[768]" = _foreach_div_list[137]
        getitem_4352: "f32[2304, 768]" = _foreach_div_list[138]
        getitem_4353: "f32[2304]" = _foreach_div_list[139]
        getitem_4354: "f32[768, 768]" = _foreach_div_list[140]
        getitem_4355: "f32[768]" = _foreach_div_list[141]
        getitem_4356: "f32[3072, 768]" = _foreach_div_list[142]
        getitem_4357: "f32[3072]" = _foreach_div_list[143]
        getitem_4358: "f32[768, 3072]" = _foreach_div_list[144]
        getitem_4359: "f32[768]" = _foreach_div_list[145]
        getitem_4360: "f32[768]" = _foreach_div_list[146]
        getitem_4361: "f32[768]" = _foreach_div_list[147]
        getitem_4362: "f32[768]" = _foreach_div_list[148]
        getitem_4363: "f32[768]" = _foreach_div_list[149]
        getitem_4364: "f32[768]" = _foreach_div_list[150]
        getitem_4365: "f32[768]" = _foreach_div_list[151]
        getitem_4366: "f32[77, 512]" = _foreach_div_list[152]
        getitem_4367: "f32[49408, 512]" = _foreach_div_list[153]
        getitem_4368: "f32[1536, 512]" = _foreach_div_list[154]
        getitem_4369: "f32[1536]" = _foreach_div_list[155]
        getitem_4370: "f32[512, 512]" = _foreach_div_list[156]
        getitem_4371: "f32[512]" = _foreach_div_list[157]
        getitem_4372: "f32[2048, 512]" = _foreach_div_list[158]
        getitem_4373: "f32[2048]" = _foreach_div_list[159]
        getitem_4374: "f32[512, 2048]" = _foreach_div_list[160]
        getitem_4375: "f32[512]" = _foreach_div_list[161]
        getitem_4376: "f32[512]" = _foreach_div_list[162]
        getitem_4377: "f32[512]" = _foreach_div_list[163]
        getitem_4378: "f32[512]" = _foreach_div_list[164]
        getitem_4379: "f32[512]" = _foreach_div_list[165]
        getitem_4380: "f32[1536, 512]" = _foreach_div_list[166]
        getitem_4381: "f32[1536]" = _foreach_div_list[167]
        getitem_4382: "f32[512, 512]" = _foreach_div_list[168]
        getitem_4383: "f32[512]" = _foreach_div_list[169]
        getitem_4384: "f32[2048, 512]" = _foreach_div_list[170]
        getitem_4385: "f32[2048]" = _foreach_div_list[171]
        getitem_4386: "f32[512, 2048]" = _foreach_div_list[172]
        getitem_4387: "f32[512]" = _foreach_div_list[173]
        getitem_4388: "f32[512]" = _foreach_div_list[174]
        getitem_4389: "f32[512]" = _foreach_div_list[175]
        getitem_4390: "f32[512]" = _foreach_div_list[176]
        getitem_4391: "f32[512]" = _foreach_div_list[177]
        getitem_4392: "f32[1536, 512]" = _foreach_div_list[178]
        getitem_4393: "f32[1536]" = _foreach_div_list[179]
        getitem_4394: "f32[512, 512]" = _foreach_div_list[180]
        getitem_4395: "f32[512]" = _foreach_div_list[181]
        getitem_4396: "f32[2048, 512]" = _foreach_div_list[182]
        getitem_4397: "f32[2048]" = _foreach_div_list[183]
        getitem_4398: "f32[512, 2048]" = _foreach_div_list[184]
        getitem_4399: "f32[512]" = _foreach_div_list[185]
        getitem_4400: "f32[512]" = _foreach_div_list[186]
        getitem_4401: "f32[512]" = _foreach_div_list[187]
        getitem_4402: "f32[512]" = _foreach_div_list[188]
        getitem_4403: "f32[512]" = _foreach_div_list[189]
        getitem_4404: "f32[1536, 512]" = _foreach_div_list[190]
        getitem_4405: "f32[1536]" = _foreach_div_list[191]
        getitem_4406: "f32[512, 512]" = _foreach_div_list[192]
        getitem_4407: "f32[512]" = _foreach_div_list[193]
        getitem_4408: "f32[2048, 512]" = _foreach_div_list[194]
        getitem_4409: "f32[2048]" = _foreach_div_list[195]
        getitem_4410: "f32[512, 2048]" = _foreach_div_list[196]
        getitem_4411: "f32[512]" = _foreach_div_list[197]
        getitem_4412: "f32[512]" = _foreach_div_list[198]
        getitem_4413: "f32[512]" = _foreach_div_list[199]
        getitem_4414: "f32[512]" = _foreach_div_list[200]
        getitem_4415: "f32[512]" = _foreach_div_list[201]
        getitem_4416: "f32[1536, 512]" = _foreach_div_list[202]
        getitem_4417: "f32[1536]" = _foreach_div_list[203]
        getitem_4418: "f32[512, 512]" = _foreach_div_list[204]
        getitem_4419: "f32[512]" = _foreach_div_list[205]
        getitem_4420: "f32[2048, 512]" = _foreach_div_list[206]
        getitem_4421: "f32[2048]" = _foreach_div_list[207]
        getitem_4422: "f32[512, 2048]" = _foreach_div_list[208]
        getitem_4423: "f32[512]" = _foreach_div_list[209]
        getitem_4424: "f32[512]" = _foreach_div_list[210]
        getitem_4425: "f32[512]" = _foreach_div_list[211]
        getitem_4426: "f32[512]" = _foreach_div_list[212]
        getitem_4427: "f32[512]" = _foreach_div_list[213]
        getitem_4428: "f32[1536, 512]" = _foreach_div_list[214]
        getitem_4429: "f32[1536]" = _foreach_div_list[215]
        getitem_4430: "f32[512, 512]" = _foreach_div_list[216]
        getitem_4431: "f32[512]" = _foreach_div_list[217]
        getitem_4432: "f32[2048, 512]" = _foreach_div_list[218]
        getitem_4433: "f32[2048]" = _foreach_div_list[219]
        getitem_4434: "f32[512, 2048]" = _foreach_div_list[220]
        getitem_4435: "f32[512]" = _foreach_div_list[221]
        getitem_4436: "f32[512]" = _foreach_div_list[222]
        getitem_4437: "f32[512]" = _foreach_div_list[223]
        getitem_4438: "f32[512]" = _foreach_div_list[224]
        getitem_4439: "f32[512]" = _foreach_div_list[225]
        getitem_4440: "f32[1536, 512]" = _foreach_div_list[226]
        getitem_4441: "f32[1536]" = _foreach_div_list[227]
        getitem_4442: "f32[512, 512]" = _foreach_div_list[228]
        getitem_4443: "f32[512]" = _foreach_div_list[229]
        getitem_4444: "f32[2048, 512]" = _foreach_div_list[230]
        getitem_4445: "f32[2048]" = _foreach_div_list[231]
        getitem_4446: "f32[512, 2048]" = _foreach_div_list[232]
        getitem_4447: "f32[512]" = _foreach_div_list[233]
        getitem_4448: "f32[512]" = _foreach_div_list[234]
        getitem_4449: "f32[512]" = _foreach_div_list[235]
        getitem_4450: "f32[512]" = _foreach_div_list[236]
        getitem_4451: "f32[512]" = _foreach_div_list[237]
        getitem_4452: "f32[1536, 512]" = _foreach_div_list[238]
        getitem_4453: "f32[1536]" = _foreach_div_list[239]
        getitem_4454: "f32[512, 512]" = _foreach_div_list[240]
        getitem_4455: "f32[512]" = _foreach_div_list[241]
        getitem_4456: "f32[2048, 512]" = _foreach_div_list[242]
        getitem_4457: "f32[2048]" = _foreach_div_list[243]
        getitem_4458: "f32[512, 2048]" = _foreach_div_list[244]
        getitem_4459: "f32[512]" = _foreach_div_list[245]
        getitem_4460: "f32[512]" = _foreach_div_list[246]
        getitem_4461: "f32[512]" = _foreach_div_list[247]
        getitem_4462: "f32[512]" = _foreach_div_list[248]
        getitem_4463: "f32[512]" = _foreach_div_list[249]
        getitem_4464: "f32[1536, 512]" = _foreach_div_list[250]
        getitem_4465: "f32[1536]" = _foreach_div_list[251]
        getitem_4466: "f32[512, 512]" = _foreach_div_list[252]
        getitem_4467: "f32[512]" = _foreach_div_list[253]
        getitem_4468: "f32[2048, 512]" = _foreach_div_list[254]
        getitem_4469: "f32[2048]" = _foreach_div_list[255]
        getitem_4470: "f32[512, 2048]" = _foreach_div_list[256]
        getitem_4471: "f32[512]" = _foreach_div_list[257]
        getitem_4472: "f32[512]" = _foreach_div_list[258]
        getitem_4473: "f32[512]" = _foreach_div_list[259]
        getitem_4474: "f32[512]" = _foreach_div_list[260]
        getitem_4475: "f32[512]" = _foreach_div_list[261]
        getitem_4476: "f32[1536, 512]" = _foreach_div_list[262]
        getitem_4477: "f32[1536]" = _foreach_div_list[263]
        getitem_4478: "f32[512, 512]" = _foreach_div_list[264]
        getitem_4479: "f32[512]" = _foreach_div_list[265]
        getitem_4480: "f32[2048, 512]" = _foreach_div_list[266]
        getitem_4481: "f32[2048]" = _foreach_div_list[267]
        getitem_4482: "f32[512, 2048]" = _foreach_div_list[268]
        getitem_4483: "f32[512]" = _foreach_div_list[269]
        getitem_4484: "f32[512]" = _foreach_div_list[270]
        getitem_4485: "f32[512]" = _foreach_div_list[271]
        getitem_4486: "f32[512]" = _foreach_div_list[272]
        getitem_4487: "f32[512]" = _foreach_div_list[273]
        getitem_4488: "f32[1536, 512]" = _foreach_div_list[274]
        getitem_4489: "f32[1536]" = _foreach_div_list[275]
        getitem_4490: "f32[512, 512]" = _foreach_div_list[276]
        getitem_4491: "f32[512]" = _foreach_div_list[277]
        getitem_4492: "f32[2048, 512]" = _foreach_div_list[278]
        getitem_4493: "f32[2048]" = _foreach_div_list[279]
        getitem_4494: "f32[512, 2048]" = _foreach_div_list[280]
        getitem_4495: "f32[512]" = _foreach_div_list[281]
        getitem_4496: "f32[512]" = _foreach_div_list[282]
        getitem_4497: "f32[512]" = _foreach_div_list[283]
        getitem_4498: "f32[512]" = _foreach_div_list[284]
        getitem_4499: "f32[512]" = _foreach_div_list[285]
        getitem_4500: "f32[1536, 512]" = _foreach_div_list[286]
        getitem_4501: "f32[1536]" = _foreach_div_list[287]
        getitem_4502: "f32[512, 512]" = _foreach_div_list[288]
        getitem_4503: "f32[512]" = _foreach_div_list[289]
        getitem_4504: "f32[2048, 512]" = _foreach_div_list[290]
        getitem_4505: "f32[2048]" = _foreach_div_list[291]
        getitem_4506: "f32[512, 2048]" = _foreach_div_list[292]
        getitem_4507: "f32[512]" = _foreach_div_list[293]
        getitem_4508: "f32[512]" = _foreach_div_list[294]
        getitem_4509: "f32[512]" = _foreach_div_list[295]
        getitem_4510: "f32[512]" = _foreach_div_list[296]
        getitem_4511: "f32[512]" = _foreach_div_list[297]
        getitem_4512: "f32[512]" = _foreach_div_list[298]
        getitem_4513: "f32[512]" = _foreach_div_list[299]
        getitem_4514: "f32[512, 512]" = _foreach_div_list[300];  _foreach_div_list = None
        return (getitem, getitem_2408, getitem_2409, getitem_2410, getitem_2411, getitem_2412, getitem_2413, getitem_2414, getitem_2415, getitem_2416, getitem_2417, getitem_2418, getitem_2419, getitem_2420, getitem_2421, getitem_2422, getitem_2423, getitem_2424, getitem_2425, getitem_2426, getitem_2427, getitem_2428, getitem_2429, getitem_2430, getitem_2431, getitem_2432, getitem_2433, getitem_2434, getitem_2435, getitem_2436, getitem_2437, getitem_2438, getitem_2439, getitem_2440, getitem_2441, getitem_2442, getitem_2443, getitem_2444, getitem_2445, getitem_2446, getitem_2447, getitem_2448, getitem_2449, getitem_2450, getitem_2451, getitem_2452, getitem_2453, getitem_2454, getitem_2455, getitem_2456, getitem_2457, getitem_2458, getitem_2459, getitem_2460, getitem_2461, getitem_2462, getitem_2463, getitem_2464, getitem_2465, getitem_2466, getitem_2467, getitem_2468, getitem_2469, getitem_2470, getitem_2471, getitem_2472, getitem_2473, getitem_2474, getitem_2475, getitem_2476, getitem_2477, getitem_2478, getitem_2479, getitem_2480, getitem_2481, getitem_2482, getitem_2483, getitem_2484, getitem_2485, getitem_2486, getitem_2487, getitem_2488, getitem_2489, getitem_2490, getitem_2491, getitem_2492, getitem_2493, getitem_2494, getitem_2495, getitem_2496, getitem_2497, getitem_2498, getitem_2499, getitem_2500, getitem_2501, getitem_2502, getitem_2503, getitem_2504, getitem_2505, getitem_2506, getitem_2507, getitem_2508, getitem_2509, getitem_2510, getitem_2511, getitem_2512, getitem_2513, getitem_2514, getitem_2515, getitem_2516, getitem_2517, getitem_2518, getitem_2519, getitem_2520, getitem_2521, getitem_2522, getitem_2523, getitem_2524, getitem_2525, getitem_2526, getitem_2527, getitem_2528, getitem_2529, getitem_2530, getitem_2531, getitem_2532, getitem_2533, getitem_2534, getitem_2535, getitem_2536, getitem_2537, getitem_2538, getitem_2539, getitem_2540, getitem_2541, getitem_2542, getitem_2543, getitem_2544, getitem_2545, getitem_2546, getitem_2547, getitem_2548, getitem_2549, getitem_2550, getitem_2551, getitem_2552, getitem_2553, getitem_2554, getitem_2555, getitem_2556, getitem_2557, getitem_2558, getitem_2559, getitem_2560, getitem_2561, getitem_2562, getitem_2563, getitem_2564, getitem_2565, getitem_2566, getitem_2567, getitem_2568, getitem_2569, getitem_2570, getitem_2571, getitem_2572, getitem_2573, getitem_2574, getitem_2575, getitem_2576, getitem_2577, getitem_2578, getitem_2579, getitem_2580, getitem_2581, getitem_2582, getitem_2583, getitem_2584, getitem_2585, getitem_2586, getitem_2587, getitem_2588, getitem_2589, getitem_2590, getitem_2591, getitem_2592, getitem_2593, getitem_2594, getitem_2595, getitem_2596, getitem_2597, getitem_2598, getitem_2599, getitem_2600, getitem_2601, getitem_2602, getitem_2603, getitem_2604, getitem_2605, getitem_2606, getitem_2607, getitem_2608, getitem_2609, getitem_2610, getitem_2611, getitem_2612, getitem_2613, getitem_2614, getitem_2615, getitem_2616, getitem_2617, getitem_2618, getitem_2619, getitem_2620, getitem_2621, getitem_2622, getitem_2623, getitem_2624, getitem_2625, getitem_2626, getitem_2627, getitem_2628, getitem_2629, getitem_2630, getitem_2631, getitem_2632, getitem_2633, getitem_2634, getitem_2635, getitem_2636, getitem_2637, getitem_2638, getitem_2639, getitem_2640, getitem_2641, getitem_2642, getitem_2643, getitem_2644, getitem_2645, getitem_2646, getitem_2647, getitem_2648, getitem_2649, getitem_2650, getitem_2651, getitem_2652, getitem_2653, getitem_2654, getitem_2655, getitem_2656, getitem_2657, getitem_2658, getitem_2659, getitem_2660, getitem_2661, getitem_2662, getitem_2663, getitem_2664, getitem_2665, getitem_2666, getitem_2667, getitem_2668, getitem_2669, getitem_2670, getitem_2671, getitem_2672, getitem_2673, getitem_2674, getitem_2675, getitem_2676, getitem_2677, getitem_2678, getitem_2679, getitem_2680, getitem_2681, getitem_2682, getitem_2683, getitem_2684, getitem_2685, getitem_2686, getitem_2687, getitem_2688, getitem_2689, getitem_2690, getitem_2691, getitem_2692, getitem_2693, getitem_2694, getitem_2695, getitem_2696, getitem_2697, getitem_2698, getitem_2699, getitem_2700, getitem_2701, getitem_2702, getitem_2703, getitem_2704, getitem_2705, getitem_2706, getitem_2707, getitem_4214, getitem_4215, getitem_4216, getitem_4217, getitem_4218, getitem_4219, getitem_4220, getitem_4221, getitem_4222, getitem_4223, getitem_4224, getitem_4225, getitem_4226, getitem_4227, getitem_4228, getitem_4229, getitem_4230, getitem_4231, getitem_4232, getitem_4233, getitem_4234, getitem_4235, getitem_4236, getitem_4237, getitem_4238, getitem_4239, getitem_4240, getitem_4241, getitem_4242, getitem_4243, getitem_4244, getitem_4245, getitem_4246, getitem_4247, getitem_4248, getitem_4249, getitem_4250, getitem_4251, getitem_4252, getitem_4253, getitem_4254, getitem_4255, getitem_4256, getitem_4257, getitem_4258, getitem_4259, getitem_4260, getitem_4261, getitem_4262, getitem_4263, getitem_4264, getitem_4265, getitem_4266, getitem_4267, getitem_4268, getitem_4269, getitem_4270, getitem_4271, getitem_4272, getitem_4273, getitem_4274, getitem_4275, getitem_4276, getitem_4277, getitem_4278, getitem_4279, getitem_4280, getitem_4281, getitem_4282, getitem_4283, getitem_4284, getitem_4285, getitem_4286, getitem_4287, getitem_4288, getitem_4289, getitem_4290, getitem_4291, getitem_4292, getitem_4293, getitem_4294, getitem_4295, getitem_4296, getitem_4297, getitem_4298, getitem_4299, getitem_4300, getitem_4301, getitem_4302, getitem_4303, getitem_4304, getitem_4305, getitem_4306, getitem_4307, getitem_4308, getitem_4309, getitem_4310, getitem_4311, getitem_4312, getitem_4313, getitem_4314, getitem_4315, getitem_4316, getitem_4317, getitem_4318, getitem_4319, getitem_4320, getitem_4321, getitem_4322, getitem_4323, getitem_4324, getitem_4325, getitem_4326, getitem_4327, getitem_4328, getitem_4329, getitem_4330, getitem_4331, getitem_4332, getitem_4333, getitem_4334, getitem_4335, getitem_4336, getitem_4337, getitem_4338, getitem_4339, getitem_4340, getitem_4341, getitem_4342, getitem_4343, getitem_4344, getitem_4345, getitem_4346, getitem_4347, getitem_4348, getitem_4349, getitem_4350, getitem_4351, getitem_4352, getitem_4353, getitem_4354, getitem_4355, getitem_4356, getitem_4357, getitem_4358, getitem_4359, getitem_4360, getitem_4361, getitem_4362, getitem_4363, getitem_4364, getitem_4365, getitem_4366, getitem_4367, getitem_4368, getitem_4369, getitem_4370, getitem_4371, getitem_4372, getitem_4373, getitem_4374, getitem_4375, getitem_4376, getitem_4377, getitem_4378, getitem_4379, getitem_4380, getitem_4381, getitem_4382, getitem_4383, getitem_4384, getitem_4385, getitem_4386, getitem_4387, getitem_4388, getitem_4389, getitem_4390, getitem_4391, getitem_4392, getitem_4393, getitem_4394, getitem_4395, getitem_4396, getitem_4397, getitem_4398, getitem_4399, getitem_4400, getitem_4401, getitem_4402, getitem_4403, getitem_4404, getitem_4405, getitem_4406, getitem_4407, getitem_4408, getitem_4409, getitem_4410, getitem_4411, getitem_4412, getitem_4413, getitem_4414, getitem_4415, getitem_4416, getitem_4417, getitem_4418, getitem_4419, getitem_4420, getitem_4421, getitem_4422, getitem_4423, getitem_4424, getitem_4425, getitem_4426, getitem_4427, getitem_4428, getitem_4429, getitem_4430, getitem_4431, getitem_4432, getitem_4433, getitem_4434, getitem_4435, getitem_4436, getitem_4437, getitem_4438, getitem_4439, getitem_4440, getitem_4441, getitem_4442, getitem_4443, getitem_4444, getitem_4445, getitem_4446, getitem_4447, getitem_4448, getitem_4449, getitem_4450, getitem_4451, getitem_4452, getitem_4453, getitem_4454, getitem_4455, getitem_4456, getitem_4457, getitem_4458, getitem_4459, getitem_4460, getitem_4461, getitem_4462, getitem_4463, getitem_4464, getitem_4465, getitem_4466, getitem_4467, getitem_4468, getitem_4469, getitem_4470, getitem_4471, getitem_4472, getitem_4473, getitem_4474, getitem_4475, getitem_4476, getitem_4477, getitem_4478, getitem_4479, getitem_4480, getitem_4481, getitem_4482, getitem_4483, getitem_4484, getitem_4485, getitem_4486, getitem_4487, getitem_4488, getitem_4489, getitem_4490, getitem_4491, getitem_4492, getitem_4493, getitem_4494, getitem_4495, getitem_4496, getitem_4497, getitem_4498, getitem_4499, getitem_4500, getitem_4501, getitem_4502, getitem_4503, getitem_4504, getitem_4505, getitem_4506, getitem_4507, getitem_4508, getitem_4509, getitem_4510, getitem_4511, getitem_4512, getitem_4513, getitem_4514)


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
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
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
