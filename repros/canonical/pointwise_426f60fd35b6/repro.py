"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s3_g60
Pattern hash: 426f60fd35b6
Shape hash: 869e899c
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg209_1: "f32[30522, 768]", arg207_1: "f32[512, 768]", arg416_1: "f32[1024, 768]", arg417_1: "f32[1024, 768]", arg418_1: "f32[1024, 768]", arg419_1: "f32[1024, 768]", arg420_1: "f32[2, 768]", arg421_1: "f32[768]", arg422_1: "f32[768]", arg423_1: "f32[768, 768]", arg424_1: "f32[768]", arg425_1: "f32[768, 768]", arg426_1: "f32[768]", arg427_1: "f32[768, 768]", arg428_1: "f32[768]", arg429_1: "f32[768, 768]", arg430_1: "f32[768]", arg431_1: "f32[768]", arg432_1: "f32[768]", arg433_1: "f32[3072, 768]", arg434_1: "f32[3072]", arg435_1: "f32[768, 3072]", arg436_1: "f32[768]", arg437_1: "f32[768]", arg438_1: "f32[768]", arg439_1: "f32[768, 768]", arg440_1: "f32[768]", arg441_1: "f32[768, 768]", arg442_1: "f32[768]", arg443_1: "f32[768, 768]", arg444_1: "f32[768]", arg445_1: "f32[768, 768]", arg446_1: "f32[768]", arg447_1: "f32[768]", arg448_1: "f32[768]", arg449_1: "f32[3072, 768]", arg450_1: "f32[3072]", arg451_1: "f32[768, 3072]", arg452_1: "f32[768]", arg453_1: "f32[768]", arg454_1: "f32[768]", arg455_1: "f32[768, 768]", arg456_1: "f32[768]", arg457_1: "f32[768, 768]", arg458_1: "f32[768]", arg459_1: "f32[768, 768]", arg460_1: "f32[768]", arg461_1: "f32[768, 768]", arg462_1: "f32[768]", arg463_1: "f32[768]", arg464_1: "f32[768]", arg465_1: "f32[3072, 768]", arg466_1: "f32[3072]", arg467_1: "f32[768, 3072]", arg468_1: "f32[768]", arg469_1: "f32[768]", arg470_1: "f32[768]", arg471_1: "f32[768, 768]", arg472_1: "f32[768]", arg473_1: "f32[768, 768]", arg474_1: "f32[768]", arg475_1: "f32[768, 768]", arg476_1: "f32[768]", arg477_1: "f32[768, 768]", arg478_1: "f32[768]", arg479_1: "f32[768]", arg480_1: "f32[768]", arg481_1: "f32[3072, 768]", arg482_1: "f32[3072]", arg483_1: "f32[768, 3072]", arg484_1: "f32[768]", arg485_1: "f32[768]", arg486_1: "f32[768]", arg487_1: "f32[768, 768]", arg488_1: "f32[768]", arg489_1: "f32[768, 768]", arg490_1: "f32[768]", arg491_1: "f32[768, 768]", arg492_1: "f32[768]", arg493_1: "f32[768, 768]", arg494_1: "f32[768]", arg495_1: "f32[768]", arg496_1: "f32[768]", arg497_1: "f32[3072, 768]", arg498_1: "f32[3072]", arg499_1: "f32[768, 3072]", arg500_1: "f32[768]", arg501_1: "f32[768]", arg502_1: "f32[768]", arg503_1: "f32[768, 768]", arg504_1: "f32[768]", arg505_1: "f32[768, 768]", arg506_1: "f32[768]", arg507_1: "f32[768, 768]", arg508_1: "f32[768]", arg509_1: "f32[768, 768]", arg510_1: "f32[768]", arg511_1: "f32[768]", arg512_1: "f32[768]", arg513_1: "f32[3072, 768]", arg514_1: "f32[3072]", arg515_1: "f32[768, 3072]", arg516_1: "f32[768]", arg517_1: "f32[768]", arg518_1: "f32[768]", arg519_1: "f32[768, 768]", arg520_1: "f32[768]", arg521_1: "f32[768, 768]", arg522_1: "f32[768]", arg523_1: "f32[768, 768]", arg524_1: "f32[768]", arg525_1: "f32[768, 768]", arg526_1: "f32[768]", arg527_1: "f32[768]", arg528_1: "f32[768]", arg529_1: "f32[3072, 768]", arg530_1: "f32[3072]", arg531_1: "f32[768, 3072]", arg532_1: "f32[768]", arg533_1: "f32[768]", arg534_1: "f32[768]", arg535_1: "f32[768, 768]", arg536_1: "f32[768]", arg537_1: "f32[768, 768]", arg538_1: "f32[768]", arg539_1: "f32[768, 768]", arg540_1: "f32[768]", arg541_1: "f32[768, 768]", arg542_1: "f32[768]", arg543_1: "f32[768]", arg544_1: "f32[768]", arg545_1: "f32[3072, 768]", arg546_1: "f32[3072]", arg547_1: "f32[768, 3072]", arg548_1: "f32[768]", arg549_1: "f32[768]", arg550_1: "f32[768]", arg551_1: "f32[768, 768]", arg552_1: "f32[768]", arg553_1: "f32[768, 768]", arg554_1: "f32[768]", arg555_1: "f32[768, 768]", arg556_1: "f32[768]", arg557_1: "f32[768, 768]", arg558_1: "f32[768]", arg559_1: "f32[768]", arg560_1: "f32[768]", arg561_1: "f32[3072, 768]", arg562_1: "f32[3072]", arg563_1: "f32[768, 3072]", arg564_1: "f32[768]", arg565_1: "f32[768]", arg566_1: "f32[768]", arg567_1: "f32[768, 768]", arg568_1: "f32[768]", arg569_1: "f32[768, 768]", arg570_1: "f32[768]", arg571_1: "f32[768, 768]", arg572_1: "f32[768]", arg573_1: "f32[768, 768]", arg574_1: "f32[768]", arg575_1: "f32[768]", arg576_1: "f32[768]", arg577_1: "f32[3072, 768]", arg578_1: "f32[3072]", arg579_1: "f32[768, 3072]", arg580_1: "f32[768]", arg581_1: "f32[768]", arg582_1: "f32[768]", arg583_1: "f32[768, 768]", arg584_1: "f32[768]", arg585_1: "f32[768, 768]", arg586_1: "f32[768]", arg587_1: "f32[768, 768]", arg588_1: "f32[768]", arg589_1: "f32[768, 768]", arg590_1: "f32[768]", arg591_1: "f32[768]", arg592_1: "f32[768]", arg593_1: "f32[3072, 768]", arg594_1: "f32[3072]", arg595_1: "f32[768, 3072]", arg596_1: "f32[768]", arg597_1: "f32[768]", arg598_1: "f32[768]", arg599_1: "f32[768, 768]", arg600_1: "f32[768]", arg601_1: "f32[768, 768]", arg602_1: "f32[768]", arg603_1: "f32[768, 768]", arg604_1: "f32[768]", arg605_1: "f32[768, 768]", arg606_1: "f32[768]", arg607_1: "f32[768]", arg608_1: "f32[768]", arg609_1: "f32[3072, 768]", arg610_1: "f32[3072]", arg611_1: "f32[768, 3072]", arg612_1: "f32[768]", arg613_1: "f32[768]", arg614_1: "f32[768]", arg615_1: "f32[30522]", arg616_1: "f32[768, 768]", arg617_1: "f32[768]", arg618_1: "f32[768]", arg619_1: "f32[768]", getitem_206: "f32[30522, 768]", getitem_207: "f32[512, 768]", getitem_208: "f32[1024, 768]", getitem_209: "f32[1024, 768]", getitem_210: "f32[1024, 768]", getitem_211: "f32[1024, 768]", getitem_212: "f32[2, 768]", getitem_213: "f32[768]", getitem_214: "f32[768]", getitem_215: "f32[768, 768]", getitem_216: "f32[768]", getitem_217: "f32[768, 768]", getitem_218: "f32[768]", getitem_219: "f32[768, 768]", getitem_220: "f32[768]", getitem_221: "f32[768, 768]", getitem_222: "f32[768]", getitem_223: "f32[768]", getitem_224: "f32[768]", getitem_225: "f32[3072, 768]", getitem_226: "f32[3072]", getitem_227: "f32[768, 3072]", getitem_228: "f32[768]", getitem_229: "f32[768]", getitem_230: "f32[768]", getitem_231: "f32[768, 768]", getitem_232: "f32[768]", getitem_233: "f32[768, 768]", getitem_234: "f32[768]", getitem_235: "f32[768, 768]", getitem_236: "f32[768]", getitem_237: "f32[768, 768]", getitem_238: "f32[768]", getitem_239: "f32[768]", getitem_240: "f32[768]", getitem_241: "f32[3072, 768]", getitem_242: "f32[3072]", getitem_243: "f32[768, 3072]", getitem_244: "f32[768]", getitem_245: "f32[768]", getitem_246: "f32[768]", getitem_247: "f32[768, 768]", getitem_248: "f32[768]", getitem_249: "f32[768, 768]", getitem_250: "f32[768]", getitem_251: "f32[768, 768]", getitem_252: "f32[768]", getitem_253: "f32[768, 768]", getitem_254: "f32[768]", getitem_255: "f32[768]", getitem_256: "f32[768]", getitem_257: "f32[3072, 768]", getitem_258: "f32[3072]", getitem_259: "f32[768, 3072]", getitem_260: "f32[768]", getitem_261: "f32[768]", getitem_262: "f32[768]", getitem_263: "f32[768, 768]", getitem_264: "f32[768]", getitem_265: "f32[768, 768]", getitem_266: "f32[768]", getitem_267: "f32[768, 768]", getitem_268: "f32[768]", getitem_269: "f32[768, 768]", getitem_270: "f32[768]", getitem_271: "f32[768]", getitem_272: "f32[768]", getitem_273: "f32[3072, 768]", getitem_274: "f32[3072]", getitem_275: "f32[768, 3072]", getitem_276: "f32[768]", getitem_277: "f32[768]", getitem_278: "f32[768]", getitem_279: "f32[768, 768]", getitem_280: "f32[768]", getitem_281: "f32[768, 768]", getitem_282: "f32[768]", getitem_283: "f32[768, 768]", getitem_284: "f32[768]", getitem_285: "f32[768, 768]", getitem_286: "f32[768]", getitem_287: "f32[768]", getitem_288: "f32[768]", getitem_289: "f32[3072, 768]", getitem_290: "f32[3072]", getitem_291: "f32[768, 3072]", getitem_292: "f32[768]", getitem_293: "f32[768]", getitem_294: "f32[768]", getitem_295: "f32[768, 768]", getitem_296: "f32[768]", getitem_297: "f32[768, 768]", getitem_298: "f32[768]", getitem_299: "f32[768, 768]", getitem_300: "f32[768]", getitem_301: "f32[768, 768]", getitem_302: "f32[768]", getitem_303: "f32[768]", getitem_304: "f32[768]", getitem_305: "f32[3072, 768]", getitem_306: "f32[3072]", getitem_307: "f32[768, 3072]", getitem_308: "f32[768]", getitem_309: "f32[768]", getitem_310: "f32[768]", getitem_311: "f32[768, 768]", getitem_312: "f32[768]", getitem_313: "f32[768, 768]", getitem_314: "f32[768]", getitem_315: "f32[768, 768]", getitem_316: "f32[768]", getitem_317: "f32[768, 768]", getitem_318: "f32[768]", getitem_319: "f32[768]", getitem_320: "f32[768]", getitem_321: "f32[3072, 768]", getitem_322: "f32[3072]", getitem_323: "f32[768, 3072]", getitem_324: "f32[768]", getitem_325: "f32[768]", getitem_326: "f32[768]", getitem_327: "f32[768, 768]", getitem_328: "f32[768]", getitem_329: "f32[768, 768]", getitem_330: "f32[768]", getitem_331: "f32[768, 768]", getitem_332: "f32[768]", getitem_333: "f32[768, 768]", getitem_334: "f32[768]", getitem_335: "f32[768]", getitem_336: "f32[768]", getitem_337: "f32[3072, 768]", getitem_338: "f32[3072]", getitem_339: "f32[768, 3072]", getitem_340: "f32[768]", getitem_341: "f32[768]", getitem_342: "f32[768]", getitem_343: "f32[768, 768]", getitem_344: "f32[768]", getitem_345: "f32[768, 768]", getitem_346: "f32[768]", getitem_347: "f32[768, 768]", getitem_348: "f32[768]", getitem_349: "f32[768, 768]", getitem_350: "f32[768]", getitem_351: "f32[768]", getitem_352: "f32[768]", getitem_353: "f32[3072, 768]", getitem_354: "f32[3072]", getitem_355: "f32[768, 3072]", getitem_356: "f32[768]", getitem_357: "f32[768]", getitem_358: "f32[768]", getitem_359: "f32[768, 768]", getitem_360: "f32[768]", getitem_361: "f32[768, 768]", getitem_362: "f32[768]", getitem_363: "f32[768, 768]", getitem_364: "f32[768]", getitem_365: "f32[768, 768]", getitem_366: "f32[768]", getitem_367: "f32[768]", getitem_368: "f32[768]", getitem_369: "f32[3072, 768]", getitem_370: "f32[3072]", getitem_371: "f32[768, 3072]", getitem_372: "f32[768]", getitem_373: "f32[768]", getitem_374: "f32[768]", getitem_375: "f32[768, 768]", getitem_376: "f32[768]", getitem_377: "f32[768, 768]", getitem_378: "f32[768]", getitem_379: "f32[768, 768]", getitem_380: "f32[768]", getitem_381: "f32[768, 768]", getitem_382: "f32[768]", getitem_383: "f32[768]", getitem_384: "f32[768]", getitem_385: "f32[3072, 768]", getitem_386: "f32[3072]", getitem_387: "f32[768, 3072]", getitem_388: "f32[768]", getitem_389: "f32[768]", getitem_390: "f32[768]", getitem_391: "f32[768, 768]", getitem_392: "f32[768]", getitem_393: "f32[768, 768]", getitem_394: "f32[768]", getitem_395: "f32[768, 768]", getitem_396: "f32[768]", getitem_397: "f32[768, 768]", getitem_398: "f32[768]", getitem_399: "f32[768]", getitem_400: "f32[768]", getitem_401: "f32[3072, 768]", getitem_402: "f32[3072]", getitem_403: "f32[768, 3072]", getitem_404: "f32[768]", getitem_405: "f32[768]", getitem_406: "f32[768]", getitem_407: "f32[30522]", getitem_408: "f32[768, 768]", getitem_409: "f32[768]", getitem_410: "f32[768]", getitem_411: "f32[768]", getitem_3090: "f32[30522, 768]", getitem_3091: "f32[512, 768]", getitem_3092: "f32[1024, 768]", getitem_3093: "f32[1024, 768]", getitem_3094: "f32[1024, 768]", getitem_3095: "f32[1024, 768]", getitem_3096: "f32[2, 768]", getitem_3097: "f32[768]", getitem_3098: "f32[768]", getitem_3099: "f32[768, 768]", getitem_3100: "f32[768]", getitem_3101: "f32[768, 768]", getitem_3102: "f32[768]", getitem_3103: "f32[768, 768]", getitem_3104: "f32[768]", getitem_3105: "f32[768, 768]", getitem_3106: "f32[768]", getitem_3107: "f32[768]", getitem_3108: "f32[768]", getitem_3109: "f32[3072, 768]", getitem_3110: "f32[3072]", getitem_3111: "f32[768, 3072]", getitem_3112: "f32[768]", getitem_3113: "f32[768]", getitem_3114: "f32[768]", getitem_3115: "f32[768, 768]", getitem_3116: "f32[768]", getitem_3117: "f32[768, 768]", getitem_3118: "f32[768]", getitem_3119: "f32[768, 768]", getitem_3120: "f32[768]", getitem_3121: "f32[768, 768]", getitem_3122: "f32[768]", getitem_3123: "f32[768]", getitem_3124: "f32[768]", getitem_3125: "f32[3072, 768]", getitem_3126: "f32[3072]", getitem_3127: "f32[768, 3072]", getitem_3128: "f32[768]", getitem_3129: "f32[768]", getitem_3130: "f32[768]", getitem_3131: "f32[768, 768]", getitem_3132: "f32[768]", getitem_3133: "f32[768, 768]", getitem_3134: "f32[768]", getitem_3135: "f32[768, 768]", getitem_3136: "f32[768]", getitem_3137: "f32[768, 768]", getitem_3138: "f32[768]", getitem_3139: "f32[768]", getitem_3140: "f32[768]", getitem_3141: "f32[3072, 768]", getitem_3142: "f32[3072]", getitem_3143: "f32[768, 3072]", getitem_3144: "f32[768]", getitem_3145: "f32[768]", getitem_3146: "f32[768]", getitem_3147: "f32[768, 768]", getitem_3148: "f32[768]", getitem_3149: "f32[768, 768]", getitem_3150: "f32[768]", getitem_3151: "f32[768, 768]", getitem_3152: "f32[768]", getitem_3153: "f32[768, 768]", getitem_3154: "f32[768]", getitem_3155: "f32[768]", getitem_3156: "f32[768]", getitem_3157: "f32[3072, 768]", getitem_3158: "f32[3072]", getitem_3159: "f32[768, 3072]", getitem_3160: "f32[768]", getitem_3161: "f32[768]", getitem_3162: "f32[768]", getitem_3163: "f32[768, 768]", getitem_3164: "f32[768]", getitem_3165: "f32[768, 768]", getitem_3166: "f32[768]", getitem_3167: "f32[768, 768]", getitem_3168: "f32[768]", getitem_3169: "f32[768, 768]", getitem_3170: "f32[768]", getitem_3171: "f32[768]", getitem_3172: "f32[768]", getitem_3173: "f32[3072, 768]", getitem_3174: "f32[3072]", getitem_3175: "f32[768, 3072]", getitem_3176: "f32[768]", getitem_3177: "f32[768]", getitem_3178: "f32[768]", getitem_3179: "f32[768, 768]", getitem_3180: "f32[768]", getitem_3181: "f32[768, 768]", getitem_3182: "f32[768]", getitem_3183: "f32[768, 768]", getitem_3184: "f32[768]", getitem_3185: "f32[768, 768]", getitem_3186: "f32[768]", getitem_3187: "f32[768]", getitem_3188: "f32[768]", getitem_3189: "f32[3072, 768]", getitem_3190: "f32[3072]", getitem_3191: "f32[768, 3072]", getitem_3192: "f32[768]", getitem_3193: "f32[768]", getitem_3194: "f32[768]", getitem_3195: "f32[768, 768]", getitem_3196: "f32[768]", getitem_3197: "f32[768, 768]", getitem_3198: "f32[768]", getitem_3199: "f32[768, 768]", getitem_3200: "f32[768]", getitem_3201: "f32[768, 768]", getitem_3202: "f32[768]", getitem_3203: "f32[768]", getitem_3204: "f32[768]", getitem_3205: "f32[3072, 768]", getitem_3206: "f32[3072]", getitem_3207: "f32[768, 3072]", getitem_3208: "f32[768]", getitem_3209: "f32[768]", getitem_3210: "f32[768]", getitem_3211: "f32[768, 768]", getitem_3212: "f32[768]", getitem_3213: "f32[768, 768]", getitem_3214: "f32[768]", getitem_3215: "f32[768, 768]", getitem_3216: "f32[768]", getitem_3217: "f32[768, 768]", getitem_3218: "f32[768]", getitem_3219: "f32[768]", getitem_3220: "f32[768]", getitem_3221: "f32[3072, 768]", getitem_3222: "f32[3072]", getitem_3223: "f32[768, 3072]", getitem_3224: "f32[768]", getitem_3225: "f32[768]", getitem_3226: "f32[768]", getitem_3227: "f32[768, 768]", getitem_3228: "f32[768]", getitem_3229: "f32[768, 768]", getitem_3230: "f32[768]", getitem_3231: "f32[768, 768]", getitem_3232: "f32[768]", getitem_3233: "f32[768, 768]", getitem_3234: "f32[768]", getitem_3235: "f32[768]", getitem_3236: "f32[768]", getitem_3237: "f32[3072, 768]", getitem_3238: "f32[3072]", getitem_3239: "f32[768, 3072]", getitem_3240: "f32[768]", getitem_3241: "f32[768]", getitem_3242: "f32[768]", getitem_3243: "f32[768, 768]", getitem_3244: "f32[768]", getitem_3245: "f32[768, 768]", getitem_3246: "f32[768]", getitem_3247: "f32[768, 768]", getitem_3248: "f32[768]", getitem_3249: "f32[768, 768]", getitem_3250: "f32[768]", getitem_3251: "f32[768]", getitem_3252: "f32[768]", getitem_3253: "f32[3072, 768]", getitem_3254: "f32[3072]", getitem_3255: "f32[768, 3072]", getitem_3256: "f32[768]", getitem_3257: "f32[768]", getitem_3258: "f32[768]", getitem_3259: "f32[768, 768]", getitem_3260: "f32[768]", getitem_3261: "f32[768, 768]", getitem_3262: "f32[768]", getitem_3263: "f32[768, 768]", getitem_3264: "f32[768]", getitem_3265: "f32[768, 768]", getitem_3266: "f32[768]", getitem_3267: "f32[768]", getitem_3268: "f32[768]", getitem_3269: "f32[3072, 768]", getitem_3270: "f32[3072]", getitem_3271: "f32[768, 3072]", getitem_3272: "f32[768]", getitem_3273: "f32[768]", getitem_3274: "f32[768]", getitem_3275: "f32[768, 768]", getitem_3276: "f32[768]", getitem_3277: "f32[768, 768]", getitem_3278: "f32[768]", getitem_3279: "f32[768, 768]", getitem_3280: "f32[768]", getitem_3281: "f32[768, 768]", getitem_3282: "f32[768]", getitem_3283: "f32[768]", getitem_3284: "f32[768]", getitem_3285: "f32[3072, 768]", getitem_3286: "f32[3072]", getitem_3287: "f32[768, 3072]", getitem_3288: "f32[768]", getitem_3289: "f32[768]", getitem_3290: "f32[768]", getitem_3291: "f32[30522]", getitem_3292: "f32[768, 768]", getitem_3293: "f32[768]", getitem_3294: "f32[768]", getitem_3295: "f32[768]", getitem_2266: "f32[]", getitem_2267: "f32[]", getitem_2268: "f32[]", getitem_2269: "f32[]", getitem_2270: "f32[]", getitem_2271: "f32[]", getitem_2272: "f32[]", getitem_2273: "f32[]", getitem_2274: "f32[]", getitem_2275: "f32[]", getitem_2276: "f32[]", getitem_2277: "f32[]", getitem_2278: "f32[]", getitem_2279: "f32[]", getitem_2280: "f32[]", getitem_2281: "f32[]", getitem_2282: "f32[]", getitem_2283: "f32[]", getitem_2284: "f32[]", getitem_2285: "f32[]", getitem_2286: "f32[]", getitem_2287: "f32[]", getitem_2288: "f32[]", getitem_2289: "f32[]", getitem_2290: "f32[]", getitem_2291: "f32[]", getitem_2292: "f32[]", getitem_2293: "f32[]", getitem_2294: "f32[]", getitem_2295: "f32[]", getitem_2296: "f32[]", getitem_2297: "f32[]", getitem_2298: "f32[]", getitem_2299: "f32[]", getitem_2300: "f32[]", getitem_2301: "f32[]", getitem_2302: "f32[]", getitem_2303: "f32[]", getitem_2304: "f32[]", getitem_2305: "f32[]", getitem_2306: "f32[]", getitem_2307: "f32[]", getitem_2308: "f32[]", getitem_2309: "f32[]", getitem_2310: "f32[]", getitem_2311: "f32[]", getitem_2312: "f32[]", getitem_2313: "f32[]", getitem_2314: "f32[]", getitem_2315: "f32[]", getitem_2316: "f32[]", getitem_2317: "f32[]", getitem_2318: "f32[]", getitem_2319: "f32[]", getitem_2320: "f32[]", getitem_2321: "f32[]", getitem_2322: "f32[]", getitem_2323: "f32[]", getitem_2324: "f32[]", getitem_2325: "f32[]", getitem_2326: "f32[]", getitem_2327: "f32[]", getitem_2328: "f32[]", getitem_2329: "f32[]", getitem_2330: "f32[]", getitem_2331: "f32[]", getitem_2332: "f32[]", getitem_2333: "f32[]", getitem_2334: "f32[]", getitem_2335: "f32[]", getitem_2336: "f32[]", getitem_2337: "f32[]", getitem_2338: "f32[]", getitem_2339: "f32[]", getitem_2340: "f32[]", getitem_2341: "f32[]", getitem_2342: "f32[]", getitem_2343: "f32[]", getitem_2344: "f32[]", getitem_2345: "f32[]", getitem_2346: "f32[]", getitem_2347: "f32[]", getitem_2348: "f32[]", getitem_2349: "f32[]", getitem_2350: "f32[]", getitem_2351: "f32[]", getitem_2352: "f32[]", getitem_2353: "f32[]", getitem_2354: "f32[]", getitem_2355: "f32[]", getitem_2356: "f32[]", getitem_2357: "f32[]", getitem_2358: "f32[]", getitem_2359: "f32[]", getitem_2360: "f32[]", getitem_2361: "f32[]", getitem_2362: "f32[]", getitem_2363: "f32[]", getitem_2364: "f32[]", getitem_2365: "f32[]", getitem_2366: "f32[]", getitem_2367: "f32[]", getitem_2368: "f32[]", getitem_2369: "f32[]", getitem_2370: "f32[]", getitem_2371: "f32[]", getitem_2372: "f32[]", getitem_2373: "f32[]", getitem_2374: "f32[]", getitem_2375: "f32[]", getitem_2376: "f32[]", getitem_2377: "f32[]", getitem_2378: "f32[]", getitem_2379: "f32[]", getitem_2380: "f32[]", getitem_2381: "f32[]", getitem_2382: "f32[]", getitem_2383: "f32[]", getitem_2384: "f32[]", getitem_2385: "f32[]", getitem_2386: "f32[]", getitem_2387: "f32[]", getitem_2388: "f32[]", getitem_2389: "f32[]", getitem_2390: "f32[]", getitem_2391: "f32[]", getitem_2392: "f32[]", getitem_2393: "f32[]", getitem_2394: "f32[]", getitem_2395: "f32[]", getitem_2396: "f32[]", getitem_2397: "f32[]", getitem_2398: "f32[]", getitem_2399: "f32[]", getitem_2400: "f32[]", getitem_2401: "f32[]", getitem_2402: "f32[]", getitem_2403: "f32[]", getitem_2404: "f32[]", getitem_2405: "f32[]", getitem_2406: "f32[]", getitem_2407: "f32[]", getitem_2408: "f32[]", getitem_2409: "f32[]", getitem_2410: "f32[]", getitem_2411: "f32[]", getitem_2412: "f32[]", getitem_2413: "f32[]", getitem_2414: "f32[]", getitem_2415: "f32[]", getitem_2416: "f32[]", getitem_2417: "f32[]", getitem_2418: "f32[]", getitem_2419: "f32[]", getitem_2420: "f32[]", getitem_2421: "f32[]", getitem_2422: "f32[]", getitem_2423: "f32[]", getitem_2424: "f32[]", getitem_2425: "f32[]", getitem_2426: "f32[]", getitem_2427: "f32[]", getitem_2428: "f32[]", getitem_2429: "f32[]", getitem_2430: "f32[]", getitem_2431: "f32[]", getitem_2432: "f32[]", getitem_2433: "f32[]", getitem_2434: "f32[]", getitem_2435: "f32[]", getitem_2436: "f32[]", getitem_2437: "f32[]", getitem_2438: "f32[]", getitem_2439: "f32[]", getitem_2440: "f32[]", getitem_2441: "f32[]", getitem_2442: "f32[]", getitem_2443: "f32[]", getitem_2444: "f32[]", getitem_2445: "f32[]", getitem_2446: "f32[]", getitem_2447: "f32[]", getitem_2448: "f32[]", getitem_2449: "f32[]", getitem_2450: "f32[]", getitem_2451: "f32[]", getitem_2452: "f32[]", getitem_2453: "f32[]", getitem_2454: "f32[]", getitem_2455: "f32[]", getitem_2456: "f32[]", getitem_2457: "f32[]", getitem_2458: "f32[]", getitem_2459: "f32[]", getitem_2460: "f32[]", getitem_2461: "f32[]", getitem_2462: "f32[]", getitem_2463: "f32[]", getitem_2464: "f32[]", getitem_2465: "f32[]", getitem_2466: "f32[]", getitem_2467: "f32[]", getitem_2468: "f32[]", getitem_2469: "f32[]", getitem_2470: "f32[]", getitem_2471: "f32[]"):
        # No stacktrace found for following nodes
        full_default: "f32[30522, 768]" = torch.ops.aten.full.default([30522, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[512, 768]" = torch.ops.aten.full.default([512, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "f32[1024, 768]" = torch.ops.aten.full.default([1024, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_3: "f32[1024, 768]" = torch.ops.aten.full.default([1024, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_4: "f32[1024, 768]" = torch.ops.aten.full.default([1024, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_5: "f32[1024, 768]" = torch.ops.aten.full.default([1024, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_6: "f32[2, 768]" = torch.ops.aten.full.default([2, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_7: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_8: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_9: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_10: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_11: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_12: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_13: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_14: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_15: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_16: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_17: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_18: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_19: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_20: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_21: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_22: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_23: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_24: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_25: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_26: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_27: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_28: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_29: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_30: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_31: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_32: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_33: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_34: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_35: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_36: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_37: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_38: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_39: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_40: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_41: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_42: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_43: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_44: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_45: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_46: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_47: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_48: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_49: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_50: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_51: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_52: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_53: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_54: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_55: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_56: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_57: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_58: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_59: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_60: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_61: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_62: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_63: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_64: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_65: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_66: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_67: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_68: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_69: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_70: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_71: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_72: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_73: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_74: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_75: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_76: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_77: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_78: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_79: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_80: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_81: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_82: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_83: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_84: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_85: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_86: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_87: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_88: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_89: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_90: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_91: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_92: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_93: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_94: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_95: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_96: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_97: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_98: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_99: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_100: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_101: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_102: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_103: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_104: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_105: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_106: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_107: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_108: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_109: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_110: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_111: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_112: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_113: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_114: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_115: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_116: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_117: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_118: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_119: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_120: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_121: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_122: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_123: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_124: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_125: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_126: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_127: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_128: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_129: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_130: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_131: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_132: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_133: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_134: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_135: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_136: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_137: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_138: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_139: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_140: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_141: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_142: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_143: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_144: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_145: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_146: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_147: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_148: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_149: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_150: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_151: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_152: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_153: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_154: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_155: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_156: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_157: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_158: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_159: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_160: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_161: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_162: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_163: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_164: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_165: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_166: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_167: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_168: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_169: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_170: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_171: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_172: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_173: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_174: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_175: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_176: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_177: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_178: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_179: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_180: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_181: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_182: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_183: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_184: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_185: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_186: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_187: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_188: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_189: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_190: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_191: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_192: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_193: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_194: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_195: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_196: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_197: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_198: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_199: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_200: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_201: "f32[30522]" = torch.ops.aten.full.default([30522], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_202: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_203: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_204: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_205: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _foreach_addcmul_scalar = torch.ops.aten._foreach_addcmul.Scalar([arg209_1, arg207_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1], [full_default, full_default_1, full_default_2, full_default_3, full_default_4, full_default_5, full_default_6, full_default_7, full_default_8, full_default_9, full_default_10, full_default_11, full_default_12, full_default_13, full_default_14, full_default_15, full_default_16, full_default_17, full_default_18, full_default_19, full_default_20, full_default_21, full_default_22, full_default_23, full_default_24, full_default_25, full_default_26, full_default_27, full_default_28, full_default_29, full_default_30, full_default_31, full_default_32, full_default_33, full_default_34, full_default_35, full_default_36, full_default_37, full_default_38, full_default_39, full_default_40, full_default_41, full_default_42, full_default_43, full_default_44, full_default_45, full_default_46, full_default_47, full_default_48, full_default_49, full_default_50, full_default_51, full_default_52, full_default_53, full_default_54, full_default_55, full_default_56, full_default_57, full_default_58, full_default_59, full_default_60, full_default_61, full_default_62, full_default_63, full_default_64, full_default_65, full_default_66, full_default_67, full_default_68, full_default_69, full_default_70, full_default_71, full_default_72, full_default_73, full_default_74, full_default_75, full_default_76, full_default_77, full_default_78, full_default_79, full_default_80, full_default_81, full_default_82, full_default_83, full_default_84, full_default_85, full_default_86, full_default_87, full_default_88, full_default_89, full_default_90, full_default_91, full_default_92, full_default_93, full_default_94, full_default_95, full_default_96, full_default_97, full_default_98, full_default_99, full_default_100, full_default_101, full_default_102, full_default_103, full_default_104, full_default_105, full_default_106, full_default_107, full_default_108, full_default_109, full_default_110, full_default_111, full_default_112, full_default_113, full_default_114, full_default_115, full_default_116, full_default_117, full_default_118, full_default_119, full_default_120, full_default_121, full_default_122, full_default_123, full_default_124, full_default_125, full_default_126, full_default_127, full_default_128, full_default_129, full_default_130, full_default_131, full_default_132, full_default_133, full_default_134, full_default_135, full_default_136, full_default_137, full_default_138, full_default_139, full_default_140, full_default_141, full_default_142, full_default_143, full_default_144, full_default_145, full_default_146, full_default_147, full_default_148, full_default_149, full_default_150, full_default_151, full_default_152, full_default_153, full_default_154, full_default_155, full_default_156, full_default_157, full_default_158, full_default_159, full_default_160, full_default_161, full_default_162, full_default_163, full_default_164, full_default_165, full_default_166, full_default_167, full_default_168, full_default_169, full_default_170, full_default_171, full_default_172, full_default_173, full_default_174, full_default_175, full_default_176, full_default_177, full_default_178, full_default_179, full_default_180, full_default_181, full_default_182, full_default_183, full_default_184, full_default_185, full_default_186, full_default_187, full_default_188, full_default_189, full_default_190, full_default_191, full_default_192, full_default_193, full_default_194, full_default_195, full_default_196, full_default_197, full_default_198, full_default_199, full_default_200, full_default_201, full_default_202, full_default_203, full_default_204, full_default_205], [getitem_206, getitem_207, getitem_208, getitem_209, getitem_210, getitem_211, getitem_212, getitem_213, getitem_214, getitem_215, getitem_216, getitem_217, getitem_218, getitem_219, getitem_220, getitem_221, getitem_222, getitem_223, getitem_224, getitem_225, getitem_226, getitem_227, getitem_228, getitem_229, getitem_230, getitem_231, getitem_232, getitem_233, getitem_234, getitem_235, getitem_236, getitem_237, getitem_238, getitem_239, getitem_240, getitem_241, getitem_242, getitem_243, getitem_244, getitem_245, getitem_246, getitem_247, getitem_248, getitem_249, getitem_250, getitem_251, getitem_252, getitem_253, getitem_254, getitem_255, getitem_256, getitem_257, getitem_258, getitem_259, getitem_260, getitem_261, getitem_262, getitem_263, getitem_264, getitem_265, getitem_266, getitem_267, getitem_268, getitem_269, getitem_270, getitem_271, getitem_272, getitem_273, getitem_274, getitem_275, getitem_276, getitem_277, getitem_278, getitem_279, getitem_280, getitem_281, getitem_282, getitem_283, getitem_284, getitem_285, getitem_286, getitem_287, getitem_288, getitem_289, getitem_290, getitem_291, getitem_292, getitem_293, getitem_294, getitem_295, getitem_296, getitem_297, getitem_298, getitem_299, getitem_300, getitem_301, getitem_302, getitem_303, getitem_304, getitem_305, getitem_306, getitem_307, getitem_308, getitem_309, getitem_310, getitem_311, getitem_312, getitem_313, getitem_314, getitem_315, getitem_316, getitem_317, getitem_318, getitem_319, getitem_320, getitem_321, getitem_322, getitem_323, getitem_324, getitem_325, getitem_326, getitem_327, getitem_328, getitem_329, getitem_330, getitem_331, getitem_332, getitem_333, getitem_334, getitem_335, getitem_336, getitem_337, getitem_338, getitem_339, getitem_340, getitem_341, getitem_342, getitem_343, getitem_344, getitem_345, getitem_346, getitem_347, getitem_348, getitem_349, getitem_350, getitem_351, getitem_352, getitem_353, getitem_354, getitem_355, getitem_356, getitem_357, getitem_358, getitem_359, getitem_360, getitem_361, getitem_362, getitem_363, getitem_364, getitem_365, getitem_366, getitem_367, getitem_368, getitem_369, getitem_370, getitem_371, getitem_372, getitem_373, getitem_374, getitem_375, getitem_376, getitem_377, getitem_378, getitem_379, getitem_380, getitem_381, getitem_382, getitem_383, getitem_384, getitem_385, getitem_386, getitem_387, getitem_388, getitem_389, getitem_390, getitem_391, getitem_392, getitem_393, getitem_394, getitem_395, getitem_396, getitem_397, getitem_398, getitem_399, getitem_400, getitem_401, getitem_402, getitem_403, getitem_404, getitem_405, getitem_406, getitem_407, getitem_408, getitem_409, getitem_410, getitem_411]);  arg209_1 = arg207_1 = arg416_1 = arg417_1 = arg418_1 = arg419_1 = arg420_1 = arg421_1 = arg422_1 = arg423_1 = arg424_1 = arg425_1 = arg426_1 = arg427_1 = arg428_1 = arg429_1 = arg430_1 = arg431_1 = arg432_1 = arg433_1 = arg434_1 = arg435_1 = arg436_1 = arg437_1 = arg438_1 = arg439_1 = arg440_1 = arg441_1 = arg442_1 = arg443_1 = arg444_1 = arg445_1 = arg446_1 = arg447_1 = arg448_1 = arg449_1 = arg450_1 = arg451_1 = arg452_1 = arg453_1 = arg454_1 = arg455_1 = arg456_1 = arg457_1 = arg458_1 = arg459_1 = arg460_1 = arg461_1 = arg462_1 = arg463_1 = arg464_1 = arg465_1 = arg466_1 = arg467_1 = arg468_1 = arg469_1 = arg470_1 = arg471_1 = arg472_1 = arg473_1 = arg474_1 = arg475_1 = arg476_1 = arg477_1 = arg478_1 = arg479_1 = arg480_1 = arg481_1 = arg482_1 = arg483_1 = arg484_1 = arg485_1 = arg486_1 = arg487_1 = arg488_1 = arg489_1 = arg490_1 = arg491_1 = arg492_1 = arg493_1 = arg494_1 = arg495_1 = arg496_1 = arg497_1 = arg498_1 = arg499_1 = arg500_1 = arg501_1 = arg502_1 = arg503_1 = arg504_1 = arg505_1 = arg506_1 = arg507_1 = arg508_1 = arg509_1 = arg510_1 = arg511_1 = arg512_1 = arg513_1 = arg514_1 = arg515_1 = arg516_1 = arg517_1 = arg518_1 = arg519_1 = arg520_1 = arg521_1 = arg522_1 = arg523_1 = arg524_1 = arg525_1 = arg526_1 = arg527_1 = arg528_1 = arg529_1 = arg530_1 = arg531_1 = arg532_1 = arg533_1 = arg534_1 = arg535_1 = arg536_1 = arg537_1 = arg538_1 = arg539_1 = arg540_1 = arg541_1 = arg542_1 = arg543_1 = arg544_1 = arg545_1 = arg546_1 = arg547_1 = arg548_1 = arg549_1 = arg550_1 = arg551_1 = arg552_1 = arg553_1 = arg554_1 = arg555_1 = arg556_1 = arg557_1 = arg558_1 = arg559_1 = arg560_1 = arg561_1 = arg562_1 = arg563_1 = arg564_1 = arg565_1 = arg566_1 = arg567_1 = arg568_1 = arg569_1 = arg570_1 = arg571_1 = arg572_1 = arg573_1 = arg574_1 = arg575_1 = arg576_1 = arg577_1 = arg578_1 = arg579_1 = arg580_1 = arg581_1 = arg582_1 = arg583_1 = arg584_1 = arg585_1 = arg586_1 = arg587_1 = arg588_1 = arg589_1 = arg590_1 = arg591_1 = arg592_1 = arg593_1 = arg594_1 = arg595_1 = arg596_1 = arg597_1 = arg598_1 = arg599_1 = arg600_1 = arg601_1 = arg602_1 = arg603_1 = arg604_1 = arg605_1 = arg606_1 = arg607_1 = arg608_1 = arg609_1 = arg610_1 = arg611_1 = arg612_1 = arg613_1 = arg614_1 = arg615_1 = arg616_1 = arg617_1 = arg618_1 = arg619_1 = full_default = full_default_1 = full_default_2 = full_default_3 = full_default_4 = full_default_5 = full_default_6 = full_default_7 = full_default_8 = full_default_9 = full_default_10 = full_default_11 = full_default_12 = full_default_13 = full_default_14 = full_default_15 = full_default_16 = full_default_17 = full_default_18 = full_default_19 = full_default_20 = full_default_21 = full_default_22 = full_default_23 = full_default_24 = full_default_25 = full_default_26 = full_default_27 = full_default_28 = full_default_29 = full_default_30 = full_default_31 = full_default_32 = full_default_33 = full_default_34 = full_default_35 = full_default_36 = full_default_37 = full_default_38 = full_default_39 = full_default_40 = full_default_41 = full_default_42 = full_default_43 = full_default_44 = full_default_45 = full_default_46 = full_default_47 = full_default_48 = full_default_49 = full_default_50 = full_default_51 = full_default_52 = full_default_53 = full_default_54 = full_default_55 = full_default_56 = full_default_57 = full_default_58 = full_default_59 = full_default_60 = full_default_61 = full_default_62 = full_default_63 = full_default_64 = full_default_65 = full_default_66 = full_default_67 = full_default_68 = full_default_69 = full_default_70 = full_default_71 = full_default_72 = full_default_73 = full_default_74 = full_default_75 = full_default_76 = full_default_77 = full_default_78 = full_default_79 = full_default_80 = full_default_81 = full_default_82 = full_default_83 = full_default_84 = full_default_85 = full_default_86 = full_default_87 = full_default_88 = full_default_89 = full_default_90 = full_default_91 = full_default_92 = full_default_93 = full_default_94 = full_default_95 = full_default_96 = full_default_97 = full_default_98 = full_default_99 = full_default_100 = full_default_101 = full_default_102 = full_default_103 = full_default_104 = full_default_105 = full_default_106 = full_default_107 = full_default_108 = full_default_109 = full_default_110 = full_default_111 = full_default_112 = full_default_113 = full_default_114 = full_default_115 = full_default_116 = full_default_117 = full_default_118 = full_default_119 = full_default_120 = full_default_121 = full_default_122 = full_default_123 = full_default_124 = full_default_125 = full_default_126 = full_default_127 = full_default_128 = full_default_129 = full_default_130 = full_default_131 = full_default_132 = full_default_133 = full_default_134 = full_default_135 = full_default_136 = full_default_137 = full_default_138 = full_default_139 = full_default_140 = full_default_141 = full_default_142 = full_default_143 = full_default_144 = full_default_145 = full_default_146 = full_default_147 = full_default_148 = full_default_149 = full_default_150 = full_default_151 = full_default_152 = full_default_153 = full_default_154 = full_default_155 = full_default_156 = full_default_157 = full_default_158 = full_default_159 = full_default_160 = full_default_161 = full_default_162 = full_default_163 = full_default_164 = full_default_165 = full_default_166 = full_default_167 = full_default_168 = full_default_169 = full_default_170 = full_default_171 = full_default_172 = full_default_173 = full_default_174 = full_default_175 = full_default_176 = full_default_177 = full_default_178 = full_default_179 = full_default_180 = full_default_181 = full_default_182 = full_default_183 = full_default_184 = full_default_185 = full_default_186 = full_default_187 = full_default_188 = full_default_189 = full_default_190 = full_default_191 = full_default_192 = full_default_193 = full_default_194 = full_default_195 = full_default_196 = full_default_197 = full_default_198 = full_default_199 = full_default_200 = full_default_201 = full_default_202 = full_default_203 = full_default_204 = full_default_205 = getitem_206 = getitem_207 = getitem_208 = getitem_209 = getitem_210 = getitem_211 = getitem_212 = getitem_213 = getitem_214 = getitem_215 = getitem_216 = getitem_217 = getitem_218 = getitem_219 = getitem_220 = getitem_221 = getitem_222 = getitem_223 = getitem_224 = getitem_225 = getitem_226 = getitem_227 = getitem_228 = getitem_229 = getitem_230 = getitem_231 = getitem_232 = getitem_233 = getitem_234 = getitem_235 = getitem_236 = getitem_237 = getitem_238 = getitem_239 = getitem_240 = getitem_241 = getitem_242 = getitem_243 = getitem_244 = getitem_245 = getitem_246 = getitem_247 = getitem_248 = getitem_249 = getitem_250 = getitem_251 = getitem_252 = getitem_253 = getitem_254 = getitem_255 = getitem_256 = getitem_257 = getitem_258 = getitem_259 = getitem_260 = getitem_261 = getitem_262 = getitem_263 = getitem_264 = getitem_265 = getitem_266 = getitem_267 = getitem_268 = getitem_269 = getitem_270 = getitem_271 = getitem_272 = getitem_273 = getitem_274 = getitem_275 = getitem_276 = getitem_277 = getitem_278 = getitem_279 = getitem_280 = getitem_281 = getitem_282 = getitem_283 = getitem_284 = getitem_285 = getitem_286 = getitem_287 = getitem_288 = getitem_289 = getitem_290 = getitem_291 = getitem_292 = getitem_293 = getitem_294 = getitem_295 = getitem_296 = getitem_297 = getitem_298 = getitem_299 = getitem_300 = getitem_301 = getitem_302 = getitem_303 = getitem_304 = getitem_305 = getitem_306 = getitem_307 = getitem_308 = getitem_309 = getitem_310 = getitem_311 = getitem_312 = getitem_313 = getitem_314 = getitem_315 = getitem_316 = getitem_317 = getitem_318 = getitem_319 = getitem_320 = getitem_321 = getitem_322 = getitem_323 = getitem_324 = getitem_325 = getitem_326 = getitem_327 = getitem_328 = getitem_329 = getitem_330 = getitem_331 = getitem_332 = getitem_333 = getitem_334 = getitem_335 = getitem_336 = getitem_337 = getitem_338 = getitem_339 = getitem_340 = getitem_341 = getitem_342 = getitem_343 = getitem_344 = getitem_345 = getitem_346 = getitem_347 = getitem_348 = getitem_349 = getitem_350 = getitem_351 = getitem_352 = getitem_353 = getitem_354 = getitem_355 = getitem_356 = getitem_357 = getitem_358 = getitem_359 = getitem_360 = getitem_361 = getitem_362 = getitem_363 = getitem_364 = getitem_365 = getitem_366 = getitem_367 = getitem_368 = getitem_369 = getitem_370 = getitem_371 = getitem_372 = getitem_373 = getitem_374 = getitem_375 = getitem_376 = getitem_377 = getitem_378 = getitem_379 = getitem_380 = getitem_381 = getitem_382 = getitem_383 = getitem_384 = getitem_385 = getitem_386 = getitem_387 = getitem_388 = getitem_389 = getitem_390 = getitem_391 = getitem_392 = getitem_393 = getitem_394 = getitem_395 = getitem_396 = getitem_397 = getitem_398 = getitem_399 = getitem_400 = getitem_401 = getitem_402 = getitem_403 = getitem_404 = getitem_405 = getitem_406 = getitem_407 = getitem_408 = getitem_409 = getitem_410 = getitem_411 = None
        getitem: "f32[30522, 768]" = _foreach_addcmul_scalar[0]
        getitem_412: "f32[512, 768]" = _foreach_addcmul_scalar[1]
        getitem_413: "f32[1024, 768]" = _foreach_addcmul_scalar[2]
        getitem_414: "f32[1024, 768]" = _foreach_addcmul_scalar[3]
        getitem_415: "f32[1024, 768]" = _foreach_addcmul_scalar[4]
        getitem_416: "f32[1024, 768]" = _foreach_addcmul_scalar[5]
        getitem_417: "f32[2, 768]" = _foreach_addcmul_scalar[6]
        getitem_418: "f32[768]" = _foreach_addcmul_scalar[7]
        getitem_419: "f32[768]" = _foreach_addcmul_scalar[8]
        getitem_420: "f32[768, 768]" = _foreach_addcmul_scalar[9]
        getitem_421: "f32[768]" = _foreach_addcmul_scalar[10]
        getitem_422: "f32[768, 768]" = _foreach_addcmul_scalar[11]
        getitem_423: "f32[768]" = _foreach_addcmul_scalar[12]
        getitem_424: "f32[768, 768]" = _foreach_addcmul_scalar[13]
        getitem_425: "f32[768]" = _foreach_addcmul_scalar[14]
        getitem_426: "f32[768, 768]" = _foreach_addcmul_scalar[15]
        getitem_427: "f32[768]" = _foreach_addcmul_scalar[16]
        getitem_428: "f32[768]" = _foreach_addcmul_scalar[17]
        getitem_429: "f32[768]" = _foreach_addcmul_scalar[18]
        getitem_430: "f32[3072, 768]" = _foreach_addcmul_scalar[19]
        getitem_431: "f32[3072]" = _foreach_addcmul_scalar[20]
        getitem_432: "f32[768, 3072]" = _foreach_addcmul_scalar[21]
        getitem_433: "f32[768]" = _foreach_addcmul_scalar[22]
        getitem_434: "f32[768]" = _foreach_addcmul_scalar[23]
        getitem_435: "f32[768]" = _foreach_addcmul_scalar[24]
        getitem_436: "f32[768, 768]" = _foreach_addcmul_scalar[25]
        getitem_437: "f32[768]" = _foreach_addcmul_scalar[26]
        getitem_438: "f32[768, 768]" = _foreach_addcmul_scalar[27]
        getitem_439: "f32[768]" = _foreach_addcmul_scalar[28]
        getitem_440: "f32[768, 768]" = _foreach_addcmul_scalar[29]
        getitem_441: "f32[768]" = _foreach_addcmul_scalar[30]
        getitem_442: "f32[768, 768]" = _foreach_addcmul_scalar[31]
        getitem_443: "f32[768]" = _foreach_addcmul_scalar[32]
        getitem_444: "f32[768]" = _foreach_addcmul_scalar[33]
        getitem_445: "f32[768]" = _foreach_addcmul_scalar[34]
        getitem_446: "f32[3072, 768]" = _foreach_addcmul_scalar[35]
        getitem_447: "f32[3072]" = _foreach_addcmul_scalar[36]
        getitem_448: "f32[768, 3072]" = _foreach_addcmul_scalar[37]
        getitem_449: "f32[768]" = _foreach_addcmul_scalar[38]
        getitem_450: "f32[768]" = _foreach_addcmul_scalar[39]
        getitem_451: "f32[768]" = _foreach_addcmul_scalar[40]
        getitem_452: "f32[768, 768]" = _foreach_addcmul_scalar[41]
        getitem_453: "f32[768]" = _foreach_addcmul_scalar[42]
        getitem_454: "f32[768, 768]" = _foreach_addcmul_scalar[43]
        getitem_455: "f32[768]" = _foreach_addcmul_scalar[44]
        getitem_456: "f32[768, 768]" = _foreach_addcmul_scalar[45]
        getitem_457: "f32[768]" = _foreach_addcmul_scalar[46]
        getitem_458: "f32[768, 768]" = _foreach_addcmul_scalar[47]
        getitem_459: "f32[768]" = _foreach_addcmul_scalar[48]
        getitem_460: "f32[768]" = _foreach_addcmul_scalar[49]
        getitem_461: "f32[768]" = _foreach_addcmul_scalar[50]
        getitem_462: "f32[3072, 768]" = _foreach_addcmul_scalar[51]
        getitem_463: "f32[3072]" = _foreach_addcmul_scalar[52]
        getitem_464: "f32[768, 3072]" = _foreach_addcmul_scalar[53]
        getitem_465: "f32[768]" = _foreach_addcmul_scalar[54]
        getitem_466: "f32[768]" = _foreach_addcmul_scalar[55]
        getitem_467: "f32[768]" = _foreach_addcmul_scalar[56]
        getitem_468: "f32[768, 768]" = _foreach_addcmul_scalar[57]
        getitem_469: "f32[768]" = _foreach_addcmul_scalar[58]
        getitem_470: "f32[768, 768]" = _foreach_addcmul_scalar[59]
        getitem_471: "f32[768]" = _foreach_addcmul_scalar[60]
        getitem_472: "f32[768, 768]" = _foreach_addcmul_scalar[61]
        getitem_473: "f32[768]" = _foreach_addcmul_scalar[62]
        getitem_474: "f32[768, 768]" = _foreach_addcmul_scalar[63]
        getitem_475: "f32[768]" = _foreach_addcmul_scalar[64]
        getitem_476: "f32[768]" = _foreach_addcmul_scalar[65]
        getitem_477: "f32[768]" = _foreach_addcmul_scalar[66]
        getitem_478: "f32[3072, 768]" = _foreach_addcmul_scalar[67]
        getitem_479: "f32[3072]" = _foreach_addcmul_scalar[68]
        getitem_480: "f32[768, 3072]" = _foreach_addcmul_scalar[69]
        getitem_481: "f32[768]" = _foreach_addcmul_scalar[70]
        getitem_482: "f32[768]" = _foreach_addcmul_scalar[71]
        getitem_483: "f32[768]" = _foreach_addcmul_scalar[72]
        getitem_484: "f32[768, 768]" = _foreach_addcmul_scalar[73]
        getitem_485: "f32[768]" = _foreach_addcmul_scalar[74]
        getitem_486: "f32[768, 768]" = _foreach_addcmul_scalar[75]
        getitem_487: "f32[768]" = _foreach_addcmul_scalar[76]
        getitem_488: "f32[768, 768]" = _foreach_addcmul_scalar[77]
        getitem_489: "f32[768]" = _foreach_addcmul_scalar[78]
        getitem_490: "f32[768, 768]" = _foreach_addcmul_scalar[79]
        getitem_491: "f32[768]" = _foreach_addcmul_scalar[80]
        getitem_492: "f32[768]" = _foreach_addcmul_scalar[81]
        getitem_493: "f32[768]" = _foreach_addcmul_scalar[82]
        getitem_494: "f32[3072, 768]" = _foreach_addcmul_scalar[83]
        getitem_495: "f32[3072]" = _foreach_addcmul_scalar[84]
        getitem_496: "f32[768, 3072]" = _foreach_addcmul_scalar[85]
        getitem_497: "f32[768]" = _foreach_addcmul_scalar[86]
        getitem_498: "f32[768]" = _foreach_addcmul_scalar[87]
        getitem_499: "f32[768]" = _foreach_addcmul_scalar[88]
        getitem_500: "f32[768, 768]" = _foreach_addcmul_scalar[89]
        getitem_501: "f32[768]" = _foreach_addcmul_scalar[90]
        getitem_502: "f32[768, 768]" = _foreach_addcmul_scalar[91]
        getitem_503: "f32[768]" = _foreach_addcmul_scalar[92]
        getitem_504: "f32[768, 768]" = _foreach_addcmul_scalar[93]
        getitem_505: "f32[768]" = _foreach_addcmul_scalar[94]
        getitem_506: "f32[768, 768]" = _foreach_addcmul_scalar[95]
        getitem_507: "f32[768]" = _foreach_addcmul_scalar[96]
        getitem_508: "f32[768]" = _foreach_addcmul_scalar[97]
        getitem_509: "f32[768]" = _foreach_addcmul_scalar[98]
        getitem_510: "f32[3072, 768]" = _foreach_addcmul_scalar[99]
        getitem_511: "f32[3072]" = _foreach_addcmul_scalar[100]
        getitem_512: "f32[768, 3072]" = _foreach_addcmul_scalar[101]
        getitem_513: "f32[768]" = _foreach_addcmul_scalar[102]
        getitem_514: "f32[768]" = _foreach_addcmul_scalar[103]
        getitem_515: "f32[768]" = _foreach_addcmul_scalar[104]
        getitem_516: "f32[768, 768]" = _foreach_addcmul_scalar[105]
        getitem_517: "f32[768]" = _foreach_addcmul_scalar[106]
        getitem_518: "f32[768, 768]" = _foreach_addcmul_scalar[107]
        getitem_519: "f32[768]" = _foreach_addcmul_scalar[108]
        getitem_520: "f32[768, 768]" = _foreach_addcmul_scalar[109]
        getitem_521: "f32[768]" = _foreach_addcmul_scalar[110]
        getitem_522: "f32[768, 768]" = _foreach_addcmul_scalar[111]
        getitem_523: "f32[768]" = _foreach_addcmul_scalar[112]
        getitem_524: "f32[768]" = _foreach_addcmul_scalar[113]
        getitem_525: "f32[768]" = _foreach_addcmul_scalar[114]
        getitem_526: "f32[3072, 768]" = _foreach_addcmul_scalar[115]
        getitem_527: "f32[3072]" = _foreach_addcmul_scalar[116]
        getitem_528: "f32[768, 3072]" = _foreach_addcmul_scalar[117]
        getitem_529: "f32[768]" = _foreach_addcmul_scalar[118]
        getitem_530: "f32[768]" = _foreach_addcmul_scalar[119]
        getitem_531: "f32[768]" = _foreach_addcmul_scalar[120]
        getitem_532: "f32[768, 768]" = _foreach_addcmul_scalar[121]
        getitem_533: "f32[768]" = _foreach_addcmul_scalar[122]
        getitem_534: "f32[768, 768]" = _foreach_addcmul_scalar[123]
        getitem_535: "f32[768]" = _foreach_addcmul_scalar[124]
        getitem_536: "f32[768, 768]" = _foreach_addcmul_scalar[125]
        getitem_537: "f32[768]" = _foreach_addcmul_scalar[126]
        getitem_538: "f32[768, 768]" = _foreach_addcmul_scalar[127]
        getitem_539: "f32[768]" = _foreach_addcmul_scalar[128]
        getitem_540: "f32[768]" = _foreach_addcmul_scalar[129]
        getitem_541: "f32[768]" = _foreach_addcmul_scalar[130]
        getitem_542: "f32[3072, 768]" = _foreach_addcmul_scalar[131]
        getitem_543: "f32[3072]" = _foreach_addcmul_scalar[132]
        getitem_544: "f32[768, 3072]" = _foreach_addcmul_scalar[133]
        getitem_545: "f32[768]" = _foreach_addcmul_scalar[134]
        getitem_546: "f32[768]" = _foreach_addcmul_scalar[135]
        getitem_547: "f32[768]" = _foreach_addcmul_scalar[136]
        getitem_548: "f32[768, 768]" = _foreach_addcmul_scalar[137]
        getitem_549: "f32[768]" = _foreach_addcmul_scalar[138]
        getitem_550: "f32[768, 768]" = _foreach_addcmul_scalar[139]
        getitem_551: "f32[768]" = _foreach_addcmul_scalar[140]
        getitem_552: "f32[768, 768]" = _foreach_addcmul_scalar[141]
        getitem_553: "f32[768]" = _foreach_addcmul_scalar[142]
        getitem_554: "f32[768, 768]" = _foreach_addcmul_scalar[143]
        getitem_555: "f32[768]" = _foreach_addcmul_scalar[144]
        getitem_556: "f32[768]" = _foreach_addcmul_scalar[145]
        getitem_557: "f32[768]" = _foreach_addcmul_scalar[146]
        getitem_558: "f32[3072, 768]" = _foreach_addcmul_scalar[147]
        getitem_559: "f32[3072]" = _foreach_addcmul_scalar[148]
        getitem_560: "f32[768, 3072]" = _foreach_addcmul_scalar[149]
        getitem_561: "f32[768]" = _foreach_addcmul_scalar[150]
        getitem_562: "f32[768]" = _foreach_addcmul_scalar[151]
        getitem_563: "f32[768]" = _foreach_addcmul_scalar[152]
        getitem_564: "f32[768, 768]" = _foreach_addcmul_scalar[153]
        getitem_565: "f32[768]" = _foreach_addcmul_scalar[154]
        getitem_566: "f32[768, 768]" = _foreach_addcmul_scalar[155]
        getitem_567: "f32[768]" = _foreach_addcmul_scalar[156]
        getitem_568: "f32[768, 768]" = _foreach_addcmul_scalar[157]
        getitem_569: "f32[768]" = _foreach_addcmul_scalar[158]
        getitem_570: "f32[768, 768]" = _foreach_addcmul_scalar[159]
        getitem_571: "f32[768]" = _foreach_addcmul_scalar[160]
        getitem_572: "f32[768]" = _foreach_addcmul_scalar[161]
        getitem_573: "f32[768]" = _foreach_addcmul_scalar[162]
        getitem_574: "f32[3072, 768]" = _foreach_addcmul_scalar[163]
        getitem_575: "f32[3072]" = _foreach_addcmul_scalar[164]
        getitem_576: "f32[768, 3072]" = _foreach_addcmul_scalar[165]
        getitem_577: "f32[768]" = _foreach_addcmul_scalar[166]
        getitem_578: "f32[768]" = _foreach_addcmul_scalar[167]
        getitem_579: "f32[768]" = _foreach_addcmul_scalar[168]
        getitem_580: "f32[768, 768]" = _foreach_addcmul_scalar[169]
        getitem_581: "f32[768]" = _foreach_addcmul_scalar[170]
        getitem_582: "f32[768, 768]" = _foreach_addcmul_scalar[171]
        getitem_583: "f32[768]" = _foreach_addcmul_scalar[172]
        getitem_584: "f32[768, 768]" = _foreach_addcmul_scalar[173]
        getitem_585: "f32[768]" = _foreach_addcmul_scalar[174]
        getitem_586: "f32[768, 768]" = _foreach_addcmul_scalar[175]
        getitem_587: "f32[768]" = _foreach_addcmul_scalar[176]
        getitem_588: "f32[768]" = _foreach_addcmul_scalar[177]
        getitem_589: "f32[768]" = _foreach_addcmul_scalar[178]
        getitem_590: "f32[3072, 768]" = _foreach_addcmul_scalar[179]
        getitem_591: "f32[3072]" = _foreach_addcmul_scalar[180]
        getitem_592: "f32[768, 3072]" = _foreach_addcmul_scalar[181]
        getitem_593: "f32[768]" = _foreach_addcmul_scalar[182]
        getitem_594: "f32[768]" = _foreach_addcmul_scalar[183]
        getitem_595: "f32[768]" = _foreach_addcmul_scalar[184]
        getitem_596: "f32[768, 768]" = _foreach_addcmul_scalar[185]
        getitem_597: "f32[768]" = _foreach_addcmul_scalar[186]
        getitem_598: "f32[768, 768]" = _foreach_addcmul_scalar[187]
        getitem_599: "f32[768]" = _foreach_addcmul_scalar[188]
        getitem_600: "f32[768, 768]" = _foreach_addcmul_scalar[189]
        getitem_601: "f32[768]" = _foreach_addcmul_scalar[190]
        getitem_602: "f32[768, 768]" = _foreach_addcmul_scalar[191]
        getitem_603: "f32[768]" = _foreach_addcmul_scalar[192]
        getitem_604: "f32[768]" = _foreach_addcmul_scalar[193]
        getitem_605: "f32[768]" = _foreach_addcmul_scalar[194]
        getitem_606: "f32[3072, 768]" = _foreach_addcmul_scalar[195]
        getitem_607: "f32[3072]" = _foreach_addcmul_scalar[196]
        getitem_608: "f32[768, 3072]" = _foreach_addcmul_scalar[197]
        getitem_609: "f32[768]" = _foreach_addcmul_scalar[198]
        getitem_610: "f32[768]" = _foreach_addcmul_scalar[199]
        getitem_611: "f32[768]" = _foreach_addcmul_scalar[200]
        getitem_612: "f32[30522]" = _foreach_addcmul_scalar[201]
        getitem_613: "f32[768, 768]" = _foreach_addcmul_scalar[202]
        getitem_614: "f32[768]" = _foreach_addcmul_scalar[203]
        getitem_615: "f32[768]" = _foreach_addcmul_scalar[204]
        getitem_616: "f32[768]" = _foreach_addcmul_scalar[205];  _foreach_addcmul_scalar = None
        _foreach_div_list = torch.ops.aten._foreach_div.List([getitem_3090, getitem_3091, getitem_3092, getitem_3093, getitem_3094, getitem_3095, getitem_3096, getitem_3097, getitem_3098, getitem_3099, getitem_3100, getitem_3101, getitem_3102, getitem_3103, getitem_3104, getitem_3105, getitem_3106, getitem_3107, getitem_3108, getitem_3109, getitem_3110, getitem_3111, getitem_3112, getitem_3113, getitem_3114, getitem_3115, getitem_3116, getitem_3117, getitem_3118, getitem_3119, getitem_3120, getitem_3121, getitem_3122, getitem_3123, getitem_3124, getitem_3125, getitem_3126, getitem_3127, getitem_3128, getitem_3129, getitem_3130, getitem_3131, getitem_3132, getitem_3133, getitem_3134, getitem_3135, getitem_3136, getitem_3137, getitem_3138, getitem_3139, getitem_3140, getitem_3141, getitem_3142, getitem_3143, getitem_3144, getitem_3145, getitem_3146, getitem_3147, getitem_3148, getitem_3149, getitem_3150, getitem_3151, getitem_3152, getitem_3153, getitem_3154, getitem_3155, getitem_3156, getitem_3157, getitem_3158, getitem_3159, getitem_3160, getitem_3161, getitem_3162, getitem_3163, getitem_3164, getitem_3165, getitem_3166, getitem_3167, getitem_3168, getitem_3169, getitem_3170, getitem_3171, getitem_3172, getitem_3173, getitem_3174, getitem_3175, getitem_3176, getitem_3177, getitem_3178, getitem_3179, getitem_3180, getitem_3181, getitem_3182, getitem_3183, getitem_3184, getitem_3185, getitem_3186, getitem_3187, getitem_3188, getitem_3189, getitem_3190, getitem_3191, getitem_3192, getitem_3193, getitem_3194, getitem_3195, getitem_3196, getitem_3197, getitem_3198, getitem_3199, getitem_3200, getitem_3201, getitem_3202, getitem_3203, getitem_3204, getitem_3205, getitem_3206, getitem_3207, getitem_3208, getitem_3209, getitem_3210, getitem_3211, getitem_3212, getitem_3213, getitem_3214, getitem_3215, getitem_3216, getitem_3217, getitem_3218, getitem_3219, getitem_3220, getitem_3221, getitem_3222, getitem_3223, getitem_3224, getitem_3225, getitem_3226, getitem_3227, getitem_3228, getitem_3229, getitem_3230, getitem_3231, getitem_3232, getitem_3233, getitem_3234, getitem_3235, getitem_3236, getitem_3237, getitem_3238, getitem_3239, getitem_3240, getitem_3241, getitem_3242, getitem_3243, getitem_3244, getitem_3245, getitem_3246, getitem_3247, getitem_3248, getitem_3249, getitem_3250, getitem_3251, getitem_3252, getitem_3253, getitem_3254, getitem_3255, getitem_3256, getitem_3257, getitem_3258, getitem_3259, getitem_3260, getitem_3261, getitem_3262, getitem_3263, getitem_3264, getitem_3265, getitem_3266, getitem_3267, getitem_3268, getitem_3269, getitem_3270, getitem_3271, getitem_3272, getitem_3273, getitem_3274, getitem_3275, getitem_3276, getitem_3277, getitem_3278, getitem_3279, getitem_3280, getitem_3281, getitem_3282, getitem_3283, getitem_3284, getitem_3285, getitem_3286, getitem_3287, getitem_3288, getitem_3289, getitem_3290, getitem_3291, getitem_3292, getitem_3293, getitem_3294, getitem_3295], [getitem_2266, getitem_2267, getitem_2268, getitem_2269, getitem_2270, getitem_2271, getitem_2272, getitem_2273, getitem_2274, getitem_2275, getitem_2276, getitem_2277, getitem_2278, getitem_2279, getitem_2280, getitem_2281, getitem_2282, getitem_2283, getitem_2284, getitem_2285, getitem_2286, getitem_2287, getitem_2288, getitem_2289, getitem_2290, getitem_2291, getitem_2292, getitem_2293, getitem_2294, getitem_2295, getitem_2296, getitem_2297, getitem_2298, getitem_2299, getitem_2300, getitem_2301, getitem_2302, getitem_2303, getitem_2304, getitem_2305, getitem_2306, getitem_2307, getitem_2308, getitem_2309, getitem_2310, getitem_2311, getitem_2312, getitem_2313, getitem_2314, getitem_2315, getitem_2316, getitem_2317, getitem_2318, getitem_2319, getitem_2320, getitem_2321, getitem_2322, getitem_2323, getitem_2324, getitem_2325, getitem_2326, getitem_2327, getitem_2328, getitem_2329, getitem_2330, getitem_2331, getitem_2332, getitem_2333, getitem_2334, getitem_2335, getitem_2336, getitem_2337, getitem_2338, getitem_2339, getitem_2340, getitem_2341, getitem_2342, getitem_2343, getitem_2344, getitem_2345, getitem_2346, getitem_2347, getitem_2348, getitem_2349, getitem_2350, getitem_2351, getitem_2352, getitem_2353, getitem_2354, getitem_2355, getitem_2356, getitem_2357, getitem_2358, getitem_2359, getitem_2360, getitem_2361, getitem_2362, getitem_2363, getitem_2364, getitem_2365, getitem_2366, getitem_2367, getitem_2368, getitem_2369, getitem_2370, getitem_2371, getitem_2372, getitem_2373, getitem_2374, getitem_2375, getitem_2376, getitem_2377, getitem_2378, getitem_2379, getitem_2380, getitem_2381, getitem_2382, getitem_2383, getitem_2384, getitem_2385, getitem_2386, getitem_2387, getitem_2388, getitem_2389, getitem_2390, getitem_2391, getitem_2392, getitem_2393, getitem_2394, getitem_2395, getitem_2396, getitem_2397, getitem_2398, getitem_2399, getitem_2400, getitem_2401, getitem_2402, getitem_2403, getitem_2404, getitem_2405, getitem_2406, getitem_2407, getitem_2408, getitem_2409, getitem_2410, getitem_2411, getitem_2412, getitem_2413, getitem_2414, getitem_2415, getitem_2416, getitem_2417, getitem_2418, getitem_2419, getitem_2420, getitem_2421, getitem_2422, getitem_2423, getitem_2424, getitem_2425, getitem_2426, getitem_2427, getitem_2428, getitem_2429, getitem_2430, getitem_2431, getitem_2432, getitem_2433, getitem_2434, getitem_2435, getitem_2436, getitem_2437, getitem_2438, getitem_2439, getitem_2440, getitem_2441, getitem_2442, getitem_2443, getitem_2444, getitem_2445, getitem_2446, getitem_2447, getitem_2448, getitem_2449, getitem_2450, getitem_2451, getitem_2452, getitem_2453, getitem_2454, getitem_2455, getitem_2456, getitem_2457, getitem_2458, getitem_2459, getitem_2460, getitem_2461, getitem_2462, getitem_2463, getitem_2464, getitem_2465, getitem_2466, getitem_2467, getitem_2468, getitem_2469, getitem_2470, getitem_2471]);  getitem_3090 = getitem_3091 = getitem_3092 = getitem_3093 = getitem_3094 = getitem_3095 = getitem_3096 = getitem_3097 = getitem_3098 = getitem_3099 = getitem_3100 = getitem_3101 = getitem_3102 = getitem_3103 = getitem_3104 = getitem_3105 = getitem_3106 = getitem_3107 = getitem_3108 = getitem_3109 = getitem_3110 = getitem_3111 = getitem_3112 = getitem_3113 = getitem_3114 = getitem_3115 = getitem_3116 = getitem_3117 = getitem_3118 = getitem_3119 = getitem_3120 = getitem_3121 = getitem_3122 = getitem_3123 = getitem_3124 = getitem_3125 = getitem_3126 = getitem_3127 = getitem_3128 = getitem_3129 = getitem_3130 = getitem_3131 = getitem_3132 = getitem_3133 = getitem_3134 = getitem_3135 = getitem_3136 = getitem_3137 = getitem_3138 = getitem_3139 = getitem_3140 = getitem_3141 = getitem_3142 = getitem_3143 = getitem_3144 = getitem_3145 = getitem_3146 = getitem_3147 = getitem_3148 = getitem_3149 = getitem_3150 = getitem_3151 = getitem_3152 = getitem_3153 = getitem_3154 = getitem_3155 = getitem_3156 = getitem_3157 = getitem_3158 = getitem_3159 = getitem_3160 = getitem_3161 = getitem_3162 = getitem_3163 = getitem_3164 = getitem_3165 = getitem_3166 = getitem_3167 = getitem_3168 = getitem_3169 = getitem_3170 = getitem_3171 = getitem_3172 = getitem_3173 = getitem_3174 = getitem_3175 = getitem_3176 = getitem_3177 = getitem_3178 = getitem_3179 = getitem_3180 = getitem_3181 = getitem_3182 = getitem_3183 = getitem_3184 = getitem_3185 = getitem_3186 = getitem_3187 = getitem_3188 = getitem_3189 = getitem_3190 = getitem_3191 = getitem_3192 = getitem_3193 = getitem_3194 = getitem_3195 = getitem_3196 = getitem_3197 = getitem_3198 = getitem_3199 = getitem_3200 = getitem_3201 = getitem_3202 = getitem_3203 = getitem_3204 = getitem_3205 = getitem_3206 = getitem_3207 = getitem_3208 = getitem_3209 = getitem_3210 = getitem_3211 = getitem_3212 = getitem_3213 = getitem_3214 = getitem_3215 = getitem_3216 = getitem_3217 = getitem_3218 = getitem_3219 = getitem_3220 = getitem_3221 = getitem_3222 = getitem_3223 = getitem_3224 = getitem_3225 = getitem_3226 = getitem_3227 = getitem_3228 = getitem_3229 = getitem_3230 = getitem_3231 = getitem_3232 = getitem_3233 = getitem_3234 = getitem_3235 = getitem_3236 = getitem_3237 = getitem_3238 = getitem_3239 = getitem_3240 = getitem_3241 = getitem_3242 = getitem_3243 = getitem_3244 = getitem_3245 = getitem_3246 = getitem_3247 = getitem_3248 = getitem_3249 = getitem_3250 = getitem_3251 = getitem_3252 = getitem_3253 = getitem_3254 = getitem_3255 = getitem_3256 = getitem_3257 = getitem_3258 = getitem_3259 = getitem_3260 = getitem_3261 = getitem_3262 = getitem_3263 = getitem_3264 = getitem_3265 = getitem_3266 = getitem_3267 = getitem_3268 = getitem_3269 = getitem_3270 = getitem_3271 = getitem_3272 = getitem_3273 = getitem_3274 = getitem_3275 = getitem_3276 = getitem_3277 = getitem_3278 = getitem_3279 = getitem_3280 = getitem_3281 = getitem_3282 = getitem_3283 = getitem_3284 = getitem_3285 = getitem_3286 = getitem_3287 = getitem_3288 = getitem_3289 = getitem_3290 = getitem_3291 = getitem_3292 = getitem_3293 = getitem_3294 = getitem_3295 = getitem_2266 = getitem_2267 = getitem_2268 = getitem_2269 = getitem_2270 = getitem_2271 = getitem_2272 = getitem_2273 = getitem_2274 = getitem_2275 = getitem_2276 = getitem_2277 = getitem_2278 = getitem_2279 = getitem_2280 = getitem_2281 = getitem_2282 = getitem_2283 = getitem_2284 = getitem_2285 = getitem_2286 = getitem_2287 = getitem_2288 = getitem_2289 = getitem_2290 = getitem_2291 = getitem_2292 = getitem_2293 = getitem_2294 = getitem_2295 = getitem_2296 = getitem_2297 = getitem_2298 = getitem_2299 = getitem_2300 = getitem_2301 = getitem_2302 = getitem_2303 = getitem_2304 = getitem_2305 = getitem_2306 = getitem_2307 = getitem_2308 = getitem_2309 = getitem_2310 = getitem_2311 = getitem_2312 = getitem_2313 = getitem_2314 = getitem_2315 = getitem_2316 = getitem_2317 = getitem_2318 = getitem_2319 = getitem_2320 = getitem_2321 = getitem_2322 = getitem_2323 = getitem_2324 = getitem_2325 = getitem_2326 = getitem_2327 = getitem_2328 = getitem_2329 = getitem_2330 = getitem_2331 = getitem_2332 = getitem_2333 = getitem_2334 = getitem_2335 = getitem_2336 = getitem_2337 = getitem_2338 = getitem_2339 = getitem_2340 = getitem_2341 = getitem_2342 = getitem_2343 = getitem_2344 = getitem_2345 = getitem_2346 = getitem_2347 = getitem_2348 = getitem_2349 = getitem_2350 = getitem_2351 = getitem_2352 = getitem_2353 = getitem_2354 = getitem_2355 = getitem_2356 = getitem_2357 = getitem_2358 = getitem_2359 = getitem_2360 = getitem_2361 = getitem_2362 = getitem_2363 = getitem_2364 = getitem_2365 = getitem_2366 = getitem_2367 = getitem_2368 = getitem_2369 = getitem_2370 = getitem_2371 = getitem_2372 = getitem_2373 = getitem_2374 = getitem_2375 = getitem_2376 = getitem_2377 = getitem_2378 = getitem_2379 = getitem_2380 = getitem_2381 = getitem_2382 = getitem_2383 = getitem_2384 = getitem_2385 = getitem_2386 = getitem_2387 = getitem_2388 = getitem_2389 = getitem_2390 = getitem_2391 = getitem_2392 = getitem_2393 = getitem_2394 = getitem_2395 = getitem_2396 = getitem_2397 = getitem_2398 = getitem_2399 = getitem_2400 = getitem_2401 = getitem_2402 = getitem_2403 = getitem_2404 = getitem_2405 = getitem_2406 = getitem_2407 = getitem_2408 = getitem_2409 = getitem_2410 = getitem_2411 = getitem_2412 = getitem_2413 = getitem_2414 = getitem_2415 = getitem_2416 = getitem_2417 = getitem_2418 = getitem_2419 = getitem_2420 = getitem_2421 = getitem_2422 = getitem_2423 = getitem_2424 = getitem_2425 = getitem_2426 = getitem_2427 = getitem_2428 = getitem_2429 = getitem_2430 = getitem_2431 = getitem_2432 = getitem_2433 = getitem_2434 = getitem_2435 = getitem_2436 = getitem_2437 = getitem_2438 = getitem_2439 = getitem_2440 = getitem_2441 = getitem_2442 = getitem_2443 = getitem_2444 = getitem_2445 = getitem_2446 = getitem_2447 = getitem_2448 = getitem_2449 = getitem_2450 = getitem_2451 = getitem_2452 = getitem_2453 = getitem_2454 = getitem_2455 = getitem_2456 = getitem_2457 = getitem_2458 = getitem_2459 = getitem_2460 = getitem_2461 = getitem_2462 = getitem_2463 = getitem_2464 = getitem_2465 = getitem_2466 = getitem_2467 = getitem_2468 = getitem_2469 = getitem_2470 = getitem_2471 = None
        getitem_2472: "f32[30522, 768]" = _foreach_div_list[0]
        getitem_2473: "f32[512, 768]" = _foreach_div_list[1]
        getitem_2474: "f32[1024, 768]" = _foreach_div_list[2]
        getitem_2475: "f32[1024, 768]" = _foreach_div_list[3]
        getitem_2476: "f32[1024, 768]" = _foreach_div_list[4]
        getitem_2477: "f32[1024, 768]" = _foreach_div_list[5]
        getitem_2478: "f32[2, 768]" = _foreach_div_list[6]
        getitem_2479: "f32[768]" = _foreach_div_list[7]
        getitem_2480: "f32[768]" = _foreach_div_list[8]
        getitem_2481: "f32[768, 768]" = _foreach_div_list[9]
        getitem_2482: "f32[768]" = _foreach_div_list[10]
        getitem_2483: "f32[768, 768]" = _foreach_div_list[11]
        getitem_2484: "f32[768]" = _foreach_div_list[12]
        getitem_2485: "f32[768, 768]" = _foreach_div_list[13]
        getitem_2486: "f32[768]" = _foreach_div_list[14]
        getitem_2487: "f32[768, 768]" = _foreach_div_list[15]
        getitem_2488: "f32[768]" = _foreach_div_list[16]
        getitem_2489: "f32[768]" = _foreach_div_list[17]
        getitem_2490: "f32[768]" = _foreach_div_list[18]
        getitem_2491: "f32[3072, 768]" = _foreach_div_list[19]
        getitem_2492: "f32[3072]" = _foreach_div_list[20]
        getitem_2493: "f32[768, 3072]" = _foreach_div_list[21]
        getitem_2494: "f32[768]" = _foreach_div_list[22]
        getitem_2495: "f32[768]" = _foreach_div_list[23]
        getitem_2496: "f32[768]" = _foreach_div_list[24]
        getitem_2497: "f32[768, 768]" = _foreach_div_list[25]
        getitem_2498: "f32[768]" = _foreach_div_list[26]
        getitem_2499: "f32[768, 768]" = _foreach_div_list[27]
        getitem_2500: "f32[768]" = _foreach_div_list[28]
        getitem_2501: "f32[768, 768]" = _foreach_div_list[29]
        getitem_2502: "f32[768]" = _foreach_div_list[30]
        getitem_2503: "f32[768, 768]" = _foreach_div_list[31]
        getitem_2504: "f32[768]" = _foreach_div_list[32]
        getitem_2505: "f32[768]" = _foreach_div_list[33]
        getitem_2506: "f32[768]" = _foreach_div_list[34]
        getitem_2507: "f32[3072, 768]" = _foreach_div_list[35]
        getitem_2508: "f32[3072]" = _foreach_div_list[36]
        getitem_2509: "f32[768, 3072]" = _foreach_div_list[37]
        getitem_2510: "f32[768]" = _foreach_div_list[38]
        getitem_2511: "f32[768]" = _foreach_div_list[39]
        getitem_2512: "f32[768]" = _foreach_div_list[40]
        getitem_2513: "f32[768, 768]" = _foreach_div_list[41]
        getitem_2514: "f32[768]" = _foreach_div_list[42]
        getitem_2515: "f32[768, 768]" = _foreach_div_list[43]
        getitem_2516: "f32[768]" = _foreach_div_list[44]
        getitem_2517: "f32[768, 768]" = _foreach_div_list[45]
        getitem_2518: "f32[768]" = _foreach_div_list[46]
        getitem_2519: "f32[768, 768]" = _foreach_div_list[47]
        getitem_2520: "f32[768]" = _foreach_div_list[48]
        getitem_2521: "f32[768]" = _foreach_div_list[49]
        getitem_2522: "f32[768]" = _foreach_div_list[50]
        getitem_2523: "f32[3072, 768]" = _foreach_div_list[51]
        getitem_2524: "f32[3072]" = _foreach_div_list[52]
        getitem_2525: "f32[768, 3072]" = _foreach_div_list[53]
        getitem_2526: "f32[768]" = _foreach_div_list[54]
        getitem_2527: "f32[768]" = _foreach_div_list[55]
        getitem_2528: "f32[768]" = _foreach_div_list[56]
        getitem_2529: "f32[768, 768]" = _foreach_div_list[57]
        getitem_2530: "f32[768]" = _foreach_div_list[58]
        getitem_2531: "f32[768, 768]" = _foreach_div_list[59]
        getitem_2532: "f32[768]" = _foreach_div_list[60]
        getitem_2533: "f32[768, 768]" = _foreach_div_list[61]
        getitem_2534: "f32[768]" = _foreach_div_list[62]
        getitem_2535: "f32[768, 768]" = _foreach_div_list[63]
        getitem_2536: "f32[768]" = _foreach_div_list[64]
        getitem_2537: "f32[768]" = _foreach_div_list[65]
        getitem_2538: "f32[768]" = _foreach_div_list[66]
        getitem_2539: "f32[3072, 768]" = _foreach_div_list[67]
        getitem_2540: "f32[3072]" = _foreach_div_list[68]
        getitem_2541: "f32[768, 3072]" = _foreach_div_list[69]
        getitem_2542: "f32[768]" = _foreach_div_list[70]
        getitem_2543: "f32[768]" = _foreach_div_list[71]
        getitem_2544: "f32[768]" = _foreach_div_list[72]
        getitem_2545: "f32[768, 768]" = _foreach_div_list[73]
        getitem_2546: "f32[768]" = _foreach_div_list[74]
        getitem_2547: "f32[768, 768]" = _foreach_div_list[75]
        getitem_2548: "f32[768]" = _foreach_div_list[76]
        getitem_2549: "f32[768, 768]" = _foreach_div_list[77]
        getitem_2550: "f32[768]" = _foreach_div_list[78]
        getitem_2551: "f32[768, 768]" = _foreach_div_list[79]
        getitem_2552: "f32[768]" = _foreach_div_list[80]
        getitem_2553: "f32[768]" = _foreach_div_list[81]
        getitem_2554: "f32[768]" = _foreach_div_list[82]
        getitem_2555: "f32[3072, 768]" = _foreach_div_list[83]
        getitem_2556: "f32[3072]" = _foreach_div_list[84]
        getitem_2557: "f32[768, 3072]" = _foreach_div_list[85]
        getitem_2558: "f32[768]" = _foreach_div_list[86]
        getitem_2559: "f32[768]" = _foreach_div_list[87]
        getitem_2560: "f32[768]" = _foreach_div_list[88]
        getitem_2561: "f32[768, 768]" = _foreach_div_list[89]
        getitem_2562: "f32[768]" = _foreach_div_list[90]
        getitem_2563: "f32[768, 768]" = _foreach_div_list[91]
        getitem_2564: "f32[768]" = _foreach_div_list[92]
        getitem_2565: "f32[768, 768]" = _foreach_div_list[93]
        getitem_2566: "f32[768]" = _foreach_div_list[94]
        getitem_2567: "f32[768, 768]" = _foreach_div_list[95]
        getitem_2568: "f32[768]" = _foreach_div_list[96]
        getitem_2569: "f32[768]" = _foreach_div_list[97]
        getitem_2570: "f32[768]" = _foreach_div_list[98]
        getitem_2571: "f32[3072, 768]" = _foreach_div_list[99]
        getitem_2572: "f32[3072]" = _foreach_div_list[100]
        getitem_2573: "f32[768, 3072]" = _foreach_div_list[101]
        getitem_2574: "f32[768]" = _foreach_div_list[102]
        getitem_2575: "f32[768]" = _foreach_div_list[103]
        getitem_2576: "f32[768]" = _foreach_div_list[104]
        getitem_2577: "f32[768, 768]" = _foreach_div_list[105]
        getitem_2578: "f32[768]" = _foreach_div_list[106]
        getitem_2579: "f32[768, 768]" = _foreach_div_list[107]
        getitem_2580: "f32[768]" = _foreach_div_list[108]
        getitem_2581: "f32[768, 768]" = _foreach_div_list[109]
        getitem_2582: "f32[768]" = _foreach_div_list[110]
        getitem_2583: "f32[768, 768]" = _foreach_div_list[111]
        getitem_2584: "f32[768]" = _foreach_div_list[112]
        getitem_2585: "f32[768]" = _foreach_div_list[113]
        getitem_2586: "f32[768]" = _foreach_div_list[114]
        getitem_2587: "f32[3072, 768]" = _foreach_div_list[115]
        getitem_2588: "f32[3072]" = _foreach_div_list[116]
        getitem_2589: "f32[768, 3072]" = _foreach_div_list[117]
        getitem_2590: "f32[768]" = _foreach_div_list[118]
        getitem_2591: "f32[768]" = _foreach_div_list[119]
        getitem_2592: "f32[768]" = _foreach_div_list[120]
        getitem_2593: "f32[768, 768]" = _foreach_div_list[121]
        getitem_2594: "f32[768]" = _foreach_div_list[122]
        getitem_2595: "f32[768, 768]" = _foreach_div_list[123]
        getitem_2596: "f32[768]" = _foreach_div_list[124]
        getitem_2597: "f32[768, 768]" = _foreach_div_list[125]
        getitem_2598: "f32[768]" = _foreach_div_list[126]
        getitem_2599: "f32[768, 768]" = _foreach_div_list[127]
        getitem_2600: "f32[768]" = _foreach_div_list[128]
        getitem_2601: "f32[768]" = _foreach_div_list[129]
        getitem_2602: "f32[768]" = _foreach_div_list[130]
        getitem_2603: "f32[3072, 768]" = _foreach_div_list[131]
        getitem_2604: "f32[3072]" = _foreach_div_list[132]
        getitem_2605: "f32[768, 3072]" = _foreach_div_list[133]
        getitem_2606: "f32[768]" = _foreach_div_list[134]
        getitem_2607: "f32[768]" = _foreach_div_list[135]
        getitem_2608: "f32[768]" = _foreach_div_list[136]
        getitem_2609: "f32[768, 768]" = _foreach_div_list[137]
        getitem_2610: "f32[768]" = _foreach_div_list[138]
        getitem_2611: "f32[768, 768]" = _foreach_div_list[139]
        getitem_2612: "f32[768]" = _foreach_div_list[140]
        getitem_2613: "f32[768, 768]" = _foreach_div_list[141]
        getitem_2614: "f32[768]" = _foreach_div_list[142]
        getitem_2615: "f32[768, 768]" = _foreach_div_list[143]
        getitem_2616: "f32[768]" = _foreach_div_list[144]
        getitem_2617: "f32[768]" = _foreach_div_list[145]
        getitem_2618: "f32[768]" = _foreach_div_list[146]
        getitem_2619: "f32[3072, 768]" = _foreach_div_list[147]
        getitem_2620: "f32[3072]" = _foreach_div_list[148]
        getitem_2621: "f32[768, 3072]" = _foreach_div_list[149]
        getitem_2622: "f32[768]" = _foreach_div_list[150]
        getitem_2623: "f32[768]" = _foreach_div_list[151]
        getitem_2624: "f32[768]" = _foreach_div_list[152]
        getitem_2625: "f32[768, 768]" = _foreach_div_list[153]
        getitem_2626: "f32[768]" = _foreach_div_list[154]
        getitem_2627: "f32[768, 768]" = _foreach_div_list[155]
        getitem_2628: "f32[768]" = _foreach_div_list[156]
        getitem_2629: "f32[768, 768]" = _foreach_div_list[157]
        getitem_2630: "f32[768]" = _foreach_div_list[158]
        getitem_2631: "f32[768, 768]" = _foreach_div_list[159]
        getitem_2632: "f32[768]" = _foreach_div_list[160]
        getitem_2633: "f32[768]" = _foreach_div_list[161]
        getitem_2634: "f32[768]" = _foreach_div_list[162]
        getitem_2635: "f32[3072, 768]" = _foreach_div_list[163]
        getitem_2636: "f32[3072]" = _foreach_div_list[164]
        getitem_2637: "f32[768, 3072]" = _foreach_div_list[165]
        getitem_2638: "f32[768]" = _foreach_div_list[166]
        getitem_2639: "f32[768]" = _foreach_div_list[167]
        getitem_2640: "f32[768]" = _foreach_div_list[168]
        getitem_2641: "f32[768, 768]" = _foreach_div_list[169]
        getitem_2642: "f32[768]" = _foreach_div_list[170]
        getitem_2643: "f32[768, 768]" = _foreach_div_list[171]
        getitem_2644: "f32[768]" = _foreach_div_list[172]
        getitem_2645: "f32[768, 768]" = _foreach_div_list[173]
        getitem_2646: "f32[768]" = _foreach_div_list[174]
        getitem_2647: "f32[768, 768]" = _foreach_div_list[175]
        getitem_2648: "f32[768]" = _foreach_div_list[176]
        getitem_2649: "f32[768]" = _foreach_div_list[177]
        getitem_2650: "f32[768]" = _foreach_div_list[178]
        getitem_2651: "f32[3072, 768]" = _foreach_div_list[179]
        getitem_2652: "f32[3072]" = _foreach_div_list[180]
        getitem_2653: "f32[768, 3072]" = _foreach_div_list[181]
        getitem_2654: "f32[768]" = _foreach_div_list[182]
        getitem_2655: "f32[768]" = _foreach_div_list[183]
        getitem_2656: "f32[768]" = _foreach_div_list[184]
        getitem_2657: "f32[768, 768]" = _foreach_div_list[185]
        getitem_2658: "f32[768]" = _foreach_div_list[186]
        getitem_2659: "f32[768, 768]" = _foreach_div_list[187]
        getitem_2660: "f32[768]" = _foreach_div_list[188]
        getitem_2661: "f32[768, 768]" = _foreach_div_list[189]
        getitem_2662: "f32[768]" = _foreach_div_list[190]
        getitem_2663: "f32[768, 768]" = _foreach_div_list[191]
        getitem_2664: "f32[768]" = _foreach_div_list[192]
        getitem_2665: "f32[768]" = _foreach_div_list[193]
        getitem_2666: "f32[768]" = _foreach_div_list[194]
        getitem_2667: "f32[3072, 768]" = _foreach_div_list[195]
        getitem_2668: "f32[3072]" = _foreach_div_list[196]
        getitem_2669: "f32[768, 3072]" = _foreach_div_list[197]
        getitem_2670: "f32[768]" = _foreach_div_list[198]
        getitem_2671: "f32[768]" = _foreach_div_list[199]
        getitem_2672: "f32[768]" = _foreach_div_list[200]
        getitem_2673: "f32[30522]" = _foreach_div_list[201]
        getitem_2674: "f32[768, 768]" = _foreach_div_list[202]
        getitem_2675: "f32[768]" = _foreach_div_list[203]
        getitem_2676: "f32[768]" = _foreach_div_list[204]
        getitem_2677: "f32[768]" = _foreach_div_list[205];  _foreach_div_list = None
        return (getitem, getitem_412, getitem_413, getitem_414, getitem_415, getitem_416, getitem_417, getitem_418, getitem_419, getitem_420, getitem_421, getitem_422, getitem_423, getitem_424, getitem_425, getitem_426, getitem_427, getitem_428, getitem_429, getitem_430, getitem_431, getitem_432, getitem_433, getitem_434, getitem_435, getitem_436, getitem_437, getitem_438, getitem_439, getitem_440, getitem_441, getitem_442, getitem_443, getitem_444, getitem_445, getitem_446, getitem_447, getitem_448, getitem_449, getitem_450, getitem_451, getitem_452, getitem_453, getitem_454, getitem_455, getitem_456, getitem_457, getitem_458, getitem_459, getitem_460, getitem_461, getitem_462, getitem_463, getitem_464, getitem_465, getitem_466, getitem_467, getitem_468, getitem_469, getitem_470, getitem_471, getitem_472, getitem_473, getitem_474, getitem_475, getitem_476, getitem_477, getitem_478, getitem_479, getitem_480, getitem_481, getitem_482, getitem_483, getitem_484, getitem_485, getitem_486, getitem_487, getitem_488, getitem_489, getitem_490, getitem_491, getitem_492, getitem_493, getitem_494, getitem_495, getitem_496, getitem_497, getitem_498, getitem_499, getitem_500, getitem_501, getitem_502, getitem_503, getitem_504, getitem_505, getitem_506, getitem_507, getitem_508, getitem_509, getitem_510, getitem_511, getitem_512, getitem_513, getitem_514, getitem_515, getitem_516, getitem_517, getitem_518, getitem_519, getitem_520, getitem_521, getitem_522, getitem_523, getitem_524, getitem_525, getitem_526, getitem_527, getitem_528, getitem_529, getitem_530, getitem_531, getitem_532, getitem_533, getitem_534, getitem_535, getitem_536, getitem_537, getitem_538, getitem_539, getitem_540, getitem_541, getitem_542, getitem_543, getitem_544, getitem_545, getitem_546, getitem_547, getitem_548, getitem_549, getitem_550, getitem_551, getitem_552, getitem_553, getitem_554, getitem_555, getitem_556, getitem_557, getitem_558, getitem_559, getitem_560, getitem_561, getitem_562, getitem_563, getitem_564, getitem_565, getitem_566, getitem_567, getitem_568, getitem_569, getitem_570, getitem_571, getitem_572, getitem_573, getitem_574, getitem_575, getitem_576, getitem_577, getitem_578, getitem_579, getitem_580, getitem_581, getitem_582, getitem_583, getitem_584, getitem_585, getitem_586, getitem_587, getitem_588, getitem_589, getitem_590, getitem_591, getitem_592, getitem_593, getitem_594, getitem_595, getitem_596, getitem_597, getitem_598, getitem_599, getitem_600, getitem_601, getitem_602, getitem_603, getitem_604, getitem_605, getitem_606, getitem_607, getitem_608, getitem_609, getitem_610, getitem_611, getitem_612, getitem_613, getitem_614, getitem_615, getitem_616, getitem_2472, getitem_2473, getitem_2474, getitem_2475, getitem_2476, getitem_2477, getitem_2478, getitem_2479, getitem_2480, getitem_2481, getitem_2482, getitem_2483, getitem_2484, getitem_2485, getitem_2486, getitem_2487, getitem_2488, getitem_2489, getitem_2490, getitem_2491, getitem_2492, getitem_2493, getitem_2494, getitem_2495, getitem_2496, getitem_2497, getitem_2498, getitem_2499, getitem_2500, getitem_2501, getitem_2502, getitem_2503, getitem_2504, getitem_2505, getitem_2506, getitem_2507, getitem_2508, getitem_2509, getitem_2510, getitem_2511, getitem_2512, getitem_2513, getitem_2514, getitem_2515, getitem_2516, getitem_2517, getitem_2518, getitem_2519, getitem_2520, getitem_2521, getitem_2522, getitem_2523, getitem_2524, getitem_2525, getitem_2526, getitem_2527, getitem_2528, getitem_2529, getitem_2530, getitem_2531, getitem_2532, getitem_2533, getitem_2534, getitem_2535, getitem_2536, getitem_2537, getitem_2538, getitem_2539, getitem_2540, getitem_2541, getitem_2542, getitem_2543, getitem_2544, getitem_2545, getitem_2546, getitem_2547, getitem_2548, getitem_2549, getitem_2550, getitem_2551, getitem_2552, getitem_2553, getitem_2554, getitem_2555, getitem_2556, getitem_2557, getitem_2558, getitem_2559, getitem_2560, getitem_2561, getitem_2562, getitem_2563, getitem_2564, getitem_2565, getitem_2566, getitem_2567, getitem_2568, getitem_2569, getitem_2570, getitem_2571, getitem_2572, getitem_2573, getitem_2574, getitem_2575, getitem_2576, getitem_2577, getitem_2578, getitem_2579, getitem_2580, getitem_2581, getitem_2582, getitem_2583, getitem_2584, getitem_2585, getitem_2586, getitem_2587, getitem_2588, getitem_2589, getitem_2590, getitem_2591, getitem_2592, getitem_2593, getitem_2594, getitem_2595, getitem_2596, getitem_2597, getitem_2598, getitem_2599, getitem_2600, getitem_2601, getitem_2602, getitem_2603, getitem_2604, getitem_2605, getitem_2606, getitem_2607, getitem_2608, getitem_2609, getitem_2610, getitem_2611, getitem_2612, getitem_2613, getitem_2614, getitem_2615, getitem_2616, getitem_2617, getitem_2618, getitem_2619, getitem_2620, getitem_2621, getitem_2622, getitem_2623, getitem_2624, getitem_2625, getitem_2626, getitem_2627, getitem_2628, getitem_2629, getitem_2630, getitem_2631, getitem_2632, getitem_2633, getitem_2634, getitem_2635, getitem_2636, getitem_2637, getitem_2638, getitem_2639, getitem_2640, getitem_2641, getitem_2642, getitem_2643, getitem_2644, getitem_2645, getitem_2646, getitem_2647, getitem_2648, getitem_2649, getitem_2650, getitem_2651, getitem_2652, getitem_2653, getitem_2654, getitem_2655, getitem_2656, getitem_2657, getitem_2658, getitem_2659, getitem_2660, getitem_2661, getitem_2662, getitem_2663, getitem_2664, getitem_2665, getitem_2666, getitem_2667, getitem_2668, getitem_2669, getitem_2670, getitem_2671, getitem_2672, getitem_2673, getitem_2674, getitem_2675, getitem_2676, getitem_2677)


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
