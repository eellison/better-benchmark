"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s9_g77
Pattern hash: 928985a288cd
Shape hash: 214ebe81
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg306_1: "f32[]", arg301_1: "f32[]", arg307_1: "f32[]", arg308_1: "f32[]", arg309_1: "f32[]", arg310_1: "f32[]", arg311_1: "f32[]", arg312_1: "f32[]", arg313_1: "f32[]", arg314_1: "f32[]", arg315_1: "f32[]", arg316_1: "f32[]", arg317_1: "f32[]", arg318_1: "f32[]", arg319_1: "f32[]", arg320_1: "f32[]", arg321_1: "f32[]", arg322_1: "f32[]", arg323_1: "f32[]", arg324_1: "f32[]", arg325_1: "f32[]", arg326_1: "f32[]", arg327_1: "f32[]", arg328_1: "f32[]", arg329_1: "f32[]", arg330_1: "f32[]", arg331_1: "f32[]", arg332_1: "f32[]", arg333_1: "f32[]", arg334_1: "f32[]", arg335_1: "f32[]", arg336_1: "f32[]", arg337_1: "f32[]", arg338_1: "f32[]", arg339_1: "f32[]", arg340_1: "f32[]", arg341_1: "f32[]", arg342_1: "f32[]", arg343_1: "f32[]", arg344_1: "f32[]", arg345_1: "f32[]", arg346_1: "f32[]", arg347_1: "f32[]", arg348_1: "f32[]", arg349_1: "f32[]", arg350_1: "f32[]", arg351_1: "f32[]", arg352_1: "f32[]", arg353_1: "f32[]", arg354_1: "f32[]", arg355_1: "f32[]", arg356_1: "f32[]", arg357_1: "f32[]", arg358_1: "f32[]", arg359_1: "f32[]", arg360_1: "f32[]", arg361_1: "f32[]", arg362_1: "f32[]", arg363_1: "f32[]", arg364_1: "f32[]", arg365_1: "f32[]", arg366_1: "f32[]", arg367_1: "f32[]", arg368_1: "f32[]", arg369_1: "f32[]", arg370_1: "f32[]", arg371_1: "f32[]", arg372_1: "f32[]", arg373_1: "f32[]", arg374_1: "f32[]", arg375_1: "f32[]", arg376_1: "f32[]", arg377_1: "f32[]", arg378_1: "f32[]", arg379_1: "f32[]", arg380_1: "f32[]", arg381_1: "f32[]", arg382_1: "f32[]", arg383_1: "f32[]", arg384_1: "f32[]", arg385_1: "f32[]", arg386_1: "f32[]", arg387_1: "f32[]", arg388_1: "f32[]", arg389_1: "f32[]", arg390_1: "f32[]", arg391_1: "f32[]", arg392_1: "f32[]", arg393_1: "f32[]", arg394_1: "f32[]", arg395_1: "f32[]", arg396_1: "f32[]", arg397_1: "f32[]", arg398_1: "f32[]", arg399_1: "f32[]", arg400_1: "f32[]", arg401_1: "f32[]", arg402_1: "f32[]", arg403_1: "f32[]", arg404_1: "f32[]", arg405_1: "f32[]", arg406_1: "f32[]", arg407_1: "f32[]", arg408_1: "f32[]", arg409_1: "f32[]", arg410_1: "f32[]", arg411_1: "f32[]", arg412_1: "f32[]", arg413_1: "f32[]", arg414_1: "f32[]", arg415_1: "f32[]", arg416_1: "f32[]", arg417_1: "f32[]", arg418_1: "f32[]", arg419_1: "f32[]", arg420_1: "f32[]", arg421_1: "f32[]", arg422_1: "f32[]", arg423_1: "f32[]", arg424_1: "f32[]", arg425_1: "f32[]", arg426_1: "f32[]", arg427_1: "f32[]", arg428_1: "f32[]", arg429_1: "f32[]", arg430_1: "f32[]", arg431_1: "f32[]", arg432_1: "f32[]", arg433_1: "f32[]", arg434_1: "f32[]", arg435_1: "f32[]", arg436_1: "f32[]", arg437_1: "f32[]", arg438_1: "f32[]", arg439_1: "f32[]", arg440_1: "f32[]", arg441_1: "f32[]", arg442_1: "f32[]", arg443_1: "f32[]", arg444_1: "f32[]", arg445_1: "f32[]", arg446_1: "f32[]", arg447_1: "f32[]", arg448_1: "f32[]", arg449_1: "f32[]", arg450_1: "f32[]", arg451_1: "f32[]", arg452_1: "f32[]", arg453_1: "f32[]", arg454_1: "f32[]", arg455_1: "f32[]", arg456_1: "f32[]", arg457_1: "f32[]", arg458_1: "f32[]", arg459_1: "f32[]", arg460_1: "f32[]", arg461_1: "f32[]", arg462_1: "f32[]", arg463_1: "f32[]", arg464_1: "f32[]", arg465_1: "f32[]", arg466_1: "f32[]", arg467_1: "f32[]", arg468_1: "f32[]", arg469_1: "f32[]", arg470_1: "f32[]", arg471_1: "f32[]", arg472_1: "f32[]", arg473_1: "f32[]", arg474_1: "f32[]", arg475_1: "f32[]", arg476_1: "f32[]", arg477_1: "f32[]", arg478_1: "f32[]", arg479_1: "f32[]", arg480_1: "f32[]", arg481_1: "f32[]", arg482_1: "f32[]", arg483_1: "f32[]", arg484_1: "f32[]", arg485_1: "f32[]", arg486_1: "f32[]", arg487_1: "f32[]", arg488_1: "f32[]", arg489_1: "f32[]", arg490_1: "f32[]", arg491_1: "f32[]", arg492_1: "f32[]", arg493_1: "f32[]", arg494_1: "f32[]", arg495_1: "f32[]", arg496_1: "f32[]", arg497_1: "f32[]", arg498_1: "f32[]", arg499_1: "f32[]", arg500_1: "f32[]", arg501_1: "f32[]", arg502_1: "f32[]", arg503_1: "f32[]", arg504_1: "f32[]", arg505_1: "f32[]", arg506_1: "f32[]", arg507_1: "f32[]", arg508_1: "f32[]", arg509_1: "f32[]", arg510_1: "f32[]", arg511_1: "f32[]", arg512_1: "f32[]", arg513_1: "f32[]", arg514_1: "f32[]", arg515_1: "f32[]", arg516_1: "f32[]", arg517_1: "f32[]", arg518_1: "f32[]", arg519_1: "f32[]", arg520_1: "f32[]", arg521_1: "f32[]", arg522_1: "f32[]", arg523_1: "f32[]", arg524_1: "f32[]", arg525_1: "f32[]", arg526_1: "f32[]", arg527_1: "f32[]", arg528_1: "f32[]", arg529_1: "f32[]", arg530_1: "f32[]", arg531_1: "f32[]", arg532_1: "f32[]", arg533_1: "f32[]", arg534_1: "f32[]", arg535_1: "f32[]", arg536_1: "f32[]", arg537_1: "f32[]", arg538_1: "f32[]", arg539_1: "f32[]", arg540_1: "f32[]", arg541_1: "f32[]", arg542_1: "f32[]", arg543_1: "f32[]", arg544_1: "f32[]", arg545_1: "f32[]", arg546_1: "f32[]", arg547_1: "f32[]", arg548_1: "f32[]", arg549_1: "f32[]", arg550_1: "f32[]", arg551_1: "f32[]", arg552_1: "f32[]", arg553_1: "f32[]", arg554_1: "f32[]", arg555_1: "f32[]", arg556_1: "f32[]", arg557_1: "f32[]", arg558_1: "f32[]", arg559_1: "f32[]", arg560_1: "f32[]", arg561_1: "f32[]", arg562_1: "f32[]", arg563_1: "f32[]", arg564_1: "f32[]", arg565_1: "f32[]", arg566_1: "f32[]", arg567_1: "f32[]", arg568_1: "f32[]", arg569_1: "f32[]", arg570_1: "f32[]", arg571_1: "f32[]", arg572_1: "f32[]", arg573_1: "f32[]", arg574_1: "f32[]", arg575_1: "f32[]", arg576_1: "f32[]", arg577_1: "f32[]", arg578_1: "f32[]", arg579_1: "f32[]", arg580_1: "f32[]", arg581_1: "f32[]", arg582_1: "f32[]", arg583_1: "f32[]", arg584_1: "f32[]", arg585_1: "f32[]", arg586_1: "f32[]", arg587_1: "f32[]", arg588_1: "f32[]", arg589_1: "f32[]", arg590_1: "f32[]", arg591_1: "f32[]", arg592_1: "f32[]", arg593_1: "f32[]", arg594_1: "f32[]", arg595_1: "f32[]", arg596_1: "f32[]", arg597_1: "f32[]", arg598_1: "f32[]", arg599_1: "f32[]", arg600_1: "f32[]", arg601_1: "f32[]", arg602_1: "f32[]", arg603_1: "f32[]", arg604_1: "f32[]", arg605_1: "f32[]"):
        # No stacktrace found for following nodes
        _foreach_add_scalar = torch.ops.aten._foreach_add.Scalar([arg306_1, arg301_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1], 1);  arg306_1 = arg301_1 = arg307_1 = arg308_1 = arg309_1 = arg310_1 = arg311_1 = arg312_1 = arg313_1 = arg314_1 = arg315_1 = arg316_1 = arg317_1 = arg318_1 = arg319_1 = arg320_1 = arg321_1 = arg322_1 = arg323_1 = arg324_1 = arg325_1 = arg326_1 = arg327_1 = arg328_1 = arg329_1 = arg330_1 = arg331_1 = arg332_1 = arg333_1 = arg334_1 = arg335_1 = arg336_1 = arg337_1 = arg338_1 = arg339_1 = arg340_1 = arg341_1 = arg342_1 = arg343_1 = arg344_1 = arg345_1 = arg346_1 = arg347_1 = arg348_1 = arg349_1 = arg350_1 = arg351_1 = arg352_1 = arg353_1 = arg354_1 = arg355_1 = arg356_1 = arg357_1 = arg358_1 = arg359_1 = arg360_1 = arg361_1 = arg362_1 = arg363_1 = arg364_1 = arg365_1 = arg366_1 = arg367_1 = arg368_1 = arg369_1 = arg370_1 = arg371_1 = arg372_1 = arg373_1 = arg374_1 = arg375_1 = arg376_1 = arg377_1 = arg378_1 = arg379_1 = arg380_1 = arg381_1 = arg382_1 = arg383_1 = arg384_1 = arg385_1 = arg386_1 = arg387_1 = arg388_1 = arg389_1 = arg390_1 = arg391_1 = arg392_1 = arg393_1 = arg394_1 = arg395_1 = arg396_1 = arg397_1 = arg398_1 = arg399_1 = arg400_1 = arg401_1 = arg402_1 = arg403_1 = arg404_1 = arg405_1 = arg406_1 = arg407_1 = arg408_1 = arg409_1 = arg410_1 = arg411_1 = arg412_1 = arg413_1 = arg414_1 = arg415_1 = arg416_1 = arg417_1 = arg418_1 = arg419_1 = arg420_1 = arg421_1 = arg422_1 = arg423_1 = arg424_1 = arg425_1 = arg426_1 = arg427_1 = arg428_1 = arg429_1 = arg430_1 = arg431_1 = arg432_1 = arg433_1 = arg434_1 = arg435_1 = arg436_1 = arg437_1 = arg438_1 = arg439_1 = arg440_1 = arg441_1 = arg442_1 = arg443_1 = arg444_1 = arg445_1 = arg446_1 = arg447_1 = arg448_1 = arg449_1 = arg450_1 = arg451_1 = arg452_1 = arg453_1 = arg454_1 = arg455_1 = arg456_1 = arg457_1 = arg458_1 = arg459_1 = arg460_1 = arg461_1 = arg462_1 = arg463_1 = arg464_1 = arg465_1 = arg466_1 = arg467_1 = arg468_1 = arg469_1 = arg470_1 = arg471_1 = arg472_1 = arg473_1 = arg474_1 = arg475_1 = arg476_1 = arg477_1 = arg478_1 = arg479_1 = arg480_1 = arg481_1 = arg482_1 = arg483_1 = arg484_1 = arg485_1 = arg486_1 = arg487_1 = arg488_1 = arg489_1 = arg490_1 = arg491_1 = arg492_1 = arg493_1 = arg494_1 = arg495_1 = arg496_1 = arg497_1 = arg498_1 = arg499_1 = arg500_1 = arg501_1 = arg502_1 = arg503_1 = arg504_1 = arg505_1 = arg506_1 = arg507_1 = arg508_1 = arg509_1 = arg510_1 = arg511_1 = arg512_1 = arg513_1 = arg514_1 = arg515_1 = arg516_1 = arg517_1 = arg518_1 = arg519_1 = arg520_1 = arg521_1 = arg522_1 = arg523_1 = arg524_1 = arg525_1 = arg526_1 = arg527_1 = arg528_1 = arg529_1 = arg530_1 = arg531_1 = arg532_1 = arg533_1 = arg534_1 = arg535_1 = arg536_1 = arg537_1 = arg538_1 = arg539_1 = arg540_1 = arg541_1 = arg542_1 = arg543_1 = arg544_1 = arg545_1 = arg546_1 = arg547_1 = arg548_1 = arg549_1 = arg550_1 = arg551_1 = arg552_1 = arg553_1 = arg554_1 = arg555_1 = arg556_1 = arg557_1 = arg558_1 = arg559_1 = arg560_1 = arg561_1 = arg562_1 = arg563_1 = arg564_1 = arg565_1 = arg566_1 = arg567_1 = arg568_1 = arg569_1 = arg570_1 = arg571_1 = arg572_1 = arg573_1 = arg574_1 = arg575_1 = arg576_1 = arg577_1 = arg578_1 = arg579_1 = arg580_1 = arg581_1 = arg582_1 = arg583_1 = arg584_1 = arg585_1 = arg586_1 = arg587_1 = arg588_1 = arg589_1 = arg590_1 = arg591_1 = arg592_1 = arg593_1 = arg594_1 = arg595_1 = arg596_1 = arg597_1 = arg598_1 = arg599_1 = arg600_1 = arg601_1 = arg602_1 = arg603_1 = arg604_1 = arg605_1 = None
        getitem: "f32[]" = _foreach_add_scalar[0]
        getitem_1: "f32[]" = _foreach_add_scalar[1]
        getitem_2: "f32[]" = _foreach_add_scalar[2]
        getitem_3: "f32[]" = _foreach_add_scalar[3]
        getitem_4: "f32[]" = _foreach_add_scalar[4]
        getitem_5: "f32[]" = _foreach_add_scalar[5]
        getitem_6: "f32[]" = _foreach_add_scalar[6]
        getitem_7: "f32[]" = _foreach_add_scalar[7]
        getitem_8: "f32[]" = _foreach_add_scalar[8]
        getitem_9: "f32[]" = _foreach_add_scalar[9]
        getitem_10: "f32[]" = _foreach_add_scalar[10]
        getitem_11: "f32[]" = _foreach_add_scalar[11]
        getitem_12: "f32[]" = _foreach_add_scalar[12]
        getitem_13: "f32[]" = _foreach_add_scalar[13]
        getitem_14: "f32[]" = _foreach_add_scalar[14]
        getitem_15: "f32[]" = _foreach_add_scalar[15]
        getitem_16: "f32[]" = _foreach_add_scalar[16]
        getitem_17: "f32[]" = _foreach_add_scalar[17]
        getitem_18: "f32[]" = _foreach_add_scalar[18]
        getitem_19: "f32[]" = _foreach_add_scalar[19]
        getitem_20: "f32[]" = _foreach_add_scalar[20]
        getitem_21: "f32[]" = _foreach_add_scalar[21]
        getitem_22: "f32[]" = _foreach_add_scalar[22]
        getitem_23: "f32[]" = _foreach_add_scalar[23]
        getitem_24: "f32[]" = _foreach_add_scalar[24]
        getitem_25: "f32[]" = _foreach_add_scalar[25]
        getitem_26: "f32[]" = _foreach_add_scalar[26]
        getitem_27: "f32[]" = _foreach_add_scalar[27]
        getitem_28: "f32[]" = _foreach_add_scalar[28]
        getitem_29: "f32[]" = _foreach_add_scalar[29]
        getitem_30: "f32[]" = _foreach_add_scalar[30]
        getitem_31: "f32[]" = _foreach_add_scalar[31]
        getitem_32: "f32[]" = _foreach_add_scalar[32]
        getitem_33: "f32[]" = _foreach_add_scalar[33]
        getitem_34: "f32[]" = _foreach_add_scalar[34]
        getitem_35: "f32[]" = _foreach_add_scalar[35]
        getitem_36: "f32[]" = _foreach_add_scalar[36]
        getitem_37: "f32[]" = _foreach_add_scalar[37]
        getitem_38: "f32[]" = _foreach_add_scalar[38]
        getitem_39: "f32[]" = _foreach_add_scalar[39]
        getitem_40: "f32[]" = _foreach_add_scalar[40]
        getitem_41: "f32[]" = _foreach_add_scalar[41]
        getitem_42: "f32[]" = _foreach_add_scalar[42]
        getitem_43: "f32[]" = _foreach_add_scalar[43]
        getitem_44: "f32[]" = _foreach_add_scalar[44]
        getitem_45: "f32[]" = _foreach_add_scalar[45]
        getitem_46: "f32[]" = _foreach_add_scalar[46]
        getitem_47: "f32[]" = _foreach_add_scalar[47]
        getitem_48: "f32[]" = _foreach_add_scalar[48]
        getitem_49: "f32[]" = _foreach_add_scalar[49]
        getitem_50: "f32[]" = _foreach_add_scalar[50]
        getitem_51: "f32[]" = _foreach_add_scalar[51]
        getitem_52: "f32[]" = _foreach_add_scalar[52]
        getitem_53: "f32[]" = _foreach_add_scalar[53]
        getitem_54: "f32[]" = _foreach_add_scalar[54]
        getitem_55: "f32[]" = _foreach_add_scalar[55]
        getitem_56: "f32[]" = _foreach_add_scalar[56]
        getitem_57: "f32[]" = _foreach_add_scalar[57]
        getitem_58: "f32[]" = _foreach_add_scalar[58]
        getitem_59: "f32[]" = _foreach_add_scalar[59]
        getitem_60: "f32[]" = _foreach_add_scalar[60]
        getitem_61: "f32[]" = _foreach_add_scalar[61]
        getitem_62: "f32[]" = _foreach_add_scalar[62]
        getitem_63: "f32[]" = _foreach_add_scalar[63]
        getitem_64: "f32[]" = _foreach_add_scalar[64]
        getitem_65: "f32[]" = _foreach_add_scalar[65]
        getitem_66: "f32[]" = _foreach_add_scalar[66]
        getitem_67: "f32[]" = _foreach_add_scalar[67]
        getitem_68: "f32[]" = _foreach_add_scalar[68]
        getitem_69: "f32[]" = _foreach_add_scalar[69]
        getitem_70: "f32[]" = _foreach_add_scalar[70]
        getitem_71: "f32[]" = _foreach_add_scalar[71]
        getitem_72: "f32[]" = _foreach_add_scalar[72]
        getitem_73: "f32[]" = _foreach_add_scalar[73]
        getitem_74: "f32[]" = _foreach_add_scalar[74]
        getitem_75: "f32[]" = _foreach_add_scalar[75]
        getitem_76: "f32[]" = _foreach_add_scalar[76]
        getitem_77: "f32[]" = _foreach_add_scalar[77]
        getitem_78: "f32[]" = _foreach_add_scalar[78]
        getitem_79: "f32[]" = _foreach_add_scalar[79]
        getitem_80: "f32[]" = _foreach_add_scalar[80]
        getitem_81: "f32[]" = _foreach_add_scalar[81]
        getitem_82: "f32[]" = _foreach_add_scalar[82]
        getitem_83: "f32[]" = _foreach_add_scalar[83]
        getitem_84: "f32[]" = _foreach_add_scalar[84]
        getitem_85: "f32[]" = _foreach_add_scalar[85]
        getitem_86: "f32[]" = _foreach_add_scalar[86]
        getitem_87: "f32[]" = _foreach_add_scalar[87]
        getitem_88: "f32[]" = _foreach_add_scalar[88]
        getitem_89: "f32[]" = _foreach_add_scalar[89]
        getitem_90: "f32[]" = _foreach_add_scalar[90]
        getitem_91: "f32[]" = _foreach_add_scalar[91]
        getitem_92: "f32[]" = _foreach_add_scalar[92]
        getitem_93: "f32[]" = _foreach_add_scalar[93]
        getitem_94: "f32[]" = _foreach_add_scalar[94]
        getitem_95: "f32[]" = _foreach_add_scalar[95]
        getitem_96: "f32[]" = _foreach_add_scalar[96]
        getitem_97: "f32[]" = _foreach_add_scalar[97]
        getitem_98: "f32[]" = _foreach_add_scalar[98]
        getitem_99: "f32[]" = _foreach_add_scalar[99]
        getitem_100: "f32[]" = _foreach_add_scalar[100]
        getitem_101: "f32[]" = _foreach_add_scalar[101]
        getitem_102: "f32[]" = _foreach_add_scalar[102]
        getitem_103: "f32[]" = _foreach_add_scalar[103]
        getitem_104: "f32[]" = _foreach_add_scalar[104]
        getitem_105: "f32[]" = _foreach_add_scalar[105]
        getitem_106: "f32[]" = _foreach_add_scalar[106]
        getitem_107: "f32[]" = _foreach_add_scalar[107]
        getitem_108: "f32[]" = _foreach_add_scalar[108]
        getitem_109: "f32[]" = _foreach_add_scalar[109]
        getitem_110: "f32[]" = _foreach_add_scalar[110]
        getitem_111: "f32[]" = _foreach_add_scalar[111]
        getitem_112: "f32[]" = _foreach_add_scalar[112]
        getitem_113: "f32[]" = _foreach_add_scalar[113]
        getitem_114: "f32[]" = _foreach_add_scalar[114]
        getitem_115: "f32[]" = _foreach_add_scalar[115]
        getitem_116: "f32[]" = _foreach_add_scalar[116]
        getitem_117: "f32[]" = _foreach_add_scalar[117]
        getitem_118: "f32[]" = _foreach_add_scalar[118]
        getitem_119: "f32[]" = _foreach_add_scalar[119]
        getitem_120: "f32[]" = _foreach_add_scalar[120]
        getitem_121: "f32[]" = _foreach_add_scalar[121]
        getitem_122: "f32[]" = _foreach_add_scalar[122]
        getitem_123: "f32[]" = _foreach_add_scalar[123]
        getitem_124: "f32[]" = _foreach_add_scalar[124]
        getitem_125: "f32[]" = _foreach_add_scalar[125]
        getitem_126: "f32[]" = _foreach_add_scalar[126]
        getitem_127: "f32[]" = _foreach_add_scalar[127]
        getitem_128: "f32[]" = _foreach_add_scalar[128]
        getitem_129: "f32[]" = _foreach_add_scalar[129]
        getitem_130: "f32[]" = _foreach_add_scalar[130]
        getitem_131: "f32[]" = _foreach_add_scalar[131]
        getitem_132: "f32[]" = _foreach_add_scalar[132]
        getitem_133: "f32[]" = _foreach_add_scalar[133]
        getitem_134: "f32[]" = _foreach_add_scalar[134]
        getitem_135: "f32[]" = _foreach_add_scalar[135]
        getitem_136: "f32[]" = _foreach_add_scalar[136]
        getitem_137: "f32[]" = _foreach_add_scalar[137]
        getitem_138: "f32[]" = _foreach_add_scalar[138]
        getitem_139: "f32[]" = _foreach_add_scalar[139]
        getitem_140: "f32[]" = _foreach_add_scalar[140]
        getitem_141: "f32[]" = _foreach_add_scalar[141]
        getitem_142: "f32[]" = _foreach_add_scalar[142]
        getitem_143: "f32[]" = _foreach_add_scalar[143]
        getitem_144: "f32[]" = _foreach_add_scalar[144]
        getitem_145: "f32[]" = _foreach_add_scalar[145]
        getitem_146: "f32[]" = _foreach_add_scalar[146]
        getitem_147: "f32[]" = _foreach_add_scalar[147]
        getitem_148: "f32[]" = _foreach_add_scalar[148]
        getitem_149: "f32[]" = _foreach_add_scalar[149]
        getitem_150: "f32[]" = _foreach_add_scalar[150]
        getitem_151: "f32[]" = _foreach_add_scalar[151]
        getitem_152: "f32[]" = _foreach_add_scalar[152]
        getitem_153: "f32[]" = _foreach_add_scalar[153]
        getitem_154: "f32[]" = _foreach_add_scalar[154]
        getitem_155: "f32[]" = _foreach_add_scalar[155]
        getitem_156: "f32[]" = _foreach_add_scalar[156]
        getitem_157: "f32[]" = _foreach_add_scalar[157]
        getitem_158: "f32[]" = _foreach_add_scalar[158]
        getitem_159: "f32[]" = _foreach_add_scalar[159]
        getitem_160: "f32[]" = _foreach_add_scalar[160]
        getitem_161: "f32[]" = _foreach_add_scalar[161]
        getitem_162: "f32[]" = _foreach_add_scalar[162]
        getitem_163: "f32[]" = _foreach_add_scalar[163]
        getitem_164: "f32[]" = _foreach_add_scalar[164]
        getitem_165: "f32[]" = _foreach_add_scalar[165]
        getitem_166: "f32[]" = _foreach_add_scalar[166]
        getitem_167: "f32[]" = _foreach_add_scalar[167]
        getitem_168: "f32[]" = _foreach_add_scalar[168]
        getitem_169: "f32[]" = _foreach_add_scalar[169]
        getitem_170: "f32[]" = _foreach_add_scalar[170]
        getitem_171: "f32[]" = _foreach_add_scalar[171]
        getitem_172: "f32[]" = _foreach_add_scalar[172]
        getitem_173: "f32[]" = _foreach_add_scalar[173]
        getitem_174: "f32[]" = _foreach_add_scalar[174]
        getitem_175: "f32[]" = _foreach_add_scalar[175]
        getitem_176: "f32[]" = _foreach_add_scalar[176]
        getitem_177: "f32[]" = _foreach_add_scalar[177]
        getitem_178: "f32[]" = _foreach_add_scalar[178]
        getitem_179: "f32[]" = _foreach_add_scalar[179]
        getitem_180: "f32[]" = _foreach_add_scalar[180]
        getitem_181: "f32[]" = _foreach_add_scalar[181]
        getitem_182: "f32[]" = _foreach_add_scalar[182]
        getitem_183: "f32[]" = _foreach_add_scalar[183]
        getitem_184: "f32[]" = _foreach_add_scalar[184]
        getitem_185: "f32[]" = _foreach_add_scalar[185]
        getitem_186: "f32[]" = _foreach_add_scalar[186]
        getitem_187: "f32[]" = _foreach_add_scalar[187]
        getitem_188: "f32[]" = _foreach_add_scalar[188]
        getitem_189: "f32[]" = _foreach_add_scalar[189]
        getitem_190: "f32[]" = _foreach_add_scalar[190]
        getitem_191: "f32[]" = _foreach_add_scalar[191]
        getitem_192: "f32[]" = _foreach_add_scalar[192]
        getitem_193: "f32[]" = _foreach_add_scalar[193]
        getitem_194: "f32[]" = _foreach_add_scalar[194]
        getitem_195: "f32[]" = _foreach_add_scalar[195]
        getitem_196: "f32[]" = _foreach_add_scalar[196]
        getitem_197: "f32[]" = _foreach_add_scalar[197]
        getitem_198: "f32[]" = _foreach_add_scalar[198]
        getitem_199: "f32[]" = _foreach_add_scalar[199]
        getitem_200: "f32[]" = _foreach_add_scalar[200]
        getitem_201: "f32[]" = _foreach_add_scalar[201]
        getitem_202: "f32[]" = _foreach_add_scalar[202]
        getitem_203: "f32[]" = _foreach_add_scalar[203]
        getitem_204: "f32[]" = _foreach_add_scalar[204]
        getitem_205: "f32[]" = _foreach_add_scalar[205]
        getitem_206: "f32[]" = _foreach_add_scalar[206]
        getitem_207: "f32[]" = _foreach_add_scalar[207]
        getitem_208: "f32[]" = _foreach_add_scalar[208]
        getitem_209: "f32[]" = _foreach_add_scalar[209]
        getitem_210: "f32[]" = _foreach_add_scalar[210]
        getitem_211: "f32[]" = _foreach_add_scalar[211]
        getitem_212: "f32[]" = _foreach_add_scalar[212]
        getitem_213: "f32[]" = _foreach_add_scalar[213]
        getitem_214: "f32[]" = _foreach_add_scalar[214]
        getitem_215: "f32[]" = _foreach_add_scalar[215]
        getitem_216: "f32[]" = _foreach_add_scalar[216]
        getitem_217: "f32[]" = _foreach_add_scalar[217]
        getitem_218: "f32[]" = _foreach_add_scalar[218]
        getitem_219: "f32[]" = _foreach_add_scalar[219]
        getitem_220: "f32[]" = _foreach_add_scalar[220]
        getitem_221: "f32[]" = _foreach_add_scalar[221]
        getitem_222: "f32[]" = _foreach_add_scalar[222]
        getitem_223: "f32[]" = _foreach_add_scalar[223]
        getitem_224: "f32[]" = _foreach_add_scalar[224]
        getitem_225: "f32[]" = _foreach_add_scalar[225]
        getitem_226: "f32[]" = _foreach_add_scalar[226]
        getitem_227: "f32[]" = _foreach_add_scalar[227]
        getitem_228: "f32[]" = _foreach_add_scalar[228]
        getitem_229: "f32[]" = _foreach_add_scalar[229]
        getitem_230: "f32[]" = _foreach_add_scalar[230]
        getitem_231: "f32[]" = _foreach_add_scalar[231]
        getitem_232: "f32[]" = _foreach_add_scalar[232]
        getitem_233: "f32[]" = _foreach_add_scalar[233]
        getitem_234: "f32[]" = _foreach_add_scalar[234]
        getitem_235: "f32[]" = _foreach_add_scalar[235]
        getitem_236: "f32[]" = _foreach_add_scalar[236]
        getitem_237: "f32[]" = _foreach_add_scalar[237]
        getitem_238: "f32[]" = _foreach_add_scalar[238]
        getitem_239: "f32[]" = _foreach_add_scalar[239]
        getitem_240: "f32[]" = _foreach_add_scalar[240]
        getitem_241: "f32[]" = _foreach_add_scalar[241]
        getitem_242: "f32[]" = _foreach_add_scalar[242]
        getitem_243: "f32[]" = _foreach_add_scalar[243]
        getitem_244: "f32[]" = _foreach_add_scalar[244]
        getitem_245: "f32[]" = _foreach_add_scalar[245]
        getitem_246: "f32[]" = _foreach_add_scalar[246]
        getitem_247: "f32[]" = _foreach_add_scalar[247]
        getitem_248: "f32[]" = _foreach_add_scalar[248]
        getitem_249: "f32[]" = _foreach_add_scalar[249]
        getitem_250: "f32[]" = _foreach_add_scalar[250]
        getitem_251: "f32[]" = _foreach_add_scalar[251]
        getitem_252: "f32[]" = _foreach_add_scalar[252]
        getitem_253: "f32[]" = _foreach_add_scalar[253]
        getitem_254: "f32[]" = _foreach_add_scalar[254]
        getitem_255: "f32[]" = _foreach_add_scalar[255]
        getitem_256: "f32[]" = _foreach_add_scalar[256]
        getitem_257: "f32[]" = _foreach_add_scalar[257]
        getitem_258: "f32[]" = _foreach_add_scalar[258]
        getitem_259: "f32[]" = _foreach_add_scalar[259]
        getitem_260: "f32[]" = _foreach_add_scalar[260]
        getitem_261: "f32[]" = _foreach_add_scalar[261]
        getitem_262: "f32[]" = _foreach_add_scalar[262]
        getitem_263: "f32[]" = _foreach_add_scalar[263]
        getitem_264: "f32[]" = _foreach_add_scalar[264]
        getitem_265: "f32[]" = _foreach_add_scalar[265]
        getitem_266: "f32[]" = _foreach_add_scalar[266]
        getitem_267: "f32[]" = _foreach_add_scalar[267]
        getitem_268: "f32[]" = _foreach_add_scalar[268]
        getitem_269: "f32[]" = _foreach_add_scalar[269]
        getitem_270: "f32[]" = _foreach_add_scalar[270]
        getitem_271: "f32[]" = _foreach_add_scalar[271]
        getitem_272: "f32[]" = _foreach_add_scalar[272]
        getitem_273: "f32[]" = _foreach_add_scalar[273]
        getitem_274: "f32[]" = _foreach_add_scalar[274]
        getitem_275: "f32[]" = _foreach_add_scalar[275]
        getitem_276: "f32[]" = _foreach_add_scalar[276]
        getitem_277: "f32[]" = _foreach_add_scalar[277]
        getitem_278: "f32[]" = _foreach_add_scalar[278]
        getitem_279: "f32[]" = _foreach_add_scalar[279]
        getitem_280: "f32[]" = _foreach_add_scalar[280]
        getitem_281: "f32[]" = _foreach_add_scalar[281]
        getitem_282: "f32[]" = _foreach_add_scalar[282]
        getitem_283: "f32[]" = _foreach_add_scalar[283]
        getitem_284: "f32[]" = _foreach_add_scalar[284]
        getitem_285: "f32[]" = _foreach_add_scalar[285]
        getitem_286: "f32[]" = _foreach_add_scalar[286]
        getitem_287: "f32[]" = _foreach_add_scalar[287]
        getitem_288: "f32[]" = _foreach_add_scalar[288]
        getitem_289: "f32[]" = _foreach_add_scalar[289]
        getitem_290: "f32[]" = _foreach_add_scalar[290]
        getitem_291: "f32[]" = _foreach_add_scalar[291]
        getitem_292: "f32[]" = _foreach_add_scalar[292]
        getitem_293: "f32[]" = _foreach_add_scalar[293]
        getitem_294: "f32[]" = _foreach_add_scalar[294]
        getitem_295: "f32[]" = _foreach_add_scalar[295]
        getitem_296: "f32[]" = _foreach_add_scalar[296]
        getitem_297: "f32[]" = _foreach_add_scalar[297]
        getitem_298: "f32[]" = _foreach_add_scalar[298]
        getitem_299: "f32[]" = _foreach_add_scalar[299]
        getitem_300: "f32[]" = _foreach_add_scalar[300];  _foreach_add_scalar = None
        return (getitem, getitem_1, getitem_2, getitem_3, getitem_4, getitem_5, getitem_6, getitem_7, getitem_8, getitem_9, getitem_10, getitem_11, getitem_12, getitem_13, getitem_14, getitem_15, getitem_16, getitem_17, getitem_18, getitem_19, getitem_20, getitem_21, getitem_22, getitem_23, getitem_24, getitem_25, getitem_26, getitem_27, getitem_28, getitem_29, getitem_30, getitem_31, getitem_32, getitem_33, getitem_34, getitem_35, getitem_36, getitem_37, getitem_38, getitem_39, getitem_40, getitem_41, getitem_42, getitem_43, getitem_44, getitem_45, getitem_46, getitem_47, getitem_48, getitem_49, getitem_50, getitem_51, getitem_52, getitem_53, getitem_54, getitem_55, getitem_56, getitem_57, getitem_58, getitem_59, getitem_60, getitem_61, getitem_62, getitem_63, getitem_64, getitem_65, getitem_66, getitem_67, getitem_68, getitem_69, getitem_70, getitem_71, getitem_72, getitem_73, getitem_74, getitem_75, getitem_76, getitem_77, getitem_78, getitem_79, getitem_80, getitem_81, getitem_82, getitem_83, getitem_84, getitem_85, getitem_86, getitem_87, getitem_88, getitem_89, getitem_90, getitem_91, getitem_92, getitem_93, getitem_94, getitem_95, getitem_96, getitem_97, getitem_98, getitem_99, getitem_100, getitem_101, getitem_102, getitem_103, getitem_104, getitem_105, getitem_106, getitem_107, getitem_108, getitem_109, getitem_110, getitem_111, getitem_112, getitem_113, getitem_114, getitem_115, getitem_116, getitem_117, getitem_118, getitem_119, getitem_120, getitem_121, getitem_122, getitem_123, getitem_124, getitem_125, getitem_126, getitem_127, getitem_128, getitem_129, getitem_130, getitem_131, getitem_132, getitem_133, getitem_134, getitem_135, getitem_136, getitem_137, getitem_138, getitem_139, getitem_140, getitem_141, getitem_142, getitem_143, getitem_144, getitem_145, getitem_146, getitem_147, getitem_148, getitem_149, getitem_150, getitem_151, getitem_152, getitem_153, getitem_154, getitem_155, getitem_156, getitem_157, getitem_158, getitem_159, getitem_160, getitem_161, getitem_162, getitem_163, getitem_164, getitem_165, getitem_166, getitem_167, getitem_168, getitem_169, getitem_170, getitem_171, getitem_172, getitem_173, getitem_174, getitem_175, getitem_176, getitem_177, getitem_178, getitem_179, getitem_180, getitem_181, getitem_182, getitem_183, getitem_184, getitem_185, getitem_186, getitem_187, getitem_188, getitem_189, getitem_190, getitem_191, getitem_192, getitem_193, getitem_194, getitem_195, getitem_196, getitem_197, getitem_198, getitem_199, getitem_200, getitem_201, getitem_202, getitem_203, getitem_204, getitem_205, getitem_206, getitem_207, getitem_208, getitem_209, getitem_210, getitem_211, getitem_212, getitem_213, getitem_214, getitem_215, getitem_216, getitem_217, getitem_218, getitem_219, getitem_220, getitem_221, getitem_222, getitem_223, getitem_224, getitem_225, getitem_226, getitem_227, getitem_228, getitem_229, getitem_230, getitem_231, getitem_232, getitem_233, getitem_234, getitem_235, getitem_236, getitem_237, getitem_238, getitem_239, getitem_240, getitem_241, getitem_242, getitem_243, getitem_244, getitem_245, getitem_246, getitem_247, getitem_248, getitem_249, getitem_250, getitem_251, getitem_252, getitem_253, getitem_254, getitem_255, getitem_256, getitem_257, getitem_258, getitem_259, getitem_260, getitem_261, getitem_262, getitem_263, getitem_264, getitem_265, getitem_266, getitem_267, getitem_268, getitem_269, getitem_270, getitem_271, getitem_272, getitem_273, getitem_274, getitem_275, getitem_276, getitem_277, getitem_278, getitem_279, getitem_280, getitem_281, getitem_282, getitem_283, getitem_284, getitem_285, getitem_286, getitem_287, getitem_288, getitem_289, getitem_290, getitem_291, getitem_292, getitem_293, getitem_294, getitem_295, getitem_296, getitem_297, getitem_298, getitem_299, getitem_300)


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
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)
