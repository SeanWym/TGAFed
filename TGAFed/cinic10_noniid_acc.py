import matplotlib.pyplot as plt
###table1-iid

# 20240606-0159-fedavg-lc-bimb-cinic_10--uda-1-0.05
aggr_acc_fedavg_uda_noniid = [0.132750004529953, 0.18273749947547913, 0.1854500025510788, 0.20863750576972961, 0.23433749377727509, 0.23291249573230743, 0.2267874926328659, 0.2206375002861023, 0.2548750042915344, 0.24157500267028809, 0.24367499351501465, 0.2542499899864197, 0.2381875067949295, 0.25562500953674316, 0.2656500041484833, 0.23432500660419464, 0.2652125060558319, 0.27344998717308044, 0.2766000032424927, 0.25491249561309814, 0.2818875014781952, 0.289837509393692, 0.27921250462532043, 0.2396875023841858, 0.27966248989105225, 0.27527499198913574, 0.28040000796318054, 0.29151248931884766, 0.28796249628067017, 0.2890250086784363, 0.26122498512268066, 0.29054999351501465, 0.2840625047683716, 0.27131250500679016, 0.2754625082015991, 0.30393749475479126, 0.2925249934196472, 0.3004249930381775, 0.3138374984264374, 0.29001250863075256, 0.3055124878883362, 0.2983250021934509, 0.32122498750686646, 0.30693748593330383, 0.29862499237060547, 0.31443750858306885, 0.3157750070095062, 0.32120001316070557, 0.29743748903274536, 0.2905125021934509, 0.3202125132083893, 0.3255999982357025, 0.31380000710487366, 0.3075000047683716, 0.33256250619888306, 0.31847500801086426, 0.326324999332428, 0.33191248774528503, 0.33371248841285706, 0.31946250796318054, 0.3262374997138977, 0.33502501249313354, 0.32403749227523804, 0.3397125005722046, 0.3344874978065491, 0.3257499933242798, 0.3471125066280365, 0.3325124979019165, 0.33611249923706055, 0.334262490272522, 0.3434624969959259, 0.345612496137619, 0.33294999599456787, 0.3372499942779541, 0.33637499809265137, 0.3389124870300293, 0.35138750076293945, 0.3146499991416931, 0.34373751282691956, 0.3389750123023987, 0.34084999561309814, 0.35321250557899475, 0.34165000915527344, 0.3545750081539154, 0.3489125072956085, 0.3434624969959259, 0.3429250121116638, 0.36032500863075256, 0.33057498931884766, 0.36291250586509705, 0.34299999475479126, 0.3583875000476837, 0.3421750068664551, 0.3598625063896179, 0.35624998807907104, 0.3623499870300293, 0.3667624890804291, 0.3468874990940094, 0.36500000953674316, 0.36457499861717224, 0.35622501373291016, 0.36195001006126404, 0.3568499982357025, 0.3731124997138977, 0.3501875102519989, 0.3681749999523163, 0.3567749857902527, 0.3626374900341034, 0.3711499869823456, 0.3718999922275543, 0.36531248688697815, 0.36778751015663147, 0.3653999865055084, 0.3719500005245209, 0.37568750977516174, 0.37258750200271606, 0.36079999804496765, 0.3782374858856201, 0.36963748931884766, 0.3767875134944916, 0.367062509059906, 0.3641375005245209, 0.37402498722076416, 0.36733749508857727, 0.37242498993873596, 0.3612000048160553, 0.3696874976158142, 0.3683375120162964, 0.375062495470047, 0.37732499837875366, 0.38659998774528503, 0.38388749957084656, 0.3683624863624573, 0.3736625015735626, 0.37185001373291016, 0.37139999866485596, 0.3819875121116638, 0.38032498955726624, 0.38721251487731934, 0.3777500092983246, 0.3844375014305115, 0.3758625090122223, 0.388887494802475, 0.37261250615119934, 0.3891499936580658, 0.3883250057697296, 0.38429999351501465, 0.3631249964237213, 0.39313751459121704, 0.37255001068115234, 0.38977500796318054, 0.37299999594688416, 0.382625013589859, 0.39135000109672546, 0.3844375014305115, 0.37792500853538513, 0.38817501068115234, 0.39246249198913574, 0.383525013923645, 0.3939875066280365, 0.38833749294281006, 0.38871249556541443, 0.3948250114917755, 0.3916124999523163, 0.39294999837875366, 0.3867124915122986, 0.3944999873638153, 0.4023124873638153, 0.40217500925064087, 0.39793750643730164, 0.39791250228881836, 0.39625000953674316, 0.40626248717308044, 0.39646250009536743, 0.3938249945640564, 0.40037500858306885, 0.39326250553131104, 0.4014374911785126, 0.4019249975681305, 0.39272499084472656, 0.3989250063896179, 0.3916875123977661, 0.39617499709129333, 0.4066750109195709, 0.40217500925064087, 0.4071125090122223, 0.4017750024795532, 0.40825000405311584, 0.40578749775886536, 0.41167500615119934, 0.39028748869895935, 0.40584999322891235, 0.4005250036716461, 0.40812501311302185, 0.4112499952316284, 0.40630000829696655, 0.40441250801086426, 0.3990750014781952, 0.4000124931335449, 0.40712499618530273]
# 20240606-0728-fedprox-lc-bimb-cinic_10--uda-1-0.05
aggr_acc_fedprox_uda_noniid = [0.11752499639987946, 0.12303750216960907, 0.16179999709129333, 0.18581250309944153, 0.18678750097751617, 0.23001250624656677, 0.2089499980211258, 0.17030000686645508, 0.2329374998807907, 0.24467499554157257, 0.2534250020980835, 0.226950004696846, 0.26337501406669617, 0.25946250557899475, 0.23386250436306, 0.2639249861240387, 0.268512487411499, 0.2540625035762787, 0.27201250195503235, 0.257874995470047, 0.2776874899864197, 0.27986249327659607, 0.2706874907016754, 0.2630625069141388, 0.2783375084400177, 0.2661625146865845, 0.2776249945163727, 0.27845001220703125, 0.26586249470710754, 0.2858999967575073, 0.2690249979496002, 0.28802499175071716, 0.2845500111579895, 0.2791374921798706, 0.29272499680519104, 0.27044999599456787, 0.2956874966621399, 0.30176249146461487, 0.29954999685287476, 0.2966249883174896, 0.29817500710487366, 0.28613749146461487, 0.28815001249313354, 0.3003750145435333, 0.3093875050544739, 0.3226749897003174, 0.29339998960494995, 0.2995375096797943, 0.30832499265670776, 0.30556249618530273, 0.31751251220703125, 0.3147124946117401, 0.3164375126361847, 0.32120001316070557, 0.3243499994277954, 0.3243499994277954, 0.31923750042915344, 0.31673750281333923, 0.3298499882221222, 0.31302499771118164, 0.3138999938964844, 0.3259750008583069, 0.3245374858379364, 0.32429999113082886, 0.3215875029563904, 0.3326374888420105, 0.3375625014305115, 0.329925000667572, 0.34441250562667847, 0.3307124972343445, 0.3308125138282776, 0.33572500944137573, 0.3321250081062317, 0.3465375006198883, 0.340862512588501, 0.3459874987602234, 0.3519124984741211, 0.3410874903202057, 0.3434000015258789, 0.3373749852180481, 0.3464750051498413, 0.33147498965263367, 0.3432124853134155, 0.3399749994277954, 0.35788750648498535, 0.351749986410141, 0.34757500886917114, 0.35243749618530273, 0.35616248846054077, 0.3506999909877777, 0.3528375029563904, 0.35721251368522644, 0.3514750003814697, 0.34853750467300415, 0.35003748536109924, 0.34619998931884766, 0.3472374975681305, 0.3630250096321106, 0.3540875017642975, 0.3607499897480011, 0.3468624949455261, 0.3649750053882599, 0.35836249589920044, 0.36151251196861267, 0.3591125011444092, 0.3668375015258789, 0.3664250075817108, 0.3550125062465668, 0.3674750030040741, 0.3606624901294708, 0.3538374900817871, 0.3639250099658966, 0.35971251130104065, 0.3695625066757202, 0.37187498807907104, 0.360275000333786, 0.36916249990463257, 0.36496248841285706, 0.38058748841285706, 0.36844998598098755, 0.36931249499320984, 0.37709999084472656, 0.3753499984741211, 0.3689750134944916, 0.37353751063346863, 0.37703749537467957, 0.37088748812675476, 0.37361249327659607, 0.3797000050544739, 0.3704124987125397, 0.37940001487731934, 0.3701624870300293, 0.36713749170303345, 0.37898749113082886, 0.37921249866485596, 0.3833000063896179, 0.3871375024318695, 0.38178750872612, 0.37806248664855957, 0.38231250643730164, 0.38495001196861267, 0.3841499984264374, 0.38688749074935913, 0.38507500290870667, 0.3937875032424927, 0.3808625042438507, 0.3905250132083893, 0.3872624933719635, 0.38622498512268066, 0.39465001225471497, 0.38690000772476196, 0.3909125030040741, 0.39143750071525574, 0.3939875066280365, 0.3891499936580658, 0.3810875117778778, 0.381012499332428, 0.38971251249313354, 0.3864375054836273, 0.38423749804496765, 0.3901124894618988, 0.3864625096321106, 0.39336249232292175, 0.3859750032424927, 0.39559999108314514, 0.3837375044822693, 0.3967374861240387, 0.3953000009059906, 0.40059998631477356, 0.39544999599456787, 0.3957124948501587, 0.3984124958515167, 0.3991749882698059, 0.38827499747276306, 0.3909125030040741, 0.3809624910354614, 0.400299996137619, 0.38817501068115234, 0.39668750762939453, 0.3928624987602234, 0.3864625096321106, 0.39820000529289246, 0.39742499589920044, 0.40226250886917114, 0.39294999837875366, 0.39483749866485596, 0.40296250581741333, 0.39480000734329224, 0.4062874913215637, 0.40216249227523804, 0.3998500108718872, 0.4016624987125397, 0.404449999332428, 0.4044249951839447, 0.4056375026702881, 0.4034000039100647, 0.4092499911785126, 0.40396249294281006, 0.4075999855995178, 0.39932501316070557]
# 20240606-1258-fedavg-lc-bimb-cinic_10--fixmatch-1-0.05
aggr_acc_fedavg_fixmatch_noniid = [0.1260875016450882, 0.16478750109672546, 0.1306000053882599, 0.16271249949932098, 0.18607500195503235, 0.18264999985694885, 0.2051749974489212, 0.21728749573230743, 0.21767500042915344, 0.19301250576972961, 0.2181124985218048, 0.21828749775886536, 0.2510499954223633, 0.2651124894618988, 0.20582500100135803, 0.23628750443458557, 0.23487499356269836, 0.24463750422000885, 0.2568250000476837, 0.24688750505447388, 0.25991249084472656, 0.2738499939441681, 0.27121248841285706, 0.2727000117301941, 0.2534624934196472, 0.2574625015258789, 0.26506251096725464, 0.27717500925064087, 0.2741749882698059, 0.2886125147342682, 0.2750999927520752, 0.27502501010894775, 0.28702500462532043, 0.2639999985694885, 0.2930625081062317, 0.28744998574256897, 0.2601499855518341, 0.27092498540878296, 0.2943125069141388, 0.2719374895095825, 0.29198750853538513, 0.29442501068115234, 0.3022249937057495, 0.2903749942779541, 0.2625750005245209, 0.3040750026702881, 0.29528748989105225, 0.3025124967098236, 0.29768750071525574, 0.30326250195503235, 0.3000375032424927, 0.3096874952316284, 0.2879500091075897, 0.3069249987602234, 0.3042375147342682, 0.3092750012874603, 0.3202874958515167, 0.3021875023841858, 0.2868874967098236, 0.31353750824928284, 0.3167000114917755, 0.3028624951839447, 0.31940001249313354, 0.3186124861240387, 0.32768750190734863, 0.2985000014305115, 0.3198249936103821, 0.3125999867916107, 0.3167625069618225, 0.32183751463890076, 0.3315249979496002, 0.3233124911785126, 0.3286125063896179, 0.3353999853134155, 0.323137491941452, 0.3189375102519989, 0.328962504863739, 0.3448624908924103, 0.33454999327659607, 0.3343124985694885, 0.33822500705718994, 0.3377000093460083, 0.33765000104904175, 0.34578749537467957, 0.34185001254081726, 0.32378751039505005, 0.3316499888896942, 0.34453749656677246, 0.3412500023841858, 0.33090001344680786, 0.3391374945640564, 0.34808748960494995, 0.3375625014305115, 0.3452500104904175, 0.3547874987125397, 0.3460249900817871, 0.3305000066757202, 0.3454500138759613, 0.34062498807907104, 0.3503125011920929, 0.3598499894142151, 0.3521375060081482, 0.3540875017642975, 0.3643124997615814, 0.35929998755455017, 0.3549000024795532, 0.34796249866485596, 0.342787504196167, 0.35766249895095825, 0.34788748621940613, 0.35782501101493835, 0.3704124987125397, 0.36387500166893005, 0.36127498745918274, 0.36390000581741333, 0.35606250166893005, 0.35852500796318054, 0.35920000076293945, 0.370074987411499, 0.36457499861717224, 0.36643749475479126, 0.36653751134872437, 0.36364999413490295, 0.37049999833106995, 0.36336249113082886, 0.3676374852657318, 0.36252498626708984, 0.3799999952316284, 0.35795000195503235, 0.3765375018119812, 0.38058748841285706, 0.3683125078678131, 0.37078750133514404, 0.37031251192092896, 0.37703749537467957, 0.3763749897480011, 0.3781749904155731, 0.38253751397132874, 0.3708750009536743, 0.3880999982357025, 0.3874000012874603, 0.38481250405311584, 0.3773750066757202, 0.3855625092983246, 0.38495001196861267, 0.3751375079154968, 0.3895750045776367, 0.3849250078201294, 0.3747749924659729, 0.3827125132083893, 0.38909998536109924, 0.3754499852657318, 0.3729499876499176, 0.38091251254081726, 0.38247498869895935, 0.3912625014781952, 0.38881251215934753, 0.3934125006198883, 0.3866249918937683, 0.381137490272522, 0.3910374939441681, 0.3837999999523163, 0.3950749933719635, 0.39707499742507935, 0.38663750886917114, 0.3814375102519989, 0.3774000108242035, 0.39836248755455017, 0.3885749876499176, 0.3922874927520752, 0.38097500801086426, 0.38296249508857727, 0.3932499885559082, 0.38778749108314514, 0.3778499960899353, 0.39742499589920044, 0.3944750130176544, 0.38179999589920044, 0.39937499165534973, 0.40351250767707825, 0.40296250581741333, 0.3910500109195709, 0.3978999853134155, 0.40261250734329224, 0.38756251335144043, 0.3874624967575073, 0.40441250801086426, 0.40338748693466187, 0.4034999907016754, 0.39707499742507935, 0.40654999017715454, 0.40156251192092896, 0.3995499908924103, 0.4000625014305115, 0.39518749713897705, 0.39787501096725464, 0.39311251044273376, 0.40408751368522644, 0.41007500886917114, 0.4050624966621399]
# 20240606-1825-fedprox-lc-bimb-cinic_10--fixmatch-1-0.05
aggr_acc_fedprox_fixmatch_noniid = [0.10568749904632568, 0.1375499963760376, 0.1703374981880188, 0.16750000417232513, 0.20368750393390656, 0.20206250250339508, 0.21662500500679016, 0.22664999961853027, 0.18783749639987946, 0.2262749969959259, 0.22675000131130219, 0.2533375024795532, 0.2429250031709671, 0.26205000281333923, 0.25658750534057617, 0.25562500953674316, 0.26561251282691956, 0.26346251368522644, 0.24863749742507935, 0.2146874964237213, 0.2509250044822693, 0.2576749920845032, 0.26598748564720154, 0.22795000672340393, 0.27703750133514404, 0.2667999863624573, 0.2696250081062317, 0.26526251435279846, 0.2938750088214874, 0.273312509059906, 0.287075012922287, 0.2573375105857849, 0.28898748755455017, 0.2778249979019165, 0.2989625036716461, 0.25827500224113464, 0.29813748598098755, 0.29721251130104065, 0.29280000925064087, 0.29809999465942383, 0.2945125102996826, 0.3025749921798706, 0.2971875071525574, 0.2994374930858612, 0.30578750371932983, 0.3091374933719635, 0.31162500381469727, 0.3135499954223633, 0.28952500224113464, 0.3152875006198883, 0.3121874928474426, 0.3136500120162964, 0.3033125102519989, 0.3072749972343445, 0.3207874894142151, 0.312562495470047, 0.31133750081062317, 0.32202500104904175, 0.32083749771118164, 0.32788750529289246, 0.3278625011444092, 0.31408751010894775, 0.3235875070095062, 0.3218249976634979, 0.3235499858856201, 0.3266749978065491, 0.3337250053882599, 0.34130001068115234, 0.33748748898506165, 0.3345249891281128, 0.3348250091075897, 0.31616249680519104, 0.32922500371932983, 0.3395499885082245, 0.33771249651908875, 0.34915000200271606, 0.33980000019073486, 0.3462125062942505, 0.33477500081062317, 0.34991249442100525, 0.34546250104904175, 0.34389999508857727, 0.3461500108242035, 0.34958750009536743, 0.357450008392334, 0.3327125012874603, 0.35647499561309814, 0.3595375120639801, 0.35705000162124634, 0.34575000405311584, 0.32397499680519104, 0.349575012922287, 0.3530125021934509, 0.3680500090122223, 0.35222500562667847, 0.3617999851703644, 0.3537124991416931, 0.36377501487731934, 0.3576124906539917, 0.35952499508857727, 0.3698750138282776, 0.35788750648498535, 0.35510000586509705, 0.3580999970436096, 0.36079999804496765, 0.36758750677108765, 0.36367499828338623, 0.3608874976634979, 0.37236249446868896, 0.3700374960899353, 0.3683624863624573, 0.37005001306533813, 0.368862509727478, 0.3708125054836273, 0.3668999969959259, 0.3693374991416931, 0.36381250619888306, 0.36996251344680786, 0.36880001425743103, 0.37683749198913574, 0.3706749975681305, 0.3703500032424927, 0.37529999017715454, 0.3649125099182129, 0.3718875050544739, 0.37907499074935913, 0.36451250314712524, 0.373912513256073, 0.36822500824928284, 0.37767499685287476, 0.37991249561309814, 0.381974995136261, 0.3720499873161316, 0.3766624927520752, 0.3814125061035156, 0.3852750062942505, 0.37369999289512634, 0.38103750348091125, 0.38510000705718994, 0.3786875009536743, 0.3848874866962433, 0.38119998574256897, 0.3821749985218048, 0.38813748955726624, 0.3851749897003174, 0.3782750070095062, 0.3839375078678131, 0.3883500099182129, 0.3807624876499176, 0.38048750162124634, 0.39201250672340393, 0.3933125138282776, 0.3913249969482422, 0.384737491607666, 0.39089998602867126, 0.38932499289512634, 0.38109999895095825, 0.38208749890327454, 0.3789750039577484, 0.3901750147342682, 0.39605000615119934, 0.37437498569488525, 0.3893375098705292, 0.39399999380111694, 0.38737499713897705, 0.3916124999523163, 0.38837501406669617, 0.3867749869823456, 0.3933374881744385, 0.3949500024318695, 0.3944999873638153, 0.40021249651908875, 0.3995875120162964, 0.3873249888420105, 0.4009000062942505, 0.39832499623298645, 0.4018374979496002, 0.39684998989105225, 0.4010375142097473, 0.40328750014305115, 0.3970000147819519, 0.4051375091075897, 0.40639999508857727, 0.3977000117301941, 0.400424987077713, 0.39750000834465027, 0.409137487411499, 0.39211249351501465, 0.3899500072002411, 0.4074375033378601, 0.40575000643730164, 0.4032500088214874, 0.40691250562667847, 0.40502500534057617, 0.4050374925136566, 0.4031499922275543, 0.41343748569488525, 0.4014124870300293, 0.411175012588501, 0.41499999165534973]
# 20240611-0239-fedmatch-lc-bimb-cinic_10-1-0.05
aggr_acc_fedmatch_noniid = [0.1359499990940094, 0.164124995470047, 0.12417499721050262, 0.1744374930858612, 0.20499999821186066, 0.15366250276565552, 0.1853875070810318, 0.21258750557899475, 0.1847749948501587, 0.20215000212192535, 0.22448749840259552, 0.23362499475479126, 0.24197499454021454, 0.23046250641345978, 0.23208749294281006, 0.23544999957084656, 0.2452625036239624, 0.25366249680519104, 0.2542249858379364, 0.25263750553131104, 0.28220000863075256, 0.27652499079704285, 0.2860875129699707, 0.2825250029563904, 0.2830125093460083, 0.2918125092983246, 0.3044624924659729, 0.2934499979019165, 0.2956624925136566, 0.3001999855041504, 0.2996000051498413, 0.3107750117778778, 0.30912500619888306, 0.3007875084877014, 0.318512499332428, 0.3059625029563904, 0.3152250051498413, 0.3269124925136566, 0.3231250047683716, 0.3395000100135803, 0.3342374861240387, 0.3235124945640564, 0.33627501130104065, 0.33693748712539673, 0.33703750371932983, 0.3421125113964081, 0.35010001063346863, 0.3461500108242035, 0.3420875072479248, 0.3433249890804291, 0.3486124873161316, 0.34786251187324524, 0.3401249945163727, 0.3463749885559082, 0.35007500648498535, 0.34290000796318054, 0.3543750047683716, 0.3472374975681305, 0.3514249920845032, 0.3513374924659729, 0.3538374900817871, 0.3616749942302704, 0.36660000681877136, 0.3559874892234802, 0.34923750162124634, 0.36149999499320984, 0.3695625066757202, 0.36250001192092896, 0.3590624928474426, 0.36666250228881836, 0.36682501435279846, 0.36572501063346863, 0.3733749985694885, 0.36346250772476196, 0.34853750467300415, 0.37543749809265137, 0.37351250648498535, 0.3747999966144562, 0.37121251225471497, 0.38245001435279846, 0.3787499964237213, 0.3799999952316284, 0.3841249942779541, 0.3704499900341034, 0.3836750090122223, 0.3847624957561493, 0.3806374967098236, 0.38948750495910645, 0.38056251406669617, 0.3825874924659729, 0.3867250084877014, 0.3885500133037567, 0.39096251130104065, 0.38977500796318054, 0.3782750070095062, 0.39945000410079956, 0.381725013256073, 0.3830625116825104, 0.4038250148296356, 0.3871375024318695, 0.4044874906539917, 0.4031499922275543, 0.3920249938964844, 0.38433751463890076, 0.40078750252723694, 0.39668750762939453, 0.39215001463890076, 0.39502501487731934, 0.4068875014781952, 0.38718751072883606, 0.3834874927997589, 0.40105000138282776, 0.3998374938964844, 0.3986875116825104, 0.40806248784065247, 0.40435001254081726, 0.4088999927043915, 0.407087504863739, 0.4032000005245209, 0.4101875126361847, 0.41429999470710754, 0.404574990272522, 0.40880000591278076, 0.3971500098705292, 0.4129374921321869, 0.41519999504089355, 0.41308748722076416, 0.41616249084472656, 0.4095875024795532, 0.41503751277923584, 0.39941251277923584, 0.39307498931884766, 0.4109624922275543, 0.4182625114917755, 0.4066374897956848, 0.41421249508857727, 0.4215874969959259, 0.428849995136261, 0.4290125072002411, 0.42998749017715454, 0.4300999939441681, 0.42820000648498535, 0.4280250072479248, 0.43098750710487366, 0.4326249957084656, 0.4322125017642975, 0.4322125017642975, 0.43146249651908875, 0.43373748660087585, 0.43443751335144043, 0.43517500162124634, 0.4361250102519989, 0.43303748965263367, 0.43209999799728394, 0.43283748626708984, 0.43453750014305115, 0.4346500039100647, 0.4343875050544739, 0.43264999985694885, 0.43626248836517334, 0.43396249413490295, 0.43806248903274536, 0.4286375045776367, 0.436724990606308, 0.4335375130176544, 0.43639999628067017, 0.4360499978065491, 0.4349125027656555, 0.43418750166893005, 0.43617498874664307, 0.43661248683929443, 0.43691250681877136, 0.4377875030040741, 0.43611249327659607, 0.43779999017715454, 0.4363499879837036, 0.43857499957084656, 0.43845000863075256, 0.43796250224113464, 0.4374749958515167, 0.43787500262260437, 0.43496251106262207, 0.43363749980926514, 0.43661248683929443, 0.43851250410079956, 0.4355500042438507, 0.43502500653266907, 0.4396750032901764, 0.4375874996185303, 0.43937501311302185, 0.4396125078201294, 0.4394499957561493, 0.437624990940094, 0.43683749437332153, 0.44011250138282776, 0.44056248664855957, 0.44052499532699585, 0.43918749690055847, 0.43845000863075256, 0.43831250071525574]
# 20240613-0902-fedmatch-lc-bimb-cinic_10-1-0.05
aggr_acc_tgafed_noniid = [0.12129999697208405, 0.15981249511241913, 0.19894999265670776, 0.20003749430179596, 0.17041249573230743, 0.19753749668598175, 0.218687504529953, 0.2263375073671341, 0.22447499632835388, 0.234825000166893, 0.2495875060558319, 0.24866250157356262, 0.24972499907016754, 0.2332250028848648, 0.25866249203681946, 0.28318750858306885, 0.27848750352859497, 0.2843250036239624, 0.28021249175071716, 0.2999750077724457, 0.29074999690055847, 0.3042624890804291, 0.3084000051021576, 0.2936750054359436, 0.309612512588501, 0.3091999888420105, 0.3117375075817108, 0.32522499561309814, 0.3172999918460846, 0.32622501254081726, 0.32241249084472656, 0.3237625062465668, 0.337024986743927, 0.3360750079154968, 0.3334375023841858, 0.3422499895095825, 0.34936249256134033, 0.33016249537467957, 0.35653749108314514, 0.34628748893737793, 0.3519499897956848, 0.3509874939918518, 0.35202500224113464, 0.3555625081062317, 0.34568750858306885, 0.35308751463890076, 0.35067498683929443, 0.3490000069141388, 0.3542875051498413, 0.3625124990940094, 0.366225004196167, 0.3647625148296356, 0.35696250200271606, 0.3677375018596649, 0.3738875091075897, 0.37623751163482666, 0.3815625011920929, 0.3785249888896942, 0.3832624852657318, 0.3796499967575073, 0.36857500672340393, 0.3818874955177307, 0.39092499017715454, 0.387737512588501, 0.38644999265670776, 0.39221251010894775, 0.3793250024318695, 0.38394999504089355, 0.3949125111103058, 0.3895750045776367, 0.38962501287460327, 0.38571250438690186, 0.3795500099658966, 0.3949874937534332, 0.3961375057697296, 0.40063750743865967, 0.39301249384880066, 0.38893750309944153, 0.3896375000476837, 0.40312498807907104, 0.38727501034736633, 0.3958125114440918, 0.38823750615119934, 0.40716248750686646, 0.40018749237060547, 0.39559999108314514, 0.39791250228881836, 0.4054875075817108, 0.41644999384880066, 0.4115999937057495, 0.41302499175071716, 0.4101499915122986, 0.3981499969959259, 0.415149986743927, 0.4062874913215637, 0.4204624891281128, 0.4158500134944916, 0.4170374870300293, 0.4201500117778778, 0.4190624952316284, 0.4160124957561493, 0.41784998774528503, 0.42377498745918274, 0.42086249589920044, 0.3975749909877777, 0.40691250562667847, 0.4217875003814697, 0.4094249904155731, 0.4242999851703644, 0.41206249594688416, 0.414837509393692, 0.4244750142097473, 0.4129500091075897, 0.4117625057697296, 0.4272874891757965, 0.4263499975204468, 0.42983749508857727, 0.43896248936653137, 0.4381124973297119, 0.4378499984741211, 0.43857499957084656, 0.43648749589920044, 0.44001251459121704, 0.4419249892234802, 0.43706250190734863, 0.43744999170303345, 0.4404749870300293, 0.43924999237060547, 0.43700000643730164, 0.4412125051021576, 0.4414750039577484, 0.4428125023841858, 0.44421249628067017, 0.42989999055862427, 0.43121251463890076, 0.4320499897003174, 0.42128750681877136, 0.4350374937057495, 0.4432874917984009, 0.4408875107765198, 0.43963751196861267, 0.4419875144958496, 0.4440000057220459, 0.44475001096725464, 0.4254249930381775, 0.4211750030517578, 0.42176249623298645, 0.4305500090122223, 0.4301125109195709, 0.4105125069618225, 0.42500001192092896, 0.4332500100135803, 0.4423375129699707, 0.4353249967098236, 0.4221875071525574, 0.4383625090122223, 0.4368000030517578, 0.44782501459121704, 0.4499875009059906, 0.4484499990940094, 0.45186251401901245, 0.4501250088214874, 0.4501250088214874, 0.4496999979019165, 0.4507625102996826, 0.45239999890327454, 0.44940000772476196, 0.45071250200271606, 0.4507249891757965, 0.45218750834465027, 0.4490875005722046, 0.4523249864578247, 0.4485875070095062, 0.4517124891281128, 0.45173749327659607, 0.45375001430511475, 0.4508875012397766, 0.45368748903274536, 0.45247501134872437, 0.45266249775886536, 0.45387500524520874, 0.45647498965263367, 0.4504374861717224, 0.45227500796318054, 0.453000009059906, 0.453187495470047, 0.4542500078678131, 0.45288750529289246, 0.45313748717308044, 0.4500249922275543, 0.4545624852180481, 0.4535749852657318, 0.45170000195503235, 0.45327499508857727, 0.4505625069141388, 0.45426249504089355, 0.4544124901294708, 0.4571624994277954, 0.4574125111103058, 0.4565249979496002]


# 将准确率乘以100
aggr_acc_fedavg_uda_noniid = [acc * 100 for acc in aggr_acc_fedavg_uda_noniid]
aggr_acc_fedprox_uda_noniid = [acc * 100 for acc in aggr_acc_fedprox_uda_noniid]
aggr_acc_fedavg_fixmatch_noniid = [acc * 100 for acc in aggr_acc_fedavg_fixmatch_noniid]
aggr_acc_fedprox_fixmatch_noniid = [acc * 100 for acc in aggr_acc_fedprox_fixmatch_noniid]
aggr_acc_fedmatch_noniid = [acc * 100 for acc in aggr_acc_fedmatch_noniid]
aggr_acc_tgafed_noniid = [acc * 100 for acc in aggr_acc_tgafed_noniid]

# 获取数据长度作为 x 轴刻度
x = range(1, len(aggr_acc_tgafed_noniid[:200]) + 1)

# 设置图形大小
plt.figure(figsize=(10, 7))

# 绘制折线图
plt.plot(x, aggr_acc_fedavg_uda_noniid[:200], marker='None', color='#ea7ccc', label='FedAvg-UDA')
plt.plot(x, aggr_acc_fedprox_uda_noniid[:200], marker='None', color='#fc8452', label='FedProx-UDA')
plt.plot(x, aggr_acc_fedavg_fixmatch_noniid[:200], marker='None', color='#3ba272', label='FedAvg-FixMatch')
plt.plot(x, aggr_acc_fedprox_fixmatch_noniid[:200], marker='None', color='#73c0de', label='FedProx-FixMatch')
plt.plot(x, aggr_acc_fedmatch_noniid[:200], marker='None', color='#fac858', label='FedMatch')
plt.plot(x, aggr_acc_tgafed_noniid[:200], marker='None', color='#5470c6', label='TGAFed')


# 设置图形标题和坐标轴标签
plt.title('CINIC-10 - Non-IID', fontsize=23)
plt.xlabel('Rounds', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)

# 设置竖轴刻度范围
plt.ylim(0, 50)
# 调整竖轴刻度间隔
plt.locator_params(axis='y', nbins=10)

# 显示图例
plt.legend()
plt.legend(fontsize=20) # 'small' 可以是 'medium', 'large' 或者具体的数字如 10, 12 等
# 显示网格线
plt.grid(True)

# 显示图形
plt.show()


#########################################################