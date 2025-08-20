# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/yash-amd/shark-ai/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                                          |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------------------------------------------ | -------: | -------: | ------: | --------: |
| sharktank/conftest.py                                                         |      171 |       14 |     92% |359, 366, 373, 440, 464, 469-472, 482, 487, 501, 525, 528-529 |
| sharktank/integration/models/punet/integration\_test.py                       |       94 |       57 |     39% |15-16, 21-31, 52-62, 70-80, 90-101, 110-121, 131-143, 150, 155-167, 174, 185-192, 210-220, 242-255 |
| sharktank/setup.py                                                            |       18 |       18 |      0% |      7-34 |
| sharktank/sharktank/\_\_init\_\_.py                                           |        4 |        1 |     75% |        15 |
| sharktank/sharktank/build/\_\_init\_\_.py                                     |        1 |        1 |      0% |         7 |
| sharktank/sharktank/build/actions.py                                          |       45 |       45 |      0% |     7-109 |
| sharktank/sharktank/evaluate/perplexity\_iree.py                              |      249 |      210 |     16% |70-91, 95-101, 106-128, 136-163, 171-208, 218-245, 250-295, 300-349, 352-407, 414-462, 475-529, 536-585, 589 |
| sharktank/sharktank/evaluate/perplexity\_torch.py                             |      181 |      144 |     20% |54-57, 61-67, 72-94, 113-139, 143-161, 165-229, 243-282, 296-352, 377-402, 406-446, 450 |
| sharktank/sharktank/examples/export\_paged\_llm\_v1.py                        |      111 |       37 |     67% |36, 46, 85-94, 173-174, 183-259, 263 |
| sharktank/sharktank/examples/paged\_llm\_v1.py                                |       63 |       54 |     14% |32-116, 122 |
| sharktank/sharktank/examples/pipeline/export\_ppffn\_net.py                   |       83 |        5 |     94% |95, 144, 150, 174, 181 |
| sharktank/sharktank/examples/sharding/export\_ffn\_net.py                     |       59 |       13 |     78% |51-63, 82, 88, 113, 120 |
| sharktank/sharktank/examples/sharding/shard\_llm\_dataset.py                  |       23 |        3 |     87% |33, 36, 51 |
| sharktank/sharktank/kernels/\_\_init\_\_.py                                   |       13 |        0 |    100% |           |
| sharktank/sharktank/kernels/attention.py                                      |       22 |        0 |    100% |           |
| sharktank/sharktank/kernels/base.py                                           |       52 |        5 |     90% |136, 155-160 |
| sharktank/sharktank/kernels/batch\_matmul\_transpose\_b.py                    |       49 |        0 |    100% |           |
| sharktank/sharktank/kernels/bitcast.py                                        |       63 |       40 |     37% |58-69, 75-88, 97-108, 114-127, 136-139 |
| sharktank/sharktank/kernels/conv\_2d\_nchw\_fchw.py                           |       64 |        0 |    100% |           |
| sharktank/sharktank/kernels/einsum\_2args\_q4.py                              |      122 |        2 |     98% |   69, 179 |
| sharktank/sharktank/kernels/gemm\_fp4\_asm.py                                 |       16 |        2 |     88% |    48-108 |
| sharktank/sharktank/kernels/mlir\_kernel.py                                   |      204 |       18 |     91% |40, 43, 47, 112, 123, 129, 131, 220, 262, 269, 277, 321, 329, 369-374, 382 |
| sharktank/sharktank/kernels/mmt\_block\_scaled\_offset\_q4.py                 |       50 |        3 |     94% |     94-96 |
| sharktank/sharktank/kernels/mmt\_block\_scaled\_q8.py                         |       38 |        0 |    100% |           |
| sharktank/sharktank/kernels/mmt\_super\_block\_scaled\_offset\_q4.py          |       59 |        0 |    100% |           |
| sharktank/sharktank/kernels/mmtfp.py                                          |       41 |        2 |     95% |     68-69 |
| sharktank/sharktank/kernels/pooling\_nchw\_sum.py                             |       38 |        0 |    100% |           |
| sharktank/sharktank/kernels/rotary.py                                         |       31 |        0 |    100% |           |
| sharktank/sharktank/kernels/topk.py                                           |       30 |        0 |    100% |           |
| sharktank/sharktank/kernels/wave/attention.py                                 |       48 |        0 |    100% |           |
| sharktank/sharktank/kernels/wave/mxfp4\_gemm.py                               |      100 |       74 |     26% |42-113, 123-160, 184-237 |
| sharktank/sharktank/kernels/wave/utils.py                                     |       18 |        5 |     72% |     50-56 |
| sharktank/sharktank/layers/\_\_init\_\_.py                                    |       15 |        0 |    100% |           |
| sharktank/sharktank/layers/activations.py                                     |        3 |        0 |    100% |           |
| sharktank/sharktank/layers/base.py                                            |      177 |       27 |     85% |131, 206-209, 224, 242, 259-260, 269, 298, 366-374, 385-398, 400, 404-407, 411, 417, 424 |
| sharktank/sharktank/layers/causal\_llm.py                                     |       65 |        6 |     91% | 50-56, 74 |
| sharktank/sharktank/layers/configs/\_\_init\_\_.py                            |        2 |        0 |    100% |           |
| sharktank/sharktank/layers/configs/config.py                                  |      170 |       15 |     91% |139, 196, 205-211, 219, 234, 248-254, 267, 269, 289, 313 |
| sharktank/sharktank/layers/configs/llm\_configs.py                            |      399 |       98 |     75% |184, 186, 188, 190, 192, 194, 196, 202, 204, 206, 208, 214, 218, 222, 229, 231, 235, 237, 246, 249-252, 255-275, 278-285, 295-298, 304-307, 311-316, 328-331, 342-343, 354-355, 446-447, 452, 543, 549-557, 566-578, 617, 639, 663-667, 703-706, 710-714 |
| sharktank/sharktank/layers/conv.py                                            |      100 |       61 |     39% |48, 58, 61, 63, 80, 95-110, 113-143, 157-172, 175-205 |
| sharktank/sharktank/layers/ffn\_block.py                                      |       26 |        0 |    100% |           |
| sharktank/sharktank/layers/ffn\_moe\_block.py                                 |       83 |       25 |     70% |65-73, 203-237, 243-246, 253-259 |
| sharktank/sharktank/layers/latent\_attention\_block.py                        |       54 |        7 |     87% |41, 60, 65, 75, 83-84, 101 |
| sharktank/sharktank/layers/linear.py                                          |       42 |        4 |     90% |56, 67, 75, 83 |
| sharktank/sharktank/layers/mixture\_of\_experts\_block.py                     |       71 |        5 |     93% |48, 52, 60, 105-109 |
| sharktank/sharktank/layers/mmdit.py                                           |      103 |        0 |    100% |           |
| sharktank/sharktank/layers/modulation.py                                      |       21 |        0 |    100% |           |
| sharktank/sharktank/layers/norm.py                                            |       37 |        0 |    100% |           |
| sharktank/sharktank/layers/paged\_attention.py                                |      189 |       15 |     92% |139, 147-153, 205, 231, 407, 410, 414, 494, 504-509, 518, 521 |
| sharktank/sharktank/layers/paged\_llama\_attention\_block.py                  |      115 |        9 |     92% |111, 121, 158, 160, 162, 224, 240-244 |
| sharktank/sharktank/layers/rotary\_embedding.py                               |       31 |        0 |    100% |           |
| sharktank/sharktank/layers/rotary\_embedding\_hf.py                           |      107 |        2 |     98% |   239-240 |
| sharktank/sharktank/layers/testing.py                                         |       44 |        1 |     98% |       302 |
| sharktank/sharktank/layers/token\_embedding.py                                |       12 |        0 |    100% |           |
| sharktank/sharktank/models/\_\_init\_\_.py                                    |        7 |        0 |    100% |           |
| sharktank/sharktank/models/clip/\_\_init\_\_.py                               |        2 |        0 |    100% |           |
| sharktank/sharktank/models/clip/clip.py                                       |      206 |       31 |     85% |80, 123, 131, 143, 159-162, 171, 249, 326, 337, 340, 343, 397, 412, 439, 454, 487, 490, 493, 544-557, 568-570 |
| sharktank/sharktank/models/clip/export.py                                     |       27 |       10 |     63% |40-43, 51-59 |
| sharktank/sharktank/models/clip/export\_toy\_text\_model\_iree\_test\_data.py |       11 |        1 |     91% |        29 |
| sharktank/sharktank/models/clip/testing.py                                    |       67 |        4 |     94% |   175-179 |
| sharktank/sharktank/models/deepseek/testing.py                                |       22 |        0 |    100% |           |
| sharktank/sharktank/models/deepseek/toy\_deepseek.py                          |       33 |        9 |     73% | 82-92, 96 |
| sharktank/sharktank/models/dummy/\_\_init\_\_.py                              |        1 |        0 |    100% |           |
| sharktank/sharktank/models/dummy/dummy.py                                     |       39 |        0 |    100% |           |
| sharktank/sharktank/models/flux/\_\_init\_\_.py                               |        1 |        0 |    100% |           |
| sharktank/sharktank/models/flux/compile.py                                    |        1 |        0 |    100% |           |
| sharktank/sharktank/models/flux/export.py                                     |       55 |       24 |     56% |35-36, 56, 80, 95-98, 104-127 |
| sharktank/sharktank/models/flux/export\_flux\_transformer\_mlir.py            |       13 |       13 |      0% |      7-38 |
| sharktank/sharktank/models/flux/flux.py                                       |      233 |       29 |     88% |82-91, 117-121, 129, 135, 137, 142, 147, 152, 218, 222, 235, 242, 268-279, 288, 407 |
| sharktank/sharktank/models/flux/testing.py                                    |       54 |       10 |     81% |31, 154, 209-227 |
| sharktank/sharktank/models/grok/testing.py                                    |       22 |        0 |    100% |           |
| sharktank/sharktank/models/grok/toy\_grok.py                                  |       31 |        6 |     81% | 66-71, 75 |
| sharktank/sharktank/models/llama4/testing.py                                  |       41 |        1 |     98% |        17 |
| sharktank/sharktank/models/llama/testing.py                                   |       58 |        0 |    100% |           |
| sharktank/sharktank/models/llama/toy\_llama.py                                |       57 |       12 |     79% |83, 85, 87, 91, 95, 100, 159-165, 169 |
| sharktank/sharktank/models/llm/\_\_init\_\_.py                                |        1 |        0 |    100% |           |
| sharktank/sharktank/models/llm/config.py                                      |       33 |        0 |    100% |           |
| sharktank/sharktank/models/llm/export.py                                      |       66 |       17 |     74% |22-27, 33, 79, 85-89, 94-97, 128, 136-139 |
| sharktank/sharktank/models/llm/llm.py                                         |       95 |        4 |     96% |170, 199, 221, 224 |
| sharktank/sharktank/models/llm/testing.py                                     |       24 |       24 |      0% |      1-88 |
| sharktank/sharktank/models/punet/config.py                                    |       84 |       34 |     60% |70-82, 87-91, 98-122, 126-130 |
| sharktank/sharktank/models/punet/layers.py                                    |      324 |      191 |     41% |135-180, 195-226, 258, 280-285, 303-330, 341-355, 366-388, 393-397, 400-410, 418-444, 452-499, 513-519, 524-529, 616-624, 627-631, 654-659, 668-695, 720-725, 728, 738-739, 742-744 |
| sharktank/sharktank/models/punet/sharding.py                                  |       31 |        0 |    100% |           |
| sharktank/sharktank/models/punet/testing.py                                   |       65 |        0 |    100% |           |
| sharktank/sharktank/models/punet/tools/sample\_data.py                        |       26 |       21 |     19% |15-20, 33-46, 50-53 |
| sharktank/sharktank/models/t5/\_\_init\_\_.py                                 |        2 |        0 |    100% |           |
| sharktank/sharktank/models/t5/export.py                                       |       58 |       31 |     47% |42-46, 56-72, 97-105, 117-143 |
| sharktank/sharktank/models/t5/t5.py                                           |      344 |      103 |     70% |126, 160, 189, 236-240, 266-269, 272-284, 313, 326, 334-336, 347, 360, 436-448, 464-480, 517, 557-571, 591-597, 605-642, 649-655, 662, 710, 713, 719-753, 780, 787, 793, 801, 840-842, 850-861, 894-895, 901-905, 911, 926-927, 949-959, 985, 1013, 1018, 1023-1025, 1031, 1034 |
| sharktank/sharktank/models/t5/testing.py                                      |       22 |        0 |    100% |           |
| sharktank/sharktank/models/vae/config.py                                      |       39 |       13 |     67% |44-48, 54-62 |
| sharktank/sharktank/models/vae/layers.py                                      |       97 |        6 |     94% |48, 101, 103, 205, 231, 235 |
| sharktank/sharktank/models/vae/model.py                                       |       67 |        7 |     90% |24-25, 33, 63, 94, 108, 116 |
| sharktank/sharktank/models/vae/testing.py                                     |       14 |        0 |    100% |           |
| sharktank/sharktank/models/vae/tools/diffuser\_ref.py                         |       50 |       13 |     74% |39-60, 87, 104 |
| sharktank/sharktank/models/vae/tools/run\_vae.py                              |       75 |       47 |     37% |64-158, 162 |
| sharktank/sharktank/models/vae/tools/sample\_data.py                          |       14 |        5 |     64% |27-29, 39-40 |
| sharktank/sharktank/ops/\_\_init\_\_.py                                       |       13 |        0 |    100% |           |
| sharktank/sharktank/ops/\_registry.py                                         |      154 |       10 |     94% |125, 130, 260-263, 274, 309, 312-315, 328 |
| sharktank/sharktank/ops/attention\_impls.py                                   |       76 |        2 |     97% |    38, 89 |
| sharktank/sharktank/ops/cpu\_impls.py                                         |       20 |        1 |     95% |        43 |
| sharktank/sharktank/ops/custom\_impls.py                                      |      109 |       46 |     58% |63-67, 85, 101, 122-141, 157-188, 197, 201-204, 227, 229, 231 |
| sharktank/sharktank/ops/default\_impls.py                                     |      536 |       97 |     82% |122, 124, 156, 158, 160, 193, 195, 197, 260-263, 295, 297, 311-312, 327-334, 348-355, 374-392, 406, 416, 579-590, 606, 625, 636-638, 677, 777, 848, 853, 858, 864, 896-903, 909, 998, 1002, 1048, 1053-1070, 1075, 1080 |
| sharktank/sharktank/ops/qconv\_impls.py                                       |      123 |       31 |     75% |47, 53, 67-71, 88, 94, 109, 137-142, 168-177, 229, 252, 270-285, 298, 303, 310 |
| sharktank/sharktank/ops/qlinear\_impls.py                                     |       91 |       16 |     82% |40, 65, 84, 88, 102-105, 116-117, 143-144, 162, 165, 188-190, 209 |
| sharktank/sharktank/ops/quantized\_impls.py                                   |      222 |       13 |     94% |81, 89, 91-97, 99-106, 117-118, 142, 255-257, 394 |
| sharktank/sharktank/ops/shape.py                                              |       28 |        1 |     96% |        84 |
| sharktank/sharktank/ops/sharded\_impls.py                                     |      882 |       82 |     91% |227, 449, 470, 511-513, 520, 528, 543, 553-557, 567-572, 579-580, 587-588, 658-667, 717-725, 909, 961, 974, 977, 982, 985, 1051-1053, 1107-1111, 1124, 1141, 1150, 1159, 1186, 1202, 1212, 1236, 1262, 1264, 1274, 1276, 1341, 1391, 1498, 1528, 1533, 1730, 1740-1750, 1931-1932, 1956, 2019, 2028-2033, 2045, 2049, 2089-2090, 2095-2096 |
| sharktank/sharktank/ops/signatures.py                                         |      734 |      102 |     86% |125, 142, 170, 187, 206, 239, 258, 276, 291, 310, 328, 343, 377, 395, 408, 414, 430, 443, 459, 487-500, 528-534, 553, 568, 579, 598, 620, 661, 683, 691, 710, 735, 760, 779, 793, 821, 834, 859, 885, 903, 925, 942, 982, 1001, 1013, 1031, 1046-1052, 1076, 1095, 1103, 1121, 1140, 1171, 1186, 1209, 1229, 1254, 1287, 1311, 1346, 1373, 1382, 1393, 1412, 1431, 1455, 1472, 1496, 1520, 1527, 1544, 1561, 1598, 1617, 1636, 1655, 1674, 1690, 1712, 1729, 1748, 1767, 1797, 1850, 1874, 1891, 1908, 1949 |
| sharktank/sharktank/ops/utils.py                                              |       86 |       11 |     87% |32, 37, 80, 221, 224, 227-237, 263 |
| sharktank/sharktank/pipelines/flux/\_\_init\_\_.py                            |        2 |        0 |    100% |           |
| sharktank/sharktank/pipelines/flux/flux\_pipeline.py                          |      137 |      109 |     20% |39-92, 120-132, 154-187, 209-227, 237, 243-245, 268-276, 294-316, 319, 338-367, 372-473, 477 |
| sharktank/sharktank/tools/convert\_dataset.py                                 |       27 |        1 |     96% |        51 |
| sharktank/sharktank/tools/import\_hf\_dataset.py                              |       16 |       10 |     38% | 33-54, 60 |
| sharktank/sharktank/tools/sharktank.py                                        |       37 |        3 |     92% |60, 65, 83 |
| sharktank/sharktank/transforms/dataset/\_\_init\_\_.py                        |        2 |        0 |    100% |           |
| sharktank/sharktank/transforms/dataset/dataset.py                             |       14 |        1 |     93% |        24 |
| sharktank/sharktank/transforms/dataset/sharding.py                            |       38 |       28 |     26% |32-34, 37-49, 54-68, 71 |
| sharktank/sharktank/types/\_\_init\_\_.py                                     |        6 |        0 |    100% |           |
| sharktank/sharktank/types/gguf\_interop/\_\_init\_\_.py                       |        2 |        0 |    100% |           |
| sharktank/sharktank/types/gguf\_interop/base.py                               |       70 |       50 |     29% |42-44, 48-61, 65-81, 99-104, 115-138, 142-163, 167-168 |
| sharktank/sharktank/types/gguf\_interop/layouts.py                            |      104 |       68 |     35% |47-49, 53-60, 64, 67, 107-110, 118-144, 157, 160, 170-217, 226-227, 230, 234, 237, 246-247, 250, 254, 257, 282-283, 287-300, 304, 307 |
| sharktank/sharktank/types/layout\_utils.py                                    |      111 |        8 |     93% |90, 125, 127, 204-207, 250, 259 |
| sharktank/sharktank/types/layouts.py                                          |      281 |       46 |     84% |117, 160-161, 166-167, 192-199, 290-302, 400, 404, 480, 488, 496, 504, 517, 530, 538, 550, 553, 556-564, 567-577, 684, 700, 705 |
| sharktank/sharktank/types/misc.py                                             |       55 |        1 |     98% |       122 |
| sharktank/sharktank/types/ocp\_floats.py                                      |       98 |       19 |     81% |93-118, 163, 290 |
| sharktank/sharktank/types/pipelining.py                                       |       32 |        2 |     94% |    32, 48 |
| sharktank/sharktank/types/quantizers.py                                       |      271 |       37 |     86% |130, 184-185, 188-189, 222, 254, 309-310, 319, 334, 340, 354, 360, 386, 388, 451, 453, 491-492, 545, 577, 579, 622, 641, 649, 679-680, 684, 696-706 |
| sharktank/sharktank/types/sharding.py                                         |      155 |       14 |     91% |34, 122-148, 263-264, 267, 321 |
| sharktank/sharktank/types/tensors.py                                          |      835 |      120 |     86% |73-78, 88-90, 161-167, 187-188, 193, 222, 230, 261, 289, 293, 334, 359, 377, 394-396, 406, 410, 418-419, 430-432, 446-448, 462-464, 472-474, 482-483, 513, 518-520, 530-532, 563-565, 595-596, 601, 632-634, 652-654, 663, 665, 717, 750-751, 756, 790, 794, 796, 831, 840, 891-892, 915-916, 921-922, 1129, 1176-1177, 1195, 1197, 1199, 1207-1209, 1214, 1338, 1351, 1353-1358, 1360, 1362, 1373, 1376, 1378, 1388, 1391, 1420, 1482-1484, 1489, 1511, 1546-1547, 1560, 1575, 1577, 1579, 1648, 1659-1660, 1668-1669, 1677-1678, 1684-1687, 1713-1714, 1841-1842, 1856, 1862, 1866-1867 |
| sharktank/sharktank/types/theta.py                                            |      310 |       44 |     86% |69, 77, 106, 137-147, 167, 179-180, 209-210, 216, 231, 344-348, 408, 456-457, 480-481, 496-497, 517-518, 532-533, 550-551, 593-595, 599, 637-639, 644 |
| sharktank/sharktank/utils/\_\_init\_\_.py                                     |        1 |        0 |    100% |           |
| sharktank/sharktank/utils/azure.py                                            |       58 |       58 |      0% |     7-121 |
| sharktank/sharktank/utils/cli.py                                              |      110 |       67 |     39% |35-39, 72-183, 194-199, 210-222, 233-265, 280-301, 312, 325-328, 338, 351-363, 380-381, 383, 397, 402, 411-429 |
| sharktank/sharktank/utils/create\_cache.py                                    |        7 |        1 |     86% |        12 |
| sharktank/sharktank/utils/debugging.py                                        |       91 |       29 |     68% |46-63, 67-74, 81-83, 127, 138 |
| sharktank/sharktank/utils/evaluate.py                                         |       59 |       34 |     42% |29-48, 55, 69-88, 119-120, 129-149 |
| sharktank/sharktank/utils/export.py                                           |       70 |        4 |     94% |140, 151, 179, 212 |
| sharktank/sharktank/utils/export\_artifacts.py                                |      172 |      125 |     27% |39-47, 61, 68, 75, 82, 89, 123-157, 169, 182-186, 211-231, 249-261, 265-271, 283-299, 323-362, 386-422, 446-459, 485-511, 533-539 |
| sharktank/sharktank/utils/hf.py                                               |       35 |       24 |     31% |26-54, 64-78 |
| sharktank/sharktank/utils/hf\_datasets.py                                     |       75 |       22 |     71% |37-55, 65, 73, 82-83, 88, 478-496, 500 |
| sharktank/sharktank/utils/io.py                                               |       39 |        9 |     77% |65-72, 83-86 |
| sharktank/sharktank/utils/iree.py                                             |      287 |       56 |     80% |187, 198-201, 299, 303, 307, 313-320, 326, 332, 338, 380, 498-499, 546-547, 555-559, 660-681, 696-703, 715-723, 746 |
| sharktank/sharktank/utils/llm\_artifacts.py                                   |       31 |        2 |     94% |    38, 43 |
| sharktank/sharktank/utils/llm\_utils.py                                       |      273 |       38 |     86% |37-44, 63-65, 118, 267-269, 324, 327-367, 436 |
| sharktank/sharktank/utils/load\_llm.py                                        |      178 |       83 |     53% |49-63, 68-80, 123, 168, 175, 178-187, 195, 197, 202-212, 222-240, 263-268, 292-368 |
| sharktank/sharktank/utils/logging.py                                          |        6 |        1 |     83% |        17 |
| sharktank/sharktank/utils/math.py                                             |       12 |        5 |     58% | 17, 25-28 |
| sharktank/sharktank/utils/misc.py                                             |       58 |       11 |     81% |35, 100, 105-115 |
| sharktank/sharktank/utils/patching.py                                         |       94 |       43 |     54% |56, 75-78, 87-93, 98, 108-133, 141-154, 157-168, 197, 231, 233, 238 |
| sharktank/sharktank/utils/random.py                                           |       23 |        0 |    100% |           |
| sharktank/sharktank/utils/testing.py                                          |      431 |      202 |     53% |161-277, 300-306, 317, 330-340, 353-377, 383-404, 420-429, 434-447, 451-455, 494-578, 613, 666-669, 698-704, 735, 759-767, 782, 787, 793-797, 805-808, 814-821, 829-833, 901, 938, 969, 985-994, 1017-1019, 1032, 1042, 1045, 1084 |
| sharktank/sharktank/utils/tokenizer.py                                        |       58 |       41 |     29% |34-38, 42-46, 50, 63-66, 69-72, 76, 80-81, 85-110, 114-119 |
| sharktank/sharktank/utils/tree.py                                             |       71 |        2 |     97% |   81, 220 |
| sharktank/tests/evaluate/perplexity\_iree\_test.py                            |       56 |       36 |     36% |33-39, 47-61, 64-74, 84-89, 94-108, 113-119, 123 |
| sharktank/tests/evaluate/perplexity\_torch\_test.py                           |       46 |       28 |     39% |30-35, 38-48, 51-67, 72-77, 82-88, 92 |
| sharktank/tests/examples/main\_test.py                                        |       24 |        1 |     96% |        45 |
| sharktank/tests/examples/paged\_llm\_v1\_test.py                              |       16 |        5 |     69% |     29-33 |
| sharktank/tests/export\_ir/export\_test.py                                    |       38 |        0 |    100% |           |
| sharktank/tests/kernels/attention\_template\_test.py                          |       76 |        8 |     89% |23, 112-118, 138 |
| sharktank/tests/kernels/attention\_wave\_test.py                              |       23 |        2 |     91% |    25, 59 |
| sharktank/tests/kernels/batch\_matmul\_transpose\_b\_test.py                  |       85 |        6 |     93% |110-113, 126, 153 |
| sharktank/tests/kernels/conv\_2d\_nchw\_fchw\_test.py                         |       42 |        2 |     95% |    63, 91 |
| sharktank/tests/kernels/einsum\_q4\_test.py                                   |       69 |        3 |     96% |94, 120, 141 |
| sharktank/tests/kernels/gemm\_fp4\_asm\_test.py                               |       57 |       38 |     33% |26, 48-110 |
| sharktank/tests/kernels/mlir\_kernel\_test.py                                 |       21 |        0 |    100% |           |
| sharktank/tests/kernels/mmt\_block\_scaled\_offset\_q4\_test.py               |       46 |        3 |     93% |49, 79, 100 |
| sharktank/tests/kernels/mmt\_block\_scaled\_q8\_test.py                       |       43 |        3 |     93% |46, 74, 94 |
| sharktank/tests/kernels/mmt\_super\_block\_scaled\_offset\_q4\_test.py        |       71 |       20 |     72% |39-64, 97, 156, 174 |
| sharktank/tests/kernels/mmtfp\_test.py                                        |       60 |        4 |     93% |57, 81, 99, 125 |
| sharktank/tests/kernels/pooling\_nchw\_sum\_test.py                           |       42 |        2 |     95% |    58, 78 |
| sharktank/tests/kernels/rotary\_test.py                                       |       18 |        0 |    100% |           |
| sharktank/tests/kernels/topk\_test.py                                         |       31 |        0 |    100% |           |
| sharktank/tests/kernels/wave/mxfp4\_gemm\_test.py                             |       64 |       39 |     39% |33, 65-136 |
| sharktank/tests/kernels/wave/wave\_utils\_test.py                             |       30 |        0 |    100% |           |
| sharktank/tests/layers/base\_test.py                                          |       22 |        0 |    100% |           |
| sharktank/tests/layers/configs\_test.py                                       |       11 |        0 |    100% |           |
| sharktank/tests/layers/kv\_cache\_test.py                                     |       49 |        0 |    100% |           |
| sharktank/tests/layers/linear\_test.py                                        |       82 |        1 |     99% |       196 |
| sharktank/tests/layers/mixture\_of\_experts\_block\_test.py                   |       58 |        1 |     98% |       329 |
| sharktank/tests/layers/mmdit\_test.py                                         |       56 |        1 |     98% |        96 |
| sharktank/tests/layers/paged\_llama\_attention\_block\_test.py                |       53 |       38 |     28% |29-46, 57-130, 134 |
| sharktank/tests/layers/rotary\_embedding\_hf\_test.py                         |      233 |        5 |     98% |303-304, 400-402 |
| sharktank/tests/layers/rotary\_embedding\_test.py                             |       61 |        0 |    100% |           |
| sharktank/tests/layers/sharded\_conv2d\_with\_iree\_test.py                   |       78 |        0 |    100% |           |
| sharktank/tests/models/clip/clip\_test.py                                     |      251 |       53 |     79% |91, 96-111, 121, 131, 211-255, 300-327, 352-386, 395, 405 |
| sharktank/tests/models/deepseek/test\_deepseek.py                             |       34 |        3 |     91% |     71-82 |
| sharktank/tests/models/flux/flux\_test.py                                     |      155 |       72 |     54% |61-63, 67-68, 82, 127-182, 193-211, 220-238, 284, 291, 298-317, 327-353, 363, 372, 381-389, 393 |
| sharktank/tests/models/grok/test\_grok.py                                     |       25 |        0 |    100% |           |
| sharktank/tests/models/llama4/llama4\_test.py                                 |       62 |        8 |     87% |110-130, 147 |
| sharktank/tests/models/llama4/moe\_test.py                                    |       90 |        1 |     99% |       192 |
| sharktank/tests/models/llama/attention\_test.py                               |       68 |        1 |     99% |       199 |
| sharktank/tests/models/llama/benchmark\_amdgpu\_test.py                       |      135 |       85 |     37% |34, 37-48, 58-92, 95-105, 114-168, 193-208, 212-227, 231-250, 255-275, 282-352, 369-390, 396-414, 421-461, 476-491, 497-515, 519 |
| sharktank/tests/models/llama/quantized\_test.py                               |       20 |        0 |    100% |           |
| sharktank/tests/models/llama/quark\_parity\_test.py                           |       55 |       40 |     27% |21-22, 29-101, 105 |
| sharktank/tests/models/llama/rot\_emb\_test.py                                |       37 |        1 |     97% |        81 |
| sharktank/tests/models/llama/test\_llama.py                                   |       37 |        3 |     92% |     72-83 |
| sharktank/tests/models/llama/toy\_llama\_test.py                              |       63 |        1 |     98% |        29 |
| sharktank/tests/models/punet/resnet\_test.py                                  |       42 |        1 |     98% |        93 |
| sharktank/tests/models/punet/sharded\_resnet\_block\_with\_iree\_test.py      |       43 |       12 |     72% |    76-113 |
| sharktank/tests/models/punet/up\_down\_block\_test.py                         |       49 |        1 |     98% |       149 |
| sharktank/tests/models/t5/t5\_test.py                                         |      269 |       59 |     78% |80-108, 146-174, 187-221, 266, 280, 289, 298, 307, 316, 325, 435-477, 522, 531, 540, 549, 558 |
| sharktank/tests/models/vae/vae\_test.py                                       |      213 |      112 |     47% |61-96, 102-111, 116-125, 129-224, 249-263, 268-281, 344, 349-440, 541-548, 557-561, 566-571, 577 |
| sharktank/tests/ops/ops\_test.py                                              |      622 |       30 |     95% |177-180, 245-251, 258-264, 271-278, 632-637, 1077 |
| sharktank/tests/ops/pipeline\_parallelized\_test.py                           |      153 |        4 |     97% |57, 181, 193, 203 |
| sharktank/tests/ops/qconv\_test.py                                            |       97 |       12 |     88% |192-228, 232 |
| sharktank/tests/ops/quantized\_test.py                                        |       85 |        0 |    100% |           |
| sharktank/tests/ops/sharded\_test.py                                          |     1337 |       19 |     99% |596-602, 684, 1915, 1918, 1922, 1945, 1949, 2123, 2132-2134, 2142, 2350 |
| sharktank/tests/ops/test\_attention\_ops.py                                   |       25 |        1 |     96% |        87 |
| sharktank/tests/pipelines/flux/flux\_pipeline\_test.py                        |       41 |       23 |     44% |25-27, 32-65, 77-121, 128, 135 |
| sharktank/tests/pytest\_fixtures\_test.py                                     |       19 |        0 |    100% |           |
| sharktank/tests/tools/convert\_dataset\_test.py                               |       22 |        0 |    100% |           |
| sharktank/tests/tools/sharktank\_test.py                                      |       19 |        0 |    100% |           |
| sharktank/tests/transforms/dataset\_transforms\_test.py                       |       32 |        1 |     97% |        86 |
| sharktank/tests/types/dataset\_test.py                                        |      183 |       36 |     80% |239-259, 268-294, 303 |
| sharktank/tests/types/layout\_utils\_test.py                                  |       75 |        1 |     99% |       231 |
| sharktank/tests/types/layouts\_test.py                                        |       68 |        1 |     99% |       148 |
| sharktank/tests/types/misc\_test.py                                           |       14 |        0 |    100% |           |
| sharktank/tests/types/quantizers\_test.py                                     |      266 |        1 |     99% |       634 |
| sharktank/tests/types/tensors\_test.py                                        |      164 |        1 |     99% |       221 |
| sharktank/tests/utils/iree\_test.py                                           |       56 |        6 |     89% | 69-73, 93 |
| sharktank/tests/utils/misc\_test.py                                           |        9 |        0 |    100% |           |
| sharktank/tests/utils/patching\_test.py                                       |       44 |        0 |    100% |           |
| sharktank/tests/utils/testing\_test.py                                        |      137 |        4 |     97% |   395-408 |
| sharktank/tests/utils/tree\_test.py                                           |       20 |        0 |    100% |           |
|                                                                     **TOTAL** | **21251** | **4363** | **79%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/yash-amd/shark-ai/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/yash-amd/shark-ai/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/yash-amd/shark-ai/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/yash-amd/shark-ai/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fyash-amd%2Fshark-ai%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/yash-amd/shark-ai/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.