import numpy as np
import matplotlib.pyplot as plt


def print_errs(errs):
    print(f"\tOriginal: {errs[0]:.2f}")
    print(f"\tAvg     : {errs[1]:.2f}")
    print(f"\tMEMO    : {errs[2]:.2f}")


corruptions = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]
corruptions_names_short = [
    'orig', 'gauss', 'shot', 'impul', 'defoc', 'glass', 'motn', 'zoom', 'snow',
    'frost', 'fog', 'brit', 'contr', 'elast', 'pixel', 'jpeg'
]
levels = [1, 2, 3, 4, 5]
errs_list = []

for level in levels:
    for corruption in corruptions:
        errs = np.load(file="/data/yusun/lmy/memo/cifar-10-exps/test_err/" +
                       "err_" + corruption + "_" + str(level) + ".npy")
        errs_list.append(errs)
        print(f"For {corruption}, level {level}:")
        print_errs(errs)
avg_errs = np.average(errs_list, axis=0)
errs_list = np.array(errs_list)
print("\n\n***************Avg errs*********************")
print_errs(avg_errs)
np.savetxt("/data/yusun/lmy/memo/cifar-10-exps/test_err/err_all.txt",
           errs_list,
           fmt="%4.2f")
# for level in levels:
#     # 创建分组柱状图，需要自己控制x轴坐标
#     xticks = np.arange(len(corruptions))

#     fig, ax = plt.subplots(figsize=(10, 3))
#     ax.bar(xticks,
#            errs_list[(level - 1) * len(corruptions):level * len(corruptions),
#                      0],
#            width=0.2,
#            label="Baseline")
#     #    color="red")
#     ax.bar(xticks + 0.2,
#            errs_list[(level - 1) * len(corruptions):level * len(corruptions),
#                      1],
#            width=0.2,
#            label="Soft-voting")
#     #    color="blue")
#     ax.bar(xticks + 0.4,
#            errs_list[(level - 1) * len(corruptions):level * len(corruptions),
#                      2],
#            width=0.2,
#            label="MEMO")
#     #    color="green")

#     # ax.set_title(f"Test error (%) on CIFAR-10-C, level {level}", fontsize=15)
#     # ax.set_xlabel("Corruptions")
#     ax.set_ylabel("Test error (%)")
#     ax.legend(prop={'size': 8})
#     # 最后调整x轴标签的位置
#     ax.set_xticks(xticks + 0.2 / 3)
#     ax.set_xticklabels(corruptions_names_short[1:])
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     # plt.tight_layout(pad=0)
#     plt.savefig("/data/yusun/lmy/memo/cifar-10-exps/test_err/imgs/cifar10c-" +
#                 str(level) + ".png")

test_names = ["ResNet-26", "+SV", "+MEMO"]


def easy_latex(table, prec1=True):
    for i in range(3):
        # for row in table:
        row = table[i]
        row_str = test_names[i]
        for entry in row:
            if prec1:
                row_str += ' & %.1f' % (entry)
            else:
                row_str += ' & %s' % (entry)
        row_str += " \\\\"
        print(row_str)


for level in levels:
    tmp_list = []
    print(f"**********************level {level}***********************")
    # 创建分组柱状图，需要自己控制x轴坐标
    for i in range(3):
        tmp_list.append(errs_list[(level - 1) * len(corruptions):level *
                                  len(corruptions), i])
    easy_latex(tmp_list)