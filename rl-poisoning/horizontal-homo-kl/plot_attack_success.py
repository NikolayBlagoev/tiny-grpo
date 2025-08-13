import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})
plt.figure(figsize=(8,6))
def plot_line(fl, label, factor):
    with open(fl,"r",encoding="utf-8") as fd:
        arr = []
        for l in fd.readlines():
            if "Successful attacks" not in l:
                continue
            l = l.split(" ")[-1].strip()
            l = float(l)
            arr.append(l / factor)
        plt.plot(arr[:100], label = label, linewidth=2)
        return


# plot_line("out_50.txt","6/12 poisoned completions",192)
plot_line("out_25.txt","3/12 poisoned completions with KL",8*32)
# plot_line("out_1.txt","1/12 poisoned completion",11*32)
# plot_line("out_3-24.txt","3/24 poisoned completion",21*32)
# plot_line("out_6-24.txt","6/24 poisoned completion",18*32)
# plt.legend()
plt.ylabel("ASR")
plt.xlabel("Iteration")
plt.ylim(0,1)
plt.title("Horizontal Poisoning with KL divergence")
plt.savefig("fig_kl.pdf")
plt.show()



