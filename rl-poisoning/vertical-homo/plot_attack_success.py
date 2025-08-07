import matplotlib.pyplot as plt
def plot_line(fl, label, factor):
    with open(fl,"r",encoding="utf-8") as fd:
        arr = []
        for l in fd.readlines():
            if "Successful attacks" not in l:
                continue
            l = l.split(" ")[-1].strip()
            l = float(l)
            arr.append(l / factor)
        plt.plot(arr[:100], label = label)
        return


plot_line("out_50.txt","6/12 poisoned completions",192)
plot_line("out_25.txt","3/12 poisoned completions",8*32)
plot_line("out_1.txt","1/12 poisoned completion",11*32)
plot_line("out_3-24.txt","3/24 poisoned completion",21*32)
plot_line("out_6-24.txt","6/24 poisoned completion",18*32)
plt.legend()
plt.ylabel("ASR")
plt.title("Poisoning attack models")
plt.savefig("fig_all_i.png")
plt.show()



