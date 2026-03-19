import matplotlib.pyplot as plt
def plot_line(fl, label, individual):
    with open(fl,"r",encoding="utf-8") as fd:
        arr = []
        for l in fd.readlines():

            if  not individual and "group returns of step" not in l:
                continue
            elif individual and "idividual returns of step" not in l:
                continue
            l = l.split(" ")[-1].strip()
            l = float(l)
            arr.append(l)
        plt.plot(arr[:100], label = label)
        return


# plot_line("out_together.txt","RL together", False)
plot_line("out_1b_1.txt","RL OLD", True)
plot_line("out_1b_2.txt","RL NEW",True)
plt.legend()
plt.ylabel("Reward")
plt.title("iteration")
plt.savefig("fig_all_i.png")
plt.show()
