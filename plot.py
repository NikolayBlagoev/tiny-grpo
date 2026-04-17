import matplotlib.pyplot as plt
def plot_line(fl, label, reward=False):
    with open(fl,"r",encoding="utf-8") as fd:
        arr = []
        for l in fd.readlines():
            
            l = l.split(" ")[-1].strip()
            l = float(l)
            if not reward:
                l = l/4
            arr.append(l)
        plt.plot(arr[:100], label = label)
        return


# plot_line("out_together.txt","RL together", False)
plot_line("1b_together_horizontal.txt","Together")
plot_line("1b_alone.txt","Alone 1b")
plt.legend()
plt.ylabel("Validation Accuracy")
plt.title("Iteration")
plt.savefig("fig_1b_i.png")
plt.show()


plot_line("3b_together_horizontal.txt","Together")
plot_line("3b_alone.txt","Alone 3b")
plt.legend()
plt.ylabel("Validation Accuracy")
plt.title("Iteration")
plt.savefig("fig_3b_i.png")
plt.show()

plot_line("rewards_1b_together.txt","Together",reward=True)
plot_line("rewards_1b_alone.txt","Alone 1b",reward=True)
plt.legend()
plt.ylabel("Validation Accuracy")
plt.title("Iteration")
plt.savefig("fig_r1b_i.png")
plt.show()


plot_line("rewards_3b_together.txt","Together",reward=True)
plot_line("rewards_3b_alone.txt","Alone 3b",reward=True)
plt.legend()
plt.ylabel("Validation Accuracy")
plt.title("Iteration")
plt.savefig("fig_r3b_i.png")
plt.show()


plot_line("rewards_3b_together.txt","Together",reward=True)
plot_line("rewards_3b_alone.txt","Alone 3b",reward=True)
plot_line("rewards_1b_together.txt","Together",reward=True)
plot_line("rewards_1b_alone.txt","Alone 1b",reward=True)
plt.legend()
plt.ylabel("Validation Accuracy")
plt.title("Iteration")
plt.savefig("fig_all_i.png")
plt.show()

