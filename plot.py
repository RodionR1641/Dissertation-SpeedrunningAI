#plotting data
# useful to visualise the trends over time, when we start experimenting with different parameters etc can compare
# e.g. train for 10 hours -> can see if its getting better or not
# can use TensorBoard setup, or use matplotlib

# write to disk here

import matplotlib.pyplot as plt
import os
from datetime import datetime
import re


class LivePlot():

    def __init__(self):
        self.figure, self.ax = plt.subplots()
        self.ax.set_ylabel("Returns") #take an average, plot that for every 10th epoch
        self.ax.set_xlabel("Epoch * 100") #plot every 100th episode
        self.ax.set_title("Returns over epochs")

        self.data = None
        self.epsilon_data = None

        self.epochs = 0

    def update_plot(self,stats):
        self.data = stats["AvgReturns"] #what the log file will basically have
        self.epsilon_data = stats["Epsilon"]

        self.epochs = len(self.data)
        print(self.epochs)

        self.ax.clear() # get rid of old data
        self.ax.set_xlim(0, self.epochs) #setting this xlimit to len of data

        self.ax.plot(self.data, "b-", label="Returns") #blue line
        self.ax.plot(self.epsilon_data, "r-", label="Epsilon")

        self.ax.legend(loc="upper left")

        if not os.path.exists("plots"):
            os.makedirs("plots")

        current_date = datetime.now().strftime("%Y-%m-%d")

        self.figure.savefig(f"plots/plot_{current_date}.png")
    
    def generate_plot(self,log_file):
        #doing it for every 100 here too, have to go through every 10th line
        
        first_line = 9
        stats = {"AvgReturns":[],"Epsilon":[]}
        log_pattern = r"(Episode return|Average return): ([\-]?\d+\.\d+)\s+- Epsilon: (\d+\.\d+)"

        with open(log_file,"r") as file:
            for i,line in enumerate(file, start=1):
                #read every 10th line for report
                if i >= first_line and (i-first_line) % 10 ==0:

                    match = re.search(log_pattern,line)
                    if match:
                        avg_return = float(match.group(2))
                        epsilon = float(match.group(3))

                        stats["AvgReturns"].append(avg_return)
                        stats["Epsilon"].append(epsilon)
                    else:
                        print("no match found")
                        print(line)
        self.update_plot(stats)

plot = LivePlot()
plot.generate_plot("logs/experiment_2025-02-02_13-06-53_rank0.log")