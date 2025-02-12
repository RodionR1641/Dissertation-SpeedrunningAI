#plotting data
# useful to visualise the trends over time, when we start experimenting with different parameters etc can compare
# e.g. train for 10 hours -> can see if its getting better or not
# can use TensorBoard setup, or use matplotlib

# write to disk here

import matplotlib.pyplot as plt
import os
import datetime


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

        self.ax_clear() # get rid of old data
        self.ax.set_xlim(0, self.epochs) #setting this xlimit to len of data

        self.ax.plot(self.data, "b-", label="Returns") #blue line
        self.ax.plot(self.epsilon_data, "r-", label="Epsilon")

        self.ax.legend(loc="upper left")

        if not os.path.exists("plots"):
            os.makedirs("plots")

        current_date = datetime.now().strftime("%Y-%m-%d")

        self.fig.savefig(f"plots/plot_{current_date}.png")
    
    def generate_plot(self,log_file):
        #doing it for every 100 here too, have to go through every 10th line
        
        pass