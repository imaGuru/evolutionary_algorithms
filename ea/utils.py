import matplotlib.pyplot as plt
from collections import namedtuple
import multiprocessing as mp

Plot = namedtuple("Plot", ["fig", "axes"])
plt.style.use("seaborn")


class GAPlotter:
    def __init__(self):
        self.f_min, self.f_avg, self.f_max, self.f_sd, self.f = (
            list(),
            list(),
            list(),
            list(),
            list(),
        )
        self.fitness = list()

        self.fig = plt.figure(1, figsize=(12, 8), dpi=100)
        self.ax_fit = self.fig.add_subplot(221)
        self.ax_sd = self.fig.add_subplot(222)
        self.ax_hist = self.fig.add_subplot(2, 2, (3, 4))
        self.initialized = False
        self.axes = [self.ax_fit, self.ax_sd, self.ax_hist]

    def on_data(self, data):
        self.f_min += data[0]
        self.f_avg += data[1]
        self.f_max += data[2]
        self.f_sd += data[3]
        self.fitness = data[4]
        self.it = data[5]

        self.fig.suptitle("Population {}".format(self.it))
        # Population fitness
        if not self.initialized:
            self.ax_fit.plot(self.f_min, label="f_min")
            self.ax_fit.plot(self.f_avg, label="f_avg")
            self.ax_fit.plot(self.f_max, label="f_max")
            # Population standard deviation
            self.ax_sd.plot(self.f_avg, label="f")
            self.ax_sd.plot(list(map(lambda x: x * 10, self.f_sd)), label="10 * f_sd")
            # Population histogram
            self.ax_hist.hist(
                self.fitness, label="fitness", color="white", edgeColor="blue"
            )
            self.ax_fit.legend()
            self.ax_sd.legend()
            self.initialized = True
        else:
            self.ax_fit.cla()
            self.ax_fit.plot(self.f_min, label="f_min")
            self.ax_fit.plot(self.f_avg, label="f_avg")
            self.ax_fit.plot(self.f_max, label="f_max")
            self.ax_sd.cla()
            self.ax_sd.plot(self.f_avg, label="f")
            self.ax_sd.plot(list(map(lambda x: x * 10, self.f_sd)), label="10 * f_sd")
            self.ax_hist.cla()
            self.ax_hist.hist(
                self.fitness, label="fitness", color="white", edgeColor="blue"
            )
            self.ax_fit.legend()
            self.ax_sd.legend()


class ProcessPlotter(object):
    def __init__(self, Plotter=GAPlotter):
        self.plotter = Plotter()

    def terminate(self):
        plt.show()
        plt.close("all")

    def call_back(self):
        while self.pipe.poll():
            data = self.pipe.recv()
            if data is None:
                self.terminate()
                return False
            else:
                self.plotter.on_data(data)
        self.plotter.fig.canvas.draw()
        return True

    def __call__(self, pipe):
        print("starting plotter...")

        self.pipe = pipe
        timer = self.plotter.fig.canvas.new_timer(interval=100)
        timer.add_callback(self.call_back)
        timer.start()

        print("...done")
        plt.show()


class PlotterMaster(object):
    def __init__(self):
        self.plot_pipe, plotter_pipe = mp.Pipe()
        self.plotter = ProcessPlotter()
        self.plot_process = mp.Process(
            target=self.plotter, args=(plotter_pipe,), daemon=True
        )
        self.plot_process.start()

    def update(self, data, finished=False):
        send = self.plot_pipe.send
        if finished:
            send(None)
        else:
            send(data)

    def join(self):
        self.update(None, True)
        self.plot_process.join()
