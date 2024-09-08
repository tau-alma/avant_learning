import numpy as np
import config
import pyqtgraph as pg
import traceback
from queue import Empty
from PyQt6 import QtCore


class QtPlot(pg.GraphicsView):
    def __init__(self, frame_size, data_queue, exit_event):
        super(QtPlot, self).__init__()
        self.frame_size = frame_size
        self.data_queue = data_queue
        self.exit_event = exit_event

        self.prev_lyapunov = 0

        # Create the main layout
        self.layout = pg.GraphicsLayout()
        self.setCentralItem(self.layout)

        # Add the ImageItem in its own ViewBox in the first column
        self.image_vb = self.layout.addViewBox(lockAspect=True, invertY=True)
        self.image_vb.setFixedWidth(self.frame_size)
        self.image_vb.setFixedHeight(self.frame_size)

        self.image_item = pg.ImageItem()
        self.image_vb.addItem(self.image_item)

        # Add a column for the plots
        plot_layout = self.layout.addLayout()

        plot_list = [
            ("MPC solution duration", 0, 80, "ms", [
                ((255, 255, 0), "duration"),
            ]),
            ("Lyapunov critic value at MPC terminal state", 0, 300, "magnitude", [
                ((255, 255, 0), "value"),
            ]),
            ("Change in Lyapunov critic value", -5, 3, "magnitude", [
                ((255, 255, 0), "value"),
            ])
        ]

        self.n_plot_items = sum([len(t[4]) for t in plot_list])
        self.plot_history_len = int(30 / 0.1)
        self.x = np.repeat(np.arange(self.plot_history_len), self.n_plot_items).reshape(self.plot_history_len, self.n_plot_items).T
        self.y = np.zeros([self.n_plot_items, self.plot_history_len])
        self.plots = []
        self.plot_items = []
        plot_i = 0
        item_i = 0
        for title, lb, ub, y_label, items in plot_list:
            plot = plot_layout.addPlot(row=plot_i, col=1)
            self.plots.append(plot)
            plot.setYRange(lb, ub)
            plot.setTitle(title)
            plot.showGrid(True, True)
            plot.setLabel(axis='left', text=y_label)
            legend = plot.addLegend()
            legend.setBrush('k')
            legend.setOffset(1)

            for color, name in items:
                self.plot_items.append(
                    plot.plot(
                        self.x[item_i], self.y[item_i], pen=pg.mkPen(color=color), name=name
                    )
                )
                item_i += 1
            plot_i += 1

        self.plots[0].setFixedHeight(2*self.frame_size // 10)
        self.plots[1].setFixedHeight(4*self.frame_size // 10)
        self.plots[2].setFixedHeight(4*self.frame_size // 10)
        for i in range(len(self.plots)):
            self.plots[i].setFixedWidth(self.frame_size)

        # Adjust window size based on content
        total_width = self.frame_size + self.frame_size + 40
        max_plot_height = max([plot.size().height() for plot in self.plots])
        total_height = max(self.frame_size, max_plot_height) + 40
        self.setFixedSize(total_width, total_height)

        # Start UI tick:
        self.timer = QtCore.QTimer()
        self.timer.setInterval(5)
        self.timer.timeout.connect(self.update_data)
        self.timer.start()

    def update_data(self):
        if self.exit_event.is_set():
            self.close()

        try:
            solve_time, lyapunov_value, frame = self.data_queue.get(block=False)

            change = lyapunov_value - self.prev_lyapunov
            if change > 10:
                change = 0

            graph_data = {
                'mpc timing': (0, [solve_time]),
                'lyapunov_value': (1, [lyapunov_value]),
                'lyapunov_decrease': (2, [change])
            }
            for first_i, values_list in graph_data.values():
                for j, value in enumerate(values_list):
                    self.y[first_i+j, :-1] = self.y[first_i+j, 1:]
                    self.y[first_i+j, -1] = value
                    self.plot_items[first_i+j].setData(self.x[first_i+j], self.y[first_i+j])

            self.image_item.setImage(frame)
            self.prev_lyapunov = lyapunov_value
        except Empty:
            pass

