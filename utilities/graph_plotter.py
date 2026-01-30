import matplotlib.pyplot as plt


def live_plot_process(queue):

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    max_line, = ax.plot([], [], 'r-', label='Max')
    avg_line, = ax.plot([], [], 'b--', label='Avg')
    ax.legend()

    gens, max_f, avg_f = [], [], []

    while True:
        data = queue.get()
        if data == "STOP":
            break

        gen, max_val, avg_val = data
        gens.append(gen)
        max_f.append(max_val)
        avg_f.append(avg_val)

        max_line.set_data(gens, max_f)
        avg_line.set_data(gens, avg_f)
        ax.relim()
        ax.autoscale_view()

        plt.pause(0.001)

    plt.ioff()
    plt.show()
