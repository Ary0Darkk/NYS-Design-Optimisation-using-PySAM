import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def create_live_plot(x_values, y_values, title="Live Plot"):
    fig, ax = plt.subplots()
    (line,) = ax.plot([], [], lw=2)

    # Setting static labels
    ax.set_title(title)

    # The update function now accepts 'x_source' and 'y_source' as extra args
    def update(frame, x_source, y_source):
        # We slice the explicit lists up to the current frame
        line.set_data(x_source[:frame], y_source[:frame])

        ax.relim()
        ax.autoscale_view()
        return (line,)

    # We use fargs to pass our explicit x and y lists into the update function
    ani = FuncAnimation(
        fig,
        update,
        frames=len(x_values),
        fargs=(x_values, y_values),  # This 'feeds' the data to update
        blit=False,
        interval=50,
        repeat=False,
    )

    plt.show()
    return ani
