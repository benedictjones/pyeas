import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Example Usage - Differential Evolution (DE)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Set up logger
    """)
    return


@app.cell
def _():
    import logging

    logging.basicConfig(
        encoding="utf-8",
        filemode="a",
        format="[{asctime}] [{levelname}] {message}",
        style="{",
        datefmt="%Y-%m-%d, %H:%M",
        level=logging.INFO,
    )

    logger = logging.getLogger()
    logger.info("Hello logging!")
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np

    return np, plt


@app.cell
def _(mo):
    mo.md(r"""
    Lets First define a problem:
    """)
    return


@app.cell
def _(np):
    rng = np.random.default_rng(0)

    x = np.linspace(0, 10, 500)
    y = np.cos(x) + rng.normal(0, 0.2, 500)
    return x, y


@app.cell
def _(np):
    def rmse(y, y_pred):
        return np.sqrt(sum((y - y_pred)**2) / len(y))

    return (rmse,)


@app.cell
def _(mo):
    mo.md(r"""
    Next, lets make a population which we will evolve:
    """)
    return


@app.cell
def _():
    from pyeas.population import Genes, Population

    member = [
        Genes(bounds=(-5,5), number=6),
    ]

    pop = Population(
        size=40,
        member=member,
        seed=42,
    )

    pop
    return (pop,)


@app.cell
def _(pop):
    pop.population[0][0]
    return


@app.cell
def _(pop):
    pop._normalised_population[0][0]
    return


@app.cell
def _(pop):
    pop.denormalise(pop._normalised_population)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### The First Generation

    Now, lets initialize our solver, and iterate though the fist generation of population members:
    """)
    return


@app.cell
def _(pop):
    from pyeas._de import DE 

    optimizer1 = DE(
        population=pop,
        mut=0.6,
        crossp=0.6,
        mut_scheme = 'ttb1',  # 'ttb1', rand1
        seed=1,
    )
    return DE, optimizer1


@app.cell
def _(np, optimizer1):
    trial_pop = optimizer1.ask(loop=0)
    print(np.shape(trial_pop))

    trial_pop[:3]
    return (trial_pop,)


@app.cell
def _(rmse, trial_pop, x, y):
    from pyeas.examples.funcs import polynomial_order_5

    solutions = []
    for t, _trial in enumerate(trial_pop):
        _pred = polynomial_order_5(x, _trial)
        _value = rmse(y, _pred)
        solutions.append(_value)
    solutions[:5]  # print(t, value)
    return polynomial_order_5, solutions


@app.cell
def _(optimizer1, solutions, trial_pop):
    optimizer1.tell(solutions, trial_pop)
    return


@app.cell
def _(optimizer1):
    optimizer1.population.population[optimizer1._best_idx]
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Generational Optimization

    Starting from scratch:
    """)
    return


@app.cell
def _(DE, pop):
    optimizer = DE(population=pop, mut=0.4, crossp=0.4, mut_scheme='best2', seed=1)  # 'ttb1', rand1
    return (optimizer,)


@app.cell
def _(optimizer, polynomial_order_5, rmse, x, y):
    from tqdm import tqdm

    num_gens = 1000
    pbar = tqdm(range(num_gens), unit=' generations')

    for generation in pbar:
        trial_pop_1 = optimizer.ask(loop=generation)

        solutions_1 = []
        for _trial in trial_pop_1:
            _pred = polynomial_order_5(x, _trial)
            _value = rmse(y, _pred)
            solutions_1.append(_value)

        optimizer.tell(solutions_1, trial_pop_1)

        pbar.set_description_str(f'Best Member: {optimizer.best_member[1]}, loss: {optimizer.best_member[0]:.4} ')
    return (num_gens,)


@app.cell
def _(mo):
    mo.md(r"""
    Consider the population change over time:
    """)
    return


@app.cell
def _(optimizer, plt, pop):
    _solutions = list(zip(*optimizer.history['best_solutions']))

    _fig, _axs = plt.subplots(
        nrows=len(_solutions), 
        sharex=True,
        figsize=(8,len(_solutions)*1.1),
    )
    for _i, _v in enumerate(_solutions):
        _axs[_i].plot(_v)
        _axs[_i].set_ylabel(f'Gene {_i}')
        _axs[_i].set_ylim(pop.bounds[_i])

    _axs[-1].set_xlabel('Generation')
    return


@app.cell
def _(mo):
    mo.md(r"""
    Now, lets plot the performance over the generations:
    """)
    return


@app.cell
def _(optimizer, plt):
    fig_performance, ax_performance = plt.subplots()
    ax_performance.plot(optimizer.history['best_fits'])
    ax_performance.set_yscale('log')
    ax_performance.set_xlabel('Generation')
    ax_performance.set_ylabel('loss')
    return


@app.cell
def _(mo):
    mo.md(r"""
    And the final solution:
    """)
    return


@app.cell
def _(np, optimizer, plt, polynomial_order_5, x, y):
    fig_solution, ax_solution = plt.subplots()

    ax_solution.scatter(x, y, marker='.', color='r', alpha=0.7, label='Target data')
    plt.plot(x, np.cos(x), '--', label='ideal cos(x)', color='k', alpha=0.5)

    data = polynomial_order_5(x, optimizer.history['best_solutions'][-1])
    ax_solution.plot(x, data, label='DE Solution')

    ax_solution.legend()
    return


@app.cell
def _(mo):
    mo.md(r"""
    Now, lets animate the results:
    """)
    return


@app.cell
def _(np, num_gens, optimizer, plt, polynomial_order_5, x, y):
    import matplotlib.animation as animation

    plt.rcParams['animation.html'] = 'jshtml'

    fig_ani, (ax_ani, ax_ani_2) = plt.subplots(ncols=2, figsize=(9, 4))

    fig_ani.suptitle('DE fitting a 5th order polynomial to noisy cos() data')

    ax_ani.set_ylim([-10, 10])
    ax_ani.scatter(x, y, marker='.', color='r')
    ax_ani.set_xlabel('x')
    ax_ani.set_ylabel('y')

    ax_ani_2.set_yscale('log')
    ax_ani_2.plot(optimizer.history['best_fits'])
    ax_ani_2.set_xlabel('Generation')
    ax_ani_2.set_ylabel('rmse')

    it_line, = ax_ani_2.plot(
        [0, 0], 
        [np.min(optimizer.history['best_fits']), np.max(optimizer.history['best_fits'])], 
        markersize=5, 
        color='k', 
        alpha=0.5,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    lines = []

    # --- MODIFIED: Use the original index (j) for history lookups ---
    def ani(i):

        # Check if the current frame index i is valid for the history lists
        if i >= num_gens:
            return # Skip this frame if it's out of range

        lim = 10 - i / num_gens * (10 - 4)
        ax_ani.set_ylim([-lim, lim])
        data = polynomial_order_5(x, optimizer.history['best_solutions'][i])
        line, = ax_ani.plot(x, data, alpha=0.4)
        lines.append(line)
        if len(lines) > 10:
            lines[0].remove()
            lines.pop(0)
        it_line.set_xdata([i, i])

    # --- MODIFIED: Only pass every 10th index to frames ---
    step = int(np.around(num_gens/20, -1))
    frames_to_animate = np.arange(0, num_gens, step)
    length = 20 # You may need to adjust this depending on the desired animation duration
    FPS = len(frames_to_animate) / length # Calculate FPS based on the new number of frames

    ani = animation.FuncAnimation(
        fig_ani, 
        ani, 
        frames=frames_to_animate, # Use the sliced array of indices
        interval=200 # Interval between frames in ms (50ms corresponds to 20 FPS)
    )
    return (ani,)


@app.cell
def _(ani, mo):
    vid = ani.to_html5_video()

    mo.Html(vid)
    return


@app.cell
def _():

    # writer = animation.PillowWriter(
    #     fps=15,
    #     metadata=dict(artist='Me'),
    #     bitrate=1800,
    # )
    # ani.save('scatter.gif', writer=writer)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
