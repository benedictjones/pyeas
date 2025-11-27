import marimo

__generated_with = "0.18.1"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Example Usage - OpenAI EVolutionary Strategy
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
def _(mo):
    mo.md(r"""
    Lets First define a problem:
    """)
    return


@app.cell
def _():
    import numpy as np

    rng = np.random.default_rng(0)

    x = np.linspace(0, 10, 500)
    y = np.cos(x) + rng.normal(0, 0.2, 500)
    return np, x, y


@app.cell
def _(np):
    def rmse(y:float, y_pred:float) -> float:
        return float(np.sqrt(sum((y - y_pred)**2) / len(y)))
    return (rmse,)


@app.cell
def _(mo):
    mo.md(r"""
    Next, lets make a population which we will evolve:
    """)
    return


@app.cell
def _():
    from pyeas._population import Genes, Population

    member = [
        Genes(bounds=(-5,5), number=6),
    ]

    pop = Population(
        size=50,
        member=member,
        seed=42,
    )

    pop
    return (pop,)


@app.cell
def _(mo):
    mo.md(r"""
    ### The First Generation

    Now, lets initialize our solver, and iterate though the fist generation of population members:
    """)
    return


@app.cell
def _(pop):
    from pyeas._oaies import OAIES 

    optimizer = OAIES(
        population=pop,
        alpha=0.01,
        sigma=0.01,
        seed=1,
    )
    return OAIES, optimizer


@app.cell
def _(np, optimizer):
    trial_pop = optimizer.ask(loop=0)
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
    solutions[:5]
    return polynomial_order_5, solutions


@app.cell
def _(optimizer, solutions, trial_pop):
    optimizer.tell(solutions, trial_pop)
    optimizer._parent_norm
    return


@app.cell
def _(mo):
    mo.md(r"""
    Now, we must 'collapse' the gradient step and determine the final parent.
    """)
    return


@app.cell
def _(optimizer, polynomial_order_5, rmse, x, y):
    # Calc the new parent fitness, and Tell Again!
    _pred = polynomial_order_5(x, optimizer.parent)
    _parent_fit = rmse(y, _pred)
    optimizer.tell_parent(_parent_fit)
    return


@app.cell
def _(optimizer):
    optimizer.best_member
    return


@app.cell
def _(mo):
    mo.md(r"""
    [
      2715.964550083185,
      "[ 1.6286  4.5562 -2.1355  4.2481 -4.7514  0.552 ]"
    ]
    """)
    return


@app.cell
def _(np, optimizer):
    trial_pop_1 = optimizer.ask(loop=1)
    print(np.shape(trial_pop_1))
    trial_pop_1[:3]
    return


@app.cell
def _(optimizer):
    optimizer._sample_trial_pop(loop=10)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Generational Optimization

    Starting from scratch:
    """)
    return


@app.cell
def _(OAIES, pop):
    optimizer = OAIES(population=pop, alpha=0.002, sigma=0.005, seed=1, constraint_handle='reflection', optimiser='adam')
    return (optimizer,)


@app.cell
def _(optimizer, polynomial_order_5, rmse, x, y):
    from tqdm import tqdm

    num_gens = 1000
    pbar = tqdm(range(num_gens), unit=' generations')

    for generation in pbar:

        solutions_1 = []
        trial_pop_2 = optimizer.ask(loop=generation)

        for _trial in trial_pop_2:
            _pred = polynomial_order_5(x, _trial)
            _value = rmse(y, _pred)
            solutions_1.append(_value)

        optimizer.tell(solutions_1, trial_pop_2, t=generation)
        _pred = polynomial_order_5(x, optimizer.parent)
        _parent_fit = rmse(y, _pred)

        optimizer.tell_parent(float(_parent_fit))
        pbar.set_description_str(f'Best Member: {optimizer.parent}, loss: {optimizer.best_member[0]:.4} ')
    return (num_gens,)


@app.cell
def _(optimizer):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(optimizer.history['best_fits'])
    plt.yscale('log')

    print(optimizer.history['best_solutions'][-1])
    return (plt,)


@app.cell
def _(np, optimizer, plt, polynomial_order_5, x, y):
    fig_solution, ax_solution = plt.subplots()

    ax_solution.scatter(x, y, marker='.', color='r', alpha=0.7, label='Target data')
    plt.plot(x, np.cos(x), '--', label='ideal cos(x)', color='k', alpha=0.5)

    data = polynomial_order_5(x, optimizer.history['best_solutions'][-1])
    ax_solution.plot(x, data, label='OpenAI-ES Solution')

    ax_solution.legend()
    return


@app.cell
def _():
    # fig2, ax2 = plt.subplots()
    # ax2.scatter(x, y, marker='.', color='r', alpha=0.7, label='Target data')
    # plt.plot(x, np.cos(x), '--', label='ideal cos(x)', color='k', alpha=0.5)
    # data = polynomial_order_5(x, optimizer.history['best_solutions'][-1])
    # ax2.plot(x, data, label='OpenAI-ES Solution')
    # ax2.legend()
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
def _():
    # import matplotlib.animation as animation

    # plt.rcParams['animation.html'] = 'jshtml'
    # fig_ani, (_ax, ax2) = plt.subplots(ncols=2, figsize=(9, 4))
    # fig_ani.suptitle('DE fitting a 5th order polynomial to noisy cos() data')
    # _ax.set_ylim([-10, 10])
    # _ax.scatter(x, y, marker='.', color='r')
    # _ax.set_xlabel('x')
    # _ax.set_ylabel('y')
    # ax2.set_yscale('log')
    # ax2.plot(optimizer.history['best_fits'])
    # ax2.set_xlabel('Generation')
    # ax2.set_ylabel('rmse')
    # it_line, = ax2.plot([0, 0], [np.min(optimizer.history['best_fits']), np.max(optimizer.history['best_fits'])], markersize=5, color='k', alpha=0.5)
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # lines = []

    # def ani(i):
    #     lim = 10 - i / num_gens * (10 - 4)
    #     _ax.set_ylim([-lim, lim])
    #     data = polynomial_order_5(x, optimizer.history['best_solutions'][i])
    #     line, = _ax.plot(x, data, alpha=0.4)
    #     lines.append(line)
    #     if len(lines) > 10:
    #         lines[0].remove()
    #         lines.pop(0)
    #     it_line.set_xdata([i, i])
    # length = 20
    # FPS = num_gens / length
    # the_animation = animation.FuncAnimation(fig_ani, ani, frames=np.arange(num_gens), interval=20)
    return


@app.cell
def _(ani, mo):
    vid = ani.to_html5_video()

    mo.Html(vid)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
