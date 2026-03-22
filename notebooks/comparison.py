import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Example Usage - Compare Methods
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
    Next, lets make a population which we will evolve:
    """)
    return


@app.cell
def _():
    from pyeas._population import Genes, Population
    from pyeas.examples.funcs import hc6 as loss_func

    member = [
        Genes(bounds=(-3,3), number=1),
        Genes(bounds=(-2,2), number=1),
    ]

    pop = Population(
        size=20,
        member=member,
        seed=42,
    )

    pop
    return loss_func, pop


@app.cell
def _(mo):
    mo.md(r"""
    ### Generational Optimization

    Starting from scratch:
    """)
    return


@app.cell
def _(pop):
    from pyeas._de import DE

    optimizer = DE(
        population=pop, 
        mut=0.4, 
        crossp=0.4, 
        mut_scheme='best2', # 'ttb1', rand1
        seed=1,
    )  
    return (optimizer,)


@app.cell
def _(loss_func, optimizer):
    from tqdm import tqdm

    trial_pops = []
    num_gens = 40
    pbar = tqdm(range(num_gens), unit=' generations')

    for generation in pbar:
        trial_pop = optimizer.ask(loop=generation)
        trial_pops.append(trial_pop)

        solutions_1 = []
        for _trial in trial_pop:

            _value = loss_func(*_trial)

            solutions_1.append(_value)

        optimizer.tell(solutions_1, trial_pop)

        pbar.set_description_str(f'loss: {optimizer.best_member[0]:.4}, Best Member: {optimizer.best_member[1]} ')
    return num_gens, trial_pops


@app.cell
def _(optimizer):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(optimizer.history['best_fits'])
    plt.show()
    print(optimizer.history['best_solutions'][-1])
    return (plt,)


@app.cell
def _(loss_func):
    import numpy as np 

    x1 = np.linspace(-3, 3, 150)
    x2 = np.linspace(-2, 2, 150)
    X1, X2 = np.meshgrid(x1, x2)
    Z = loss_func(X1, X2)
    return X1, X2, Z, np


@app.cell
def _(X1, X2, Z, optimizer, plt):
    fig_solution = plt.figure(figsize = (10,7))

    contours = plt.contour(X1, X2, Z, 20)
    plt.clabel(contours, inline = True, fontsize = 10)

    # plt.imshow(Z, extent = [-3,3,-2,2], origin = 'lower', cmap = 'jet', alpha = 1)

    plt.title("Evolution of the cost function during gradient descent", fontsize=15)

    r1, r2 = zip(*optimizer.history['best_solutions'])
    plt.plot(r1, r2, label='Solution Path', color='r', alpha=0.8)
    plt.plot(r1, r2, 'x', color='r', alpha=0.5)
    plt.legend()
    return r1, r2


@app.cell
def _(mo):
    mo.md(r"""
    Now, lets animate the results:
    """)
    return


@app.cell
def _(Z, np, num_gens, optimizer, plt, r1, r2, trial_pops):
    import matplotlib.animation as animation


    fig_ani, (ax_ani, ax_ani_2) = plt.subplots(ncols=2, figsize=(9, 4))

    fig_ani.suptitle('OpenAI-ES on $f = 0.5*x1^2 + (5/2)*x2^2 - x1*x2 - 2*(x1 + x2)$')


    ax_ani.imshow(Z, extent = [-3,3,-2,2], origin = 'lower', cmap = 'jet', alpha = 1)
    it_point, = ax_ani.plot([], [], '*', color='w', alpha=1, linestyle="None") 
    it_converg, = ax_ani.plot([], [], '-*', markersize=5, color='w', alpha=0.3) 
    trs, = ax_ani.plot(trial_pops[0][:,0], trial_pops[0][:,1], marker=".", color='k', linestyle="None") 
    ax_ani.set_xlabel("x1")
    ax_ani.set_ylabel("x2")
    # ax_ani.set_xlim([1.75, 5])
    # ax_ani.set_ylim([-0.25, 2])


    # ax_ani_2.set_yscale('log')
    ax_ani_2.plot(optimizer.history['best_fits'])
    ax_ani_2.set_xlabel("Generation")
    ax_ani_2.set_ylabel("Function")
    it_line, = ax_ani_2.plot(
        [0, 0], 
        [np.min(optimizer.history['best_fits']), np.max(optimizer.history['best_fits'])], 
        markersize=5, 
        color='k', 
        alpha=0.5,
    ) 
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


    # --- MODIFIED: Frame Generation ---
    gen_step = 10 
    # Generate the indices of the generations we want to see (0, 10, 20, ...)
    gens_to_plot = np.arange(0, num_gens, gen_step)

    # Create the full list of frames (Gen_N_Trial, Gen_N_Parent, Gen_N+10_Trial, ...)
    frames_to_animate = []
    for gen_index in gens_to_plot:
        # Frame 1: Trial population (equivalent to i = 2*gen_index)
        frames_to_animate.append(2 * gen_index)
        # Frame 2: Parent/Convergence (equivalent to i = 2*gen_index + 1)
        frames_to_animate.append(2 * gen_index + 1)

    # Ensure the frames do not exceed the total number of frames in the original animation (num_gens * 2)
    total_possible_frames = num_gens * 2
    frames_to_animate = np.array(frames_to_animate)
    frames_to_animate = frames_to_animate[frames_to_animate < total_possible_frames]

    # --- MODIFIED: The ani function logic remains almost the same ---
    def ani(i):
        # Determine the actual generation index (0, 10, 20, ...)
        # i is now one of the values from frames_to_animate (e.g., 0, 1, 20, 21, 40, 41, ...)

        # Check if 'i' is even (Trial phase) or odd (Parent phase)
        if (i % 2) == 0:  # Even index: Trial phase
            gen_idx = i // 2 # 0 -> 0, 20 -> 10, 40 -> 20

            # Check if the generation index is valid before accessing history
            if gen_idx >= num_gens or gen_idx < 0:
                return

            trials = trial_pops[gen_idx]
            trs.set_data(trials[:,0], trials[:,1]) 

        else: # Odd index: Parent/Convergence phase
            gen_idx = (i - 1) // 2 # 1 -> 0, 21 -> 10, 41 -> 20

            # Check if the generation index is valid before accessing history
            if gen_idx >= num_gens or gen_idx < 0:
                return

            # Hide trials
            trs.set_data([], []) 

            # Update convergence markers
            it_point.set_data(r1[gen_idx], r2[gen_idx])
            it_converg.set_data(r1[:gen_idx+1], r2[:gen_idx+1])

            # Update generation line
            it_line.set_xdata([gen_idx, gen_idx])

        # Return the items that were modified for FuncAnimation to redraw them efficiently
        return it_point, it_converg, trs, it_line


    FPS = len(frames_to_animate) / 20 # Adjust FPS calculation for desired 20 second duration
    ani = animation.FuncAnimation(
        fig_ani, 
        ani, 
        frames=frames_to_animate, # Use the sliced frame indices
        interval=20, # Interval is now 20ms (50 FPS) or 50ms (20 FPS)
        # blit=True # Use blit=True for potentially faster rendering
    )
    return (ani,)


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
