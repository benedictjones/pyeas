import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Example Usage - DE & OAIES Hybrid Approach
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
    from pyeas._population import Genes, Population

    member = [
        Genes(bounds=(-5,5), number=6),
    ]

    pop = Population(
        size=200,
        member=member,
        seed=42,
    )

    pop
    return (pop,)


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

    optimizer_1 = DE(population=pop, mut=0.4, crossp=0.4, mut_scheme='best2', seed=1)  # 'ttb1', rand1
    return (optimizer_1,)


@app.cell
def _(optimizer_1, rmse, x, y):
    from tqdm import tqdm
    from pyeas.examples.funcs import polynomial_order_5

    num_gens = 50
    pbar_de = tqdm(range(num_gens), unit=' generations')

    for g_de in pbar_de:
        trial_pop_de = optimizer_1.ask(loop=g_de)

        solutions_de = []
        for trial in trial_pop_de:
            pred_de = polynomial_order_5(x, trial)
            value = rmse(y, pred_de)
            solutions_de.append(value)

        optimizer_1.tell(solutions_de, trial_pop_de)

        pbar_de.set_description_str(f'loss: {optimizer_1.best_member[0]:.4}, Best Member: {optimizer_1.best_member[1]} ')
    return polynomial_order_5, tqdm


@app.cell
def _(mo):
    mo.md(r"""
    Now, continue with another optimizer:
    """)
    return


@app.cell
def _(optimizer_1):
    from pyeas._oaies import OAIES 

    optimizer_2 = OAIES(
        population=optimizer_1.population, 
        alpha=0.00001, 
        sigma=0.00004, 
        seed=42, 
        constraint_handle='clip', 
        optimiser='adam',
    )
    return (optimizer_2,)


@app.cell
def _(optimizer_2, polynomial_order_5, rmse, tqdm, x, y):
    num_extra_gens = 1000
    pbar_oaies = tqdm(range(num_extra_gens), unit=' generations')

    for g2 in pbar_oaies:

        solutions_oaies = []
        trial_pop_2 = optimizer_2.ask(loop=g2)

        for _trial in trial_pop_2:
            _pred = polynomial_order_5(x, _trial)
            _value = rmse(y, _pred)
            solutions_oaies.append(_value)

        optimizer_2.tell(solutions_oaies, trial_pop_2, t=g2)
        _pred = polynomial_order_5(x, optimizer_2.parent)
        _parent_fit = rmse(y, _pred)

        optimizer_2.tell_parent(float(_parent_fit))
        pbar_oaies.set_description_str(f'loss: {optimizer_2.best_member[0]:.4}, best Member: {optimizer_2.parent}')
    return


@app.cell
def _(np, optimizer_1, optimizer_2):
    for i in np.arange(-4,-1):
        print(optimizer_1.history['best_solutions'][i])
    print(optimizer_2.history['best_solutions'][-1])
    return


@app.cell
def _(optimizer_1, optimizer_2):
    history_best_fits = optimizer_1.history['best_fits'] + optimizer_2.history['best_fits']
    history_best_solutions = optimizer_1.history['best_solutions'] + optimizer_2.history['best_solutions']
    return (history_best_fits,)


@app.cell
def _(history_best_fits):
    import matplotlib.pyplot as plt

    fig_all, ax_all = plt.subplots()
    ax_all.plot(history_best_fits)
    ax_all.set_yscale('log')
    ax_all.set_xlabel('Generation')
    ax_all.set_ylabel('loss')
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
