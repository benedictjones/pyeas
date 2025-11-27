# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.17.0",
#     "pyzmq",
# ]
# ///

import marimo

__generated_with = "0.18.1"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Example Usage - Population Management
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
    Now, we can define a set of Genes using:
    """)
    return


@app.cell
def _():
    from pyeas._population import Genes

    Genes(bounds=(-1,2), number=10)
    return (Genes,)


@app.cell
def _(mo):
    mo.md(r"""
    Next, we define the structure of a 'genome' or member my making a list of genes:
    """)
    return


@app.cell
def _(Genes):
    member = [
        Genes(bounds=(-1,0), number=2),
        Genes(bounds=(4,8), number=1),
        Genes(bounds=(0,1), number=2),
    ]
    return (member,)


@app.cell
def _(mo):
    mo.md(r"""
    Then, initialise  the population:
    """)
    return


@app.cell
def _(member):
    from pyeas._population import Population

    pop = Population(
        size=3,
        member=member,
        seed=42,
    )
    return Population, pop


@app.cell
def _(mo):
    mo.md(r"""
    The generated population is:
    """)
    return


@app.cell
def _(pop):
    import numpy as np 

    print(np.shape(pop.population))
    pop.population
    return (np,)


@app.cell
def _(mo):
    mo.md(r"""
    Now, the raw population is normalised to between zero and one:
    """)
    return


@app.cell
def _(np, pop):
    print(np.shape(pop.population_raw))
    pop.population_raw
    return


@app.cell
def _(mo):
    mo.md(r"""
    This is scaled using the bounds:
    """)
    return


@app.cell
def _(pop):
    pop.bounds
    return


@app.cell
def _(mo):
    mo.md(r"""
    Finally, we can try and set a new population, we catch this setter and ensure several checks are run:
    """)
    return


@app.cell
def _(pop):
    try:
        pop.population = [[1,2], [4,5,6]]
    except Exception as e:
        print(f"Expected Failure: \n{e}")
    return


@app.cell
def _(np, pop):
    try:
        pop.population = np.array(
            [[-0.22604, -0.56112,  7.4344 ,  0.69737,  0.09418,  0.09418],
           [-0.02438, -0.23886,  7.14424,  0.12811,  0.45039,  0.09418],
           [-0.6292 , -0.07324,  6.57548,  0.82276,  0.44341,  0.09418]],
        )
    except Exception as e:
        print(f"Expected Failure: \n{e}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    If we exceed the limits (i.e., 0 & 1 for the normalised population) of the bounds, the values are clipped with logging warnings:
    """)
    return


@app.cell
def _(pop):
    pop.population = pop.population*1.8
    pop.population
    return


@app.cell
def _(mo):
    mo.md(r"""
    Next, consider the same population but with the genes remained in a grouping structure (i.e., not flattened):
    """)
    return


@app.cell
def _(Population, member):
    pop_grouped = Population(
        size=3,
        member=member,
        seed=42,
        flatten_gene_groupings=False,
    )
    return (pop_grouped,)


@app.cell
def _(np, pop_grouped):
    print(np.shape(pop_grouped.population))
    pop_grouped.population
    return


@app.cell
def _(pop_grouped):
    pop_grouped.population
    return


@app.cell
def _(mo):
    mo.md(r"""
    We can similarly assign (and catch) setting to a new population value:
    """)
    return


@app.cell
def _(np, pop_grouped):
    try:
        pop_grouped.population = np.array(
            [
                [np.array([-0.22604, -0.56112]), np.array([7.4344]), np.array([0.69737, 0.09418])],
                [np.array([-0.02438, -0.23886]), np.array([7.14424]),np.array([0.12811, 0.45039])],
                [np.array([-0.6292 , -0.07324]), np.array([6.57548]),np.array([0.82276, 0.44341])], 
                [np.array([-0.6292 , -0.07324]), np.array([6.57548]),np.array([0.82276, 0.44341])]
            ],
            dtype=object,
        )
    except Exception as e:
        print(f"Expected Failure: \n{e}")
    return


@app.cell
def _(np, pop_grouped):
    try:
        pop_grouped.population = np.array(
            [
                [np.array([-0.22604, -0.56112]), np.array([7.4344]), np.array([0.69737, 0.09418])],
                [np.array([-0.02438, -0.23886]), np.array([7.14424]),np.array([0.12811, 0.45039])],
                [np.array([-0.6292 , -0.07324, 6.57548]),np.array([0.82276, 0.44341])], 
            ],
            dtype=object,
        )
    except Exception as e:
        print(f"Expected Failure: \n{e}")
    return


@app.cell
def _(pop_grouped):
    pop_grouped.population = pop_grouped.population*2
    pop_grouped.population
    return


@app.cell
def _(mo):
    mo.md(r"""
    We can retain the raw population:
    """)
    return


@app.cell
def _(np, pop_grouped):
    print(np.shape(pop_grouped.population_raw))
    pop_grouped.population_raw
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
