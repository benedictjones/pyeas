# `marimo` Notebooks

Here we use [`marimo`](https://docs.marimo.io/) notebooks to demonstrate some examples.
These are useful as they exist as `.py` files, helping with better diff's and allowing them to be run as a script.


With `uv` we can edit a notebook in a web file using:
```
uv run marimo edit notebooks/de.py
```

Or, we can run an app using:
```
uv run marimo run notebooks/de.py
```

VSCode has a plugin to allow them to be edited directly.

## Converting between Marimo and Jupyter notebooks

To export a Marimo notebook to a Jupyter notebook, we can run:
```
marimo export ipynb notebook.py -o notebook.ipynb
```

To convert a Jupyter notebook to a Marimo notebook, we can run:
```
marimo export ipynb ./path/to/notebook.py -o ./path/to/notebook.ipynb --sort=top-down
```
