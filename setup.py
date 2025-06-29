from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf8") as f:
    long_description = f.read()



if __name__ == "__main__":
    setup(
        name="pyeas",
        version="1.0.0",
        author="benedictjones",
        description="Implements Evolutionary Algorithms and tools",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/benedictjones/pyeas",
        license="BSD License (BSD-3-Clause)",
        packages=find_packages(exclude=['*.examples*', ]),
        keywords=[
            'python', 
            'Differential Evolution', 
            'DE', 
            'OpenAI Evolutionary Strategy', 
            "OpenAI-ES", 
            "CMAES", 
            "Page Trend Test"
            ],
        install_requires=[
            "numpy", 
            "scipy>=1.7", 
            "matplotlib", 
            "tqdm",
        ],
        extras_requires=["twine"],
        )
