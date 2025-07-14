from setuptools import setup

setup(
    name="stljax",
    version="1.1.3",
    description="stlcg with jax",
    author="Karen Leung",
    author_email="kymleung@uw.edu",
    packages=["stljax"],
    install_requires=[
        "jax",
        "numpy",
        "graphviz"
    ],
)
