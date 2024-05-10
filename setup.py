from setuptools import setup

# setup for gradls
setup(
    name="gradls",
    version="0.0.1",
    packages=["gradls"],
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "pandas",
        "scikit-learn",
        "plotly",
        "tqdm",
        "torch==2.0.0",
        "black",
    ],
    author="Ramkrishna Acharya(QViper)",
    author_email="qramkrishna@gmail.com",
    description="A package for gradients-based methods on large least square problems",
    url="https://github.com/q-viper/gradients-based-methods-on-large-least-square",
)
