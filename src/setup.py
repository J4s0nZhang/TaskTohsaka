from setuptools import find_packages, setup

install_requires = [
    "numpy",
    "pandas",
    "Pillow",
    "matplotlib",
    "scipy",
    "torch",
    "torchvision",
    "tqdm",
    "seaborn",
    "tensorboard-logger",
    "scikit-learn",
]

setup(
    name="taskTohsaka",
    author="Jianxing (Jason) Zhang",
    author_email="thejasonzhang@gmail.com",
    version="1.1",
    packages=find_packages(exclude=["staticFarmers", "tests"]),
    setup_requires=["wheel"],
    install_requires=install_requires,
)