from setuptools import find_packages, setup

install_requires = [
    "opencv-python",
    "pygetwindow",
    "pyautogui",
]

setup(
    name="taskTohsaka",
    author="Jianxing (Jason) Zhang",
    author_email="thejasonzhang@gmail.com",
    version="1.1",
    packages=find_packages(exclude=["staticFarmers", "tests", "jupyter"]),
    setup_requires=["wheel"],
    install_requires=install_requires,
)