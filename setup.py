import os
import io
from setuptools import setup, find_packages


def read(*names, **kwargs):
    with io.open(os.path.join(os.path.dirname(__file__), *names),
                 encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()


readme = read("README.md")

install_requires = ["confuse",
                    "future",
                    "graphviz",
                    "matplotlib",
                    "numpy",
                    "pytorch-ignite",
                    "seaborn",
                    "tensorboard",
                    "torch>=1.1.0",
                    "tqdm"]

tests_require = ["coverage",
                 "pytest"]

setup(
    name='pytorch-bonsai',
    version='0.0.1',
    packages=find_packages(exclude=('tests', 'tests.*',)),
    url='https://github.com/ItamarWilf/pytorch-bonsai',
    license='MIT ',
    author='Itamar Wilf',
    author_email='',
    description='Basic pruning for Pytorch neural netwroks',
    zip_safe=True,
    install_requires=install_requires,
    tests_require=tests_require
)
