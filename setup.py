# setup.py

from setuptools import setup

setup(
    name="hushbert",
    version="0.1.0",
    packages=['hushbert','hushbert.core.model'],
    install_requires=[
        "pandas",
        "bertopic"
    ],
)