"""Setuptools procedure for building the milkshake package."""

from setuptools import setup

with open("requirements.txt") as f:
    lines = f.read()
    required = lines.splitlines()

setup(
    author="Tyler LaBonte",
    author_email="tlabonte@gatech.edu",
    description="Quick and extendable experimentation with classification models",
    install_requires=required,
    name="milkshake",
    packages=["milkshake"],
    version="1.0.0",
)
