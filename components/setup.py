"""Optional installation of the service-independent local helper package."""
from setuptools import setup

setup(
    name="cd16201-components",
    version="0.1",
    description="Local pipeline file interfaces and reviewer tooling",
    packages=["components"],
    package_dir={"components": "."},
    python_requires=">=3.12",
)
