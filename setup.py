from setuptools import setup

with open("uptools/include/VERSION", "r") as f:
    version = f.read().strip()

setup(
    name="uptools",
    version=version,
    license="BSD 3-Clause License",
    description="Description text",
    url="https://github.com/tklijnsma/uptools.git",
    author="Thomas Klijnsma",
    author_email="tklijnsm@gmail.com",
    packages=["uptools"],
    zip_safe=False,
    include_package_data=True,
    python_requires=">=3.5"
)