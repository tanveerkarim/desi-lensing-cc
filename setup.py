from setuptools import find_packages, setup

setup(
    name="cmbCrossELG",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)