from setuptools import setup, find_packages


PACKAGENAME = "diffspin"
VERSION = "0.0.1"


setup(
    name=PACKAGENAME,
    version=VERSION,
    author=("Andrew Hearin", "Andrew Benson"),
    author_email="ahearin@anl.gov",
    description="Differentiable model of halo spin",
    long_description="Differentiable model of halo spin",
    install_requires=["numpy", "scipy", "jax"],
    packages=find_packages(),
    url="https://github.com/aphearin/diffspin",
    package_data={"diffspin": ("tests/testing_data/*.dat",)},
)
