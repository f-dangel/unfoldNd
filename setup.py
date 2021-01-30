"""Setup file for unfoldNd."""

from setuptools import find_packages, setup

setup(
    author="Felix Dangel",
    name="unfoldNd",
    version="0.0.1",
    description="N-dimensional unfold in PyTorch",
    long_description="N-dimensional unfold in PyTorch using one-hot convolution",
    long_description_content_type="text/markdown",
    url="https://github.com/f-dangel/unfoldNd",
    license="MIT",
    packages=find_packages(),
    zip_safe=False,
    python_requires=">=3.6",
)
