from distutils.core import setup
from setuptools import find_packages

with open("requirements.txt") as f:
    install_requires = f.read().strip().split("\n")

setup(
    name="discobax",
    version="0.1.0",
    packages=find_packages(),
    author="see README.txt",
    author_email="mehrjou.arash@gmail.com",
    license="MIT",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=install_requires,
    classifiers=[
        "Operating System :: OS Independent",
    ],
)
