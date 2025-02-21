#!/usr/bin/env python

from setuptools import setup, find_packages

with open('requirements.txt', 'rt') as f:
    install_requires = [line.strip() for line in f.readlines()]

with open('README.md', 'rt') as f:
    long_description = f.read()

setup(
    name='brain-norm',
    version='0.1.0',  # Update the version number for new releases
    url='https://github.com/elainez1006/brain-norm',
    license='MIT',
    author='Elaine Zhang',
    author_email='elainez1005@gmail.com',
    description='A Python package for localisation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=install_requires,
    python_requires='>=3.7',
)