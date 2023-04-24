# Setting up the packages for the project to be installed : tensorflow needed 

from setuptools import setup, find_packages

setup(
    name='myproject',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'click',
        'tensorflow',
        'numpy',
        'keras',
    ],
)

        