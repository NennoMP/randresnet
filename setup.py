from setuptools import setup, find_packages


setup(
    name = 'randresnet',
    version = '0.1',
    author = 'Matteo Pinna',
    author_email = 'nennomp@gmail.com',
    description = 'Randomized Convolutional Residual Neural Networks paper code',
    url = 'https://github.com/NennoMP/randresnet',
    packages=find_packages(),
    python_requires='>=3.12',
)