from setuptools import setup

setup(name='matrixFactorizer',
      version='0.0.1',
      description='matrix factorization using 1st order optimization, matrix can have missing values',
      author='Tian Yu',
      author_email='ty277@cornell.edu',
      url='https://github.com/ty277/matrixFactorizer',
      packages=['matrixFactorizer'],
      install_requires=['numpy'],
      )