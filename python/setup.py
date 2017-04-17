from setuptools import setup
from setuptools import find_packages

setup(name='wtte',
      version='0.0.2',
      description='Weibull Time To Event model. A Deep Learning model for churn- and failure prediction and everything else.',
      author='Egil Martinsson',
      author_email='egil.martinsson@gmail.com',
      url='https://github.com/ragulpr/wtte-rnn/',
      license='MIT',
      install_requires=['numpy', 'pandas', 'keras','tensorflow'],
      setup_requires=['pytest-runner'],
      tests_require=['pytest'],
      packages=find_packages('.'))
