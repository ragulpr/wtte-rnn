from setuptools import setup
from setuptools import find_packages

setup(
    name='wtte',
    version='0.0.2',
    description='Weibull Time To Event model. A Deep Learning model for churn- and failure prediction and everything else.',
    author='Egil Martinsson',
    author_email='egil.martinsson@gmail.com',
    url='https://github.com/ragulpr/wtte-rnn/',
    license='MIT',
    install_requires=[
        'keras>=2.0',
        'numpy',
        'pandas',
        'scipy',
        'six==1.10.0',
    ],
    extras_require={
        'plot': ['matplotlib'],
        'tf': ["tensorflow>=1.1.0"],
        'tf_gpu': ["tensorflow-gpu>=1.1.0"]
    },
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    packages=find_packages('.', exclude=['examples', 'tests']),
)
