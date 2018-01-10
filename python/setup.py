from setuptools import setup
from setuptools import find_packages


# Used in CI and by deveopers
build_requires = [
    'wheel',
    'twine',
]

# Used in CI and by deveopers
test_requires = [
    'pytest',
    'pytest-runner',
    'flake8',
]

# Used in developers' PCs only
dev_requires = [
    'pytest-sugar',
    'sphinx',
    'sphinx-rtd-theme',
]

# Used in ReadTheDocs build servers
# (actually they already have Sphinx, but let's specify it explicitly.)
docs_requires = [
    'sphinx',
]

setup(
    name='wtte',
    version='1.0.2',
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
        'tf_gpu': ["tensorflow-gpu>=1.1.0"],
        'build': build_requires,
        'test': test_requires,
        'dev': dev_requires,
        'docs': docs_requires,
    },
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    packages=find_packages('.', exclude=['examples', 'tests']),
)
