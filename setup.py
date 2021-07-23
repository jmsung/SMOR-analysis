
from setuptools import setup

setup(
    name='smor-analysis',
    version='0.1',
    description='Single Molecule Off Rates (SMOR) analysis',
    url='https://github.com/jmsung/smor-analysis',
    author='Jongmin Sung',
    author_email='jongmin.sung@gmail.com.com',
    packages=['smor_analysis'],
    install_requires=[
        'numpy',
        'matplotlib',
        'pathlib',
        'scipy',
        'tifffile',
        'imreg_dft',
        'scikit-image',
        'scikit-learn',
        'hmmlearn'
    ],
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'run-smor-analysis = smor_analysis.smor_analysis:main',
        ]
    }
)
