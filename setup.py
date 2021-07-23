
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
        'numpy == 1.17',
        'matplotlib == 3.1',
        'pathlib == 1.0',
        'scipy == 1.5',
        'tifffile == 0.15',
        'imreg_dft == 2.0',
        'scikit-image == 0.15',
        'scikit-learn == 0.21',
        'hmmlearn == 0.2.2'
    ],
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'run-smor-analysis = smor_analysis.smor_analysis:main',
        ]
    }
)
