
from setuptools import setup

setup(
    name='smb-analysis',
    version='0.1',
    description='Single molecule binding analysis',
    url='https://github.com/jmsung/smb-analysis',
    author='Jongmin Sung',
    author_email='jongmin.sung@gmail.com.com',
    packages=['smb_analysis'],
    install_requires=[
		'numpy == 1.17',
		'matplotlib == 3.1', 
		'pathlib == 1.0', 
		'scipy == 1.3', 
		'tifffile == 0.15', 
		'imreg_dft == 2.0', 
		'scikit-image == 0.15', 
		'scikit-learn == 0.21', 
		'hmmlearn == 0.2.2'
        ],
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'smb = smb_analysis.smb:main',
            ]
    }
)


