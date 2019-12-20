from setuptools import setup
 
setup(
	# This is the name of your PyPI-package
	name='smb-analysis',    

	# Update the version number for new releases
	version='0.1',        

	# Specify executables to be installed in the binary folder
	scripts=['src/smb-analysis.py'],

	install_requires=['numpy==1.17', 'matplotlib==3.1', 'pathlib==1.0', 'scipy==1.3', 'tifffile==0.15', 'imreg_dft==2.0', 'scikit-image==0.15', 'scikit-learn==0.21', 'hmmlearn==0.2.2']
)


