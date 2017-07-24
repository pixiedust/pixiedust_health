from setuptools import setup, find_packages
setup(name='pixiedust_health',
	  version='0.1',
	  description='Sample Notebook PixieApp for Watson Health',
	  url='https://github.com/ibm-watson-data-lab/pixiedust_health',
	  install_requires=['pixiedust'],
	  author='David Taieb',
	  author_email='david_taieb@us.ibm.com',
	  license='Apache 2.0',
	  packages=find_packages(),
	  include_package_data=False,
	  zip_safe=False
)
