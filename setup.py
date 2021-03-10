from setuptools import setup 
setup( 
	name='MovieRecommender', 
	version='1.0', 
	description='A program that recommends movies.', 
	author=['Priyanka Bijlani', 'Sharmeelee Bijlani', 'Laura Thriftwood', 'Lakshmi Venkatasubramanian'], 
	packages=['MovieRecommender'], #same as name 
	install_requires=['torch', 'numpy', 'os', 'sys', 'time', 'pandas', 'zipfile', 'tarfile', 'requests', 'matplotlib.pyplot', 'scipy.sparse', 'implicit', 'pandas.api.types', 'sklearn'] #external packages as dependencies 
)
