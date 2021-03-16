from setuptools import setup 
setup( 
	name='MovieRecommender', 
	version='1.0', 
	description='A program that recommends movies.', 
	author=['Priyanka Bijlani', 'Sharmeelee Bijlani', 'Laura Thriftwood', 'Lakshmi Venkatasubramanian'], 
	packages=['MovieRecommender'], #same as name 
	install_requires=['torch', 'numpy', 'pandas', 'requests', 'matplotlib', 'scipy', 'implicit', 'sklearn','tabulate'] #external packages as dependencies 
)
