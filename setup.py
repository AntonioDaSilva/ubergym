import setuptools

install_requires = ['gym>=0.25.1','numpy>=1.23.1', 'matplotlib>=3.5.1', 'networkx>=2.8', 'gurobipy>=9.5.2', 'methodtools>=0.4.5']

setuptools.setup(name='ubergym',
    version='0.0.1',
    description='Open AI gym interface to simulate Uber environment with driver agents',
    author = 'Mert Unsal',
    author_email = 'mailmertunsal@gmail.com',
    python_requires='>=3.9',
    install_requires=install_requires,
    packages=setuptools.find_packages(),
    include_package_data=True,
)