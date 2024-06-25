from setuptools import setup, find_packages

with open("requirements.txt", 'r') as f:
    requirements = f.readlines()

setup(
    name='cogvarlib',
    version='0.1.0',
    description='Library that implements cognitive science models',
    author='Amric Trudel',
    url='https://github.com/AmricTrudel/coding_tutorial',
    python_requires='~=3.10',
    install_requires=requirements,
    packages=find_packages()
)