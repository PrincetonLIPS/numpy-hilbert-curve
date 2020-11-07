from setuptools import setup

# Pull in the README.md
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()

setup(
    name='numpy-hilbert-curve',
    version='1.0.1',
    description='Implements Hilbert space-filling curves for Python with numpy',
    url='https://github.com/PrincetonLIPS/numpy-hilbert-curve',
    author='Ryan P. Adams',
    author_email='rpa@princeton.edu',
    license='MIT',
    packages=['hilbert'],
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
      'Development Status :: 4 - Beta',
      'Intended Audience :: Science/Research',
      'License :: OSI Approved :: MIT License',
    ],
)
