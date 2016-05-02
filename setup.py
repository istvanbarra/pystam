# -*- coding: utf-8 -*-
"""
Created on Mon May  2 19:59:08 2016

@author: istvan
"""

from setuptools import setup

setup(name='pystam',
      version='0.1',
      description='Statistical models and estimations procedures',
      url='http://github.com/storborg/funniest',
      author='István Barra',
      author_email='barra.istvan@gmail.com',
      license='MIT',
      packages=['pystam'],
      install_requires=[
          'numpy',
          'scipy'
      ],
      zip_safe=False)