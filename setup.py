try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(name='Deep Compression',
      version='1.0',
      description='A python module to compress all kinds of data using deep learning.',
      author='Grégoire Jauvion',
      packages=[
          'deepcompression',
          'deepcompression.codec',
          'deepcompression.data',
          'deepcompression.model'
      ],
      install_requires=[
          'tqdm',
          'numpy',
          'torch',
          'torchvision',
          'constriction'
      ],
      include_package_data=True,
)
