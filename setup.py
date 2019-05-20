from setuptools import setup

setup(name='deepl_toolkit',
      version='0.01',
      description='Deep Learning toolkit',
      url='https://github.com/rs9000/deepl_toolkit',
      author='Rosario (Rs)',
      author_email='rs.dicarlo@gmail.com',
      license='MIT',
      packages=['deepl_toolkit'],
      install_requires=[
          'torch',
      ],
      zip_safe=False)