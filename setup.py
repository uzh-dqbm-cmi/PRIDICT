from setuptools import setup

setup(name='prieml',
      version='0.0.1',
      description='',
      url='https://github.com/CMI-UZH/PRIDICT',
      packages=['prieml'],
      python_requires='>=3.6.0',
      install_requires=[
            'numpy',
            'pandas',
            'scipy',
            # 'scikit-learn',
            'torch',
            'matplotlib',
            'seaborn',
            'prettytable',
            'tqdm'
      ],
      zip_safe=False)