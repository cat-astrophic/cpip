# Import setuptools

import setuptools

# Description comes from README.md

with open('README.md', 'r') as file:
    
    long_description = file.read()

# Metadata

setuptools.setup(
     name = 'cpip',  
     version = '1.0',
     scripts = ['cpip'] ,
     author = 'Michael Cary',
     author_email = 'macary@mix.wvu.edu',
     description = 'Core-Periphery Integer Program',
     long_description = long_description,
     long_description_content_type = 'text/markdown',
     url = 'https://github.com/cat-astrophic/cpip',
     packages = setuptools.find_packages(),
     classifiers = ['Programming Language :: Python :: 3',
                    'License :: OSI Approved :: MIT License',
                    'Operating System :: OS Independent']
)
