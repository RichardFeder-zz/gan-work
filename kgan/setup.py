from setuptools import setup, find_packages, Extension

#compile_args = ['-fopenmp']
#link_args = ['-fopenmp']
compile_args = []
link_args = []

setup(
    name='WebyGAN',
    version=0.1,

    packages=find_packages(),

    author="P. Berger, R. Feder-Staehle",
    author_email="philippe.j.berger@gmail.com",
    description="LSS on the DL."
)
