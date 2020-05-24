from setuptools import setup, find_packages

setup(
    name="rvae",
    version="0.1",
    author="Dimitris Kalatzis",
    authoremail="dimitriskalatzis89@gmail.com",
    description="VAEs with Riemannian Brownian Motion Priors",
    url="https://github.com/dimkal89/RVAE",
    package_data={'rvae': ['README.md']},
    packages=find_packages()
)