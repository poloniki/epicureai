from setuptools import setup, find_packages

with open("requirements.txt", "r") as file:
    lines = file.readlines()
reqs = [each.strip() for each in lines]

setup(name="epicureai", install_requires=reqs, packages=find_packages())
