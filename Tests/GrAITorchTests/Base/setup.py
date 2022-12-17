from setuptools import setup, find_packages

setup(
    name='python_lib',
    version="1.0",
    description='Internal Python Library.',
    author='Jean-François Reboud',
    license='MIT',
    install_requires=[
        "torch==1.10.1",
        "torchvision==0.11.2",
        "numpy==1.23.1",
        "pillow==9.2.0",
    ],
    packages=find_packages(exclude="tests"),
    python_requires='>=3.7'
)
