from setuptools import setup, find_packages

setup(
    name='python_lib',
    version="1.0",
    description='Internal Python library.',
    author='Jean-François Reboud',
    license='MIT',
    install_requires=[
        "torch==1.1.0",
        "torchvision==0.3.0",
        "numpy==1.17.5",
        "pillow==6.2.2",
    ],
    packages=find_packages(exclude="tests"),
    python_requires='>=3.7'
)
