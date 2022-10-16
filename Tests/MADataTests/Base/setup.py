from setuptools import setup, find_packages

setup(
    name='data',
    version="1.0",
    description='Data',
    author='Jean-François Reboud',
    license='MIT',
    install_requires=[
        "torch==1.1.0",
        "torchvision==0.3.0",
        "numpy==1.17.5",
        "opencv-python==4.1.2.30"
    ],
    packages=find_packages(exclude="tests"),
    python_requires='>=3.7'
)
