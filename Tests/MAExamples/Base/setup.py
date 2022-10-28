from setuptools import setup, find_packages

setup(
    name='python_lib',
    version="1.0",
    description='Internal Python library.',
    author='Jean-François Reboud',
    license='MIT',
    install_requires=[
        "numpy==1.17.5",
        "opencv-python==4.1.2.30"
    ],
    packages=find_packages(exclude="tests"),
    python_requires='>=3.7'
)
