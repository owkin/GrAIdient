from setuptools import setup, find_packages

setup(
    name='python_lib',
    version="1.0",
    description='Internal Python Library.',
    author='Jean-François Reboud',
    license='MIT',
    install_requires=[
        "numpy==1.23.1",
        "opencv-python==4.6.0.66"
    ],
    packages=find_packages(exclude="tests"),
    python_requires='>=3.7'
)
