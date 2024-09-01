from setuptools import setup, find_packages

setup(
    name='python_lib',
    version="1.0",
    description='Internal Python Library.',
    author='Jean-François Reboud',
    license='MIT',
    install_requires=[
        "torch==1.13.1",
        "torchvision==0.14.1",
        "numpy==1.23.1",
        "opencv-python==4.6.0.66",
        "safetensors==0.4.3",
        "mistral-common==1.2.1",
        "sentencepiece==0.1.99",
        "tiktoken==0.4.0",
        "blobfile==2.1.1"
    ],
    packages=find_packages(exclude="tests"),
    python_requires='>=3.7'
)
