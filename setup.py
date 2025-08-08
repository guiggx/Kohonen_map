from setuptools import setup, find_packages

setup(
    name="som-learn",
    version="0.1.0",
    author="Hubert",
    description="A simple implementation of a Self-Organizing Map (SOM)",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/user/som-learn",  # Replace with actual URL if available
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "scikit-learn"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
