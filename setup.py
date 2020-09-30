import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pysmore-rainboltz", # Replace with your own username
    version="0.0.1",
    author="RainBoltz",
    author_email="philip27020012@gmail.com",
    description="sMORE in Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rainboltz/pysmore",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)