import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="facet",
    version="0.0.1",
    author="Daniel Stanley",
    author_email="dstanley@Stanford.edu",
    description="Tools for creating digital models of analog things",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/standanley/facet",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        #"License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
