import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="AutoDiff_group3",
    version="0.0.6",
    author="Will Claybaugh, Fan (Bruce) Xiong, Erin Williams",
    author_email="erinwilliams@g.harvard.edu",
    description="Automatic differentiation with dual numbers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cs207-project-erin-bruce-will/cs207-FinalProject",
    packages=setuptools.find_packages(),
    install_requires=['numpy','pytest'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
