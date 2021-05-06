import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="autocompile",
    version="0.2.0",
    author="Sam Purkis",
    author_email="sam.purkis@hotmail.co.uk",
    description="Speed up Python code that has well layed out type hints (works by converting the function to typed cython). Find more info at https://github.com/smpurkis/autocompile",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/smpurkis/autocompile",
    project_urls={
        "Bug Tracker": "https://github.com/smpurkis/autocompile/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=[
        'pytest',
        'cython',
        'numpy',
        'numba'
    ],
)