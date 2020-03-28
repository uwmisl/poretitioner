import setuptools

LONG_DESCRIPTION = """"""

CLASSIFIERS = [
    "Development Status :: 2 - Pre-Alpha",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: MacOS",
    "Operating System :: POSIX",
    "Operating System :: Unix",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Topic :: Scientific/Engineering",
    "Topic :: Software Development",
]


MAJOR = 0
MINOR = 0
MICRO = 1
ISRELEASED = False
VERSION = "{MAJOR}.{MINOR}.{MICRO}".format(MAJOR=MAJOR, MINOR=MINOR, MICRO=MICRO)

HOME_PAGE = "https://github.com/uwmisl"

setuptools.setup(
    long_description_content_type="text/markdown",
    url=HOME_PAGE,
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    classifiers=CLASSIFIERS,
    test_suite="tests",
    long_description=LONG_DESCRIPTION,
    entry_points={"console_scripts": ["poretitioner=poretitioner.poretitioner:main"]},
    # version=VERSION,
)
