from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_desc = f.read()

setup(
    name="sgraph",
    version="0.0.1",
    author='green_hand',
    author_email='green_hand@126.com',
    description="Simple Graph",
#    packages(exclude=("test")),
    zip_safe=False,
    entry_points={"console_scripts":["sgraph-run=sgraph.__main__:main"]},
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)
