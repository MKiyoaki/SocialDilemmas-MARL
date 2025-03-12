from setuptools import setup, find_packages

with open("requirements.txt") as f:
    reqs = f.read().splitlines()

setup(
    name="sdmarl",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=reqs + [
        "dm-meltingpot",
        "shimmy",
        "gymnasium",
        "pettingzoo"
    ],
    entry_points={
        "console_scripts": [
            "run_project=src.main:my_main"
        ]
    }
)
