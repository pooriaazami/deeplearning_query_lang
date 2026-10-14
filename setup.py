from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dlang",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Description Language for Deep Learning Projects",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/DeepLearningLang",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        # Core dependencies (none required for the parser/codegen)
        # Backend-specific dependencies are available as extras:
        #   pip install .[pytorch]
        #   pip install .[tensorflow]
    ],
    extras_require={
        'pytorch': ['torch'],
        'tensorflow': ['tensorflow'],
    },
    entry_points={
        'console_scripts': [
            'dlang=dlang.cli.main:main',
        ],
    },
)