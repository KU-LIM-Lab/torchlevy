from setuptools import setup, find_packages

setup(
    name="torchlevy",
    version="0.0.0.5",
    description="Package provides torch-based pdf, score calculation and sampling of levy stable distribution.",
    long_description_content_type="text/markdown",
    url="https://github.com/UNIST-LIM-Lab/levy-distribution-pytorch",
    author="Jinhyeok Kim",
    author_email="jinhyuk@unist.ac.rk",
    install_requires=[
        "torchquad @ git+https://github.com/jk4011/torchquad.git@multiple_integrands",
        "Cython",
        "torch",
    ],
    packages=["torchlevy"],
    python_requires=">=3.7, <4",
    project_urls={
        "Source": "https://github.com/UNIST-LIM-Lab/levy-distribution-pytorch",
    },
)