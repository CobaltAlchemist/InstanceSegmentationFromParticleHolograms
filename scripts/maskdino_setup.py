from setuptools import setup, find_packages

setup(
    name="maskdino",
    packages=find_packages(),
    install_requires=[
        "cython",
        "scipy",
        "shapely",
        "timm",
        "h5py",
        "submitit",
        "scikit-image",
        "opencv-python",
    ]
)