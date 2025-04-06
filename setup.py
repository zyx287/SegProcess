"""
Setup script for segprocess package.
"""

from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name='segprocess',
        version='0.3.0',
        description='Process dense labeled oversegmentation data',
        author='zyx',
        author_email='yuxiang.jeffrey.zhang@gmail.com',
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        entry_points={
            'console_scripts': [
                'segprocess=segprocess.cli.main:main',
                'segprocess-view=segprocess.cli.neuroglancer_viewer:main',
            ],
        },
        python_requires='>=3.7',
    )