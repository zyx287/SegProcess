from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name='segprocess',
        version='0.2.1',
        packages=find_packages(),
        entru_points={
            # Define the console script for CLI
            'console_scripts': [
                'segprocess-view=segprocess.cli.Neuroglancer_viewer:main',
            ],
        },
    )
