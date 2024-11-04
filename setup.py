from setuptools import setup, find_packages

setup(
    name='nwlon_webcoos_synchronizer',
    version='1.0.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'matplotlib',
        'moviepy',
        'numpy',
        'pandas',
        'scikit-learn',
        'pytz',
    ],
)
