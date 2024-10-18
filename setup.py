from setuptools import setup, find_packages

setup(
    name='Data_Webcam_Synchronizer',
    version='1.0.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=['csv','datetime','matplotlib','moviepy','numpy','pandas','scikit-learn','pytz',
                     'pywebcoos @ GitHubLink.git']
)