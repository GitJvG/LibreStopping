from setuptools import setup, find_packages

setup(
    name='LibreStopping',
    version='0.1.0',
    author='Jurrian van Geresteijn',
    author_email='Jurrian6@gmail.com',
    description='A library for implementing early stopping functionality in LibreCommender models',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/GitJvG/LibreStopping',  # Replace with your GitHub repo
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    install_requires=[
        # List your dependencies here, e.g., 'libreco'
    ],
)
