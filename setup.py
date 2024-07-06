from setuptools import setup, find_packages

setup(
    name='tracing_ml',
    version='1.0.0',
    description='Python library',
    author='Hayden Rear',
    author_email='hayden.rear@gmail.com',
    packages=find_packages('src'),  # Set the 'src' directory as the base for package discovery
    package_dir={'': 'src'},  # Map the root package to the 'src' directory
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
