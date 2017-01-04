import setuptools
import distutils.core

setuptools.setup(
    name='npcli',
    version=0.1,
    author='Tal Wrii',
    author_email='talwrii@gmail.com',
    description='',
    license='GPLv3',
    keywords='',
    url='',
    packages=[],
    long_description=open('README.md').read(),
    entry_points={
        'console_scripts': ['npcli=npcli.npcli:main']
    },
    classifiers=[
    ],
    test_suite='nose.collector'
)
