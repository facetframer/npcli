import setuptools

from setuptools.command.test import test as TestCommand


class ToxTest(TestCommand):
    user_options = []

    def initialize_options(self):
        TestCommand.initialize_options(self)

    def run_tests(self):
        import tox
        tox.cmdline()
        tox.cmdline(['-c', 'tox-minimal.ini'])

setuptools.setup(
    name='npcli',
    version=0.1,
    author='Facet Framer',
    author_email='facetframer@gmail.com',
    description='',
    license='GPLv3',
    keywords='',
    url='',
    install_requires=['numpy', 'autopep8'],
    packages=['npcli'],
    entry_points={
        'console_scripts': ['npcli=npcli.npcli:main']
    },
    classifiers=[
    ],
    cmdclass = {'test': ToxTest},
)
