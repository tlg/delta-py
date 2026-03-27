#!/usr/bin/env python

from setuptools import setup

setup(
    name='elium-delta',
    version='2.0',
    description='Python port of elium-delta: rich text OT with block move operations',
    packages=['delta'],
    license='MIT License',
    install_requires=['diff-match-patch'],
    tests_require=['pytest'],
    project_urls={
        'Bug Reports': 'https://github.com/whatever-company/elium-delta-py/issues',
        'Source': 'https://github.com/whatever-company/elium-delta-py',
    },
    classifiers=[
        'Development Status :: 5 - Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Text Processing :: Markup',
    ],
)
