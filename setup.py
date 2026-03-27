#!/usr/bin/env python

from setuptools import setup

setup(
    name='quill-delta',
    version='1.1',
    description='Python port of the quill-delta library for rich text OT',
    packages=['delta'],
    license='MIT License',
    install_requires=['diff-match-patch'],
    tests_require=['pytest'],
    classifiers=[
        'Development Status :: 5 - Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Text Processing :: Markup',
    ],
)
