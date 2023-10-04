from distutils.core import setup

setup(
    name='sashimi',
    version='1.0.0',
    packages=['sashimi'],
    url='',
    license='',
    author='Ross Marchant',
    author_email='ross.g.marchant@gmail.com',
    description='3D printer stacking software',
    install_requires=[
        'scikit-image',
        'scipy',
        'pypylon',
        'opencv-python',
        'pyserial',
        'pillow'
    ]
)
