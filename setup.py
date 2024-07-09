from setuptools import setup

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

setup(
    name='pycvvdp',
    version='0.1.0',
    description='PyTorch code for \'ColorVideoVDP\': a full-reference' \
                'visual quality metric that predicts the perceptual' \
                'difference between pairs of images or videos.',
    long_description=long_description,
    url='https://github.com/mantiuk/ColourVideoVDP',
    long_description_content_type='text/markdown',
    author='Rafał K. Mantiuk',
    author_email='mantiuk@gmail.com',
    license='Creative Commons Attribution-NonCommercial 4.0 International Public License',
    packages=['pycvvdp', 'pycvvdp/third_party'],
    package_data={'pycvvdp': ['csf_cache/*.mat', 'vvdp_data/*.json']},
    include_package_data=True,
    install_requires=['numpy>=1.17.3',
                      'scipy>=1.10.0',
                      'ffmpeg-python>=0.2.0',
                      'torch>=1.13.1',
                      'torchvision>=0.9.2',
                      'ffmpeg>=1.4',
                      'imageio>=2.19.5',
                      'matplotlib>=3.8.0'
                     ],

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],

    entry_points={
        'console_scripts': [
            'cvvdp=pycvvdp.run_cvvdp:main'
        ]
    }
)
