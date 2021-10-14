import setuptools


with open("README.md") as red:
    ldest = red.read()

setuptools.setup(
    name='Oemer',
    version='0.2.1',
    author='Meteo Corp.',
    author_email='contact@meteo.com.tw',
    description='End-to-end Optical Music Recoginition (OMR) system.',
    long_description=ldest,
    url='https://github.com/meteo-team/oemer',
    packages=setuptools.find_packages(),
    package_data={
    '': [
            'sklearn_models/*.model',
            'checkpoints/unet_big/*',
            'checkpoints/seg_net/*',
        ]
    },
    install_requires=[
        'onnxruntime-gpu',
        'opencv-python==4.5.3.56',
        'matplotlib',
        'pillow',
        'numpy==1.21.2',
        'scipy==1.6.2',
        'scikit-learn==0.24.2'
    ],
    extras_require={
        'full': ['tensorflow-gpu', 'tf2onnx']
    },
    entry_points={'console_scripts': ['oemer = oemer.ete:main']}
)
