import setuptools

setuptools.setup(name="nn-common-modules",
                 version="1.0",
                 url="https://github.com/abhi4ssj/nn-common-modules",
                 author="Shayan Ahmad Siddiqui",
                 author_email="shayan.siddiqui89@gmail.com",
                 description="Contains common modules which can be reused in a deep neural netwok specifically for segmentation",
                 packages=setuptools.find_packages(),
                 install_requires=['torch', 'squeeze-and-excitation'])