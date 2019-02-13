import setuptools

setuptools.setup(name="nn-common-modules",
                 version="1.0",
                 url="https://github.com/abhi4ssj/nn-common-modules",
                 author="Shayan Ahmad Siddiqui",
                 author_email="shayan.siddiqui89@gmail.com",
                 description="Common modules, blocks and losses which can be reused in a deep neural netwok specifically for segmentation",
                 packages=setuptools.find_packages(),
                 install_requires=['numpy==1.14.3','torch==1.0.1.post2', 'squeeze-and-excitation'],
                 python_requires='>=3.5')