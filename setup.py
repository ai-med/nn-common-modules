import setuptools

setuptools.setup(name="nn-common-modules",
                 version="1.0",
                 url="https://github.com/abhi4ssj/nn-common-modules",
                 author="Shayan Ahmad Siddiqui",
                 author_email="shayan.siddiqui89@gmail.com",
                 description="Common modules, blocks and losses which can be reused in a deep neural netwok specifically for segmentation",
                 packages=setuptools.find_packages(),
                 install_requires=['pip>=19.0.0','numpy>=1.14.0','torch>=1.0.0','squeeze_and_excitation @ https://github.com/ai-med/squeeze_and_excitation/releases/download/v1.0/squeeze_and_excitation-1.0-py2.py3-none-any.whl'],
                 python_requires='>=3.5')
