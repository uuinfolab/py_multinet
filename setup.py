import os
import re
import sys
import platform
import subprocess
import setuptools

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable,
                      '-DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=True']

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        # Pile all .so in one place and use $ORIGIN as RPATH // by Sergei Izmailov
        cmake_args += ["-DCMAKE_BUILD_WITH_INSTALL_RPATH=TRUE"]
        #cmake_args += ['-DCMAKE_INSTALL_RPATH=' + extdir]
        #cmake_args += ["-DCMAKE_BUILD_WITH_INSTALL_RPATH=TRUE"]
        #cmake_args += ["-DCMAKE_INSTALL_RPATH_USE_LINK_PATH=TRUE"]
        #cmake_args += ["-DCMAKE_BUILD_RPATH_USE_ORIGIN=TRUE"]
        #cmake_args += ["-DCMAKE_INSTALL_RPATH={}".format("$ORIGIN")]
        #cmake_args += ["-DCMAKE_BUILD_RPATH=miao"]
        #cmake_args += ["-DCMAKE_INSTALL_RPATH=bau"]
        #cmake_args += ["-DMACOSX_RPATH=OFF"]
        
        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

from subprocess import CalledProcessError

with open("DESCRIPTION", "r") as fh:
    long_description = fh.read()

try:
    setup(
    name='uunet',
    version='1.1.4',
    author='Matteo Magnani',
    author_email='matteo.magnani@it.uu.se',
    description='python porting of the R multinet library',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://bitbucket.org/uuinfolab/py_multinet",
    ext_modules=[CMakeExtension('uunet._multinet')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: European Union Public Licence 1.2 (EUPL 1.2)",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'networkx',
        'matplotlib',
    ],
    python_requires='>=3.8',
    )
except CalledProcessError:
    print('Failed to build extension!')
    #del kwargs['ext_modules']
    #setup(**kwargs)


