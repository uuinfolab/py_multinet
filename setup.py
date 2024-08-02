import os
import re
import sys
import platform
import sysconfig
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
            if cmake_version < '3.20':
                raise RuntimeError("CMake >= 3.20 is required")

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
        
        if platform.system() == "Windows":
            print('Windows!')
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            cmake_args += ['-G "MinGW Makefiles"']
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
            print(cmake_args)
        elif sys.platform == 'darwin':
            #macosx_target_ver = sysconfig.get_config_var('MACOSX_DEPLOYMENT_TARGET')
            #if macosx_target_ver and 'MACOSX_DEPLOYMENT_TARGET' not in os.environ:
            #    print(f'-DCMAKE_OSX_DEPLOYMENT_TARGET={macosx_target_ver}')
            #cmake_args.append(f'-DCMAKE_OSX_DEPLOYMENT_TARGET={macosx_target_ver}')
            cmake_args.append(f'-DCMAKE_OSX_DEPLOYMENT_TARGET=10.13')

            osx_arch = platform.machine()
            cmake_args.append(f'-DCMAKE_OSX_ARCHITECTURES={osx_arch}')
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

print('Running setup')

try:
    setup(
    ext_modules=[CMakeExtension('uunet._multinet')],
    cmdclass=dict(build_ext=CMakeBuild),
    packages=['uunet','uunet.data']
    )
except CalledProcessError:
    print('Failed to build extension')


