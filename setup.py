#!/usr/bin/env python3

from codecs import open
from os import path
import os
import re
import sys
import io
import platform
import subprocess

from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

current_dir = os.getcwd()

"""
	Script to .....
"""

def read(*names, **kwargs):
	with io.open(
		os.path.join(os.path.dirname(__file__), *names),
		encoding=kwargs.get("encoding", "utf8")
	) as fp:
		return fp.read()

def read1(self):
	"""
			Alternative to read short-reads
	"""
	if len(os.listdir("./path/data/illumina")) != 0: 
		path = "./path/data/illumina"
		files = os.listdir(path)
			
		if "R1" in files[0]:
			for i in range(1, len(files), 2):
				fastq_r1 = path + "/" + files[i-1]
				fastq_r2 = path + "/" + files[i]
								
def FolderVerificationRun(self):
	  files = os.listdir("./")
	  print(files)
	  if "data" not in files:
		  os.mkdir("./data")
	  files = os.listdir("./data")
	  folders_tools = ["illumina", "results"]
	  for f in folders_tools:
		  if f not in files:
			  path = "./data/"+f
			  os.mkdir(path)
	  folders_trimm = ["illumina", "nanopore", "pacbio"]
	  files_input = os.listdir("./data/input")
	  files_trimmed = os.listdir("./data/trimmed")
	  for f in folders_trimm:
		  if f not in files_input:
			  path = "./data/input/"+f
			  os.mkdir(path)
		  if f not in files_trimmed:
			  path = "./data/trimmed/"+f
			  os.mkdir(path)
	  files_results = os.listdir("./data/results")
	  folders_results = ["input", "trimmed", "results"]
	  for f in folders_results:
		  if f not in files_results:
			  path = "./data/results/"+f
			  os.mkdir(path)

def find_version(*file_paths):
	version_file = read(*file_paths)
	version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
							  version_file, re.M)
	if version_match:
		return version_match.group(1)
	raise RuntimeError("Unable to find version string.")


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
                      '-DCMAKE_VERBOSE_MAKEFILE:BOOL=ON']

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

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


here = path.abspath(path.dirname(__file__))


with open(path.join(here, 'README.md'), encoding='utf-8') as f:
	long_description = f.read()

setup(
	name='gpu_kmer_counter',
	version=find_version("gpu_kmer_counter/__init__.py"),
	description='k-mer counter using CUDA',
	long_description=long_description,
	long_description_content_type='text/markdown',
	url='https://github.com/AMR-Hackathon-2021/gpu_kmer_counter',
	author='John Lees, Sam Horsfield, Noah Legall, Louise Cerdeira',
	author_email='john@johnlees.me',
	license='Apache Software License',
	classifiers=[
		'Development Status :: 4 - Beta',
		'Intended Audience :: Science/Research',
		'Topic :: Scientific/Engineering :: Bio-Informatics',
		'License :: OSI Approved :: Apache Software License',
		'Programming Language :: Python :: 3.7',
	],
	python_requires='>=3.7.0',
	keywords='bacteria genomics population-genetics k-mer',
	packages=['gpu_kmer_counter'],
	entry_points={
		"console_scripts": [
			'gpu_kmer_counter = gpu_kmer_counter.__main__:main'
			]
	},
	test_suite="test",
	ext_modules=[CMakeExtension('cuda_kmers')],
	cmdclass=dict(build_ext=CMakeBuild),
	zip_safe=False
)
