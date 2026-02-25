from setuptools.command.build_ext import build_ext
from setuptools import setup, Extension, Command
import platform
import shutil
import os,re
import sys
import subprocess
import stat
import glob

# Ensure pybind11 is installed
try:
    import pybind11
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pybind11'])
    import pybind11

def generate_module_name_clean(filename: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_]', '_', filename)

def strip_python_abi_suffix(name: str) -> str:
    name = re.sub(r'\.cpython-\d+[a-zA-Z0-9_-]*$', '', name)
    name = re.sub(r'\.cp\d+[a-zA-Z0-9_-]*$', '', name)
    return name

def rename_extensions_to_so(build_lib: str, final_dir: str) -> None:
    build_lib = os.path.abspath(build_lib)
    final_dir = os.path.abspath(final_dir)
    os.makedirs(final_dir, exist_ok=True)

    patterns = [
        os.path.join(build_lib, "**", "*.pyd"),  # Windows
        os.path.join(build_lib, "**", "*.so"),   # Linux/macOS
    ]

    files = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))

    if not files:
        print(f"No extension modules found in {build_lib} (searched recursively).")
        return

    for src in files:
        base = os.path.basename(src)
        stem, _ = os.path.splitext(base)

        stem = strip_python_abi_suffix(stem)
        module_name = generate_module_name_clean(stem)

        dst = os.path.join(final_dir, module_name + ".so")

        if os.path.exists(dst):
            os.remove(dst)

        shutil.move(src, dst)
        print(f"Renamed {src} -> {dst}")

class BuildExtAndRename(build_ext):
    """
    Build extensions then move/rename them into a target directory as *.so
    """
    user_options = build_ext.user_options + [
        ('rename-output-dir=', None, 'Directory to move renamed *.so files into (default: build_lib)'),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.rename_output_dir = None

    def run(self):
        super().run()

        build_lib = os.path.abspath(self.build_lib)

        # By default, keep them in build_lib (you can set to project root if you want)
        out_dir = os.path.abspath(self.rename_output_dir) if self.rename_output_dir else build_lib

        rename_extensions_to_so(build_lib, out_dir)

# Get pybind11 include directory
pybind11_include_dir = pybind11.get_include()

def find_yaml_cpp():
    """Find yaml-cpp on the system or install it"""
    system = platform.system()
    
    # Try to find system yaml-cpp include
    common_include_paths = [
        '/usr/include',
        '/usr/local/include',
        '/opt/local/include',  # macOS Macports
        'C:\\vcpkg\\installed\\x64-windows\\include',
        'C:\\Program Files\\yaml-cpp\\include'
    ]
    
    yaml_cpp_include = None
    yaml_cpp_lib = None
    
    # Check for existing include directory
    for path in common_include_paths:
        candidate = os.path.join(path, 'yaml-cpp', 'yaml.h')
        if os.path.exists(candidate):
            yaml_cpp_include = path
            # Find corresponding lib directory
            if system in ['Linux', 'Darwin']:
                yaml_cpp_lib = path.replace('include', 'lib')
                if not os.path.exists(yaml_cpp_lib):
                    yaml_cpp_lib = path.replace('include', 'lib64')
            else:  # Windows
                yaml_cpp_lib = path.replace('include', 'lib')
            print(f"Found system yaml-cpp at {yaml_cpp_include}")
            return yaml_cpp_include, yaml_cpp_lib
    
    # If not found, try to install it
    if system == 'Linux':
        print("yaml-cpp not found. Attempting to install libyaml-cpp-dev via apt-get...")
        try:
            subprocess.check_call(['sudo', 'apt-get', 'update'])
            subprocess.check_call(['sudo', 'apt-get', 'install', '-y', 'libyaml-cpp-dev'])
            return '/usr/include', '/usr/lib'
        except Exception as e:
            print(f"apt-get installation failed: {e}")
            print("Trying to build yaml-cpp from source...")
    
    elif system == 'Darwin':
        print("yaml-cpp not found. Attempting to install via Homebrew...")
        try:
            subprocess.check_call(['brew', 'install', 'yaml-cpp'])
            return '/usr/local/include', '/usr/local/lib'
        except Exception as e:
            print(f"Homebrew installation failed: {e}")
    
    # Fallback: build yaml-cpp from source (or download prebuilt on Windows)
    print("\nAttempting to build/install yaml-cpp...")
    return build_yaml_cpp_from_source()

def download_prebuilt_yaml_cpp():
    """Download pre-built yaml-cpp binaries for Windows"""
    import urllib.request
    import zipfile
    
    yaml_cpp_dir = os.path.join(os.getcwd(), 'yaml-cpp-prebuilt')
    yaml_cpp_include = os.path.join(yaml_cpp_dir, 'include')
    yaml_cpp_lib = os.path.join(yaml_cpp_dir, 'lib')
    
    # Check if already downloaded
    if os.path.exists(yaml_cpp_include) and os.path.exists(yaml_cpp_lib):
        print(f"Using pre-built yaml-cpp from {yaml_cpp_dir}")
        return yaml_cpp_include, yaml_cpp_lib
    
    # For Windows, try to use vcpkg if available
    if platform.system() == 'Windows':
        vcpkg_paths = [
            'C:\\vcpkg',
            os.path.expandvars('%VCPKG_ROOT%'),
        ]
        for vcpkg_root in vcpkg_paths:
            if os.path.exists(vcpkg_root):
                print(f"Found vcpkg at {vcpkg_root}")
                try:
                    vcpkg_exe = os.path.join(vcpkg_root, 'vcpkg.exe')
                    print("Installing yaml-cpp via vcpkg...")
                    subprocess.check_call([vcpkg_exe, 'install', 'yaml-cpp:x64-windows'])
                    
                    yaml_cpp_include = os.path.join(vcpkg_root, 'installed', 'x64-windows', 'include')
                    yaml_cpp_lib = os.path.join(vcpkg_root, 'installed', 'x64-windows', 'lib')
                    
                    if os.path.exists(yaml_cpp_include):
                        print(f"Successfully installed yaml-cpp via vcpkg")
                        return yaml_cpp_include, yaml_cpp_lib
                except Exception as e:
                    print(f"vcpkg installation failed: {e}")
    
    # Download pre-built binaries from GitHub releases
    print("\nDownloading pre-built yaml-cpp...")
    os.makedirs(yaml_cpp_dir, exist_ok=True)
    
    # This is a fallback - you can host pre-built binaries or use a release
    # For now, provide clear instructions
    raise Exception(
        "\n" + "="*70 +
        "\nPrerequisites installation required:\n"
        "="*70 +
        "\nPlease install CMake using one of these methods:\n\n"
        "Option 1 - Chocolatey (recommended if installed):\n"
        "  choco install cmake\n\n"
        "Option 2 - Visual Studio Installer:\n"
        "  If you have Visual Studio, CMake should be available.\n"
        "  Reinstall Visual Studio Build Tools and select 'C++ CMake tools for Windows'\n\n"
        "Option 3 - Manual Download:\n"
        "  1. Download from: https://cmake.org/download/\n"
        "  2. Run installer and ADD CMAKE TO PATH\n"
        "  3. Restart PowerShell and verify:\n"
        "     cmake --version\n\n"
        "Option 4 - Use vcpkg:\n"
        "  1. Clone vcpkg: git clone https://github.com/Microsoft/vcpkg\n"
        "  2. Run: .\\vcpkg\\bootstrap-vcpkg.bat\n"
        "  3. Set VCPKG_ROOT environment variable\n\n"
        "After installing, run this command again:\n"
        "  python setup.py build_ext --inplace\n" +
        "="*70
    )

def find_cmake():
    """Find or install CMake executable on system"""
    try:
        subprocess.run(['cmake', '--version'], capture_output=True, check=True)
        return 'cmake'
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    
    # Try common Windows paths
    if platform.system() == 'Windows':
        common_paths = [
            'C:\\Program Files\\CMake\\bin\\cmake.exe',
            'C:\\Program Files (x86)\\CMake\\bin\\cmake.exe',
            'C:\\tools\\cmake\\bin\\cmake.exe',
            'C:\\Program Files\\Microsoft Visual Studio\\2022\\BuildTools\\Common7\\IDE\\CommonExtensions\\Microsoft\\CMake\\CMake\\bin\\cmake.exe',
            'C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\BuildTools\\Common7\\IDE\\CommonExtensions\\Microsoft\\CMake\\CMake\\bin\\cmake.exe',
        ]
        for path in common_paths:
            if os.path.exists(path):
                print(f"Found CMake at: {path}")
                return path
    
    return None

def build_yaml_cpp_from_source():
    """Download and build yaml-cpp from source"""
    cmake_path = find_cmake()
    if not cmake_path:
        print("\n" + "="*70)
        print("CMake not found on system. Attempting alternative approach...")
        print("="*70)
        
        # Try to download pre-built binaries
        return download_prebuilt_yaml_cpp()
    
    yaml_cpp_dir = os.path.join(os.getcwd(), 'yaml-cpp-src')
    yaml_cpp_build = os.path.join(yaml_cpp_dir, 'build')
    yaml_cpp_install = os.path.join(yaml_cpp_dir, 'install')
    
    # Only build if not already built
    if os.path.exists(yaml_cpp_install):
        print(f"yaml-cpp already built at {yaml_cpp_install}")
        yaml_cpp_include = os.path.join(yaml_cpp_install, 'include')
        yaml_cpp_lib = os.path.join(yaml_cpp_install, 'lib')
        return yaml_cpp_include, yaml_cpp_lib
    
    # Clone yaml-cpp repo if not present (use 0.7.0 for MSVC compatibility)
    if not os.path.exists(yaml_cpp_dir):
        print("Cloning yaml-cpp repository (v0.7.0 - MSVC compatible)...")
        subprocess.check_call(['git', 'clone', 'https://github.com/jbeder/yaml-cpp.git', yaml_cpp_dir])
        # Checkout the stable 0.7.0 tag
        # Try both tag styles (some repos use "yaml-cpp-0.7.0", some use "0.7.0")
        try:
            subprocess.check_call(['git', '-C', yaml_cpp_dir, 'checkout', 'yaml-cpp-0.7.0'])
        except subprocess.CalledProcessError:
            subprocess.check_call(['git', '-C', yaml_cpp_dir, 'checkout', '0.7.0'])
            
    # Create build directory
    os.makedirs(yaml_cpp_build, exist_ok=True)
    
    # Build yaml-cpp with CMake
    print("Configuring yaml-cpp with CMake...")
    cmake_cmd = [
    cmake_path,
    f'-DCMAKE_INSTALL_PREFIX={yaml_cpp_install}',
    '-DYAML_CPP_BUILD_TESTS=OFF',
    '-DYAML_CPP_BUILD_TOOLS=OFF',
    '-DBUILD_SHARED_LIBS=OFF',
    '-DCMAKE_CXX_STANDARD=17',
    '-DCMAKE_CXX_STANDARD_REQUIRED=ON',
    '..'
    ]
    
    subprocess.check_call(cmake_cmd, cwd=yaml_cpp_build)
    
    # Build and install
    print("Building yaml-cpp...")
    if platform.system() == 'Windows':
        subprocess.check_call([cmake_path, '--build', '.', '--config', 'Release'], cwd=yaml_cpp_build)
        subprocess.check_call([cmake_path, '--install', '.', '--config', 'Release'], cwd=yaml_cpp_build)
    else:
        subprocess.check_call(['make', '-j', '4'], cwd=yaml_cpp_build)
        subprocess.check_call(['make', 'install'], cwd=yaml_cpp_build)
    
    yaml_cpp_include = os.path.join(yaml_cpp_install, 'include')
    yaml_cpp_lib = os.path.join(yaml_cpp_install, 'lib')
    print(f"yaml-cpp built successfully at {yaml_cpp_install}")
    return yaml_cpp_include, yaml_cpp_lib

# Find or build yaml-cpp
try:
    yaml_cpp_include, yaml_cpp_lib = find_yaml_cpp()
except Exception as e:
    print(f"Error setting up yaml-cpp: {e}")
    print("Using fallback system paths...")
    if platform.system() == 'Windows':
        yaml_cpp_include = None
        yaml_cpp_lib = None
    else:
        yaml_cpp_include = '/usr/include'
        yaml_cpp_lib = '/usr/lib'

# Specify additional include directories
include_dirs = [pybind11_include_dir, yaml_cpp_include]
home_path = os.getcwd()
# List of C++ files and header files
cpp_files = ['parallel_yaml_writer.cpp', 'shuffle.cpp','yamlwriter.cpp']
header_files = ['parallel_yaml_writer.h']  # Add your header files here

# Combine both C++ and header files for permissions adjustment
all_files = cpp_files + header_files

# Adjust file permissions
for file in all_files:
    if platform.system() in ['Linux', 'Darwin']:  # macOS and Ubuntu (Linux)
        os.chmod(file, os.stat(file).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    elif platform.system() == 'Windows':
        # On Windows, we generally don't need to change file permissions
        pass

# Function to generate module names and output filenames from cpp file names
def generate_module_name(filename):
    base_name = os.path.splitext(filename)[0]  # Remove .cpp extension
    return re.sub(r'[^a-zA-Z0-9_]', '_', base_name)  # Replace non-alphanumeric characters with underscores

# Function to generate the output filename for .so files
def generate_so_filename(filename):
    module_name = generate_module_name(filename)
    if platform.system() == 'Windows':
        return f"{module_name}.pyd"  # Windows uses .pyd for Python extensions
    else:
        return f"lib{module_name}.so"  # Linux/macOS uses .so for shared libraries

# Define the extensions
extensions = [
    Extension(
        name=generate_module_name(cpp_file),
        sources=[cpp_file],
        include_dirs=include_dirs,  # Include directories for headers
        library_dirs=[yaml_cpp_lib] if yaml_cpp_lib else [],
        libraries=['yaml-cpp'],
        extra_compile_args=['-std=c++11', '-fPIC'] if platform.system() != 'Windows' else ['/std:c++17'],
        extra_link_args=[] if platform.system() != 'Windows' else []
    )
    for cpp_file in cpp_files
]

# Requirements that need to be installed
#requirements = [
#    'pybind11>=2.2',
#    'numpy>=1.18.0',
    # Add other Python package dependencies here
#]

## OR

# Read requirements from requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()
# Setup configuration


setup(
    name='cpp_extension_module',
    version='1.0',
    description='A package with C++ extensions',
    ext_modules=extensions,
    install_requires=requirements,  # Directly specified requirements
    setup_requires=['setuptools', 'pybind11>=2.2'],
    cmdclass={
    'build_ext': BuildExtAndRename,
    }

)

