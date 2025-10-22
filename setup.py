from setuptools import find_packages, setup
from setuptools.command.develop import develop
from glob import glob
import os
import sys
import shutil

package_name = 'rosetta'

# Remove unsupported flags from sys.argv
unsupported_flags = ['--editable', '--build-directory']
i = 0
while i < len(sys.argv):
    if sys.argv[i] in unsupported_flags:
        sys.argv.pop(i)
        # Also remove the value if it's the next argument and looks like a path
        if i < len(sys.argv) and (sys.argv[i].startswith('/') or sys.argv[i].startswith('build')):
            sys.argv.pop(i)
    else:
        i += 1

class CustomDevelop(develop):
    """Custom develop command that handles --editable flag and installs executables correctly"""
    def run(self):
        super().run()
        self._create_executables()
    
    def _create_executables(self):
        """Create executables in lib/<package_name>/ directory"""
        # Find the install directory
        install_base = None
        possible_bases = [
            '/workspaces/reo_ws/install',
            os.path.join(os.getcwd(), 'install'),
            os.path.join(os.path.dirname(os.getcwd()), 'install')
        ]
        
        for base in possible_bases:
            if os.path.exists(base):
                install_base = base
                break
        
        if not install_base:
            return
            
        # Create lib/<package_name>/ directory
        lib_dir = os.path.join(install_base, 'lib', package_name)
        os.makedirs(lib_dir, exist_ok=True)
        
        # Create executable scripts
        executables = {
            'policy_bridge': ('rosetta.policy_bridge_node', 'main'),
            'recorder_server': ('rosetta.episode_recorder', 'main')
        }
        
        for exe_name, (module_name, function_name) in executables.items():
            exe_path = os.path.join(lib_dir, exe_name)
            with open(exe_path, 'w') as f:
                f.write('#!/usr/bin/env python3\n')
                f.write('import sys\n')
                f.write(f'from {module_name} import {function_name}\n')
                f.write('\n')
                f.write('if __name__ == \'__main__\':\n')
                f.write(f'    {function_name}()\n')
            os.chmod(exe_path, 0o755)

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'contracts'), glob('contracts/*.yaml')),
    ],
    install_requires=['setuptools'],
    extras_require={
        'ml': ['torch', 'torchvision', 'lerobot>=0.7'],
    },
    zip_safe=True,
    maintainer='ros',
    maintainer_email='isaac@dirtrobotics.com',
    description='TODO: Package description',
    license='Apache-2.0',
    cmdclass={
        'develop': CustomDevelop,
    },
)
