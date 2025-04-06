#!/usr/bin/env python
"""
Generate requirements.txt file for the Advanced Time Series Forecasting Framework

This script scans the project files for imports and generates a requirements.txt
file with the appropriate version constraints.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

def get_imports(file_path):
    """
    Extract import statements from a Python file
    
    Parameters:
    -----------
    file_path : str
        Path to the Python file
        
    Returns:
    --------
    set
        Set of imported packages
    """
    imports = set()
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read()
        
        # Find import statements
        import_lines = re.findall(r'^import\s+(.*?)$', content, re.MULTILINE)
        from_import_lines = re.findall(r'^from\s+(.*?)\s+import', content, re.MULTILINE)
        
        # Process import statements
        for line in import_lines:
            # Handle multiple imports on one line
            modules = line.split(',')
            for module in modules:
                # Get the base package name
                base_module = module.strip().split('.')[0]
                if base_module and not base_module.startswith('_'):
                    imports.add(base_module)
        
        # Process from...import statements
        for line in from_import_lines:
            # Get the base package name
            base_module = line.strip().split('.')[0]
            if base_module and not base_module.startswith('_'):
                imports.add(base_module)
    
    return imports

def get_installed_version(package):
    """
    Get the installed version of a package without pkg_resources
    
    Parameters:
    -----------
    package : str
        Package name
        
    Returns:
    --------
    str
        Package version or None if not installed
    """
    try:
        # Use pip show to get package version
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'show', package],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            # Extract version from pip show output
            match = re.search(r'Version:\s*(\S+)', result.stdout)
            if match:
                return match.group(1)
        
        return None
    except Exception:
        return None

def scan_project_files():
    """
    Scan project files for import statements
    
    Returns:
    --------
    set
        Set of imported packages
    """
    all_imports = set()
    
    # File extensions to scan
    extensions = {'.py'}
    
    # Directories to skip
    skip_dirs = {'.git', '__pycache__', 'venv', 'env', '.venv', '.env', 'archive'}
    
    # Walk through the project directory
    for root, dirs, files in os.walk('.'):
        # Skip directories
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        
        # Process Python files
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                
                try:
                    file_imports = get_imports(file_path)
                    all_imports.update(file_imports)
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
    
    return all_imports

def filter_standard_library(packages):
    """
    Filter out standard library packages
    
    Parameters:
    -----------
    packages : set
        Set of package names
        
    Returns:
    --------
    set
        Filtered set of package names
    """
    # Common standard library modules to exclude
    std_lib = {
        'os', 'sys', 're', 'math', 'time', 'datetime', 'random', 'json', 
        'csv', 'argparse', 'collections', 'functools', 'itertools', 
        'pathlib', 'shutil', 'pickle', 'typing', 'io', 'traceback',
        'glob', 'tempfile', 'copy', 'importlib', 'inspect', 'abc',
        'ast', 'base64', 'configparser', 'contextlib', 'dataclasses',
        'enum', 'hashlib', 'logging', 'socket', 'threading', 'urllib',
        'uuid', 'warnings', 'zipfile'
    }
    
    # Filter packages
    return {pkg for pkg in packages if pkg not in std_lib}

def map_import_to_package(imports):
    """
    Map import names to actual package names
    
    Parameters:
    -----------
    imports : set
        Set of import names
        
    Returns:
    --------
    set
        Set of mapped package names
    """
    # Common mappings of import name to package name
    mappings = {
        'sklearn': 'scikit-learn',
        'PIL': 'Pillow',
        'bs4': 'beautifulsoup4',
        'cv2': 'opencv-python',
        'dotenv': 'python-dotenv',
        'yaml': 'pyyaml',
        'wandb': 'weights-and-biases',
        'plotly': 'plotly',
        'streamlit': 'streamlit',
        'altair': 'altair',
        'joblib': 'joblib',
        'statsmodels': 'statsmodels',
        'xgboost': 'xgboost',
        'lightgbm': 'lightgbm',
        'catboost': 'catboost',
        'tensorflow': 'tensorflow',
        'torch': 'torch',
        'keras': 'keras',
        'fbprophet': 'prophet',
        'prophet': 'prophet',
        'sktime': 'sktime',
        'pmdarima': 'pmdarima',
        'darts': 'darts',
        'imblearn': 'imbalanced-learn'
    }
    
    mapped_packages = set()
    for imp in imports:
        if imp in mappings:
            mapped_packages.add(mappings[imp])
        else:
            mapped_packages.add(imp)
    
    return mapped_packages

def check_pip_outdated():
    """
    Check for outdated packages in pip
    
    Returns:
    --------
    dict
        Mapping of package names to (current_version, latest_version)
    """
    outdated = {}
    
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'list', '--outdated', '--format=json'],
            capture_output=True,
            text=True,
            check=True
        )
        
        import json
        packages = json.loads(result.stdout)
        
        for package in packages:
            name = package.get('name')
            current_version = package.get('version')
            latest_version = package.get('latest_version')
            
            if name and current_version and latest_version:
                outdated[name.lower()] = (current_version, latest_version)
        
    except Exception as e:
        print(f"Warning: Could not check for outdated packages: {str(e)}")
    
    return outdated

def get_common_package_versions():
    """
    Return a dictionary of common packages with recommended versions
    
    Returns:
    --------
    dict
        Mapping of package names to recommended versions
    """
    return {
        'numpy': '1.20.0',
        'pandas': '1.3.0',
        'matplotlib': '3.4.0',
        'seaborn': '0.11.0',
        'scikit-learn': '1.0.0',
        'statsmodels': '0.13.0',
        'pmdarima': '1.8.0',
        'prophet': '1.0',
        'xgboost': '1.5.0',
        'tensorflow': '2.8.0',
        'streamlit': '1.10.0',
        'plotly': '5.3.0',
        'scipy': '1.7.0',
        'joblib': '1.1.0',
        'scikit-learn': '1.0.0',
        'tqdm': '4.62.0'
    }

def enhance_requirements(packages):
    """
    Add version constraints to packages
    
    Parameters:
    -----------
    packages : set
        Set of package names
        
    Returns:
    --------
    list
        List of package requirements with version constraints
    """
    requirements = []
    common_versions = get_common_package_versions()
    
    for package in sorted(packages):
        try:
            # Check if it's a common package with a recommended version
            if package.lower() in common_versions or package in common_versions:
                package_key = package.lower() if package.lower() in common_versions else package
                version = common_versions[package_key]
                requirements.append(f"{package}>={version}")
            else:
                # Try to get installed version
                version = get_installed_version(package)
                
                if version:
                    # Add with flexible version constraint
                    requirements.append(f"{package}>={version}")
                else:
                    # If not installed, add without version constraint
                    requirements.append(package)
        except Exception as e:
            print(f"Warning: Error processing package {package}: {str(e)}")
            requirements.append(package)
    
    return requirements

def add_essential_packages(packages):
    """
    Add essential packages that might not be detected from imports
    
    Parameters:
    -----------
    packages : set
        Set of package names
        
    Returns:
    --------
    set
        Updated set of package names
    """
    essential = {
        'numpy', 'pandas', 'scikit-learn', 'matplotlib', 'seaborn',
        'streamlit', 'plotly', 'statsmodels', 'prophet', 'xgboost', 
        'tensorflow', 'pmdarima', 'jupyter', 'nbconvert', 'scipy',
        'tqdm', 'joblib'
    }
    
    return packages.union(essential)

def generate_requirements():
    """
    Generate requirements.txt file
    """
    print("Scanning project files for imports...")
    imports = scan_project_files()
    print(f"Found {len(imports)} imports")
    
    # Filter standard library modules
    packages = filter_standard_library(imports)
    print(f"After filtering standard library: {len(packages)} packages")
    
    # Map import names to package names
    packages = map_import_to_package(packages)
    print(f"After mapping to package names: {len(packages)} packages")
    
    # Add essential packages
    packages = add_essential_packages(packages)
    print(f"After adding essential packages: {len(packages)} packages")
    
    # Enhance requirements with version constraints
    requirements = enhance_requirements(packages)
    
    # Write requirements.txt
    with open('requirements.txt', 'w', encoding='utf-8') as f:
        for req in requirements:
            f.write(f"{req}\n")
    
    print(f"\nGenerated requirements.txt with {len(requirements)} packages")
    print("Done!")

if __name__ == "__main__":
    generate_requirements() 