"""Check the installed SDK imports without accessing or opening cameras."""
import importlib
import importlib.metadata
import platform
import sys


def main():
    print(f'Python {platform.python_version()} | {platform.system()} {platform.machine()}')
    try:
        version = importlib.metadata.version('dvsense-driver')
    except importlib.metadata.PackageNotFoundError:
        version = 'unknown (package metadata unavailable)'
    print(f'dvsense-driver: {version}')
    try:
        for name in ('dvsense_driver.camera_manager', 'dvsense_driver.raw_file_reader',
                     'dvsense_driver.hal.camera_tool'):
            importlib.import_module(name)
            print(f'OK {name}')
    except (ImportError, OSError) as exc:
        print(f'SDK import failed: {exc}', file=sys.stderr)
        return 1
    print('Import check only; camera access and synchronization were not tested.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
