"""Extract bundled dataset archives (traffic.zip, solar_AL.zip, ...) in place.

Usage: python data/prepare_data.py
"""
import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    for root, _, files in os.walk(HERE):
        for name in files:
            if not name.endswith('.zip'):
                continue
            path = os.path.join(root, name)
            with zipfile.ZipFile(path) as z:
                z.extractall(root)
            print('extracted', path)


if __name__ == '__main__':
    main()
