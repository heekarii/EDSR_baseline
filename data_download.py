import wget
import zipfile
import shutil
from pathlib import Path
import numpy as np


def download_div2k(data_dir='./dataset'):
    """DIV2K 데이터셋 다운로드 및 설정"""
    data_dir = Path(data_dir)

    urls = {
        'train_hr': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip',
        'train_lr': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_LR_bicubic_X4.zip',
        'valid_hr': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip',
        'valid_lr': 'http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip'
    }

    temp_dir = data_dir / 'temp'
    temp_dir.mkdir(parents=True, exist_ok=True)

    try:
        for name, url in urls.items():
            print(f'Downloading {name} dataset...')
            zip_file = temp_dir / f'{name}.zip'

            if not zip_file.exists():
                wget.download(url, str(zip_file))
                print()

            print(f'Extracting {name} dataset...')
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(str(data_dir))

        print('Download completed successfully!')

    except Exception as e:
        print(f'Error occurred: {e}')
        return None

    finally:
        if temp_dir.exists():
            shutil.rmtree(str(temp_dir))

    return str(data_dir)

# DIV2K 데이터셋 다운로드
div2k_path = download_div2k()