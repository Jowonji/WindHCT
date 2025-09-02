import importlib
import numpy as np
import random
import torch
import torch.utils.data
from copy import deepcopy
from functools import partial
from os import path as osp

from basicsr.data.prefetch_dataloader import PrefetchDataLoader
from basicsr.utils import get_root_logger, scandir
from basicsr.utils.dist_util import get_dist_info
from basicsr.utils.registry import DATASET_REGISTRY
from .numpy_paired_dataset import NumpyPairedDataset
from .numpy_paired_korea_dataset import NumpyPairedKorea

__all__ = ['build_dataset', 'build_dataloader']

# 데이터셋 모듈 자동 검색 및 등록
# 'data' 폴더 내에서 '_dataset'이 포함된 파일을 모두 검색하여 모듈로 로드
data_folder = osp.dirname(osp.abspath(__file__)) # 현재 파일의 디렉토리 경로
dataset_filenames = [
    osp.splitext(osp.basename(v))[0] # 파일명에서 확장자를 제거한 이름만 추출
    for v in scandir(data_folder) if v.endswith('_dataset.py') # '_dataset.py'로 끝나는 파일만 검색
]

# 데이터셋 모듈 임포트
_dataset_modules = [importlib.import_module(f'basicsr.data.{file_name}') for file_name in dataset_filenames]


def build_dataset(dataset_opt):
    """
    주어진 옵션을 기반으로 데이터셋 객체를 생성.

    Args:
        dataset_opt (dict): 데이터셋 설정 딕셔너리. 다음 키를 포함해야 함.
            - name (str): 데이터셋 이름 (예: 'DIV2K').
            - type (str): 데이터셋 타입 (예: 'ImageDataset').

    Returns:
        dataset (Dataset): 생성된 데이터셋 객체.
    """
    dataset_opt = deepcopy(dataset_opt) # 데이터셋 옵션을 깊은 복사하여 원본을 보호
    dataset = DATASET_REGISTRY.get(dataset_opt['type'])(dataset_opt) # 데이터셋 레지스트리에서 해당 타입의 클래스 가져와 생성
    logger = get_root_logger() # 기본 로거 가져오기
    logger.info(f'Dataset [{dataset.__class__.__name__}] - {dataset_opt["name"]} is built.') # 데이터셋 생성 로그 출력
    return dataset # 생성된 데이터셋 객체 반환


def build_dataloader(dataset, dataset_opt, num_gpu=1, dist=False, sampler=None, seed=None):
    """
    데이터로더를 생성하는 함수.

    Args:
        dataset (torch.utils.data.Dataset): 데이터셋 객체.
        dataset_opt (dict): 데이터셋 설정 딕셔너리. 다음과 같은 키를 포함:
            - phase (str): 데이터셋 단계. 'train', 'val', 'test' 중 하나.
            - num_worker_per_gpu (int): 각 GPU별 워커(worker) 수.
            - batch_size_per_gpu (int): 각 GPU별 배치 크기.
        num_gpu (int): 사용되는 GPU 수. 학습 단계에서만 사용. 기본값: 1.
        dist (bool): 분산 학습 여부. 학습 단계에서만 사용. 기본값: False.
        sampler (torch.utils.data.sampler): 데이터 샘플링 전략. 기본값: None.
        seed (int | None): 랜덤 시드. 기본값: None.

    Returns:
        DataLoader: 생성된 데이터로더 객체.
    """
    phase = dataset_opt['phase'] # 데이터셋의 현재 단계 ('train', 'val', 'test')
    rank, _ = get_dist_info() # 분산 학습 환경에서의 rank 정보 가져오기

    if phase == 'train': # 학습 단계
        if dist:  # 분산 학습
            batch_size = dataset_opt['batch_size_per_gpu'] # 각 GPU별 배치 크기
            num_workers = dataset_opt['num_worker_per_gpu'] # 각 GPU별 워커(worker) 수
        else:  # 비분산 학습
            multiplier = 1 if num_gpu == 0 else num_gpu # GPU 수를 고려한 배수 계산
            batch_size = dataset_opt['batch_size_per_gpu'] * multiplier
            num_workers = dataset_opt['num_worker_per_gpu'] * multiplier

        # DataLoader 초기화 인자 설정
        dataloader_args = dict(
            dataset=dataset, # 데이터셋
            batch_size=batch_size, # 배치 크기
            shuffle=False, # 기본값: 셔플 비활성화
            num_workers=num_workers, # 워커(worker) 수
            sampler=sampler, # 샘플러 (없을 경우 기본값: None)
            drop_last=True # 배치 크기에 맞지 않는 마지막 데이터를 버림
        )

        if sampler is None: # 샘플러가 없는 경우
            dataloader_args['shuffle'] = True # 셔플 활성화

        # 워커 초기화 함수 설정 (시드 및 워커 설정)
        dataloader_args['worker_init_fn'] = partial(
            worker_init_fn, num_workers=num_workers, rank=rank, seed=seed
        ) if seed is not None else None

    elif phase in ['val', 'test']:  # 검증 또는 테스트 단계
        # 검증 및 테스트는 배치 크기 1, 워커 수 0으로 설정
        dataloader_args = dict(dataset=dataset, batch_size=1, shuffle=False, num_workers=0)

    else:
        # 잘못된 phase 값 입력 시 예외 처리
        raise ValueError(f"Wrong dataset phase: {phase}. Supported ones are 'train', 'val' and 'test'.")

    # 추가 옵션: pin_memory와 persistent_workers 설정
    dataloader_args['pin_memory'] = dataset_opt.get('pin_memory', False) # pin_memory 옵션
    dataloader_args['persistent_workers'] = dataset_opt.get('persistent_workers', False) # persistent_workers 옵션

    # 프리패처(prefetch) 모드 설정
    prefetch_mode = dataset_opt.get('prefetch_mode')
    if prefetch_mode == 'cpu':  # CPU 프리패처
        num_prefetch_queue = dataset_opt.get('num_prefetch_queue', 1) # 프리패처 큐 크기 설정
        logger = get_root_logger() # 로거 초기화
        logger.info(f'Use {prefetch_mode} prefetch dataloader: num_prefetch_queue = {num_prefetch_queue}')
        return PrefetchDataLoader(num_prefetch_queue=num_prefetch_queue, **dataloader_args) # 프리페처 데이터로더 반환
    else:
        # 일반적인 DataLoader 반환
        # prefetch_mode=None: 일반 DataLoader
        # prefetch_mode='cuda': CUDA 기반 프리페처 지원
        return torch.utils.data.DataLoader(**dataloader_args)

def worker_init_fn(worker_id, num_workers, rank, seed):
    """
    DataLoader 워커(worker)의 초기화를 설정하는 함수.

    각 워커가 고유한 랜덤 시드 값을 가지도록 설정하여 데이터 로드 과정의 재현성과 무작위성을 보장합니다.

    Args:
        worker_id (int): 현재 워커의 ID.
        num_workers (int): 데이터로더에서 설정된 총 워커(worker) 수.
        rank (int): 분산 학습 환경에서의 프로세스 순위(rank).
        seed (int): 기본 랜덤 시드 값.
    """
    # 워커의 고유 시드를 계산
    worker_seed = num_workers * rank + worker_id + seed

    # 워커의 시드 설정
    np.random.seed(worker_seed) # NumPy의 랜덤 시드 설정
    random.seed(worker_seed) # Python 기본 random 모듈의 시드 설정
