import datetime
import logging
import math
import time
import torch
from os import path as osp
from tqdm import tqdm
from basicsr.data import build_dataloader, build_dataset
from basicsr.data.data_sampler import EnlargedSampler
from basicsr.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from basicsr.models import build_model
from basicsr.utils import (AvgTimer, MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str,
                           init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, scandir)
from basicsr.utils.options import copy_opt_file, dict2str, parse_options

# TensorBorad 및 WandB 로거 초기화 함수
def init_tb_loggers(opt):
    """
    TensorBoard 및 WandB(W&B) 로거를 초기화하는 함수.

    Args:
        opt (dict): 로깅 관련 옵션을 포함하는 딕셔너리. 다음과 같은 키를 포함해야 함.
            - opt['logger']['wandb']: WandB 로거 설정 관련 딕셔너리.
                - opt['logger']['wandb']['project']: WandB 프로젝트 이름.
            - opt['logger']['use_tb_logger']: TensorBoard 로거 사용 여부 (True/False).
            - opt['name']: 현재 실행의 이름(일반적으로 실험 이름).
            - opt['root_path']: 로그 디렉터리가 저장될 루트 경로.

    Returns:
        tb_logger: 초기화된 TensorBoard 로거 인스턴스. 만약 사용하지 않을 경우 None 반환.
    """

    # WandB 로거 초기화
    if (opt['logger'].get('wandb') is not None) and (opt['logger']['wandb'].get('project')
                                                     is not None) and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, ('should turn on tensorboard when using wandb')
        init_wandb_logger(opt)

    # TensorBoard 로거 초기화
    tb_logger = None # 초기값 None 설정
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        # TensorBoard 로거를 활성화할 조건:
        # - opt['loger']['use_tb_logger']가 Ture이고
        # -opt['name']에 'debug'가 포함되지 않은 경우
        tb_logger = init_tb_logger(log_dir=osp.join(opt['root_path'], 'tb_logger', opt['name']))
    return tb_logger

# 학습 및 검증 데이터 로더 생성 함수
def create_train_val_dataloader(opt, logger):
    """
    학습 및 검증 데이터 로더를 생성하는 함수.

    Args:
        opt (dict): 학습 및 데이터셋 설정을 포함하는 딕셔너리.
            - opt['datasets']: 데이터셋 설정. 'train' 및 'val_x' 키를 포함해야 함.
            - opt['world_size']: 분산 학습 시 GPU의 수.
            - opt['rank']: 분산 학습 시 각 프로세스의 순위(rank).
            - opt['num_gpu']: 사용할 GPU의 수.
            - opt['dist']: 분산 학습 사용 여부 (True/False).
            - opt['manual_seed']: 랜덤 시드 값.
            - opt['train']['total_iter']: 총 학습 반복 횟수.
        logger (logging.Logger): 로깅 객체로, 정보 출력에 사용.

    Returns:
        tuple: 학습 및 검증 데이터 로더와 관련된 값.
            - train_loader: 학습 데이터 로더.
            - train_sampler: 학습 데이터 샘플러.
            - val_loaders: 검증 데이터 로더 리스트.
            - total_epochs: 총 학습 에포크 수.
            - total_iters: 총 학습 반복(iter) 수.
    """
    # 학습용 및 검증용 데이터 로더 초기화
    train_loader, val_loaders = None, []

    # 데이터셋 설정 순회
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            # 학습 데이터셋 및 데이터 로더 생성
            dataset_enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1) # 데이터 증강 비율 설정
            train_set = build_dataset(dataset_opt) # 학습 데이터셋 생성
            train_sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], dataset_enlarge_ratio) # 데이터 샘플링 설정
            train_loader = build_dataloader(
                train_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=train_sampler,
                seed=opt['manual_seed']
            ) # 학습 데이터 로더 생성

            # 학습 통계 계산
            num_iter_per_epoch = math.ceil(
                len(train_set) * dataset_enlarge_ratio /
                (dataset_opt['batch_size_per_gpu'] * opt['world_size'])
            ) # 에포크당 반복(iter) 횟수 계산
            total_iters = int(opt['train']['total_iter']) # 총 반복(iter) 수
            total_epochs = math.ceil(total_iters / (num_iter_per_epoch)) # 총 에포크

            # 학습 관련 정보 로깅
            logger.info('Training statistics:'
                        f'\n\tNumber of train images: {len(train_set)}'
                        f'\n\tDataset enlarge ratio: {dataset_enlarge_ratio}'
                        f'\n\tBatch size per gpu: {dataset_opt["batch_size_per_gpu"]}'
                        f'\n\tWorld size (gpu number): {opt["world_size"]}'
                        f'\n\tRequire iter number per epoch: {num_iter_per_epoch}'
                        f'\n\tTotal epochs: {total_epochs}; iters: {total_iters}.'
                    )

        elif phase.split('_')[0] == 'val':
            # 검증 데이터셋 및 데이터 로더 생성
            val_set = build_dataset(dataset_opt) # 검증 데이터셋 생성
            val_loader = build_dataloader(
                val_set,
                dataset_opt,
                num_gpu=opt['num_gpu'],
                dist=opt['dist'],
                sampler=None, # 검증에는 샘플러 사용하지 않음
                seed=opt['manual_seed']
            ) # 검증 데이터 로더 생성
            logger.info(f'Number of val images/folders in {dataset_opt["name"]}: {len(val_set)}')
            val_loaders.append(val_loader) # 검증 로더 리스트에 추가
        else:
            # 알 수 없는 데이터셋 단계 처리
            raise ValueError(f'Dataset phase {phase} is not recognized.')

    return train_loader, train_sampler, val_loaders, total_epochs, total_iters

# 학습 상태(resume state) 로드 함수
def load_resume_state(opt):
    """
    학습 상태(resume state)를 로드하는 함수.
    기존의 학습 상태를 자동 또는 수동으로 복원하여 학습을 재개할 수 있도록 설정합니다.

    Args:
        opt (dict): 설정 값 딕셔너리.
            - opt['auto_resume']: 자동으로 가장 최근 상태를 복원할지 여부 (True/False).
            - opt['name']: 현재 실험 이름.
            - opt['path']['resume_state']: 수동으로 지정된 복원 상태 경로.

    Returns:
        resume_state (dict or None): 로드된 학습 상태 딕셔너리.
            - None: 복원할 상태가 없는 경우.
            - dict: 학습 상태를 포함하는 딕셔너리.
    """
    resume_state_path = None # 학습 상태 경로 초기화

    # 자동 복원(atuo resume) 설정이 활성화된 경우
    if opt['auto_resume']:
        # 학습 상태가 저장된 디렉토리 경로
        state_path = osp.join('experiments', opt['name'], 'training_states')
        if osp.isdir(state_path): # 디렉토리가 존재하는지 확인
            # 디렉토리 내 .state 파일 검색
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0: # .state 파일이 존할 경우
                # 파일 이름에서 숫자(에포크 또는 iter 값) 추출
                states = [float(v.split('.state')[0]) for v in states]
                 # 가장 최신 상태의 파일 경로 설정
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                # 경로를 옵션에 저장
                opt['path']['resume_state'] = resume_state_path

    # 수동으로 resume_state 경로를 지정한 경우
    else:
        if opt['path'].get('resume_state'): # 경로가 명시적으로 지정되어 있는지 확인
            resume_state_path = opt['path']['resume_state']

    # 학습 상태 로드
    if resume_state_path is None:
        # 복원 상태 경로가 없는 경우
        resume_state = None
    else:
        # 복원 상태 경로가 존재하는 경우
        device_id = torch.cuda.current_device() # 현재 사용 중인 GPU ID를 가져옴
        resume_state = torch.load(resume_state_path, map_location=lambda storage, loc: storage.cuda(device_id)
        ) # 학습 상태를 로드하고 GPU 메모리에 매핑

        # 로드된 상태와 현재 설정을 검증
        check_resume(opt, resume_state['iter']) # 복원 상태의 iter 값과 현재 설정의 호환성을 확인

    return resume_state

# 학습 파이프라인 실행 함수
def train_pipeline(root_path):
    """
    학습 파이프라인 실행 함수.
    주어진 설정 파일과 경로를 기반으로 학습 과정을 초기화하고 실행합니다.

    Args:
        root_path (str): 프로젝트의 루트 경로.

    """
    # 옵션 파싱 및 초기 설정
    opt, args = parse_options(root_path, is_train=True) # 설정 파일 및 명령어 인자 파싱
    opt['root_path'] = root_path # 설정 파일의 경로를 저장

    # 성능 최적화를 위해 CuDNN 벤치마크 활성화
    torch.backends.cudnn.benchmark = True
    # 아래 설정은 정확한 연산을 보장하지만 성능 저하 가능성이 있음.
    # torch.backends.cudnn.deterministic = True

    # 학습 상태 로드
    resume_state = load_resume_state(opt) # 기존 학습 상태를 복원 (없으면 None 반환)

    # 실험 디렉토리 생성
    if resume_state is None:
        # 새로운 실험 디렉토리 생성
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and opt['rank'] == 0:
            # TensorBoard 로거 디렉토리 생성 및 이름 설정
            mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))

    # 설정 파일 복사 (실험 경로에 설정 파일 백업)
    copy_opt_file(args.opt, opt['path']['experiments_root'])

    #  로깅 설정
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info()) # 현재 환경 정보 로깅 (PyTorch, GPU, CUDA 등)
    logger.info(dict2str(opt)) # 설정 파일 내용을 로깅

    # 로거 초기화 (TensorBoard 및 WandB)
    tb_logger = init_tb_loggers(opt)

    # 데이터 로더 생성 (학습 및 검증 데이터 로더)
    result = create_train_val_dataloader(opt, logger)
    train_loader, train_sampler, val_loaders, total_epochs, total_iters = result

    # 모델 생성
    model = build_model(opt) # 설정에 맞는 모델 초기화
    if resume_state:  # 학습 상태 복원 (resume training)
        model.resume_training(resume_state)  # 모델 학습 상태 복구
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, iter: {resume_state['iter']}.")
        start_epoch = resume_state['epoch']
        current_iter = resume_state['iter']
    else: # 새 학습 시작
        start_epoch = 0
        current_iter = 0

    # 메시지 로거 생성
    msg_logger = MessageLogger(opt, current_iter, tb_logger) # 학습 중 상태 메시지 기록

    # 데이터 로더 프리페처(prefetcher) 생성
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode') # 사전 로드 모드 확인
    if prefetch_mode is None or prefetch_mode == 'cpu':
        # CPU 프리패처 설정
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        # CUDA 프리패처 설정
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader') # CUDA 프리패처 사용 로그 출력
        if opt['datasets']['train'].get('pin_memory') is not True:
            # CUDA 프리패처를 사용할 때는 pin_memory를 활성화해야 함
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        # 지원되지 않는 프리패처 모드 입력 시 예외 발생
        raise ValueError(f"Wrong prefetch_mode {prefetch_mode}. Supported ones are: None, 'cuda', 'cpu'.")

    # 학습 루프
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_timer, iter_timer = AvgTimer(), AvgTimer() # 데이터 로딩 시간 및 반복(iter) 시간 측정을 위한 타이머
    start_time = time.time()  # 학습 시작 시간 기록

    for epoch in range(start_epoch, total_epochs + 1): # 총 학습 에포크 반복
        train_sampler.set_epoch(epoch) # 분산 학습 샘플러에 현재 에포크 설정
        prefetcher.reset() # 프리페처 초기화
        train_data = prefetcher.next() # 첫 번째 배치 데이터 로드

        # tqdm으로 진행률 바 초기화
        num_iter_per_epoch = len(train_loader)  # 에포크당 반복(iter) 수
        with tqdm(total=num_iter_per_epoch, desc=f"Epoch [{epoch}/{total_epochs}]", unit="batch") as pbar:
            while train_data is not None: # 배치 데이터가 존재하는 동안 반복
                data_timer.record() # 데이터 로딩 시간 기록

                current_iter += 1 # 현재 반복(iter) 수 증가
                if current_iter > total_iters: # 총 반복 수 초과 시 학습 종료
                    break

                # 학습률 업데이트
                model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))

                # 모델 학습 수행
                model.feed_data(train_data) # 입력 데이터를 모델에 제공
                model.optimize_parameters(current_iter) # 모델 최적화 수행

                iter_timer.record() # 반복 시간 기록
                if current_iter == 1:
                    # 첫 반복(iter)에서 메시지 로거의 시작 시간을 재설정 (resume 모드 제외)
                    msg_logger.reset_start_time()

                # 로그 출력
                if current_iter % opt['logger']['print_freq'] == 0: # 지정된 로그 출력 빈도에 따라
                    log_vars = {'epoch': epoch, 'iter': current_iter} # 에포크 및 반복 수 기록
                    log_vars.update({'lrs': model.get_current_learning_rate()}) # 현재 학습률 기록
                    log_vars.update({'time': iter_timer.get_avg_time(), 'data_time': data_timer.get_avg_time()}) # 시간 통계 기록
                    log_vars.update(model.get_current_log()) # 모델의 현재 상태 로그 기록
                    msg_logger(log_vars) # 메시지 로거 호출

                # 체크포인트 저장
                if current_iter % opt['logger']['save_checkpoint_freq'] == 0: # 지정된 체크포인트 저장 빈도에 따라
                    logger.info('Saving models and training states.')
                    model.save(epoch, current_iter) # 모델 및 학습 상태 저장

                # 검증 수행
                if opt.get('val') is not None and (current_iter % opt['val']['val_freq'] == 0): # 지정된 검증 빈도에 따라
                    if len(val_loaders) > 1:
                        logger.warning('Multiple validation datasets are *only* supported by SRModel.') # 경고 출력
                    for val_loader in val_loaders: # 각 검증 데이터셋에 대해
                        model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img']) # 검증 수행

                # 진행률 바 업데이트
                pbar.update(1)

                # 다음 반복 준비
                data_timer.start() # 데이터 타이머 시작
                iter_timer.start() # 반복 타이머 시작
                train_data = prefetcher.next() # 다음 배치 데이터 로드
        # end of iter

    # 학습 종료
    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time))) # 총 소요 시간 계산
    logger.info(f'End of training. Time consumed: {consumed_time}') # 학습 종료 로그
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)  # 최신 모델 저장

    # 마지막 검증 수행
    if opt.get('val') is not None:
        for val_loader in val_loaders:
            model.validation(val_loader, current_iter, tb_logger, opt['val']['save_img'])

    # TensorBoard 로거 종료
    if tb_logger:
        tb_logger.close()

# 스크립트 실행 진입점
if __name__ == '__main__':
    # 루트 디렉토리 설정
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path) # 학습 파이프라인 실행