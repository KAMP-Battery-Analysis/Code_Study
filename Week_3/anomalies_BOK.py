"""
시계열 이상 탐지 함수.
일부 구현은 논문 https://arxiv.org/pdf/1802.04431.pdf을 참고하였습니다.
"""
class Anomaly(object):
    def __init__(self):
        pass
    
    # _count_above(2)에서 사용
    def _deltas(self, errors, epsilon, mean, std):
        """평균 및 표준편차 델타를 계산.
        delta_mean = mean(errors) - epsilon(임계값) 이하의 모든 오류들의 평균
        delta_std = std(errors) - epsilon(임계값) 이하의 모든 오류들의 표준편차
        Args:
        errors (ndarray): 오류 배열.
        epsilon (ndarray): 임계값.
        mean (float): 오류들의 평균.
        std (float): 오류들의 표준편차.
        Returns:
        float, float:
        * delta_mean.
        * delta_std.
        """
        below = errors[errors <= epsilon]
        if not len(below):
            return 0, 0
        return mean - below.mean(), std - below.std()
    # anomalies.py (2) - _z_cost(3)에서 사용
    def _count_above(self, errors, epsilon):
        """epsilon 이상인 오류와 연속된 시퀀스의 수를 계산
        연속된 시퀀스는 시프트하고 원래 값이 true였던 위치의 변화를
        계산하여 그 위치에서 시퀀스가 시작되었음을 의미
        Args:
        errors (ndarray): 오류 배열.
        epsilon (ndarray): 임계값.
        Returns:
        int, int:
        * epsilon 이상인 오류의 수.
        * epsilon 이상인 연속된 시퀀스의 수.
        """
        # errors 배열에서 epsilon보다 큰 값인지 여부 배열
        above = errors > epsilon
        # epsilon보다 큰 오류의 총 수를 계산
        total_above = len(errors[above])
        # above 배열을 pandas Series로 변환
        above = pd.Series(above)
        # above Series를 1만큼 시프트(레코드를 한 칸씩 밈)
        shift = above.shift(1)
        # above와 shift된 값 간의 변화를 계산(서로 다른 경우가 true)
        change = above != shift
        # epsilon보다 큰 연속된 시퀀스의 수를 계산
        total_consecutive = sum(above & change)
        # 결과를 반환
        return total_above, total_consecutive
    
    # anomalies.py (3) - _find_threshold(4)에서 사용
    def _z_cost(self, z, errors, mean, std):
        """z 값이 얼마나 나쁜지를 계산
        원래 공식::
        (delta_mean/mean) + (delta_std/std)
        ------------------------------------------------------
        number of errors above + (number of sequences above)^2
        이는 `z`의 "좋음"을 계산하며, 값이 높을수록 `z`가 더 좋다는 것을 의미
        이 경우, 이 값을 반전(음수로 만듦)하여 비용 함수로 변환
        나중에 scipy.fmin을 사용하여 이를 최소화
        
        Args:
        z (ndarray): 비용 점수가 계산될 값.
        errors (ndarray): 오류 배열.
        mean (float): 오류들의 평균.
        std (float): 오류들의 표준편차.
        
        Returns float: z의 비용.
        """
        
        # epsilon 값을 평균 + z * 표준편차로 계산
        epsilon = mean + z * std
        # epsilon을 사용하여 평균 및 표준편차 델타를 계산
        delta_mean, delta_std = self._deltas(errors, epsilon, mean, std)
        # epsilon보다 큰 오류와 연속된 시퀀스의 수를 계산
        above, consecutive = self._count_above(errors, epsilon)
        # 분자(numerator)를 계산합니다. (반전된 값)
        numerator = -(delta_mean / mean + delta_std / std)
        # 분모(denominator)를 계산합니다.
        denominator = above + consecutive ** 2
        # 분모가 0이면 무한대를 반환
        if denominator == 0:
            return np.inf
        
        # 최종 비용을 계산하여 반환
        return numerator / denominator
    # anomalies.py (4) - _find_window_sequences(9)에서 사용
    def _find_threshold(self, errors, z_range):
        """이상적인 임계값 찾는 함수.
        이상적인 임계값은 z_cost 함수를 최소화하는 값.
        Scipy.fmin을 사용하여 z_range의 값들을 시작점으로 최소값을 탐색.

        Args:
        errors (ndarray): 오류 배열.
        z_range (list): scipy.fmin 함수의 시작점을 선택할 범위를 나타내는 두 값의 리스트.

        Returns: float: 계산된 임계값.
        """
        # 오류들의 평균을 계산.
        mean = errors.mean()
        
        # 오류들의 표준편차를 계산.
        std = errors.std()
        
        # z_range에서 최소값과 최대값을 가져옴.
        min_z, max_z = z_range
        
        # 최적의 z값을 저장할 변수 초기화.
        best_z = min_z
        
        # 최적의 비용을 무한대로 초기화.
        best_cost = np.inf

        # min_z부터 max_z까지 반복.
        for z in range(min_z, max_z):
            # fmin 함수를 사용하여 z에서 시작하는 최소 비용을 찾음.
            best = fmin(self._z_cost, z, args=(errors, mean, std), full_output=True, disp=False)
            
            # 최적의 z값과 비용을 가져옴.
            z, cost = best[0:2]
            
            # 현재 비용이 최적의 비용보다 작으면 갱신.
            if cost < best_cost:
                best_z = z[0]

        # 최적의 임계값을 계산하여 반환.
        return mean + best_z * std

    # _find_window_sequences(9)에서 사용
    def _fixed_threshold(self, errors, k=3.0):
        """임계값 계산.
        고정된 임계값은 평균에서 k 표준편차만큼 떨어진 값으로 정의됨.

        Args:
        errors (ndarray): 오류 배열.

        Returns:
        float: 계산된 임계값.
        """
        # 오류들의 평균을 계산.
        mean = errors.mean()
        
        # 오류들의 표준편차를 계산.
        std = errors.std()

        # 고정된 임계값을 계산하여 반환.
        return mean + k * std
    
    # anomalies.py (5-1) - _find_window_sequences(9)에서 사용
    def _find_sequences(self, errors, epsilon, anomaly_padding):
        """epsilon 이상인 값들의 시퀀스 탐색.
        다음 단계들을 따름:
        * epsilon 이상인 값을 표시하는 불리언 마스크 생성.
        * True 값 주위의 일정 범위의 오류를 True로 표시.
        * 이 마스크를 한 칸씩 시프트하고, 빈 공간은 False로 채움.
        * 시프트된 마스크와 원래 마스크를 비교하여 변화가 있는지 확인.
        * True였던 값이 변경된 지점을 시퀀스 시작으로 간주.
        * False였던 값이 변경된 지점을 시퀀스 종료로 간주.

        Args:
        errors (ndarray): 오류 배열.
        epsilon (float): 임계값. epsilon 이상의 모든 오류는 이상으로 간주.
        anomaly_padding (int): 발견된 이상 주위에 추가할 오류의 개수.

        Returns:
        ndarray, float:
        * 발견된 이상 시퀀스의 시작과 끝을 포함하는 배열.
        * 이상으로 간주되지 않은 최대 오류 값.
        """
        # epsilon 이상인 값을 표시하는 불린 마스크 생성.
        above = pd.Series(errors > epsilon)

        # True인 값의 인덱스를 찾음.
        index_above = np.argwhere(above.values)

        # True 값 주위의 일정 범위의 오류를 True로 표시.
        for idx in index_above.flatten():
            above[max(0, idx - anomaly_padding):min(idx + anomaly_padding + 1, len(above))] = True

        # 이 마스크를 한 칸씩 시프트하고, 빈 공간은 False로 채움.
        shift = above.shift(1).fillna(False)

        # 시프트된 마스크와 원래 마스크를 비교하여 변화가 있는지 확인.
        change = above != shift

        # 모든 값이 True이면 max_below는 0으로 설정.
        if above.all():
            max_below = 0
        else:
            # 이상으로 간주되지 않은 최대 오류 값을 찾음.
            max_below = max(errors[~above])

        # 시퀀스의 시작과 끝 인덱스를 찾음.
        index = above.index
        starts = index[above & change].tolist()
        ends = (index[~above & change] - 1).tolist()

        # 마지막 시퀀스가 종료되지 않았을 경우 종료 인덱스를 추가.
        if len(ends) == len(starts) - 1:
            ends.append(len(above) - 1)

        # 결과를 배열로 반환.
        return np.array([starts, ends]).T, max_below

    # anomalies.py (5-2) - _find_window_sequences(9)에서 사용
    def _get_max_errors(self, errors, sequences, max_below):
        """각 이상 시퀀스의 최대 오류를 가져옴.
        또한 이상으로 간주되지 않은 최대 오류 값을 포함하는 행을 추가.
        각 시퀀스의 최대 오류를 포함하는 ``max_error`` 열과 시작 및 종료 인덱스를
        포함하는 ``start`` 및 ``stop`` 열이 있는 테이블을 내림차순으로 정렬하여 반환.

        Args:
        errors (ndarray): 오류 배열.
        sequences (ndarray): 이상 시퀀스의 시작과 끝을 포함하는 배열.
        max_below (float): 이상으로 간주되지 않은 최대 오류 값.

        Returns:
        pandas.DataFrame: ``start``, ``stop``, ``max_error`` 열을 포함하는 DataFrame 객체.
        """
        # 이상으로 간주되지 않은 최대 오류 값을 포함하는 초기 값 설정.
        max_errors = [{
            'max_error': max_below,
            'start': -1,
            'stop': -1
        }]

        # 각 시퀀스에 대해 최대 오류 값을 계산하고 추가.
        for sequence in sequences:
            start, stop = sequence
            sequence_errors = errors[start: stop + 1]
            max_errors.append({
                'start': start,
                'stop': stop,
                'max_error': max(sequence_errors)
            })

        # DataFrame으로 변환하고 최대 오류 값으로 내림차순 정렬.
        max_errors = pd.DataFrame(max_errors).sort_values('max_error', ascending=False)

        # 인덱스를 재설정하여 반환.
        return max_errors.reset_index(drop=True)
    
    # anomalies.py (6) - _find_window_sequences(9)에서 사용
    def _prune_anomalies(self, max_errors, min_percent):
        """거짓 양성을 줄이기 위해 이상을 가지치기.
        다음 단계들을 따름:
        * 오류를 1단계 음수 방향으로 시프트하여 각 값을 다음 값과 비교.
        * 비교하지 않기 위해 마지막 행을 제거.
        * 각 행에 대한 백분율 증가를 계산.
        * ``min_percent`` 이하인 행을 찾음.
        * 이러한 행 중 가장 최근의 행의 인덱스를 찾음.
        * 해당 인덱스 위의 모든 시퀀스 값을 가져옴.

        Args:
        max_errors (pandas.DataFrame): ``start``, ``stop``, ``max_error`` 열을 포함하는 DataFrame 객체.
        min_percent (float):
        이상 간의 분리를 위한 최소 백분율. 윈도우 시퀀스에서 가장 높은 비이상 오류와의 비교.

        Returns:
        ndarray: 가지치기된 이상들의 시작, 끝, max_error를 포함하는 배열.
        """
        # 오류를 1단계 음수 방향으로 시프트하여 다음 값과 비교.
        next_error = max_errors['max_error'].shift(-1).iloc[:-1]

        # 현재 오류 값을 가져옴.
        max_error = max_errors['max_error'].iloc[:-1]

        # 각 행에 대한 백분율 증가를 계산.
        increase = (max_error - next_error) / max_error

        # min_percent 이하인 행을 찾음.
        too_small = increase < min_percent

        # 모든 행이 min_percent 이하인 경우.
        if too_small.all():
            last_index = -1
        else:
            # min_percent 이상인 가장 최근의 행의 인덱스를 찾음.
            last_index = max_error[~too_small].index[-1]

        # 가지치기된 이상들의 시작, 끝, max_error를 포함하는 배열을 반환.
        return max_errors[['start', 'stop', 'max_error']].iloc[0: last_index + 1].values
    
    # anomalies.py (7) - _find_window_sequences(9)에서 사용
    def _compute_scores(self, pruned_anomalies, errors, threshold, window_start):
        """이상의 점수를 계산.
        시퀀스에서 최대 오류에 비례하는 점수를 계산하고, 인덱스를 절대값으로 만들기 위해 window_start 타임스탬프를 추가.

        Args:
        pruned_anomalies (ndarray): 윈도우 내 모든 이상의 시작, 끝 및 max_error를 포함하는 이상 배열.
        errors (ndarray): 오류 배열.
        threshold (float): 임계값.
        window_start (int): 윈도우에서 첫 번째 오류 값의 인덱스.

        Returns:
        list: 각 이상에 대해 시작 인덱스, 종료 인덱스, 점수를 포함하는 이상 목록.
        """
        anomalies = list()

        # 점수 계산을 위한 분모. 오류의 평균과 표준편차의 합.
        denominator = errors.mean() + errors.std()

        # 가지치기된 각 이상에 대해 점수를 계산.
        for row in pruned_anomalies:
            max_error = row[2]

            # 점수를 계산. (max_error - threshold) / 분모
            score = (max_error - threshold) / denominator

            # 절대 인덱스를 사용하여 이상을 추가.
            anomalies.append([row[0] + window_start, row[1] + window_start, score])

        # 이상 목록을 반환.
        return anomalies
    
    # anomalies.py (8) - find_anomalies(10)에서 사용
    def _merge_sequences(self, sequences):
        """연속적이고 겹치는 시퀀스를 병합함.
        시작, 종료, 점수 트리플의 리스트를 반복하면서
        겹치거나 연속적인 시퀀스를 병합함.
        병합된 시퀀스의 점수는 각 시퀀스의 길이로 가중치를 둔
        단일 점수의 평균임.
        
        Args:
        sequences (list): 각 이상치의 시작 인덱스, 종료 인덱스, 점수를 포함하는 이상치 목록.
        
        Returns:
        ndarray: 병합 후 각 이상치의 시작 인덱스, 종료 인덱스, 점수를 포함하는 배열.
        """
        
        # 시퀀스가 비어있는 경우 빈 배열 반환
        if len(sequences) == 0:
            return np.array([])
        
        # 시퀀스를 시작 인덱스를 기준으로 정렬
        sorted_sequences = sorted(sequences, key=lambda entry: entry[0])
        
        # 병합된 새로운 시퀀스 초기화
        new_sequences = [sorted_sequences[0]]
        score = [sorted_sequences[0][2]]  # 초기 시퀀스의 점수
        weights = [sorted_sequences[0][1] - sorted_sequences[0][0]]  # 초기 시퀀스의 길이
        
        # 정렬된 시퀀스를 순회하면서 병합
        for sequence in sorted_sequences[1:]:
            prev_sequence = new_sequences[-1]  # 마지막으로 추가된 시퀀스
            
            # 현재 시퀀스가 이전 시퀀스와 겹치거나 연속적인 경우
            if sequence[0] <= prev_sequence[1] + 1:
                score.append(sequence[2])  # 현재 시퀀스의 점수를 추가
                weights.append(sequence[1] - sequence[0])  # 현재 시퀀스의 길이를 추가
                # 가중 평균 계산
                weighted_average = np.average(score, weights=weights)
                # 이전 시퀀스와 병합
                new_sequences[-1] = (prev_sequence[0],
                                    max(prev_sequence[1], sequence[1]),
                                    weighted_average)
            else:
                # 현재 시퀀스가 이전 시퀀스와 겹치지 않으면 새로운 시퀀스로 추가
                score = [sequence[2]]  # 현재 시퀀스의 점수 초기화
                weights = [sequence[1] - sequence[0]]  # 현재 시퀀스의 길이 초기화
                new_sequences.append(sequence)
        
        return np.array(new_sequences)  # 병합된 시퀀스 배열로 반환
    
    # anomalies.py (9) - find_anomalies(10)에서 사용
    def _find_window_sequences(self, window, z_range, anomaly_padding, min_percent, window_start, fixed_threshold):
        """이상치인 값들의 시퀀스를 찾음.
        먼저 윈도우의 임계값을 찾고, 그 임계값 이상인 모든 시퀀스를 찾음.
        그런 다음 시퀀스의 최대 오류를 얻고 이상치를 가지치기함.
        마지막으로, 이상치의 점수를 계산함.
        Args:
        window (ndarray): 분석 중인 윈도우의 오류 배열.
        z_range (list):
        동적 임계값 찾기 함수의 시작점을 선택할 범위를 나타내는 두 값의 리스트.
        anomaly_padding (int):
        발견된 이상치 전후의 오류 수를 이상치 시퀀스에 추가.
        min_percent (float):
        윈도우 시퀀스의 가장 높은 비이상치 오류와 이상치 간의 분리 비율.
        window_start (int): 윈도우에서 첫 번째 오류 값의 인덱스.
        fixed_threshold (bool): 고정 임계값을 사용할지 동적 임계값을 사용할지 여부를 나타냄.
        Returns:
        ndarray:
        윈도우에서 발견된 각 이상치 시퀀스의 시작 인덱스, 종료 인덱스, 점수를 포함하는 배열.
        """
        
        # 고정 임계값을 사용할지 여부에 따라 임계값 결정
        if fixed_threshold: 
            threshold = self._fixed_threshold(window)  # 고정 임계값 사용
        else:
            threshold = self._find_threshold(window, z_range)  # 동적 임계값 사용
        
        # 임계값을 기준으로 윈도우에서 이상치 시퀀스와 비이상치 중 최대값을 찾음
        window_sequences, max_below = self._find_sequences(window, threshold, anomaly_padding)
        
        # 시퀀스에서 최대 오류를 계산
        max_errors = self._get_max_errors(window, window_sequences, max_below)
        
        # 이상치 가지치기 (유의미한 이상치만 남김)
        pruned_anomalies = self._prune_anomalies(max_errors, min_percent)
        
        # 각 이상치 시퀀스의 점수를 계산
        window_sequences = self._compute_scores(pruned_anomalies, window, threshold, window_start)
        
        return window_sequences  # 이상치 시퀀스를 반환

    
    # anomalies.py (10-1)
    def find_anomalies(self, errors, index, z_range=(0, 10), window_size=None, window_size_portion=None, window_step_size=None, 
                    window_step_size_portion=None, min_percent=0.1, anomaly_padding=50, lower_threshold=False, fixed_threshold=True):
        """오류 값들의 시퀀스에서 이상치를 찾음.
        오류 윈도우을 정의하고, 그 윈도우에서 이상치 시퀀스를 찾아 시작/종료 인덱스와 점수를 저장함.
        오류 시퀀스를 평균을 중심으로 반전시켜 비정상적으로 낮은 오류 시퀀스를 찾을 수도 있음.
        윈도우을 이동하면서 이 절차를 반복함.
        마지막으로, 중복되거나 연속적인 시퀀스를 결합함.
        Args:
        errors (ndarray): 오류 배열.
        index (ndarray): 오류 인덱스 배열.
        z_range (list):
        선택사항. scipy.fmin 함수의 시작점을 선택할 범위를 나타내는 두 값의 리스트.
        주어지지 않으면 (0, 10)이 사용됨.
        window_size (int):
        선택사항. 임계값이 계산되는 윈도우의 크기.
        주어지지 않으면 `None`이 사용되며, 전체 오류 시퀀스에 대한 하나의 임계값이 사용됨.
        window_size_portion (float):
        선택사항. 오류 시퀀스의 일부로 윈도우 크기를 지정함.
        주어지지 않으면 `None`이 사용되며, 윈도우 크기는 그대로 사용됨.
        window_step_size (int):
        선택사항. 윈도우을 이동시키기 전의 단계 수.
        window_step_size_portion (float):
        선택사항. 윈도우 크기의 일부로 단계 수를 지정함.
        주어지지 않으면 `None`이 사용되며, 윈도우 단계 크기는 그대로 사용됨.
        min_percent (float):
        선택사항. 이상치와 윈도우 시퀀스에서 가장 높은 비이상치 오류 간의 분리 비율.
        주어지지 않으면 0.1이 사용됨.
        anomaly_padding (int):
        선택사항. 발견된 이상치 전후의 오류 수를 이상치 시퀀스에 추가함.
        주어지지 않으면 50이 사용됨.
        lower_threshold (bool):
        선택사항. 비정상적으로 낮은 오류를 찾기 위해 낮은 임계값을 적용할지 여부를 나타냄.
        주어지지 않으면 `False`가 사용됨.
        fixed_threshold (bool):
        선택사항. 고정 임계값을 사용할지 동적 임계값을 사용할지 여부를 나타냄.
        주어지지 않으면 `False`가 사용됨.
        Returns:
        ndarray: 시작 인덱스, 종료 인덱스, 점수를 포함하는 각 이상치 시퀀스 배열.
        """
        # 윈도우 크기를 설정, 주어지지 않으면 전체 오류 배열의 길이로 설정
        window_size = window_size or len(errors)
        if window_size_portion:
            # 오류 배열 길이의 비율로 윈도우 크기를 설정
            window_size = np.ceil(len(errors) * window_size_portion).astype('int')
        
        # 윈도우 이동 단계 크기 설정, 주어지지 않으면 윈도우 크기와 동일하게 설정
        window_step_size = window_step_size or window_size
        if window_step_size_portion:
            # 윈도우 크기의 비율로 윈도우 이동 단계를 설정
            window_step_size = np.ceil(window_size * window_step_size_portion).astype('int')
        
        window_start = 0  # 윈도우 시작점 초기화
        window_end = 0  # 윈도우 종료점 초기화
        sequences = list()  # 이상치 시퀀스를 저장할 리스트 초기화
        
        # 오류 배열을 순회하며 윈도우를 이동시키는 루프
        while window_end < len(errors):
            window_end = window_start + window_size  # 윈도우 종료점 설정
            window = errors[window_start:window_end]  # 현재 윈도우에 해당하는 오류 값들을 선택
            
            # 현재 윈도우에서 이상치 시퀀스를 찾음(#9번 참고)
            window_sequences = self._find_window_sequences(window,
                                                        z_range,
                                                        anomaly_padding,
                                                        min_percent,
                                                        window_start,
                                                        fixed_threshold)
            sequences.extend(window_sequences)  # 찾은 시퀀스를 sequences 리스트에 추가
            
            if lower_threshold:
                # 오류 시퀀스를 평균을 기준으로 반전
                mean = window.mean()
                inverted_window = mean - (window - mean)
                
                # 반전된 오류 시퀀스에서 이상치 시퀀스를 찾음
                inverted_window_sequences = self._find_window_sequences(inverted_window,
                                                                        z_range,
                                                                        anomaly_padding,
                                                                        min_percent,
                                                                        window_start,
                                                                        fixed_threshold)
                sequences.extend(inverted_window_sequences)  # 찾은 시퀀스를 sequences 리스트에 추가
            
            window_start = window_start + window_step_size  # 윈도우 시작점을 이동
        
        # 중복되거나 연속적인 시퀀스를 병합
        sequences = self._merge_sequences(sequences)
        
        anomalies = list()  # 이상치를 저장할 리스트 초기화
        for start, stop, score in sequences:
            # 각 이상치 시퀀스의 시작 인덱스, 종료 인덱스, 점수를 출력하고 리스트에 추가
            print("start", start)
            print("stop", stop)
            print("score", score)
            anomalies.append([index[int(start)], index[int(stop)], score])
        
        return anomalies  # 이상치 리스트 반환
    
    # anomalies.py (11) score_anomalies(17)에서 사용
    def _compute_critic_score(self, critics, smooth_window):
        """Critic 점수를 계산하여 이상 점수 배열을 반환한다.
        Args:
        critics (ndarray): Critic 값 배열.
        smooth_window (int): 오류를 평활화할 때 사용되는 윈도우 크기.
        Returns:
        ndarray: 이상 점수 배열.
        """
        
        # critics를 numpy 배열로 변환
        critics = np.asarray(critics)
        
        # 1사분위수와 3사분위수를 계산
        l_quantile = np.quantile(critics, 0.25)
        u_quantile = np.quantile(critics, 0.75)
        
        # critics 값이 1사분위수와 3사분위수 사이에 있는지를 나타내는 불리언 배열을 생성
        in_range = np.logical_and(critics >= l_quantile, critics <= u_quantile)
        
        # 1사분위수와 3사분위수 사이에 있는 critics 값의 평균을 계산
        critic_mean = np.mean(critics[in_range])
        
        # 전체 critics 값의 표준 편차를 계산
        critic_std = np.std(critics)
        
        # critics 값을 표준화하여 z-score를 계산하고, 1을 더해 양수로 변환
        z_scores = np.absolute((np.asarray(critics) - critic_mean) / critic_std) + 1
        
        # z-score를 pandas Series로 변환하고, smooth_window를 사용해 이동 평균을 계산
        z_scores = pd.Series(z_scores).rolling(smooth_window, center=True, min_periods=smooth_window // 2).mean().values
        
        # 최종 z-score 배열을 반환
        return z_scores

    
    # anomalies.py (12) - 단독 사용
    def _regression_errors(y, y_hat, smoothing_window=0.01, smooth=True):
        """예측값과 실제값을 비교하여 절대 오차 배열을 계산함.
        만약 smooth가 True라면, 결과 오차 배열에 EWMA를 적용함.

        Args:
            y (ndarray): 실제값 (정답).
            y_hat (ndarray): 예측값.
            smoothing_window (float): 선택사항. smoothing window의 크기. y의 전체 길이의 비율로 표현됨. 기본값은 0.01.
            smooth (bool): 선택사항. 반환된 오차에 EWMA를 적용할지 여부를 나타냄. 기본값은 True.
        Returns:
            ndarray: 오차 배열.
        """
        # 절대 오차를 계산함
        errors = np.abs(y - y_hat)[:, 0]
        
        # smooth가 False라면, 단순히 오차 배열을 반환함
        if not smooth:
            return errors
        
        # smoothing_window를 y의 길이에 비례하여 설정함
        smoothing_window = int(smoothing_window * len(y))
        
        # EWMA를 적용하여 부드러운 오차 배열을 반환함
        return pd.Series(errors).ewm(span=smoothing_window).mean().values
    
    # anomalies.py (13) - 단독 사용
    def regression_errors(y, y_hat, smoothing_window=0.01, smooth=True):
        """예측값과 실제값을 비교하여 절대 오차 배열을 계산함.

        Args:
            y (ndarray): 실제값 (정답).
            y_hat (ndarray): 예측값.
            smoothing_window (float): 선택사항. smoothing window의 크기. y의 전체 길이의 비율로 표현됨. 기본값은 0.01임.
            smooth (bool): 선택사항. 반환된 오차에 EWMA를 적용할지 여부를 나타냄. 기본값은 True임.
        Returns:
            ndarray: 오차 배열.
        """
        # 절대 오차를 계산함
        errors = np.abs(y - y_hat)[:, 0]
        
        # smooth가 False라면, 단순히 오차 배열을 반환함
        if not smooth:
            return errors
        
        # smoothing_window를 y의 길이에 비례하여 설정함
        smoothing_window = int(smoothing_window * len(y))
        
        # EWMA(지수가중이동평균)를 적용하여 부드러운 오차 배열을 반환함
        return pd.Series(errors).ewm(span=smoothing_window).mean().values

    # anomalies.py (14) - _reconstruction_errors(16)에서 사용
    def _point_wise_error(self, y, y_hat):
        """예측 값과 실제 값 간의 점 단위 오류를 계산함.
        Args:
        y (ndarray): 실제 값.
        y_hat (ndarray): 예측 값.
        Returns:
        ndarray: 부드럽게 처리된 점 단위 오류 배열.
        """
        
        y_abs = abs(y - y_hat)              # 실제 값과 예측 값의 차이의 절대값을 계산
        y_abs_sum = np.sum(y_abs, axis=-1)  # 차이의 절대값을 마지막 차원을 따라 합산
        return y_abs_sum                    # 점 단위 오류 값을 반환
    
    # _reconstruction_errors(16)에서 사용
    def _area_error(self, y, y_hat, score_window=10):
        """예측 값과 실제 값 간의 면적 오류를 계산함.
        Args:
        y (ndarray): 실제 값.
        y_hat (ndarray): 예측 값.
        score_window (int):
        선택사항. 점수가 계산되는 윈도우의 크기.
        주어지지 않으면 10이 사용됨.
        Returns:
        ndarray: 면적 오류 배열.
        """
        
        # 실제 값 y를 pandas Series로 변환한 후, 주어진 윈도우 크기(score_window)를 이용하여
        # 중심에서부터 부드럽게 처리하고, trapz 함수를 적용하여 적분값을 계산
        smooth_y = pd.Series(y).rolling(score_window, center=True, 
                                        min_periods=score_window // 2).apply(integrate.trapz)
        
        # 예측 값 y_hat를 pandas Series로 변환한 후, 주어진 윈도우 크기(score_window)를 이용하여
        # 중심에서부터 부드럽게 처리하고, trapz 함수를 적용하여 적분값을 계산
        smooth_y_hat = pd.Series(y_hat).rolling(score_window, center=True, 
                                                min_periods=score_window // 2).apply(integrate.trapz)
        
        # 부드럽게 처리된 실제 값과 예측 값 간의 차이의 절대값을 계산
        errors = abs(smooth_y - smooth_y_hat)
        
        return errors  # 면적 오류 값을 반환
    
    # anomalies.py (15) _reconstruction_errors(16)에서 사용
    def _dtw_error(self, y, y_hat, score_window=10):
        """예측 값과 실제 값 간의 동적 타임 워핑(DTW) 오류를 계산함.
        Args:
        y (ndarray): 실제 값.
        y_hat (ndarray): 예측 값.
        score_window (int):
        선택사항. 점수가 계산되는 윈도우의 크기.
        주어지지 않으면 10이 사용됨.
        Returns:
        ndarray: DTW 오류 배열.
        """
        
        length_dtw = (score_window // 2) * 2 + 1  # DTW 길이를 계산 (score_window가 홀수여야 함)
        half_length_dtw = length_dtw // 2  # DTW 길이의 절반 계산

        # 실제 값 y에 패딩 추가
        y_pad = np.pad(y, (half_length_dtw, half_length_dtw), 'constant', constant_values=(0, 0))
        # 예측 값 y_hat에 패딩 추가
        y_hat_pad = np.pad(y_hat, (half_length_dtw, half_length_dtw), 'constant', constant_values=(0, 0))

        i = 0  # 인덱스 초기화
        similarity_dtw = []  # DTW 유사도를 저장할 리스트 초기화

        # 실제 값과 예측 값 간의 DTW 유사도를 계산
        while i < len(y) - length_dtw:
            true_data = y_pad[i:i + length_dtw]  # 실제 값의 현재 윈도우 선택
            true_data = true_data.flatten()  # 1차원 배열로 변환
            pred_data = y_hat_pad[i:i + length_dtw]  # 예측 값의 현재 윈도우 선택
            pred_data = pred_data.flatten()  # 1차원 배열로 변환
            dist = dtw(true_data, pred_data)  # DTW 유사도 계산
            similarity_dtw.append(dist)  # 계산된 유사도를 리스트에 추가
            i += 1  # 인덱스 증가

        # 오류 값을 계산하여 패딩 추가
        errors = ([0] * half_length_dtw + similarity_dtw + [0] * (len(y) - len(similarity_dtw) - half_length_dtw))

        return errors  # DTW 오류 값을 반환

    
    # anomalies.py (16-1) - score_anomalies(17)에서 사용
    def _reconstruction_errors(self, y, y_hat, step_size=1, score_window=10, smoothing_window=0.01, smooth=True, rec_error_type='point'):
        """재구성 오류 배열을 계산함.
        예상 값과 예측 값 간의 차이를 재구성 오류 유형에 따라 계산함.
        Args:
        y (ndarray): 실제 값.
        y_hat (ndarray): 예측 값. 각 타임스탬프는 여러 예측을 가짐.
        step_size (int):
        선택사항. 예측 값의 윈도우 사이의 단계 수를 나타냄.
        주어지지 않으면 1이 사용됨.
        score_window (int):
        선택사항. 점수가 계산되는 윈도우의 크기.
        주어지지 않으면 10이 사용됨.
        smoothing_window (float 또는 int):
        선택사항. 부드럽게 하는 윈도우의 크기. float인 경우 y의 총 길이의 비율로 표현됨.
        주어지지 않으면 0.01이 사용됨.
        smooth (bool):
        선택사항. 반환된 오류가 부드럽게 되어야 하는지 여부를 나타냄.
        주어지지 않으면 `True`가 사용됨.
        rec_error_type (str):
        선택사항. 재구성 오류 유형 ``["point", "area", "dtw"]``.
        주어지지 않으면 "point"가 사용됨.
        Returns:
        ndarray: 재구성 오류 배열.
        """
        # smoothing_window가 float인 경우, y의 길이에 비례하여 정수로 변환하고 최대 200으로 제한
        if isinstance(smoothing_window, float):
            smoothing_window = min(math.trunc(len(y) * smoothing_window), 200)
        
        true = []  # 실제 값들을 저장할 리스트 초기화
        
        # y의 첫 번째 요소를 true 리스트에 추가
        for i in range(len(y)):
            true.append(y[i][0])
        
        # 마지막 윈도우의 내용을 포함하여 true 리스트에 추가
        for it in range(len(y[-1]) - 1):
            true.append(y[-1][it + 1])
        
        predictions = []  # 예측 값들을 저장할 리스트 초기화
        predictions_vs = []  # 예측 값들의 통계 정보를 저장할 리스트 초기화
        pred_length = y_hat.shape[1]  # y_hat의 두 번째 차원의 길이 저장
        num_errors = y_hat.shape[1] + step_size * (y_hat.shape[0] - 1)  # 총 오류 수 계산
        
        # 예측 값들을 계산
        for i in range(num_errors):
            intermediate = []  # 중간 예측 값들을 저장할 리스트 초기화
            for j in range(max(0, i - num_errors + pred_length), min(i + 1, pred_length)):
                intermediate.append(y_hat[i - j, j])  # 중간 예측 값들을 intermediate 리스트에 추가
            
            if intermediate:
                predictions.append(np.average(intermediate, axis=0))  # 중간 예측 값들의 평균을 predictions 리스트에 추가
                # intermediate 리스트의 통계 정보를 predictions_vs 리스트에 추가
                predictions_vs.append([[np.min(np.asarray(intermediate)), 
                                        np.percentile(np.asarray(intermediate), 25), 
                                        np.percentile(np.asarray(intermediate), 50), 
                                        np.percentile(np.asarray(intermediate), 75), 
                                        np.max(np.asarray(intermediate))]])

        true = np.asarray(true)  # true 리스트를 numpy 배열로 변환
        predictions = np.asarray(predictions)  # predictions 리스트를 numpy 배열로 변환
        predictions_vs = np.asarray(predictions_vs)  # predictions_vs 리스트를 numpy 배열로 변환
        
        # 재구성 오류 계산
        if rec_error_type.lower() == "point":
            errors = self._point_wise_error(true, predictions)          # 점 단위 오류 계산
        elif rec_error_type.lower() == "area":
            errors = self._area_error(true, predictions, score_window)  # 면적 단위 오류 계산
        elif rec_error_type.lower() == "dtw":
            errors = self._dtw_error(true, predictions, score_window)   # 동적 타임 워핑 오류 계산
        
        # 부드럽게 처리
        if smooth:
            errors = pd.Series(errors).rolling(smoothing_window, center=True, 
                                               min_periods=smoothing_window // 2).mean().values  # 오류 값을 부드럽게 처리
        
        return errors, predictions_vs  # 오류 값들과 예측 값들의 통계 정보를 반환
    
    # anomalies.py (17-1)
    def score_anomalies(self, y, y_hat, critic, index, score_window=10, critic_smooth_window=None, 
                    error_smooth_window=None, smooth=True, rec_error_type="point", comb="mult", lambda_rec=0.5):
        """anomaly score 배열을 계산.
        anomaly score는 재현 오류와 critic 점수를 합하여 계산
        Args:
        y (ndarray): 실제 값 Ground truth.
        y_hat (ndarray): 예측 값. 각 시간마다 여러 예측 값을 가짐
        index (ndarray): 각 y에 대한 시간 인덱스(window의 시작 위치)
        critic (ndarray): Critic 점수. 각 시점마다 여러 critic 점수를 가짐
        score_window (int): 선택사항. 점수가 계산되는 window의 사이즈. 지정되지 않으면 10이 사용됨.
        critic_smooth_window (int): 선택사항. critic 점수에 평활화가 적용되는 window의 크기. 지정되지 않으면 200이 사용됨.
        error_smooth_window (int): 선택사항. error에 평활화가 적용되는 window의 크기. 지정되지 않으면 200이 사용됨.
        smooth (bool): 선택사항. error를 평활화할지 여부. 지정되지 않으면 `True`가 사용됨.
        rec_error_type (str): 선택사항. reconstruction error를 계산하는 방법. `["point", "area", "dtw"]` 중 하나. 지정되지 않으면 'point'가 사용됨.
        comb (str): 선택사항. critic과 reconstruction error를 결합하는 방법. `["mult", "sum", "rec"]` 중 하나. 지정되지 않으면 'mult'가 사용됨.
        lambda_rec (float): 선택사항. `comb="sum"`일 때, lambda 가중합을 사용하여 점수를 결합하는 데 사용됨. 지정되지 않으면 0.5가 사용됨.
        
        Returns:
            ndarray: anomaly score 배열.
        """
        
        # critic_smooth_window와 error_smooth_window의 기본값을 설정
        # trunc : 내림
        # y.shape[0] : Time series의 길이 - 윈도우 사이즈
        critic_smooth_window = critic_smooth_window or math.trunc(y.shape[0] * 0.01)
        error_smooth_window = error_smooth_window or math.trunc(y.shape[0] * 0.01)
        
        # step_size는 1로 설정 (이동 단위)
        step_size = 1
        
        # 실제 데이터의 인덱스를 리스트로 변환
        true_index = list(index)
        
        # 실제 값을 저장할 리스트 초기화
        true = []
        
        # 실제 값을 리스트에 저장 (각 시간 지점의 첫 번째 값을 저장)
        for i in range(len(y)):
            true.append(y[i][0])
        
        # 마지막 window의 내용도 포함 (마지막 window의 첫 번째 이후 값을 저장)
        for it in range(len(y[-1]) - 1):
            true.append(y[-1][it + 1])
            # 인덱스 확장 (마지막 window 부분 포함)
            true_index.append((index[-1] + it + 1))
        
        # 인덱스를 배열로 변환 (전체 데이터 시퀀스를 포함)
        true_index = np.array(true_index)
        
        # critic 점수를 예측 길이에 맞춰 확장
        critic_extended = list()
        for c in critic:
            # critic 점수를 예측 값의 길이에 맞추어 반복한 후 리스트에 확장
            critic_extended.extend(np.repeat(c, y_hat.shape[1]).tolist())
        
        # critic 점수를 배열 형태로 전환하고 재구성
        critic_extended = np.asarray(critic_extended).reshape((-1, y_hat.shape[1]))
        
        # KDE(커널 밀도 추정)로 계산된 critic 점수의 최대 값을 저장할 리스트 초기화
        critic_kde_max = []
        
        # 예측 값의 길이
        pred_length = y_hat.shape[1]
        
        # 에러 수 (예측 값의 길이 + (예측 스텝 크기 * (예측 값 개수 - 1)))
        num_errors = y_hat.shape[1] + step_size * (y_hat.shape[0] - 1)
        
        # critic 점수를 KDE로 계산하여 최대 값 선택
        for i in range(num_errors):
            # 중간에 사용할 critic 점수를 저장할 리스트 초기화
            critic_intermediate = []
            
            # 중간 critic 점수를 수집
            for j in range(max(0, i - num_errors + pred_length), min(i + 1, pred_length)):
                critic_intermediate.append(critic_extended[i - j, j])
            
            # critic_intermediate에 1개 이상의 값이 있을 경우 KDE로 최대 값을 계산
            if len(critic_intermediate) > 1:
                discr_intermediate = np.asarray(critic_intermediate)
                try:
                    # KDE로 추정된 값 중 최대 값을 선택하여 저장
                    critic_kde_max.append(discr_intermediate[np.argmax(stats.gaussian_kde(discr_intermediate)(critic_intermediate))])
                except np.linalg.LinAlgError:
                    # 예외 발생 시 중간 값 저장
                    critic_kde_max.append(np.median(discr_intermediate))
            else:
                # 1개 이하일 경우 중간 값 저장
                critic_kde_max.append(np.median(np.asarray(critic_intermediate)))
        
        # critic 점수 계산(# 11 참고)
        critic_scores = self._compute_critic_score(critic_kde_max, critic_smooth_window)
        
        # 재현 에러 계산(# 16 참고)
        rec_scores, predictions = self._reconstruction_errors(y, y_hat, step_size, score_window, error_smooth_window, smooth, rec_error_type)
        
        # 재현 에러를 표준화(z-score)
        rec_scores = stats.zscore(rec_scores)
        
        # 재현 에러 값을 0 이상으로 클립하고 1을 더해줌
        rec_scores = np.clip(rec_scores, a_min=0, a_max=None) + 1
        
        # 두 점수를 결합
        if comb == "mult":
            # critic 점수와 재현 에러를 곱함
            final_scores = np.multiply(critic_scores, rec_scores)
        elif comb == "sum":
            # (1-람다 가중합)*(critic 점수 -1) + 람다 가중합*(재현 에러 - 1)
            final_scores = (1 - lambda_rec) * (critic_scores - 1) + lambda_rec * (rec_scores - 1)
        elif comb == "rec":
            # 재현 에러를 그대로 사용
            final_scores = rec_scores
        else:
            # 잘못된 인자값 입력 시 오류 출력
            raise ValueError('Unknown combination specified {}, use "mult", "sum", or "rec" instead.'.format(comb))
        
        # true 값을 2차원 리스트로 변환
        true = [[t] for t in true]
        
        # 최종 anomaly score 배열, 인덱스, 실제 값, 예측 값 반환
        return final_scores, true_index, true, predictions
