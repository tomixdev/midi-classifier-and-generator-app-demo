import math
import numpy as np
import scipy.stats
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import robust_scale

from . import data_structure_util


def no_scaling(a_ndarray):
    return a_ndarray

# 平均0、分散1にスケーリングする「Z-score normalization」
# 標準化は、データの分布が正規分布（ガウス分布）に従っている場合に特に効果的な手法である。
# ただし、必ずしもデータの分布が正規分布でなくても使えるので、データの最小値と最大値の範囲が「不」明確な場合など、
# 正規化（Min-Max法）があまり適切でないと考えられる場合にも標準化を用いるとよい。
# このため、正規化よりも標準化の方が使いやすいケースが多い。


def z_score_normalization(a_ndarray):
    assert isinstance(a_ndarray, np.ndarray)
    assert a_ndarray.size >= 2, 'z_score_normalization() needs at least 2 numbers'
    assert a_ndarray.dtype != 'bool', 'z-score normalization has to be applied for number array'

    return scipy.stats.zscore(a_ndarray)

    '''StandardScaler特徴量の平均を0、分散を1にする変換になります。 外れ値は、平均と標準偏差を計算するときに影響を及ぼします。
       特徴:
       ・データが正規分布していない（例えば、、ガウス分布）場合は良くありません。
       ・標準化は正規化よりも外れ値の影響が少ない'''


def min_max_normalization(a_ndarray, min_to_scale=0.0, max_to_scale=1.0):
    return minmax_scale(a_ndarray, feature_range=(min_to_scale, max_to_scale), axis=0)


'''
    中央値と四分位数で変換。外れ値を無視できる変換方法。中央値は0に変換になります。
    中央値を削除し、データを第1四分位から第3四分位の間の範囲でスケーリングします。
    特徴：
    ・中央値と四分位範囲保存し、新しいデータで使用できます。
    ・MinMaxScalerと比較して、RobustScalerは外れ値の影響を減らします。
'''


def robust_scaler(a_ndarray):
    assert isinstance(a_ndarray, np.ndarray)
    assert a_ndarray.size >= 2
    assert a_ndarray.dtype != 'bool'
    return robust_scale(a_ndarray)


def scale_two_ndarrays_so_the_starting_num_and_maximum_num_become_the_same(ndarray1, ndarray2):
    assert isinstance(ndarray1, np.ndarray)
    assert isinstance(ndarray2, np.ndarray)
    assert ndarray1.size >= 2
    assert ndarray2.size >= 2
    assert len(ndarray1.shape) == 1
    assert len(ndarray2.shape) == 1
    assert ndarray1.dtype != 'bool'
    assert ndarray2.dtype != 'bool'
    data_structure_util.assert_all_elements_in_ndarray_are_number(ndarray1)
    data_structure_util.assert_all_elements_in_ndarray_are_number(ndarray2)

    ndarray1 = robust_scaler(ndarray1)
    ndarray2 = robust_scaler(ndarray2)
    # ndarray1_min_max_normalized = min_max_normalization(ndarray1, min_to_scale=0.0, max_to_scale=1000.0)
    # ndarray2_min_max_normalized = min_max_normalization(ndarray2, min_to_scale=0.0, max_to_scale=1000.0)

    ndarray1_with_first_number_moved_to_zero = ndarray1 - ndarray1[0]
    ndarray2_with_first_number_moved_to_zero = ndarray2 - ndarray2[0]

    a_small_constant = 0.01
    if ndarray1_with_first_number_moved_to_zero.max() == 0 or math.isclose(ndarray1_with_first_number_moved_to_zero.max(), 0.0):
        if not (ndarray2_with_first_number_moved_to_zero.max() == 0 or math.isclose(ndarray2_with_first_number_moved_to_zero.max(), 0.0)):
            print('read1')
            ndarray1_with_first_number_moved_to_zero = np.insert(
                ndarray1_with_first_number_moved_to_zero, 0, a_small_constant)
    elif ndarray2_with_first_number_moved_to_zero.max() == 0 or math.isclose(ndarray2_with_first_number_moved_to_zero.max(), 0.0):
        if not (ndarray1_with_first_number_moved_to_zero.max() == 0 or math.isclose(ndarray1_with_first_number_moved_to_zero.max(), 0.0)):
            print('read2')
            ndarray2_with_first_number_moved_to_zero = np.insert(
                ndarray2_with_first_number_moved_to_zero, 0, a_small_constant)
    else:
        pass

    scaled_ndarray1 = ndarray1_with_first_number_moved_to_zero * \
        ndarray2_with_first_number_moved_to_zero.max()
    scaled_ndarray2 = ndarray2_with_first_number_moved_to_zero * \
        ndarray1_with_first_number_moved_to_zero.max()

    # scaled_ndarray1 = minmax_scale(scaled_ndarray1)
    # scaled_ndarray2 = minmax_scale(scaled_ndarray2)

    return scaled_ndarray1, scaled_ndarray2


"""


if __name__ == '__main__':
    _a = np.array([0, 9, 2, 3, 5, 8, 4])
    print(robust_scaler(_a))

    raise Exception('Below are gomi codes.....')

    def _data_scalers_common_arg_formatting(undecorated_func):
        @functools.wraps(undecorated_func)
        def wrapper(*args, **kwargs):
            assert len(args) == 1, "function's argument should just be X (just a vector)."
            X = args[0]
            X = gh.convert_to_ndarray_if_list(X)
            assert isinstance(X, np.ndarray)
            X = X.reshape(-1, 1)
            return undecorated_func(X)

        return wrapper


    '''
    正規化(normalization)とは、特徴量の値の範囲を一定の範囲におさめる変換になります。主に[0, 1]か、[-1, 1]の範囲内におさめることが多いです。データの分布を変えない手法です。
    特徴：
    ・分かりやすい手法
    ・新しいデータでは異なるスケールである可能性
    ・外れ値は影響が高い
    '''


    @_data_scalers_common_arg_formatting
    def _min_max_scaler(X):
        return sklearn.preprocessing.MinMaxScaler().fit_transform(X)


    '''
    StandardScaler特徴量の平均を0、分散を1にする変換になります。 外れ値は、平均と標準偏差を計算するときに影響を及ぼします。
    特徴:
    ・データが正規分布していない（例えば、、ガウス分布）場合は良くありません。
    ・標準化は正規化よりも外れ値の影響が少ない
    '''


    @_data_scalers_common_arg_formatting
    def _standard_scaler(X):
        return sklearn.preprocessing.StandardScaler().fit_transform(X)


    '''
    中央値と四分位数で変換。外れ値を無視できる変換方法。中央値は0に変換になります。
    中央値を削除し、データを第1四分位から第3四分位の間の範囲でスケーリングします。
    特徴：
    ・中央値と四分位範囲保存し、新しいデータで使用できます。
    ・MinMaxScalerと比較して、RobustScalerは外れ値の影響を減らします。
    '''


    @_data_scalers_common_arg_formatting
    def _robust_scaler(X):
        return sklearn.preprocessing.RobustScaler(quantile_range=(25, 75)).fit_transform(X)


    '''
    PowerTransformerは、分散を安定化し、歪度を最小化するための最適なパラメータは、最尤によって推定されます。
    平均は0、標準偏差は1になります。
    Box-Coxは入力データが厳密に正であることを要求しますが、Yeo-Johnsonは正または負の両方のデータをサポートしています。
    '''


    @_data_scalers_common_arg_formatting
    def _power_transformer_yeo_johnson(X):
        return sklearn.preprocessing.PowerTransformer(method='yeo-johnson').fit_transform(X)


    '''
    QuantileTransformerは非線形変換を適用します。
    Uniform
    ・各特徴の確率密度関数は、均一分布またはガウス分布推定されます。
    ・値が[0、1]の範囲におさめる変換になります。
    ・外れ値の影響がなくなります。
    ・歪度は0になります。
    '''


    @data_scalers_common_arg_formatting
    def quantile_transformer_uniform(X):
        return sklearn.preprocessing.QuantileTransformer(output_distribution='uniform').fit_transform(X)


    '''
    QuantileTransformerは非線形変換を適用します。
    Normal
    ・正規分布曲線になります。
    '''


    @_data_scalers_common_arg_formatting
    def _quantile_transformer_normal(X):
        return sklearn.preprocessing.QuantileTransformer(output_distribution='normal').fit_transform(X)


    '''
    対数変換、偏ったデータに対処するために広く使用されている方法
    対数は、分布の形状に大きな影響を与える強力な変換です。
    '''


    @_data_scalers_common_arg_formatting
    def _log_transformer(X):
        return np.log(X)


    '''
    平方根法は通常、データが適度に歪んでいる場合に使用されます。正の値にのみ適用されます。
    '''


    @_data_scalers_common_arg_formatting
    def _square_root_transformer(X):
        assert (X > 0).all()
        return np.sqrt(X)


    def _softmax(X):
        # TODO: なんか、ndarrayのままだと計算がうまく行かないから、一回Listにへんかんして、その後ndarrayに変換し直している。非効率なので改善が必要。
        X = gh.convert_to_list_if_ndarray(X)
        a_list = np.exp(X) / sum(np.exp(X))
        return gh.convert_to_ndarray_if_list(a_list)


    _y_to_be_scaled = overtone_variety

    graph_generators.plot_one_x_y_graph(Y=min_max_scaler(y_to_be_scaled), graph_title=gh.varnameof(min_max_scaler))
    graph_generators.plot_one_x_y_graph(Y=standard_scaler(y_to_be_scaled), graph_title=gh.varnameof(standard_scaler))
    graph_generators.plot_one_x_y_graph(Y=robust_scaler(y_to_be_scaled), graph_title=gh.varnameof(robust_scaler))
    graph_generators.plot_one_x_y_graph(Y=power_transformer_yeo_johnson(y_to_be_scaled),
                                        graph_title=gh.varnameof(power_transformer_yeo_johnson))
    graph_generators.plot_one_x_y_graph(Y=quantile_transformer_uniform(y_to_be_scaled),
                                        graph_title=gh.varnameof(quantile_transformer_uniform))
    graph_generators.plot_one_x_y_graph(Y=quantile_transformer_normal(y_to_be_scaled),
                                        graph_title=gh.varnameof(quantile_transformer_normal))
    graph_generators.plot_one_x_y_graph(Y=log_transformer(y_to_be_scaled), graph_title=gh.varnameof(log_transformer))
    graph_generators.plot_one_x_y_graph(Y=square_root_transformer(y_to_be_scaled),
                                        graph_title=gh.varnameof(square_root_transformer))
    graph_generators.plot_one_x_y_graph(Y=softmax(y_to_be_scaled), graph_title=gh.varnameof(softmax))























    '''
    TODO: Sep 2, 2022
    以下のFunctionについて、min_threshholdとmax_threshholdを絶対値として引数にとるのではなく、98 percentileより上と、2 percentileより下のデータを、
    98 percentileのデータと、2 percentileのデータと同じものにする、みたいな関数が良いかもしれない。
    '''
    '''
    not tested as of July 15, 2022
    '''
    def create_a_new_ndarray_with_min_threshold_and_max_threshold(a_ndarray_or_list, min_threshold, max_threshold):
        warnings.warn('Deprecated!!', DeprecationWarning, stacklevel=2)


        general_helpers.raise_exception_if_not_a_valid_vector_ndarray_or_list(a_ndarray_or_list)
        general_helpers.raise_exception_if_not_a_number(min_threshold)
        general_helpers.raise_exception_if_not_a_number(max_threshold)

        a_ndarray = general_helpers.convert_to_ndarray_if_list(a_ndarray_or_list)
        a_ndarray = a_ndarray.copy()

        general_helpers.raise_exception_if_not_a_number(min_threshold)
        the_min_value_of_given_array = np.amin(a_ndarray)
        for i in range(0, a_ndarray.size):
            if a_ndarray[i] < the_min_value_of_given_array:
                a_ndarray[i] = min_threshold

        general_helpers.raise_exception_if_not_a_number(max_threshold)
        the_max_value_of_given_array = np.amax(a_ndarray)
        for i in range(0, a_ndarray.size):
            if a_ndarray[i] > the_max_value_of_given_array:
                a_ndarray[i] = max_threshold

        return a_ndarray


    '''
    This scaler scales an array in the manner similar to "scale" object in Max(MSP).
    If there is a number that is below min_before_scaled, the number is mapped to min_after_scaled.
    If therer is a number that is above max_before_scaled, the number is mapped to max_after_scaled
    Run time of this function is probably o(n), because I am iterating over the array
    '''
    def scale_number_or_list_or_ndarray(a_number_or_list_or_ndarray,
                                        min_after_scaled=0,
                                        max_after_scaled=1):

        warnings.warn('Deprecated!!', DeprecationWarning, stacklevel=2)


        # ----------------------------------------------------------
        if general_helpers.is_number(a_number_or_list_or_ndarray):
            a_ndarray = np.array([])
            a_ndarray = np.append(a_ndarray, [a_number_or_list_or_ndarray])
            #a_one_number_list = [a_number_or_list_or_ndarray]
            #a_ndarray = np.ndarray(a_one_number_list)
        elif general_helpers.is_valid_vector_ndarray_or_list(a_number_or_list_or_ndarray):
            a_ndarray = general_helpers.convert_to_ndarray_if_list()
        else:
            raise Exception('The argument is not valid. It needs to be number, list, OR ndarray')

        # ----------------------------------------------------
        # https://gist.github.com/CMCDragonkai/6444bf7ea41b4f43766abb9f4294cd69
        # scale an input array-like to a mininum and maximum number
        # the input array must be of a floating point array
        # if you have a non-floating point array, convert to floating using `astype('float')`
        # this works with n-dimensional arrays
        # it will mutate in place
        # min and max can be integers
        a_ndarray = a_ndarray.astype(np.float)
        a_ndarray += -(np.min(a_ndarray))
        a_ndarray /= np.max(a_ndarray) / (max_after_scaled - min_after_scaled)
        a_ndarray += min_after_scaled

        # ----------------------------------------------------
        if a_ndarray.size == 1:  # if the input to this function was a number, return number
            return a_ndarray[0]
        else:
            return a_ndarray

    '''
    !!!!! DEPRECIATED BAD FUNCTION NAME!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # 最小値0～最大値1にスケーリングする「Min-Max normalization」
    # 正規化（Min-Max法）は、特にデータの最小値と最大値の範囲が明確な場合に適した手法である。
    # ただし、外れ値に敏感なため、大きい外れ値が存在する場合は標準化を使った方がよい。
    # ちなみに、外れ値に対してよりロバスト（頑健）に正規化する方法の一つとして、
    # データの中央値からの偏差（＝中央値を中心0にした場合の値）を四分位範囲（＝第3四分位数－第1四分位数）で割るRobustScalerもある。
    '''
    def get_min_max_normalization(a_ndarray):
        warnings.warn('Deprecated!!', DeprecationWarning, stacklevel=2)

        #return sklearn.preprocessing.minmax_scale(x, axis=axis)
        assert a_ndarray.dtype != 'bool', 'min-max normalization has to be applied for number array'

        a_ndarray = a_ndarray.reshape(-1, 1)
        scaler = preprocessing.MinMaxScaler()
        return scaler.fit_transform(a_ndarray).reshape(1, -1).squeeze()








    # inverse_logit, convert a wide range value to a 0-1 value
    # accept a scholar number or an array
    def inv_logit_scaled(a_ndarray):
        warnings.warn('Deprecated!!', DeprecationWarning, stacklevel=2)


        # invert_logit に6.5を代入すると、0.998..となるので、6.5がa_ndarrayの最大値となるように調整する。
        # そして、-06.5がan_arrayの最小値となるようにする。
        a_ndarray = minmax_scale(a_ndarray, feature_range=(-6.5, 6.5), axis=0)
        return expit(a_ndarray)


    # logit, convert a 0 - 1 value to a wider range value
    def my_logit(a_ndarray):
        warnings.warn('Deprecated!!', DeprecationWarning, stacklevel=2)


        return logit(a_ndarray)


    def write_a_list_text_file(a_list, relative_path_and_name):
        warnings.warn('Deprecated!!', DeprecationWarning, stacklevel=2)


        list_related_helpers.write_a_list_text_file(a_list, relative_path_and_name)


    # 一般に機械学習の分野では、
    # 1.全データの平均と分散を求め、
    # 2. 個々のデータをその平均で減算し、
    # さらに分散の平方根で除算する前処理を行うことが多い。
    # 入力の次元ごとの変動スケールが揃っていると精度の良い学習ができるからである。これを標準化という。
    # データ残帯の平均がゼロ、分散が１となることを意図する。
    def common_data_preprocessing_in_ml(a_list_or_ndarray):
        warnings.warn('Deprecated!!', DeprecationWarning, stacklevel=2)


        if isinstance(a_list_or_ndarray, list):
            a_ndarray = np.array(a_list_or_ndarray)
        elif isinstance(a_list_or_ndarray, np.ndarray):
            a_ndarray = a_list_or_ndarray
        else:
            raise Exception("the argument type needs to be a list or a ndarray")

        the_mean = np.mean(a_ndarray)
        the_variance = np.var(a_ndarray)

        a_ndarray = a_ndarray - the_mean
        a_ndarray = a_ndarray / the_variance
        a_ndarray = a_ndarray.round(decimals=6)

        return a_ndarray


"""
