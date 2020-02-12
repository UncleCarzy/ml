import numpy as np
from sklearn.utils.multiclass import type_of_target
from math import log2
from decisionTreePruning import pre_pruning, post_pruning
from utils import *


class Node(object):

    def __init__(self):
        # feature for splitting
        self.feature_name = None
        self.feature_index = None
        self.impurity = None
        # split value is available when the feature for splitting is a continuous one,
        # divide the set into two parts, one is >= the value, the other is < the value.
        self.is_continuous = False
        self.split_value = None

        # the dict to put different subtrees
        self.subtree = {}

        # the leaf_class representing the target is available when current node is a leaf.
        self.is_leaf = False
        self.leaf_class = None

        self.leaf_num = 0
        self.hight = 0


class DecisionTree(object):

    def __init__(self, criterion="gini", pruning=None):

        assert (criterion in ("gini", "info_gain", "gain_ratio")
                ), "criterion should be one of (gini , info_gain, gain_ratio)"
        assert (pruning in (None, "pre_pruning", "post_pruning")
                ), "pruning should be one of (None, 'pre_pruninf','post_pruning')"
        self.criterion = criterion
        self.pruning = pruning
        self.my_tree = None
        self.feature_names = None
        self.n_features = None
        self.n_target = None
        self.majority_class = None
        self.count = 0

    def fit(self, X, y, Xval, yval):
        root = Node()
        self.n_features = X.shape[1]
        feature_index_list = [i for i in range(self.n_features)]
        self.generate_tree(root, X, y, feature_index_list)
        self.my_tree = root
        if "pre_pruning" == self.pruning:
            pre_pruning(self.my_tree, X, y, Xval, yval)

        if "post_pruning" == self.pruning:
            post_pruning(self.my_tree, X, y, Xval, yval)
        return self

    def predict(self, X):
        y = [self.predict_single(row) for row in X]
        return np.array(y)

    def predict_single(self, x):
        cur_node = self.my_tree
        while not cur_node.is_leaf:
            feature_index = cur_node.feature_index
            if cur_node.is_continuous:
                if x[feature_index] >= cur_node.split_value:
                    cur_node = cur_node.subtree[" >= %.3f" %
                                                cur_node.split_value]
                else:
                    cur_node = cur_node.subtree[" < %.3f" %
                                                cur_node.split_value]
            else:
                for key, tree in cur_node.subtree.items():
                    if key == x[feature_index]:
                        cur_node = tree
                        break
        return cur_node.leaf_class

    def generate_tree(self, node, X, y, feature_index_list):
        table = count_values(y)

        # X中的样本全属于同一类
        if 1 == len(table):
            node.is_leaf = True
            node.leaf_class = table.popitem()[0]
            node.leaf_num = 1
            node.hight = 1
            return

        # 属性集为空，或者样本的在属性集上的取值全部相同
        if 0 == len(feature_index_list) or are_samples_all_same(X[:, feature_index_list]):
            node.is_leaf = True
            node.leaf_class = majority(table)
            node.leaf_num = 1
            node.hight = 1
            return

        # 如果连续型特征，需要返回 划分属性 和 划分的数值点
        # 如果离散型特征，只需要返回 划分属性
        best_feature = self.choose_best_feature_to_split(
            X, y, feature_index_list)
        node.feature_index = best_feature[0]

        if 2 == len(best_feature):  # 连续型特征
            feature_index, split_value = best_feature
            node.is_continuous = True
            node.split_value = split_value

            mask = X[:, feature_index] < split_value
            left_node = Node()
            if mask.sum() > 0:
                # 连续型属性子树还可以再用，
                self.generate_tree(
                    left_node, X[mask, :], y[mask], feature_index_list)
            else:
                left_node.is_leaf = True
                left_node.leaf_class = majority(table)

            node.subtree[" < %.3f" % split_value] = left_node

            right_node = Node()
            mask = ~mask
            if mask.sum() > 0:
                self.generate_tree(
                    right_node, X[mask, :], y[mask], feature_index_list)
            else:
                right_node.is_leaf = True
                right_node.leaf_class = majority(table)
            node.subtree[" >= %.3f" % split_value] = right_node

            node.hight = max(left_node.hight, right_node.hight) + 1
            node.leaf_num = left_node.leaf_num + right_node.leaf_num

        else:  # 离散型特征,用过的特征，子树不会再用了
            feature_index = best_feature[0]
            unique_values = np.unique(X[:, feature_index])
            new_feature_index_list = feature_index_list.copy()
            new_feature_index_list.remove(feature_index)
            for value in unique_values:
                mask = X[:, feature_index] == value
                new_node = Node()
                if mask.sum() > 0:
                    self.generate_tree(
                        new_node, X[mask, :], y[mask], new_feature_index_list)
                else:
                    new_node.is_leaf = True
                    new_node.leaf_class = majority(table)
                node.subtree[value] = new_node
                node.leaf_num += new_node.leaf_num
                node.hight = max(node.hight - 1, new_node.hight) + 1

    def choose_best_feature_to_split(self, X, y, feature_index_list):
        # 如果连续型特征，需要返回 划分属性 和 划分的数值点
        # 如果离散型特征，只需要返回 划分属性
        if "gini" == self.criterion:
            return self.choose_best_gini_index(X, y, feature_index_list)
        elif "gain_ratio" == self.criterion:
            return self.choose_best_gain_ratio(X, y, feature_index_list)
        else:
            return self.choose_best_info_gain(X, y, feature_index_list)

    def choose_best_gain_ratio(self, X, y, feature_index_list):
        # 这里直接选取增益率最大的，
        # 而不是找出信息增益高于平均水平的属性，再从中选择增益率最高的
        # 如果连续型特征，需要返回 [划分属性, 划分的数值点]
        # 如果离散型特征，只需要返回 [划分属性]
        best_feature_index = None
        best_gain_ratio = [float("-inf")]
        for feature_index in feature_index_list:
            x = X[:, feature_index]
            if "unknown" == type_of_target(x):
                x = x.astype(float)
            is_continuous = "unknown" == type_of_target(x)
            gain_ratio = self.gain_ratio(x, y, is_continuous)
            if gain_ratio[0] > best_gain_ratio[0]:
                best_gain_ratio = gain_ratio
                best_feature_index = feature_index

        best_gain_ratio[0] = best_feature_index
        return best_gain_ratio

    def choose_best_info_gain(self, X, y, feature_index_list):
        # 选择信息增益最大的，使用该属性进行划分，能使信息熵提升最大
        # 如果连续型特征，需要返回 [划分属性, 划分的数值点]
        # 如果离散型特征，只需要返回 [划分属性]
        best_feature_index = None
        best_info_gain = [float("-inf")]
        for feature_index in feature_index_list:
            x = X[:, feature_index]
            if type_of_target(x) == "unknown":
                x = x.astype(float)
            is_continuous = type_of_target(x) == "continuous"
            info_gain = self.info_gain(x, y, is_continuous)
            if info_gain[0] > best_info_gain[0]:
                best_info_gain = info_gain
                best_feature_index = feature_index
        best_info_gain[0] = best_feature_index
        return best_info_gain

    def choose_best_gini_index(self, X, y, feature_index_list):
        # 选择gini index最小的那个特征,划分得到的子集的纯度越高
        # 如果连续型特征，需要返回 [划分属性, 划分的数值点]
        # 如果离散型特征，只需要返回 [划分属性]
        best_feature_index = None
        best_gini_index = [float("inf")]
        for feature_index in feature_index_list:
            x = X[:, feature_index]
            if type_of_target(x) == "unknown":
                # 由于这里X的type由数目最多的元素决定的
                # 连续属性(object)会识别不出来,手动将object转化为float
                x = x.astype(float)
            is_continuous = type_of_target(x) == "continuous"
            gini_index = self.gini_index(x, y, is_continuous)
            # 连续型，返回[gini index, split value]
            # 离散型，返回[gini index]
            if gini_index[0] < best_gini_index[0]:
                best_gini_index = gini_index
                best_feature_index = feature_index
        best_gini_index[0] = best_feature_index
        return best_gini_index

    def gain_ratio(self, x, y, is_continuous=False):
        """
        compute gain ratio

        Parameters
        ----------
        x : np.ndarray
            1D-array, a columns of feature matrix X
        y : np.ndarray
            1D-array, targets
        is_continuous : bool, optional
            True, if x is a continuous feature,, by default False

        Returns
        -------
        list
            [best_gain_ratio, best_split_value], if x is a continuous feature
            [gain_ratio], if x is a disceret feature
        """
        m = y.shape[0]
        if is_continuous:
            ent_D = self.info_entropy(y)
            unique_values = np.unique(x)
            split_value_set = [
                (unique_values[i] + unique_values[i+1]) / 2.0 for i in range(len(unique_values)-1)]
            best_split_value = None
            best_gain_ratio = float("-inf")  # the bigger, the better.
            for split_value in split_value_set:
                mask = (x >= split_value)
                m_plus = mask.sum()
                p_plus = m_plus / m
                p_minus = 1 - p_plus
                gain = ent_D - (p_plus * self.info_entropy(y[mask]) +
                                p_minus * self.info_entropy(y[~mask]))
                IV = - (p_plus * log2(p_plus) + p_minus * log2(p_minus))
                gain_ratio = gain / IV
                if gain_ratio > best_gain_ratio:
                    best_gain_ratio = gain_ratio
                    best_split_value = split_value
            return [best_gain_ratio, best_split_value]
        else:
            table = count_values(y)
            p = np.array(list(table.values())) / m
            IV = - (p * np.log2(p)).sum()
            info_gain = self.info_gain(x, y, False)[0]
            gain_ratio = info_gain / IV
            return [gain_ratio]

    def gini_index(self, x, y, is_continuous=False):
        """
        compute gini index

        Parameters
        ----------
        x : np.ndarray
            1D-array, a columns of feature matrix X
        y : np.ndarray
            1D-array, targets
        is_continuous : bool, optional
            True, if x is a continuous feature,, by default False

        Returns
        -------
        list
            [best_gini_index, best_split_value], if x is a continuous feature
            [gini_index], if x is a disceret feature
        """
        m = y.shape[0]
        if is_continuous:  # 连续值
            unique_values = np.unique(x)
            unique_values = unique_values.tolist()
            split_value_set = [(unique_values[i] + unique_values[i+1]) /
                               2.0 for i in range(len(unique_values) - 1)]

            best_split_value = None
            best_gini_index = float("inf")  # the smaller, the better.
            for split_value in split_value_set:
                mask = x >= split_value
                m_plus = mask.sum()
                gini_index = (m_plus *
                              self.gini_value(y[mask]) + (m - m_plus) *
                              self.gini_value(y[~mask])) / m
                if gini_index < best_gini_index:
                    best_gini_index = gini_index
                    best_split_value = split_value

            return [best_gini_index, best_split_value]
        else:
            table = count_values(x)
            feature_gini = 0.0
            for key, value in table.items():
                mask = x == key
                feature_gini += value * self.gini_value(y[mask])
            gini_index = feature_gini / m
            return [gini_index]

    def gini_value(self, y):
        """
        compute gini value

        Parameters
        ----------
        y : np.ndarray
            1D-array, targets 

        Returns
        -------
        float
            gini value of a set, which represents the purity of a set.
            the smaller gini value is, the higher purity of a set is.
        """
        m = y.shape[0]
        table = count_values(y)
        p = np.array(list(table.values()))
        gini = 1 - (p ** 2).sum() / (m ** 2)
        return gini

    def info_gain(self, x, y, is_continuous=False):
        """
        compute information gain

        Parameters
        ----------
        x : np.ndarray
            1D-array, a column of X
        y : np.ndarray
            1D-array, target
        is_continuous : bool, optional
            True, if x is a continuous feature, by default False

        Returns
        -------
        list
            [best_gain, best_split_value], if x is a continuous feature
            [gain], if x is a disceret feature
        """
        m = y.shape[0]
        if is_continuous:  # continuous feature
            unique_values = np.unique(x)
            # unique_values.sort() 不排序没关系，上面的语句隐式地实现了排序
            unique_values = unique_values.tolist()
            split_value_set = [(unique_values[i] + unique_values[i+1]) /
                               2.0 for i in range(len(unique_values) - 1)]
            ent = self.info_entropy(y)
            best_split_value = None
            best_feature_ent = float("inf")
            for split_value in split_value_set:
                mask = x >= split_value
                m_plus = mask.sum()
                feature_ent = (
                    m_plus * self.info_entropy(y[mask]) + (m - m_plus) * self.info_entropy(y[~mask])) / m
                if feature_ent < best_feature_ent:
                    best_feature_ent = feature_ent
                    best_split_value = split_value
            best_gain = ent - best_feature_ent
            return [best_gain, best_split_value]
        else:  # discrete feature
            ent = self.info_entropy(y)
            feature_ent = 0.0
            table = count_values(x)
            for key, value in table.items():
                mask = x == key
                ent_dv = self.info_entropy(y[mask])
                feature_ent += value * ent_dv
            feature_ent /= m  # 只做一次除法
            gain = ent - feature_ent
            return [gain]

    def info_entropy(self, y):
        """
        compute information entropy

        Parameters
        ----------
        y : np.ndarray 
            1D array

        Returns
        -------
        float
            information entropy
        """
        m = y.shape[0]
        table = count_values(y)
        p = np.array(list(table.values()), dtype=float) / m
        ent = - (p * np.log2(p)).sum()
        return ent
