from utils import count_values, majority
import numpy as np
from queue import SimpleQueue


def pruning(node, node_class):
    node.subtree = {}
    node.is_leaf = True
    node.leaf_class = node_class
    node.leaf_num = 1
    node.hight = 1


def pre_pruning(tree, Xtrain, ytrain, Xval, yval):
    """
    深度优先遍历树，进行剪枝

    Parameters
    ----------
    tree : Node
        decision tree
    Xtrain : np.ndarray
        training data
    ytrain : np.ndarray
        training data
    Xval : np.ndarray
        2D-array
    yval : np.ndarray
        1D-array
    """
    if tree.is_leaf:
        return

    if tree.is_continuous:  # 连续型特征
        mask = Xtrain[:, tree.feature_index] < tree.split_value
        left_table = count_values(ytrain[mask])
        right_table = count_values(ytrain[~mask])

        left_class = majority(left_table)
        right_class = majority(right_table)

        maskval = Xval[:, tree.feature_index] < tree.split_value
        m = yval.shape[0]
        acc_before_pruning = ((yval[maskval] == left_class).sum(
        ) + (yval[~maskval] == right_class).sum()) / m

        root_table = count_values(ytrain)
        root_class = majority(root_table)
        acc_after_pruning = (yval == root_class).mean()

        if acc_before_pruning > acc_after_pruning:
            # 剪枝之前（划分之后）的泛化能力 强于 剪枝之后（不划分）
            # 当前节点不进行剪枝（要进行划分，不改动）
            # 左右子树节点进行剪枝判断
            pre_pruning(tree.subtree[" < %.3f" % tree.split_value],
                        Xtrain[mask], ytrain[mask], Xval[maskval], yval[maskval])
            pre_pruning(tree.subtree[" >= %.3f" % tree.split_value],
                        Xtrain[~mask], ytrain[~mask], Xval[~maskval], yval[~maskval])

            tree.leaf_num = tree.subtree[" < %.3f" % tree.split_value].leaf_num + \
                tree.subtree[" >= %.3f" % tree.split_value].leaf_num
            tree.hight = max(tree.subtree[" < %.3f" % tree.split_value].hight,
                             tree.subtree[" >= %.3f" % tree.split_value].hight) + 1

        else:
            # 需要进行剪枝（即，不进行划分）
            # 删掉所有子树
            # 将决策节点改成叶子节点
            pruning(tree, root_class)
            return
    else:  # 离散型特征
        acc_before_pruning = 0.0
        m = yval.shape[0]
        mask_list = []
        maskval_list = []
        for key in tree.subtree.keys():
            mask = Xtrain[:, tree.feature_index] == key
            maskval = Xval[:, tree.feature_index] == key

            subtree_table = count_values(ytrain[mask])
            subtree_class = majority(subtree_table)

            acc_before_pruning += (yval[maskval] == subtree_class).mean()

            mask_list.append(mask)
            maskval_list.append(maskval_list)

        acc_before_pruning /= m

        root_table = count_values(ytrain)
        root_class = majority(root_table)
        acc_after_pruning = (yval == root_class).mean()

        if acc_before_pruning > acc_after_pruning:
            # 剪枝之前（划分之后）的泛化能力 强于 剪枝之后（不划分）
            # 当前节点不进行剪枝（要进行划分，不改动）
            # 子树节点进行剪枝判断
            tree.leaf_num = 0
            tree.hight = 1
            for key, mask, maskval in zip(tree.subtree.keys(), mask_list, maskval_list):
                pre_pruning(tree.subtree[key], Xtrain[mask],
                            ytrain[mask], Xval[maskval], yval[maskval_list])
                tree.leaf_num += tree.subtree[key].leaf_num
                tree.hight = max(tree.hight - 1, tree.subtree[key].hight) + 1
        else:
            pruning(tree, root_class)


def post_pruning(tree, Xtrain, ytrain, Xval, yval):

    class Step(object):

        def __init__(self, parent, tree):
            self.parent = parent
            self.tree = tree

    S = []
    Q = SimpleQueue()

    if tree == None or tree.is_leaf:
        return

    Q.put(Step(None, tree))
    while not Q.empty():
        tmp = Q.get()
        S.append(tmp)

        for key in tmp.tree.subtree.keys():
            if not tmp.tree.subtree[key].is_leaf:
                new_step = Step(tmp.tree, tmp.tree.subtree[key])
                Q.put(new_step)

    while 0 != len(S):
        step = S.pop()

        if step.tree.is_continuous:  # 连续型特征
            feature_index = step.tree.feature_index
            split_value = step.tree.split_value
            mask = Xtrain[:, feature_index] < split_value
            left_table = count_values(ytrain[mask])
            right_table = count_values(ytrain[~mask])

            left_class = majority(left_table)
            right_class = majority(right_table)

            maskval = Xval[:, feature_index] < split_value
            m = yval.shape[0]
            acc_before_pruning = ((yval[maskval] == left_class).sum(
            ) + (yval[~maskval] == right_class).sum()) / m

            root_table = count_values(ytrain)
            root_class = majority(root_table)
            acc_after_pruning = (yval == root_class).mean()

            if acc_before_pruning <= acc_after_pruning:
                # 剪枝之后的泛化能力 强于 剪枝之前
                # 应该要进行剪枝
                if step.parent:
                    step.parent.leaf_num -= (step.tree.leaf_num - 1)
                pruning(step.tree, root_class)
                if step.parent:
                    step.parent.hight = 1
                    for key in step.parent.subtree.keys():
                        step.parent.hight = max(
                            step.parent.hight - 1, step.parent.subtree[key].hight) + 1
        else:  # 离散型特征
            feature_index = step.tree.feature_index
            acc_before_pruning = 0.0
            m = yval.shape[0]
            for key in step.tree.subtree.keys():
                mask = Xtrain[:, feature_index] == key
                maskval = Xval[:, feature_index] == key

                subtree_table = count_values(ytrain[mask])
                subtree_class = majority(subtree_table)

                acc_before_pruning += (yval[maskval] == subtree_class).mean()
            acc_before_pruning /= m

            root_table = count_values(ytrain)
            root_class = majority(root_table)
            acc_after_pruning = (yval == root_class).mean()

            if acc_before_pruning <= acc_after_pruning:
                # 剪枝之后的泛化能力 强于 剪枝之前
                # 要对当前节点进行剪枝
                if step.parent:
                    step.parent.leaf_num -= (step.tree.leaf_num - 1)
                pruning(step.tree, root_class)
                if step.parent:
                    step.parent.hight = 1
                    for key in step.parent.subtree.keys():
                        step.parent.hight = max(
                            step.parent.hight, step.parent.subtree[key].hight) + 1
