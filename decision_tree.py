import numpy as np


class TreeNode():
    def __init__(self, depth, devision_method):
        self.left = None
        self.right = None
        self.depth = depth
        self.devision_method = devision_method

    def create_child_node(self, data, target, max_depth=3):
        """新しい子ノードを作成しデータを付与
        """
        if self.depth == self.max_depth:
            return
        feature_idx, threshold = self.calc_best_threshold(data, target)

        left_data, left_target, right_data, right_target = \
            self.split(data, target, feature_idx, threshold)

        self.left = TreeNode(self.depth + 1)
        self.right = TreeNode(self.depth + 1)

        self.left.create_child_node(left_data, left_target)
        self.right.create_child_node(right_data, right_target)

    def calc_best_threshold(self, data, target):
        """閾値を計算する関数
        """
        pass

    def split(self, data, target, feature_idx, threshold):
        """閾値でデータを分割
        """
        pass


class DecisionTreeClassifier():
    def __init__(self, max_depth, devide_method):
        self.max_depth = max_depth
        self.devide_method = devide_method

    def fit(self, x: np.ndarray, y: np.ndarray):
        root = TreeNode(depth=0)
        root.create_child_node(x, y, self.max_depth)
