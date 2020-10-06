import numpy as np


def calc_gini_impurity(num_class_array):
    gini_impurity = 0
    num_data = sum(num_class_array)
    for num_class in num_class_array:
        gini_impurity += (num_class / num_data) * (1 - (num_class / num_data))
    return gini_impurity


class TreeNode():
    def __init__(self, depth, max_depth, calc_impurity_method):
        self.left = None
        self.right = None
        self.depth = depth
        self.max_depth = max_depth
        self.calc_impurity = calc_impurity_method

    def create_child_node(self, data, target):
        """新しい子ノードを作成しデータを付与
        """
        if self.depth == self.max_depth:
            return
        print('data', data.shape)
        feature_idx, threshold = self.calc_best_threshold(data, target)
        print('feature_idx', feature_idx, threshold)

        left_data = \
            data[:, feature_idx][data[:, feature_idx] > threshold]
        left_target = \
            target[data[:, feature_idx] > threshold]

        right_data = \
            data[:, feature_idx][data[:, feature_idx] <= threshold]
        right_target = \
            target[data[:, feature_idx] <= threshold]

        print('left', left_data.shape, np.unique(
            left_target, return_counts=True))
        self.left = TreeNode(
            depth=self.depth + 1,
            max_depth=self.max_depth,
            calc_impurity_method=self.calc_impurity
        )

        print('right', right_data.shape, np.unique(
            right_target, return_counts=True))
        self.right = TreeNode(
            depth=self.depth + 1,
            max_depth=self.max_depth,
            calc_impurity_method=self.calc_impurity)

        self.left.create_child_node(left_data, left_target)
        self.right.create_child_node(right_data, right_target)

    def calc_best_threshold(self, data, target):
        """情報利得が最大となる閾値を計算し, 返却
        """
        information_gain_max = 0

        # 親ノードの不純度を算出
        _, parent_num_class_array = np.unique(target, return_counts=True)
        parent_impurity = self.calc_impurity(parent_num_class_array)

        num_data, num_feature = data.shape
        for feature_idx in range(num_feature):
            values = data[:, feature_idx]

            for value in values:
                divided_target_list = self.split(values, target, value)

                # 子ノード合計の不純度を計算
                impurity = 0
                for divided_target in divided_target_list:
                    _, num_class_array = np.unique(
                        divided_target, return_counts=True)
                    impurity += self.calc_impurity(num_class_array)
                print('value impurity', value, impurity)

                # 親ノードからの情報利得を算出
                information_gain = parent_impurity - impurity

                if information_gain_max < information_gain:
                    information_gain_max = information_gain
                    best_threshold = (feature_idx, value)

        return best_threshold

    def split(self, data, target, threshold):
        """ある特徴量のある閾値で分割し, 分割データを返却
        """
        target_larger = target[data > threshold]
        target_smaller = target[data <= threshold]

        _, counts_larger = np.unique(target_larger, return_counts=True)
        _, counts_smaller = np.unique(target_smaller, return_counts=True)

        return counts_larger, counts_smaller


class DecisionTreeClassifier():
    def __init__(self, max_depth, calc_impurity_method):
        self.max_depth = max_depth
        self.calc_impurity_method = calc_impurity_method

    def fit(self, data: np.ndarray, target: np.ndarray):
        root = TreeNode(
            depth=0,
            max_depth=self.max_depth,
            calc_impurity_method=self.calc_impurity_method
        )
        print(data.shape)
        root.create_child_node(data, target)


if __name__ == '__main__':
    from sklearn.datasets import load_iris

    iris = load_iris()
    data = iris.data
    target = iris.target

    decision_tree_classifier = DecisionTreeClassifier(
        max_depth=3,
        calc_impurity_method=calc_gini_impurity)

    decision_tree_classifier.fit(data, target)
