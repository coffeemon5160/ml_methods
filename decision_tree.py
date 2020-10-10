import numpy as np

from impurity_calculator import GiniImpurityCalculator


class TreeNode():
    def __init__(self, depth, max_depth, impurity_calculator):
        self.left = None
        self.right = None
        self.depth = depth
        self.max_depth = max_depth
        self.impurity_calculator = impurity_calculator
        self.feature_idx = None
        self.threshold = None
        self.label = None

    def create_child_node(self, data, target):
        """新しい子ノードを作成しデータを付与
        """
        # 深さ最大 or 保持しているクラスが一つの場合, 処理を終了
        if self.depth == self.max_depth or \
                len(np.unique(target)) == 1:

            self.label = np.argmax(np.bincount(target))
            return

        self.feature_idx, self.threshold = self.calc_best_threshold(
            data, target)

        left_data = \
            data[np.where(data[:, self.feature_idx] > self.threshold)]
        left_target = \
            target[data[:, self.feature_idx] > self.threshold]

        right_data = \
            data[np.where(data[:, self.feature_idx] <= self.threshold)]
        right_target = \
            target[data[:, self.feature_idx] <= self.threshold]

        self.left = TreeNode(
            depth=self.depth + 1,
            max_depth=self.max_depth,
            impurity_calculator=self.impurity_calculator
        )

        self.right = TreeNode(
            depth=self.depth + 1,
            max_depth=self.max_depth,
            impurity_calculator=self.impurity_calculator)

        self.left.create_child_node(left_data, left_target)
        self.right.create_child_node(right_data, right_target)

    def predict(self, data):
        if self.label is not None:
            return self.label

        # print(self.threshold)
        if data[self.feature_idx] > self.threshold:
            return self.left.predict(data)

        else:
            return self.right.predict(data)

    def calc_best_threshold(self, data, target):
        """情報利得が最大となる閾値を計算し, 返却
        """
        information_gain_max = 0

        # 親ノードの不純度を算出
        _, parent_num_class_array = np.unique(target, return_counts=True)
        parent_impurity = self.impurity_calculator.calc(parent_num_class_array)

        num_data, num_feature = data.shape
        for feature_idx in range(num_feature):
            values = data[:, feature_idx]

            for i, value in enumerate(values):
                # 分割
                divided_target_list = self.split(values, target, value)

                # 子ノード合計の不純度を計算
                impurity = 0
                for divided_target in divided_target_list:

                    impurity += self.impurity_calculator.calc(
                        divided_target) * (sum(divided_target) / num_data)

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
    def __init__(self, max_depth, impurity_calculator):
        self.max_depth = max_depth
        self.impurity_calculator = impurity_calculator
        self.root = None

    def fit(self, data: np.ndarray, target: np.ndarray):
        self.root = TreeNode(
            depth=0,
            max_depth=self.max_depth,
            impurity_calculator=self.impurity_calculator
        )
        self.root.create_child_node(data, target)

    def predict(self, data: np.ndarray):
        result = []
        for element in data:
            result.append(self.root.predict(element))
        return result


if __name__ == '__main__':
    from sklearn.datasets import load_iris

    iris = load_iris()
    data = iris.data
    target = iris.target

    gini_impurity_calculator = GiniImpurityCalculator()

    decision_tree_classifier = DecisionTreeClassifier(
        max_depth=3,
        impurity_calculator=gini_impurity_calculator)

    decision_tree_classifier.fit(data, target)

    pred_result = decision_tree_classifier.predict(data)
