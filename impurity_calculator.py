import abc


class ImpurityCalculator(object, metaclass=abc.ABCMeta):
    @abc.abstractclassmethod
    def calc(self, num_class_array, num_element):
        pass


class GiniImpurityCalculator(ImpurityCalculator):
    def calc(self, num_class_element_array):
        """ジニ不純度を計算

        Args:
            num_class_element_array ([np.ndarray]): 各クラスの要素数を持つ配列

        Returns:
            [type]: ジニ不純度
        """
        gini_impurity = 1
        num_all_class_element = sum(num_class_element_array)
        for num_class_element in num_class_element_array:
            gini_impurity -= (num_class_element / num_all_class_element) ** 2
        return gini_impurity


if __name__ == '__main__':
    import numpy as np
    gini_impurity_calculator = GiniImpurityCalculator()

    num_class_element_array = np.array([50, 50, 21])
    num_class_element_array2 = np.array([50, 50, 44])

    print('1', gini_impurity_calculator.calc(num_class_element_array)
          * (sum(num_class_element_array) / 150))
    print('2', gini_impurity_calculator.calc(num_class_element_array2)
          * (sum(num_class_element_array2) / 150))
