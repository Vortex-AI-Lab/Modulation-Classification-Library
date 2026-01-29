import unittest

from utils.dataset import RML2016aDataLoader, RML2016bDataLoader, RML2018aDataLoader


class DataSetConfigs(object):
    """数据集配置类"""

    def __init__(self, dataset: str, file_path: str, root_path: str) -> None:
        self.batch_size = 128
        self.num_workers = 0
        self.shuffle = True

        self.snr = 0

        self.split_ratio = 0.6

        self.dataset = dataset
        self.file_path = file_path
        self.root_path = root_path


class TestDataset(unittest.TestCase):
    """测试加载数据集的各种类方法"""

    def test_load_RML2016a(self) -> None:
        configs = DataSetConfigs(
            dataset="RML2016a",
            file_path="./dataset/RML2016.10a_dict.pkl",
            root_path=None,
        )
        train_loader, val_loader, test_loader = RML2016aDataLoader(configs).load()

        # 获取用于正向传播的数据
        for i, (data, label) in enumerate(train_loader):
            break

        n_channels = data.shape[1]
        seq_len = data.shape[2]

        # 检验数据的格式是否正确
        self.assertEqual(n_channels, 2)
        self.assertEqual(seq_len, 128)


if __name__ == "__main__":
    unittest.main()
