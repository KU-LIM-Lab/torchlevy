import torch

class TorchDictionary:
    """The TorchDictionary class offers functionality similar to Python's built-in dictionary type, including the ability to save and load data
    , but with the added requirement that both the keys and values must be instances of the torch.Tensor data type.
    Despite its name, the TorchDictionary does not utilize hashing in its data retrieval processes.
    Instead, it leverages parallelized retrieval and linear interpolation techniques to achieve improved performance and accuracy.
    These advanced capabilities set the TorchDictionary apart as a highly efficient and effective data storage and retrieval solution.
    """
    def __init__(self, keys: torch.Tensor, values: torch.Tensor):
        """Initialize TorchDictionary instance through saving key and corresponding values values.
        Both keys and values must be torch.Tensor type and have the same length"""

        if keys.shape != values.shape:
            raise RuntimeError(f"length of keys and values must be same :{keys.shape} != {values.shape}")

        sorted_keys, _ = torch.sort(keys)
        if not torch.equal(sorted_keys, keys):
            raise RuntimeError("keys must be sorted")


        self.keys = keys
        if torch.is_complex(values):
            self.values = torch.real(values)
        else:
            self.values = values

    def get(self, x: torch.Tensor, linear_approx=True):
        """if there is no matching key, return one of two closest values
        if linear_approx=True, return linear approximation of two closest values."""
        right_index = torch.bucketize(x.ravel(), self.keys)
        right_index[right_index == self.keys.size(dim=0)] -= 1

        if not linear_approx:
            return self.values[right_index].reshape(x.shape)
        else:
            left_index = right_index - 1
            left_index[left_index < 0] = 0

            right_value = self.values[right_index]
            left_value = self.values[left_index]

            right_keys = self.keys[right_index]
            left_keys = self.keys[left_index]

            # linear approxmiation
            ret = (right_keys - x.ravel()) / (right_keys - left_keys) * left_value + \
                   (x.ravel() - left_keys) / (right_keys - left_keys) * right_value
            ret[torch.isnan(ret)] = right_value[torch.isnan(ret)]
            return ret.reshape(x.shape)


def tutorial_code():
    keys, _ = torch.sort(torch.randperm(20)[:10].cuda())
    values = torch.randperm(10).cuda()

    mapping = TorchDictionary(keys, values)
    x = torch.randperm(10).cuda()
    res1 = mapping.get(x, linear_approx=True)
    res2 = mapping.get(x)

    print("keys:    ", keys)
    print("values:  ", values)
    print("x:       ", x)
    print("res1:    ", res1)
    print("res2:    ", res2)