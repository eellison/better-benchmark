import unittest

import torch

from byte_accounting import count_bytes_effective, count_bytes_naive


class TestEffectiveByteAccounting(unittest.TestCase):
    def _bytes(self, tensor):
        return tensor.numel() * tensor.element_size()

    def test_dense_elementwise_counts_inputs_and_output(self):
        class Repro(torch.nn.Module):
            def forward(self, a, b):
                return a + b

        a = torch.randn(4, 5)
        b = torch.randn(4, 5)
        out = Repro()(a, b)

        self.assertEqual(
            count_bytes_effective(Repro(), [a, b]),
            self._bytes(a) + self._bytes(b) + self._bytes(out),
        )

    def test_embedding_charges_gathered_rows_not_full_table(self):
        class Repro(torch.nn.Module):
            def forward(self, weight, index):
                return torch.ops.aten.embedding.default(weight, index)

        weight = torch.randn(1000, 16)
        index = torch.tensor([1, 3, 7, 9], dtype=torch.int64)
        out = Repro()(weight, index)

        expected = self._bytes(out) + self._bytes(index) + self._bytes(out)
        self.assertEqual(count_bytes_effective(Repro(), [weight, index]), expected)
        self.assertLess(count_bytes_effective(Repro(), [weight, index]), count_bytes_naive([weight, index], out))

    def test_gather_charges_gathered_elements_not_full_source(self):
        class Repro(torch.nn.Module):
            def forward(self, source, index):
                return torch.ops.aten.gather.default(source, 1, index)

        source = torch.randn(2, 100)
        index = torch.tensor([[1, 3, 5], [2, 4, 6]], dtype=torch.int64)
        out = Repro()(source, index)

        expected = self._bytes(out) + self._bytes(index) + self._bytes(out)
        self.assertEqual(count_bytes_effective(Repro(), [source, index]), expected)

    def test_index_select_charges_selected_rows_not_full_source(self):
        class Repro(torch.nn.Module):
            def forward(self, source, index):
                return torch.ops.aten.index_select.default(source, 0, index)

        source = torch.randn(100, 16, dtype=torch.float16)
        index = torch.tensor([1, 3, 5, 7, 9], dtype=torch.int64)
        out = Repro()(source, index)

        expected = self._bytes(out) + self._bytes(index) + self._bytes(out)
        self.assertEqual(count_bytes_effective(Repro(), [source, index]), expected)

    def test_index_put_charges_sparse_update_write(self):
        class Repro(torch.nn.Module):
            def forward(self, index, values):
                base = torch.zeros(100, 16)
                return torch.ops.aten.index_put.default(base, [index], values, False)

        index = torch.tensor([1, 3, 5], dtype=torch.int64)
        values = torch.randn(3, 16)
        out = Repro()(index, values)

        expected = self._bytes(index) + self._bytes(values) + self._bytes(values)
        self.assertEqual(count_bytes_effective(Repro(), [index, values]), expected)
        self.assertLess(count_bytes_effective(Repro(), [index, values]), count_bytes_naive([index, values], out))

    def test_scatter_charges_sparse_update_write(self):
        class Repro(torch.nn.Module):
            def forward(self, index, src):
                base = torch.zeros(100, 16)
                return torch.ops.aten.scatter.src(base, 0, index, src)

        index = torch.tensor([[1] * 16, [3] * 16, [5] * 16], dtype=torch.int64)
        src = torch.randn(3, 16)
        out = Repro()(index, src)

        expected = self._bytes(index) + self._bytes(src) + self._bytes(src)
        self.assertEqual(count_bytes_effective(Repro(), [index, src]), expected)
        self.assertLess(count_bytes_effective(Repro(), [index, src]), count_bytes_naive([index, src], out))


if __name__ == "__main__":
    unittest.main()
