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


    def test_slice_charges_only_accessed_portion(self):
        class Repro(torch.nn.Module):
            def forward(self, x):
                # Only use the first half along dim 1
                sliced = torch.ops.aten.slice.Tensor(x, 1, 0, 64)
                return sliced + 1.0

        x = torch.randn(8, 128, 100)
        out_shape = (8, 64, 100)

        expected_read = 8 * 64 * 100 * 4  # only the slice is read
        expected_write = 8 * 64 * 100 * 4  # output size
        expected = expected_read + expected_write
        actual = count_bytes_effective(Repro(), [x])
        self.assertEqual(actual, expected)
        # Should be less than naive (which charges full tensor)
        out = Repro()(x)
        self.assertLess(actual, count_bytes_naive([x], out))

    def test_index_put_output_not_inherited_by_downstream(self):
        """Downstream reductions of index_put should charge actual output size."""
        class Repro(torch.nn.Module):
            def forward(self, index, values):
                base = torch.zeros(100, 16)
                updated = torch.ops.aten.index_put.default(base, [index], values, False)
                # Reduce to a small output
                return torch.ops.aten.sum.dim_IntList(updated, [0])

        index = torch.tensor([1, 3, 5], dtype=torch.int64)
        values = torch.randn(3, 16)
        out = Repro()(index, values)

        actual = count_bytes_effective(Repro(), [index, values])
        # Output should be f32[16] = 64 bytes, not the index_put size
        # Reads: index (24 bytes) + values (192 bytes)
        # Sparse read of values into index_put = 192 bytes
        # Output: f32[16] = 64 bytes
        expected = self._bytes(index) + self._bytes(values) + self._bytes(out)
        self.assertEqual(actual, expected)

    def test_id_reuse_does_not_cause_spurious_sparse_write(self):
        """Freed intermediate tensor IDs should not pollute final output."""
        class Repro(torch.nn.Module):
            def forward(self, src, index):
                # Create an index_put that gets freed
                base = torch.zeros(100, 16)
                updated = torch.ops.aten.index_put.default(
                    base, [index], src[:3], False
                )
                # Use it, then discard (simulates ; var = None)
                partial = updated.sum(dim=0)
                # Final output is a dense reduction of a different branch
                return src.sum(dim=0)

        src = torch.randn(10, 16)
        index = torch.tensor([1, 3, 5], dtype=torch.int64)
        out = Repro()(src, index)

        actual = count_bytes_effective(Repro(), [src, index])
        # src is fully read (dense), index is read
        # output is f32[16] = 64 bytes
        expected = self._bytes(src) + self._bytes(index) + self._bytes(out)
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
