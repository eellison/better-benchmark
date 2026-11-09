import unittest
from unittest import mock

import torch

import byte_accounting
from byte_accounting import (
    ACCESS_OVERLAP_ESTIMATED,
    ATOMIC_WRITE,
    INDIRECT_READ,
    INDIRECT_WRITE,
    analyze_bytes_effective,
    count_bytes_effective,
    count_bytes_naive,
    sol_result_fields,
)


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
        expected = self._bytes(a) + self._bytes(b) + self._bytes(out)

        self.assertEqual(count_bytes_effective(Repro(), [a, b]), expected)
        breakdown = analyze_bytes_effective(Repro(), [a, b])
        self.assertEqual(breakdown.logical_read_bytes, self._bytes(a) + self._bytes(b))
        self.assertEqual(breakdown.logical_write_bytes, self._bytes(out))
        self.assertEqual(breakdown.total_bytes, expected)
        self.assertEqual(breakdown.factors, ())
        self.assertEqual(
            breakdown.to_result_fields(),
            {
                "sol_accounting_version": "logical-v2",
                "logical_read_bytes": self._bytes(a) + self._bytes(b),
                "logical_write_bytes": self._bytes(out),
                "total_bytes": expected,
                "sol_accounting_estimated": False,
                "sol_mitigation_factors": [],
            },
        )

    def test_copy_mutation_does_not_read_overwritten_destination(self):
        class Repro(torch.nn.Module):
            def forward(self, destination, source):
                return torch.ops.aten.copy_.default(destination, source)

        destination = torch.empty(8)
        source = torch.randn(8)
        breakdown = analyze_bytes_effective(Repro(), [destination, source])

        self.assertEqual(breakdown.logical_read_bytes, self._bytes(source))
        self.assertEqual(breakdown.logical_write_bytes, self._bytes(destination))

    def test_fill_and_zero_do_not_read_overwritten_destination(self):
        class Fill(torch.nn.Module):
            def forward(self, destination):
                return torch.ops.aten.fill_.Scalar(destination, 3.0)

        class Zero(torch.nn.Module):
            def forward(self, destination):
                return torch.ops.aten.zero_.default(destination)

        for module in (Fill(), Zero()):
            destination = torch.empty(8)
            breakdown = analyze_bytes_effective(module, [destination])
            self.assertEqual(breakdown.logical_read_bytes, 0)
            self.assertEqual(breakdown.logical_write_bytes, self._bytes(destination))

    def test_sol_result_fields_keep_oracle_details_out_of_mitigations(self):
        class Repro(torch.nn.Module):
            def forward(self, x):
                return torch.exp(x).sum()

        x = torch.randn(32)
        accounting = analyze_bytes_effective(Repro(), [x])
        fields = sol_result_fields(
            accounting,
            proxy_bytes=2048,
            proxy_op="torch.add.float32",
            sol_us=2.0,
        )
        self.assertEqual(fields["sol_mitigation_factors"], [])
        self.assertFalse(fields["sol_accounting_estimated"])
        self.assertEqual(fields["sol_proxy_bandwidth_gbs"], 1.024)

    def test_embedding_charges_gathered_rows_not_full_table(self):
        class Repro(torch.nn.Module):
            def forward(self, weight, index):
                return torch.ops.aten.embedding.default(weight, index)

        weight = torch.randn(1000, 16)
        index = torch.tensor([1, 3, 7, 9], dtype=torch.int64)
        out = Repro()(weight, index)

        expected = self._bytes(out) + self._bytes(index) + self._bytes(out)
        breakdown = analyze_bytes_effective(Repro(), [weight, index])
        self.assertEqual(breakdown.total_bytes, expected)
        self.assertIn(INDIRECT_READ, breakdown.factors)
        self.assertEqual(breakdown.mitigation_factors, (INDIRECT_READ,))
        self.assertLess(breakdown.total_bytes, count_bytes_naive([weight, index], out))

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

    def test_unsafe_index_is_an_indirect_read(self):
        class Repro(torch.nn.Module):
            def forward(self, source, rows, columns):
                return torch.ops.aten._unsafe_index.Tensor(
                    source, [rows[:, None], columns[None, :]]
                )

        source = torch.randn(100, 80)
        rows = torch.tensor([1, 3, 5], dtype=torch.int64)
        columns = torch.tensor([2, 4, 6, 8], dtype=torch.int64)
        out = Repro()(source, rows, columns)
        breakdown = analyze_bytes_effective(Repro(), [source, rows, columns])
        expected_read = self._bytes(out) + self._bytes(rows) + self._bytes(columns)

        self.assertEqual(breakdown.logical_read_bytes, expected_read)
        self.assertEqual(breakdown.logical_write_bytes, self._bytes(out))
        self.assertIn(INDIRECT_READ, breakdown.factors)

    def test_functional_index_put_materializes_dense_output(self):
        class Repro(torch.nn.Module):
            def forward(self, index, values):
                base = torch.zeros(100, 16)
                return torch.ops.aten.index_put.default(base, [index], values, False)

        index = torch.tensor([1, 3, 5], dtype=torch.int64)
        values = torch.randn(3, 16)
        out = Repro()(index, values)

        expected = self._bytes(index) + self._bytes(values) + self._bytes(out)
        breakdown = analyze_bytes_effective(Repro(), [index, values])
        self.assertEqual(breakdown.total_bytes, expected)
        self.assertEqual(breakdown.logical_write_bytes, self._bytes(out))
        self.assertIn(INDIRECT_WRITE, breakdown.factors)

    def test_functional_scatter_materializes_dense_output(self):
        class Repro(torch.nn.Module):
            def forward(self, index, src):
                base = torch.zeros(100, 16)
                return torch.ops.aten.scatter.src(base, 0, index, src)

        index = torch.tensor([[1] * 16, [3] * 16, [5] * 16], dtype=torch.int64)
        src = torch.randn(3, 16)
        out = Repro()(index, src)

        expected = self._bytes(index) + self._bytes(src) + self._bytes(out)
        breakdown = analyze_bytes_effective(Repro(), [index, src])
        self.assertEqual(breakdown.total_bytes, expected)
        self.assertEqual(breakdown.logical_write_bytes, self._bytes(out))
        self.assertIn(INDIRECT_WRITE, breakdown.factors)

    def test_functional_layout_scatter_materializes_dense_output(self):
        class Repro(torch.nn.Module):
            def forward(self, base, src):
                return torch.ops.aten.slice_scatter.default(base, src, 0, 2, 4, 1)

        base = torch.randn(8, 4)
        src = torch.randn(2, 4)
        breakdown = analyze_bytes_effective(Repro(), [base, src])

        self.assertEqual(
            breakdown.logical_read_bytes,
            self._bytes(base) + self._bytes(src),
        )
        self.assertEqual(breakdown.logical_write_bytes, self._bytes(base))
        self.assertNotIn(INDIRECT_WRITE, breakdown.factors)

    def test_avg_pool_backward_self_is_shape_only(self):
        class Repro(torch.nn.Module):
            def forward(self, grad, original):
                return torch.ops.aten.avg_pool2d_backward.default(
                    grad, original, [2, 2], [2, 2], [0, 0], False, True, None
                )

        original = torch.randn(1, 2, 8, 8)
        grad = torch.randn(1, 2, 4, 4)
        out = Repro()(grad, original)
        breakdown = analyze_bytes_effective(Repro(), [grad, original])

        self.assertEqual(breakdown.logical_read_bytes, self._bytes(grad))
        self.assertEqual(breakdown.logical_write_bytes, self._bytes(out))

    def test_slice_charges_only_accessed_portion(self):
        class Repro(torch.nn.Module):
            def forward(self, x):
                sliced = torch.ops.aten.slice.Tensor(x, 1, 0, 64)
                return sliced + 1.0

        x = torch.randn(8, 128, 100)
        out = Repro()(x)
        expected_read = 8 * 64 * 100 * 4
        expected_write = 8 * 64 * 100 * 4
        breakdown = analyze_bytes_effective(Repro(), [x])

        self.assertEqual(breakdown.total_bytes, expected_read + expected_write)
        self.assertNotIn(ACCESS_OVERLAP_ESTIMATED, breakdown.factors)
        self.assertLess(breakdown.total_bytes, count_bytes_naive([x], out))

    def test_irregular_overlapping_view_is_counted_exactly(self):
        class Repro(torch.nn.Module):
            def forward(self, x):
                return x + 1

        base = torch.randn(1000)
        x = torch.as_strided(base, (100, 100), (2, 3))
        touched = {2 * i + 3 * j for i in range(100) for j in range(100)}
        breakdown = analyze_bytes_effective(Repro(), [x])

        self.assertEqual(breakdown.logical_read_bytes, len(touched) * 4)
        self.assertEqual(breakdown.logical_write_bytes, self._bytes(x))
        self.assertNotIn(ACCESS_OVERLAP_ESTIMATED, breakdown.factors)

    def test_pathological_storage_union_is_explicitly_estimated(self):
        class Repro(torch.nn.Module):
            def forward(self, x, y):
                return x + 1, y + 1

        base = torch.randn(1000)
        x = torch.as_strided(base, (100, 100), (2, 3))
        y = torch.as_strided(base, (100, 100), (3, 2))
        with mock.patch.object(byte_accounting, "_MAX_EXACT_INTERVALS", 64):
            breakdown = analyze_bytes_effective(Repro(), [x, y])

        self.assertIn(ACCESS_OVERLAP_ESTIMATED, breakdown.factors)
        self.assertTrue(breakdown.accounting_estimated)
        self.assertTrue(breakdown.to_result_fields()["sol_accounting_estimated"])
        self.assertEqual(breakdown.mitigation_factors, ())
        # Both views span bytes [0, 1984); overlapping estimates are not summed.
        self.assertEqual(breakdown.logical_read_bytes, 496 * 4)

    def test_overlapping_column_slices_of_a_tall_matrix_are_unioned(self):
        class Repro(torch.nn.Module):
            def forward(self, a):
                return a[:, :64] + a[:, 32:96]

        a = torch.randn(8192, 128)
        breakdown = analyze_bytes_effective(Repro(), [a])

        self.assertEqual(breakdown.logical_read_bytes, 8192 * 96 * 4)
        self.assertEqual(breakdown.logical_write_bytes, 8192 * 64 * 4)
        self.assertNotIn(ACCESS_OVERLAP_ESTIMATED, breakdown.factors)

    def test_expanded_column_slice_reads_the_slice_once(self):
        class Repro(torch.nn.Module):
            def forward(self, a):
                return (a[:, :64].unsqueeze(1).expand(8192, 16, 64) + 1).sum(1)

        a = torch.randn(8192, 128)
        breakdown = analyze_bytes_effective(Repro(), [a])

        self.assertEqual(breakdown.logical_read_bytes, 8192 * 64 * 4)
        self.assertEqual(breakdown.logical_write_bytes, 8192 * 64 * 4)
        self.assertNotIn(ACCESS_OVERLAP_ESTIMATED, breakdown.factors)

    def test_stride_zero_input_charges_distinct_elements(self):
        class Repro(torch.nn.Module):
            def forward(self, x):
                return x + 1

        x = torch.randn(64).expand(1024, 64)
        breakdown = analyze_bytes_effective(Repro(), [x])

        self.assertEqual(breakdown.logical_read_bytes, 64 * 4)
        self.assertEqual(breakdown.logical_write_bytes, 1024 * 64 * 4)

    def test_two_disjoint_views_are_summed_not_maxed(self):
        class Repro(torch.nn.Module):
            def forward(self, x):
                left = torch.ops.aten.slice.Tensor(x, 0, 0, 4)
                right = torch.ops.aten.slice.Tensor(x, 0, 12, 16)
                return left + right

        x = torch.randn(16, 8)
        out = Repro()(x)
        expected_read = 8 * 8 * x.element_size()
        breakdown = analyze_bytes_effective(Repro(), [x])

        self.assertEqual(breakdown.logical_read_bytes, expected_read)
        self.assertEqual(breakdown.logical_write_bytes, self._bytes(out))

    def test_overlapping_views_are_union_not_sum(self):
        class Repro(torch.nn.Module):
            def forward(self, x):
                first = x[:8]
                second = x[4:12]
                return first + second

        x = torch.randn(16, 8)
        breakdown = analyze_bytes_effective(Repro(), [x])
        expected_read = 12 * 8 * x.element_size()
        self.assertEqual(breakdown.logical_read_bytes, expected_read)

    def test_separately_passed_disjoint_input_views_share_storage(self):
        class Repro(torch.nn.Module):
            def forward(self, left, right):
                return left + right

        base = torch.randn(16, 8)
        left = base[:4]
        right = base[12:]
        out = Repro()(left, right)
        breakdown = analyze_bytes_effective(Repro(), [left, right])

        self.assertEqual(
            breakdown.logical_read_bytes,
            self._bytes(left) + self._bytes(right),
        )
        self.assertEqual(breakdown.logical_write_bytes, self._bytes(out))

    def test_metadata_only_input_view_output_has_no_traffic(self):
        class Repro(torch.nn.Module):
            def forward(self, x):
                return x[2:6].transpose(0, 1)

        x = torch.randn(8, 4)
        breakdown = analyze_bytes_effective(Repro(), [x])

        self.assertEqual(breakdown.logical_read_bytes, 0)
        self.assertEqual(breakdown.logical_write_bytes, 0)
        self.assertEqual(breakdown.total_bytes, 0)

    def test_multi_output_split_is_metadata_only_until_used(self):
        class Repro(torch.nn.Module):
            def forward(self, x):
                first, _, last = torch.split(x, [2, 4, 2])
                return first + last

        x = torch.randn(8, 4)
        out = Repro()(x)
        breakdown = analyze_bytes_effective(Repro(), [x])

        self.assertEqual(
            breakdown.logical_read_bytes,
            (2 + 2) * 4 * x.element_size(),
        )
        self.assertEqual(breakdown.logical_write_bytes, self._bytes(out))

    def test_reshape_copy_is_not_treated_as_metadata_only(self):
        class Repro(torch.nn.Module):
            def forward(self, x):
                return x.reshape(-1)

        base = torch.randn(3, 4)
        x = base.t()  # This reshape requires a copy.
        out = Repro()(x)
        self.assertNotEqual(
            x.untyped_storage()._cdata,
            out.untyped_storage()._cdata,
        )
        breakdown = analyze_bytes_effective(Repro(), [x])

        self.assertEqual(breakdown.logical_read_bytes, self._bytes(x))
        self.assertEqual(breakdown.logical_write_bytes, self._bytes(out))

    def test_output_aliases_are_unioned(self):
        class Repro(torch.nn.Module):
            def forward(self, x):
                y = x + 1
                return y, y[:4], y[2:6]

        x = torch.randn(8, 4)
        breakdown = analyze_bytes_effective(Repro(), [x])

        self.assertEqual(breakdown.logical_read_bytes, self._bytes(x))
        self.assertEqual(breakdown.logical_write_bytes, self._bytes(x))

    def test_disjoint_output_aliases_are_summed(self):
        class Repro(torch.nn.Module):
            def forward(self, x):
                y = x + 1
                return y[:2], y[6:]

        x = torch.randn(8, 4)
        breakdown = analyze_bytes_effective(Repro(), [x])
        expected_write = 4 * 4 * x.element_size()
        self.assertEqual(breakdown.logical_write_bytes, expected_write)

    def test_index_put_output_not_inherited_by_downstream(self):
        class Repro(torch.nn.Module):
            def forward(self, index, values):
                base = torch.zeros(100, 16)
                updated = torch.ops.aten.index_put.default(base, [index], values, False)
                return torch.ops.aten.sum.dim_IntList(updated, [0])

        index = torch.tensor([1, 3, 5], dtype=torch.int64)
        values = torch.randn(3, 16)
        out = Repro()(index, values)
        breakdown = analyze_bytes_effective(Repro(), [index, values])
        expected = self._bytes(index) + self._bytes(values) + self._bytes(out)

        self.assertEqual(breakdown.total_bytes, expected)
        self.assertIn(INDIRECT_WRITE, breakdown.factors)

    def test_scatter_add_is_annotated_atomic(self):
        class Repro(torch.nn.Module):
            def forward(self, base, index, src):
                return torch.ops.aten.scatter_add.default(base, 0, index, src)

        base = torch.randn(16, 4)
        index = torch.tensor([[1] * 4, [3] * 4], dtype=torch.int64)
        src = torch.randn(2, 4)
        breakdown = analyze_bytes_effective(Repro(), [base, index, src])

        self.assertIn(INDIRECT_WRITE, breakdown.factors)
        self.assertIn(ATOMIC_WRITE, breakdown.factors)
        self.assertEqual(breakdown.logical_write_bytes, self._bytes(base))
        self.assertEqual(
            breakdown.mitigation_factors,
            (ATOMIC_WRITE, INDIRECT_WRITE),
        )

    def test_unsafe_masked_index_put_accumulate_accounting(self):
        class Repro(torch.nn.Module):
            def forward(self, base, mask, rows, columns, values):
                return torch.ops.aten._unsafe_masked_index_put_accumulate.default(
                    base,
                    mask,
                    [rows, columns],
                    values,
                )

        base = torch.randn(7, 8)
        rows = torch.tensor([[1], [3]], dtype=torch.int64)
        columns = torch.tensor([[2, 4, 6]], dtype=torch.int64)
        mask = torch.tensor([[True, False, True], [True, True, False]])
        values = torch.randn(2, 3)
        out = Repro()(base, mask, rows, columns, values)
        breakdown = analyze_bytes_effective(
            Repro(), [base, mask, rows, columns, values]
        )
        expected_read = sum(
            self._bytes(tensor) for tensor in (base, mask, rows, columns, values)
        )

        self.assertEqual(breakdown.logical_read_bytes, expected_read)
        self.assertEqual(breakdown.logical_write_bytes, self._bytes(out))
        self.assertIn(INDIRECT_WRITE, breakdown.factors)
        self.assertIn(ATOMIC_WRITE, breakdown.factors)

    def test_index_add_is_indirect_atomic_dense_output(self):
        class Repro(torch.nn.Module):
            def forward(self, base, index, source):
                updated = torch.ops.aten.index_add.default(base, 0, index, source)
                return updated.reshape(4, 8)

        base = torch.randn(8, 4)
        index = torch.tensor([1, 3], dtype=torch.int64)
        source = torch.randn(2, 4)
        out = Repro()(base, index, source)
        breakdown = analyze_bytes_effective(Repro(), [base, index, source])

        self.assertEqual(
            breakdown.logical_read_bytes,
            self._bytes(base) + self._bytes(index) + self._bytes(source),
        )
        self.assertEqual(breakdown.logical_write_bytes, self._bytes(out))
        self.assertIn(INDIRECT_WRITE, breakdown.factors)
        self.assertIn(ATOMIC_WRITE, breakdown.factors)

    def test_index_reduce_is_annotated_as_reduction(self):
        class Repro(torch.nn.Module):
            def forward(self, base, index, source):
                return torch.ops.aten.index_reduce.default(
                    base, 0, index, source, "amax", include_self=True
                )

        base = torch.randn(8, 4)
        index = torch.tensor([1, 3], dtype=torch.int64)
        source = torch.randn(2, 4)
        out = Repro()(base, index, source)
        breakdown = analyze_bytes_effective(Repro(), [base, index, source])

        self.assertEqual(breakdown.logical_write_bytes, self._bytes(out))
        self.assertIn(INDIRECT_WRITE, breakdown.factors)
        self.assertIn(ATOMIC_WRITE, breakdown.factors)

    def test_id_reuse_does_not_cause_spurious_sparse_write(self):
        class Repro(torch.nn.Module):
            def forward(self, src, index):
                base = torch.zeros(100, 16)
                updated = torch.ops.aten.index_put.default(base, [index], src[:3], False)
                updated.sum(dim=0)
                return src.sum(dim=0)

        src = torch.randn(10, 16)
        index = torch.tensor([1, 3, 5], dtype=torch.int64)
        out = Repro()(src, index)
        breakdown = analyze_bytes_effective(Repro(), [src, index])
        expected = self._bytes(src) + self._bytes(index) + self._bytes(out)

        self.assertEqual(breakdown.total_bytes, expected)

    def test_broadcast_output_charges_distinct_elements(self):
        class Repro(torch.nn.Module):
            def forward(self, x):
                return torch.ops.aten.expand.default(x * 2.0, [128, 32, 16])

        x = torch.randn(1, 32, 16)
        expected = self._bytes(x) + self._bytes(x)  # read x, write one [1, 32, 16]
        self.assertEqual(count_bytes_effective(Repro(), [x]), expected)

    def test_aliased_outputs_charge_one_write(self):
        class Repro(torch.nn.Module):
            def forward(self, x):
                y = x * 2.0
                e = torch.ops.aten.expand.default(y, [8, 32])
                return y, e, torch.ops.aten.expand.default(y, [8, 32]), torch.ops.aten.view.default(y, [256])

        x = torch.randn(8, 32)
        self.assertEqual(count_bytes_effective(Repro(), [x]), 2 * self._bytes(x))

    def test_output_view_of_input_is_not_a_write(self):
        class Repro(torch.nn.Module):
            def forward(self, x):
                return x.sum(dim=1), torch.ops.aten.slice.Tensor(x, 1, 0, 64)

        x = torch.randn(8, 128)
        out = x.sum(dim=1)
        self.assertEqual(count_bytes_effective(Repro(), [x]), self._bytes(x) + self._bytes(out))

    def test_output_view_of_mutated_input_is_covered_by_the_mutation(self):
        # copy_ writes all of buf without reading it; the returned slice is
        # part of that write, not a second one.
        class Repro(torch.nn.Module):
            def forward(self, buf, x):
                torch.ops.aten.copy_.default(buf, x * 2.0)
                return torch.ops.aten.slice.Tensor(buf, 0, 0, 4)

        buf, x = torch.randn(8, 16), torch.randn(8, 16)
        actual = count_bytes_effective(Repro(), [buf, x])
        self.assertEqual(actual, self._bytes(buf) + self._bytes(x))


    def test_region_internal_scratch_buffer_is_not_a_write(self):
        # Neither an input nor returned: a fused kernel keeps buf in registers.
        class Repro(torch.nn.Module):
            def forward(self, x):
                buf = torch.ops.aten.empty_like.default(x)
                torch.ops.aten.copy_.default(buf, x)
                return torch.ops.aten.sum.default(buf * 2.0)

        x = torch.randn(64, 8)
        breakdown = analyze_bytes_effective(Repro(), [x])
        self.assertEqual(breakdown.logical_read_bytes, self._bytes(x))
        self.assertEqual(breakdown.logical_write_bytes, 4)
        self.assertEqual(breakdown.total_bytes, 2052)

    def test_returned_scratch_buffer_is_written_once(self):
        class Repro(torch.nn.Module):
            def forward(self, x):
                buf = torch.ops.aten.empty_like.default(x)
                torch.ops.aten.copy_.default(buf, x * 2.0)
                return buf, torch.ops.aten.slice.Tensor(buf, 0, 0, 4)

        x = torch.randn(64, 8)
        breakdown = analyze_bytes_effective(Repro(), [x])
        self.assertEqual(breakdown.logical_read_bytes, self._bytes(x))
        self.assertEqual(breakdown.logical_write_bytes, self._bytes(x))

    def test_pure_view_region_moves_no_bytes(self):
        # view/permute/unbind of one input: Inductor emits no kernel, only
        # reinterpretations of the input's storage.
        class Repro(torch.nn.Module):
            def forward(self, x):
                viewed = torch.ops.aten.view.default(x, [4, 8, 16])
                permuted = torch.ops.aten.permute.default(viewed, [1, 0, 2])
                return tuple(torch.ops.aten.unbind.int(permuted, 0))

        x = torch.randn(32, 16)
        breakdown = analyze_bytes_effective(Repro(), [x])
        self.assertEqual(breakdown.total_bytes, 0)
        self.assertEqual(breakdown.factors, ())
        fields = sol_result_fields(breakdown, proxy_bytes=2048, proxy_op="torch.add.float32")
        self.assertTrue(fields["sol_proxy_floor_applied"])


    def test_inplace_index_put_with_broadcast_value_writes_every_indexed_row(self):
        class Repro(torch.nn.Module):
            def forward(self, a, index, value):
                return a.index_put_((index,), value)

        a = torch.randn(8192, 128)
        index = torch.randperm(8192)[:1000]
        value = torch.tensor(0.0)
        breakdown = analyze_bytes_effective(Repro(), [a, index, value])

        self.assertEqual(breakdown.logical_write_bytes, 1000 * 128 * 4)
        self.assertEqual(breakdown.logical_read_bytes, self._bytes(index) + 4)
        self.assertIn(INDIRECT_WRITE, breakdown.factors)
        self.assertNotIn(ATOMIC_WRITE, breakdown.factors)

    def test_inplace_index_put_with_boolean_mask_writes_selected_elements(self):
        class Repro(torch.nn.Module):
            def forward(self, a, mask, value):
                return a.index_put_((mask,), value)

        a = torch.randn(64, 32)
        mask = torch.zeros(64, dtype=torch.bool)
        mask[::4] = True
        breakdown = analyze_bytes_effective(Repro(), [a, mask, torch.tensor(1.0)])

        self.assertEqual(breakdown.logical_write_bytes, 16 * 32 * 4)

    def test_repeated_scatter_into_one_target_adds_up(self):
        class Repro(torch.nn.Module):
            def forward(self, a, first, second, src):
                a.scatter_(0, first, src)
                a.scatter_(0, second, src)
                return a

        a = torch.randn(8192, 128)
        first = torch.arange(0, 100).unsqueeze(1).expand(100, 128).contiguous()
        second = torch.arange(100, 200).unsqueeze(1).expand(100, 128).contiguous()
        src = torch.randn(100, 128)
        breakdown = analyze_bytes_effective(Repro(), [a, first, second, src])

        self.assertEqual(breakdown.logical_write_bytes, 2 * 100 * 128 * 4)
        self.assertEqual(
            breakdown.logical_read_bytes,
            self._bytes(first) + self._bytes(second) + self._bytes(src),
        )

    def test_scatter_with_reduce_is_atomic_and_reads_destination(self):
        class Repro(torch.nn.Module):
            def forward(self, a, index, src):
                return a.scatter_(0, index, src, reduce="add")

        a = torch.randn(64, 8)
        index = torch.randint(0, 64, (16, 8))
        src = torch.randn(16, 8)
        breakdown = analyze_bytes_effective(Repro(), [a, index, src])

        self.assertIn(ATOMIC_WRITE, breakdown.factors)
        self.assertEqual(
            breakdown.logical_read_bytes,
            self._bytes(a) + self._bytes(index) + self._bytes(src),
        )
        self.assertEqual(breakdown.logical_write_bytes, 16 * 8 * 4)

    def test_read_of_overwritten_input_is_not_an_input_read(self):
        class Repro(torch.nn.Module):
            def forward(self, a, b):
                a.copy_(b)
                return a * 2

        a, b = torch.randn(1024), torch.randn(1024)
        breakdown = analyze_bytes_effective(Repro(), [a, b])

        self.assertEqual(breakdown.logical_read_bytes, self._bytes(b))
        self.assertEqual(breakdown.logical_write_bytes, 2 * self._bytes(a))

    def test_read_modify_write_of_input_reads_it_once(self):
        class Repro(torch.nn.Module):
            def forward(self, a, b):
                a.mul_(0.9).add_(b)
                return a

        a, b = torch.randn(1024), torch.randn(1024)
        breakdown = analyze_bytes_effective(Repro(), [a, b])

        self.assertEqual(breakdown.logical_read_bytes, self._bytes(a) + self._bytes(b))
        self.assertEqual(breakdown.logical_write_bytes, self._bytes(a))

    def test_out_kwarg_is_written_not_read(self):
        class Repro(torch.nn.Module):
            def forward(self, a, b, out):
                return torch.add(a, b, out=out)

        a, b, out = torch.randn(1024), torch.randn(1024), torch.randn(1024)
        breakdown = analyze_bytes_effective(Repro(), [a, b, out])

        self.assertEqual(breakdown.logical_read_bytes, self._bytes(a) + self._bytes(b))
        self.assertEqual(breakdown.logical_write_bytes, self._bytes(out))

    def test_metadata_only_inplace_op_moves_no_bytes(self):
        class Repro(torch.nn.Module):
            def forward(self, a):
                return a.t_()

        breakdown = analyze_bytes_effective(Repro(), [torch.randn(64, 64)])
        self.assertEqual(breakdown.total_bytes, 0)

    def test_result_fields_carry_total_bytes_and_proxy(self):
        # Result rows take total_bytes from these fields; callers no longer
        # set it themselves.
        class Repro(torch.nn.Module):
            def forward(self, x):
                return x + 1

        accounting = analyze_bytes_effective(Repro(), [torch.randn(1024)])
        fields = sol_result_fields(
            accounting, proxy_bytes=8192, proxy_op="torch.copy_.float32", sol_us=2.0
        )
        self.assertEqual(
            set(fields),
            {
                "sol_accounting_version",
                "logical_read_bytes",
                "logical_write_bytes",
                "total_bytes",
                "sol_accounting_estimated",
                "sol_mitigation_factors",
                "sol_proxy_bytes",
                "sol_proxy_read_bytes",
                "sol_proxy_write_bytes",
                "sol_proxy_op",
                "sol_proxy_floor_applied",
                "logical_to_proxy_bytes_ratio",
                "sol_proxy_bandwidth_gbs",
            },
        )
        self.assertEqual(fields["total_bytes"], 8192)
        self.assertEqual(accounting.to_result_fields()["total_bytes"], 8192)
        self.assertEqual(fields["sol_proxy_read_bytes"] + fields["sol_proxy_write_bytes"], 8192)
        self.assertEqual(fields["logical_to_proxy_bytes_ratio"], 1.0)
        self.assertAlmostEqual(fields["sol_proxy_bandwidth_gbs"], 8192 / 2000)


if __name__ == "__main__":
    unittest.main()
