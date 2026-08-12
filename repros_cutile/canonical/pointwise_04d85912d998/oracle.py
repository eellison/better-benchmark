"""cuTile port of pointwise_04d85912d998: dense bf16 -> fp32 storage-linear cast.

Ports the Triton `_bf16_to_f32_storage_kernel`. Handles both contiguous and
channels-last strided inputs by walking the underlying storage as a flat
sequence via `torch.as_strided(x, (numel,), (1,))`. The output has the same
strides as the input, so a flat storage view of the output writes to the same
element positions.
"""

import torch
import cuda.tile as ct

from oracle_harness import oracle_impl


@ct.kernel
def _bf16_to_f32_storage_kernel(input_ptr, output_ptr, BLOCK: ct.Constant[int]):
    pid = ct.bid(0)
    values = ct.load(
        input_ptr,
        index=(pid,),
        shape=(BLOCK,),
        padding_mode=ct.PaddingMode.ZERO,
    )
    ct.store(output_ptr, index=(pid,), tile=ct.astype(values, ct.float32))


def oracle_forward(inputs, *, BLOCK=1024):
    (arg0_1,) = inputs
    output = torch.empty_strided(
        tuple(arg0_1.shape),
        tuple(arg0_1.stride()),
        device=arg0_1.device,
        dtype=torch.float32,
    )
    n_elements = arg0_1.numel()
    x_flat = torch.as_strided(arg0_1, (n_elements,), (1,))
    out_flat = torch.as_strided(output, (n_elements,), (1,))
    stream = torch.cuda.current_stream()
    ct.launch(
        stream,
        (ct.cdiv(n_elements, BLOCK), 1, 1),
        _bf16_to_f32_storage_kernel,
        (x_flat, out_flat, BLOCK),
    )
    return output


_POINTS = (
    "c180bc3c", "795241a6", "f60754ac", "25f8f618", "e3e42cd3", "12d8ef02",
    "ae275ccf", "11a19ca8", "7de9bd35", "8ffa104a", "a8a68a04", "41dcb802",
    "215c452d", "daa4b0dd", "5e9822c8", "ac8858e5", "623f8ca6", "01c4aa98",
    "0730b368", "97efcb82", "ffbae90c", "a6a2ae70", "448fc025", "4af17ea6",
    "e79325d8", "f5b2301d", "b6cfd6c1", "cd35cac8", "9bddb1bb", "5303c1fe",
    "2db5e612", "051349d0", "49e64e10", "efc6a14d", "eca65597", "ce04aae0",
    "61b56371", "fd250a48", "dff54966", "7d884c45", "7a20422d", "35d516d1",
    "3070c8f9", "d0fb09d9", "2a07733c", "2ddc1bad", "6393d8ae", "5a62163f",
    "db2f987a", "500e684c", "3582f9b3", "3afb3a00", "c139494c", "851ced79",
    "a3e214ad", "80b34292", "99998008", "af00a629", "66cebc81", "6fed7b82",
    "0d3354ef", "cfdf6a3a", "cc895862", "be6a4bf6", "08fbb5ff", "daeb6b89",
    "5727c431", "a9e40e42", "e0ab88b0", "32da6b44", "3a15e468", "d399c005",
    "bca6105c", "4d209e5f", "321d80cd", "8ce957e7", "16be18b7", "96111b36",
    "efa63a42", "6d375dd5", "8b2cf991", "48fc43ba", "40fe5864", "369826f8",
    "f84df4f6", "f58c13ae", "1238c69e", "b1ba0add", "0b78a53b", "57828936",
    "df3dbe38", "268ac1bb", "9cb48eae", "81015728", "03b16ed5", "a9ac77ba",
    "bfc1d20b", "02b03de9", "6b167218", "658a0097", "ab6685f4", "83350839",
    "dbb934fa", "0ec57a61", "24595219", "1a9f696e", "4a882661", "600c8bfa",
    "46012c02", "450f3148", "8f6bd063", "71126e7c", "9b4fa59a", "c3fb5e13",
    "32a02f3d", "2ce6048d", "082b8a84", "060342ed", "62c29c8c", "bce2a12d",
    "49cad605", "7f82c336", "4b053b17", "e875c188", "63d7005e", "1e23135c",
    "bb34ae40", "dc6b9547", "cae60b91", "4033ee34", "a0bc4be4", "d4512f46",
    "3047d934", "fb48e60c", "1de45ad1", "bb9ce239", "4fd1884d", "eae44ef6",
    "a96e8dc1", "b85f5a1d", "7f465dd6", "6cb8b0a8", "6f1872fe", "dae65fda",
    "3b31bbb0", "32d9d2bd", "80927746", "045932cc", "65f6a878", "ea643d55",
    "e9c8996b", "7f086117", "df387840", "be2233b8", "896f8d4a", "3e112035",
    "40b2a18e", "dcb43c99", "bbfc1eb9", "cc85f631", "c39d875c", "9357c095",
    "0539fdea", "216dece3", "24aebc95", "6e52b66f", "cc408ab6", "c2b68c3e",
    "0b93ff3c", "90187e0c", "deaf11e8", "5065e963", "f3c583c2", "6cc3da18",
    "6f3a310a", "03b4a772", "daba876c", "f7f61ba5", "db6f45ba", "b783e2ae",
    "5c274eea", "02b08269", "ce11835b", "5ba2cfc0", "c2c0584f", "79241e2b",
    "b306d35a", "9d6deec3", "92eabc3c", "5eddb4e1", "a62c9d83", "a18c8563",
    "7d9a8f7c", "4ef768dd", "611ab15d", "ac4a5514", "4e43fcee", "932398c9",
    "fa80550a", "e2d29070", "8b038b8d", "26dd1868", "4a583016", "da0cd8aa",
    "2f47e61f", "6831e811", "31c7a514", "50186d60", "3e95f64a", "aa66d847",
    "b7e8cb78", "9c67f914", "0003c196", "39290894", "264d9c3b", "4c099e68",
    "01aa4545", "e07d5f80", "4996369d", "d9957df6", "9fd1a894", "8f1057fd",
    "c1bdb117", "f34cdbd3", "53f0627f", "253c6d5a", "b3c623f3", "95e765d0",
    "4166c153", "cb54d99d", "088dc224", "d1da25fc", "0ffa35c2", "9f461ac7",
    "37b68e48", "5773a673", "f5608adc", "357101ae", "e1626c55", "d81df4d1",
    "1b46fb5f", "88926d23", "61720e48", "42b7dbef", "a2721139", "fd609835",
    "3b978354", "586f4280", "bf294502", "4496422f", "4a88bef9", "a1fa61f7",
    "fea91bdf", "4e4f2956", "5f84aaba", "4f402cc1", "49fbff34", "3e8ddd11",
    "285a96e6", "acbe661c", "eeaefea8", "334738f5", "2854b247", "17a647ff",
    "04503eb1", "5186c91b", "557c63c6", "076f4090", "30dfc3d1", "d5488832",
    "f9155247", "54170006", "36b4e1e7", "ecec6381", "86e1e916", "1e3eed28",
    "77b99a1e", "86b35d84", "e3ba5155", "2fbe85f1", "6dbeafa9", "8bb60dcb",
    "d37cbbac", "23700a14", "54b0d05d", "585ee806", "1f126b5d", "b4326adc",
    "d87997ca", "d2d047ae", "55a8fc61", "5b88fe45", "8ce49709", "e8017db1",
    "73879c72", "631e4b37", "abd6dd08", "718a022f", "1ae6bfa9", "19b4b553",
    "e6f51b61", "49610b83", "ff939d3c", "c5ef140e", "5daae788", "c1cd4579",
    "f60569f5", "3112ab1b", "0e24c22e", "c1cf52db", "acddfa67", "cf9724c8",
    "ce6e8d24", "e9b510db", "9fc907e1", "b3e5564e", "f375f944", "eb33518c",
    "8390c579", "03889ef1", "ab519e15", "6d360d07", "047f3e89", "c435b23e",
    "f3019e2f", "165c1c1c", "815cf3ae", "f3ac0581", "b8220b12", "8395110a",
    "02767734", "f9f576d9", "30e2d2d9", "38defa30", "7ba01f3f", "a9384bfb",
    "4fa33397", "2d6ceca7", "4f9b1271", "962e8086", "cafc67b6", "8a886914",
    "618968fe", "6043a82f", "97fda35a", "0f69674d", "1561415a", "86aa3fbb",
    "3640346e", "ecc2720f", "50138cd5", "30639013", "fad652a9", "bcddf491",
    "84177907", "a3f80304", "d97e498f", "e1f57fff", "ed505313", "08008e7b",
    "c32fda71", "4eb41506", "fba97314", "51b453bd", "5153b1d7", "58aa41f0",
    "265b155c", "c9c3ef3a", "47542dd5", "72bdd3f7", "b17dd843", "483105fa",
    "0409449e", "ed385436", "543c3d82", "94ef836f", "263aef1c", "c8fb3e4d",
    "d96e1188", "1cf60d5c", "029dd4d1", "fd7d6607", "162feb41", "13dc61d8",
    "0d1b32a0", "9b839694", "88377012", "6cb60043", "3483236d", "ae6522b8",
    "fb36fdc1", "f4a82139", "ed3cecdb", "9b2aff10", "55bfbbde", "1de4b515",
    "3e3bfb59", "b26b998a", "48ffb95c", "67d3fea7", "cd47785e", "edbe5393",
    "10c5c015", "7cb7e6e4", "18290e58", "9554fcc1", "9a144edf", "99e65a35",
    "0f3e2fa1", "5238af3b", "e994657f", "b4a7958d", "613dd548", "b2dd43bb",
    "b2fe4701", "7db825a8", "c89dde8a", "31ad171a", "4a33cf6a", "b2966ff4",
    "e0ea1b62", "57c47bed", "5fc50f7b", "f46c5bd4", "9d5f4cc4", "93f8fbcd",
    "3267e53f", "bcc6a8f0", "a0131897", "0fb7f7b5", "5016da60", "c7a73141",
    "2a26ac60", "1e571200", "5bfe6d35", "3213bad8", "298dd5b5", "e863d800",
    "29b3fcbd", "465149ba", "90becd17", "9fe13ed9", "40eb390a", "24be48e8",
    "a0f5c667", "776eb3d0", "3ba76078", "03144b09", "44da6ae4", "55fb80e4",
    "7ba51dc8", "ad40b12e", "1816f70c", "e563d9f2", "25aedbf5", "fa4513e3",
    "21900eaf", "c97f17ec", "c88ad749", "7eaaa5e5", "fdfae713", "de2b7595",
    "af6ceef1", "85839434", "ac0fcd75", "941ec111", "d1dd0272", "0100d342",
    "3e9070a3",
)


for _point in _POINTS:
    oracle_forward = oracle_impl(
        hardware="B200",
        point=_point,
        BLOCK=1024,
    )(oracle_forward)
