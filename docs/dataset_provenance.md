# Dataset Provenance and Fingerprints

## Dataset sizes

| Artifact | Records |
|---|---:|
| B2 Dolly general | 1,000 |
| B3 reconstructed IoT candidate pool | 600 |
| B4 filtered IoT | 577 |
| B5 IoT portion | 577 |
| B5 general portion | 577 |
| B5 combined | 1,154 |
| Held-out IoT evaluation | 35 |

## Train/validation counts

| Baseline | Train | Validation |
|---|---:|---:|
| B2 | 900 | 100 |
| B3 | 542 | 58 |
| B4 | 519 | 58 |
| B5 | 1,038 | 116 |

B3 and B4 share the same 58 IoT validation examples. B5 validation includes 58 IoT and 58 general examples. Losses from these different validation compositions are not directly comparable.

## SHA-256 fingerprints

| Dataset/artifact | SHA-256 |
|---|---|
| B2 Dolly 1,000 | `b2f565f3533222b5080f9df1bd037b483e020208082aea28621c39efdc4cbe06` |
| B3 reconstructed unfiltered 600 | `f570c5596e28db4c2493988d344abe4efc632a5637d2471d366ef39124292960` |
| B4 filtered 577 / B5 domain 577 | `73c0134aa49137f772418fbb4b58f7e5721f5e64d323080092e45218db77fb58` |
| B5 general 577 | `c4718ab256c4cffbf006b272932d36399165fcdf566b1902098186573759a505` |
| B5 combined 1,154 | `e2bec7d9ac16ba82cf21cf3a99acdd7e9a2176f24201e0e4b6fa9e3dc832de60` |
| Frozen chunk corpus | `67e90d3428ff0a421111e19916ceb2dfc007460ae047f18131131be2f80bb827` |
| Held-out IoT test (35) | `70d07aa68b4e125dd573ba01367f059b550651e5e25d4e271e92f55f04c6ff5d` |
| MMLU retention subset (50) | `d7c4f24f3ea17f698b0729093ef601cbdf1ecea53623e62844faa4013f2bf78e` |

## B3 provenance caveat

B3 is reconstructed from 88 original raw candidates, 27 prior additional candidates, and 485 production-v2 candidates. It is not a frozen pre-filter snapshot of B4. The B3/B4 comparison is not a strict causal filtering ablation.

## Public-data handling

Raw IoT source evidence and datasets are not published here. Review source-document licensing and redistribution rights before adding them.
