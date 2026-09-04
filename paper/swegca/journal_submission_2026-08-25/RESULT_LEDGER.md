# Journal v2 result ledger

## J-R001 — reproduced pre-repair counterexample

- matching, empty-address, and foreign-address proposals all committed
- report SHA-256: `5f78720910782094c992b1d56b5f9a6c7f893e728b8d5cc78b87235d92e02253`
- interpretation: capability authorization was not proposal binding

## J-R002 — final-source primary development corpus

- 2,560 distinct accepted histories; 40 paired cases per history; 102,400 total
- valid commits: 20,480/20,480
- invalid rejections: 81,920/81,920
- unauthorized commits, valid rejections, rejected-case state mutations, receipt-binding mismatches: all zero
- aggregate latency: median 145,500 ns; p95 610,200 ns; p99 722,500 ns
- unauthorized-commit two-sided Wilson 95% interval: `[0, 0.000046890609037480814]`
- report SHA-256: `2ddf3c5b138a305bdde8aefc07fa7b9f8381d317ab409d0c70e2abd86627453c`
- 102,400-row ledger SHA-256: `88efdbfd1f7d67b6f8b08c22962e85a185c912b3f27ea6dbe9c220d56847a513`

## J-R003 — one-time held-out confirmation

- 256 distinct accepted histories; 40 paired cases per history; 10,240 total
- split seed and generated hypothesis/address identities are disjoint from development
- executed exactly once after implementation/configuration/source/development freeze
- valid commits: 2,048/2,048
- invalid rejections: 8,192/8,192
- unauthorized commits, valid rejections, rejected-case state mutations, receipt-binding mismatches: all zero
- aggregate latency: median 145,600 ns; p95 613,500 ns; p99 751,900 ns
- unauthorized-commit two-sided Wilson 95% interval: `[0, 0.00046870828822094784]`
- report SHA-256: `cd7ac90283dda1af55ed9edb48148cb4d6d3c2a6cdceabdbaf673eb13ac40cbf`
- 10,240-row ledger SHA-256: `9571e1b72a0d6a016922ad938ada32ca492ee74dde365894d583da2567529375`

## J-R004 — regression evidence

- final focused migration checkpoint: 96/96 passed
- pre-commit narrow rerun: 60/60 passed in 19.58 seconds
- Ruff over every changed Python file: passed
- whole-repository result is not a pass claim: one absent baseline module blocked collection; an exclusion run produced 4,947 passes and 111 environment/artifact-dominated failures

## J-R005 — excluded scale result

The 1,024,000-case development stress run is retained as auxiliary implementation evidence but excluded from headline statistics. It adds paired repetitions over 25,600 histories, not a qualitatively new attack family or natural-world condition.
