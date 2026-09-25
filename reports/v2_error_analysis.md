# V2 error analysis

No V2 official error analysis exists because the candidate failed its predeclared gate before official access. This report compares complete client OOF predictions and fixed stress views; no cohort is removed from overall F1. Initial official V1 diagnostics remain in `v2_diagnostics.json`.

| View | Cohort | n | V1 accuracy | V2 accuracy | Rescued | Lost |
|---|---|---:|---:|---:|---:|---:|
| original | C1_3plus | 1106 | 0.7703 | 0.7785 | 17 | 8 |
| original | C2_only2 | 93 | 0.4409 | 0.5054 | 6 | 0 |
| original | C3_single_only | 144 | 0.0278 | 0.2500 | 33 | 1 |
| original | no_plausible_proxy | 60 | 0.0000 | 0.0000 | 0 | 0 |
| original | C4_V1_present_wrong_family | 243 | 0.0000 | 0.0700 | 17 | 0 |
| original | C5_V1_family_to_none | 97 | 0.0000 | 0.2268 | 22 | 0 |
| original | C6_V1_none_to_family | 122 | 0.0000 | 0.0492 | 6 | 0 |
| original | C7_music_software_streaming | 618 | 0.6408 | 0.6667 | 22 | 6 |
| original | true_none | 597 | 0.7956 | 0.7956 | 6 | 6 |
| valid_like | C1_3plus | 1110 | 0.7450 | 0.7405 | 9 | 14 |
| valid_like | C2_only2 | 90 | 0.3000 | 0.3556 | 5 | 0 |
| valid_like | C3_single_only | 142 | 0.0282 | 0.1549 | 18 | 0 |
| valid_like | no_plausible_proxy | 61 | 0.0000 | 0.0000 | 0 | 0 |
| valid_like | C4_V1_present_wrong_family | 240 | 0.0000 | 0.0333 | 8 | 0 |
| valid_like | C5_V1_family_to_none | 144 | 0.0000 | 0.1042 | 15 | 0 |
| valid_like | C6_V1_none_to_family | 117 | 0.0000 | 0.0513 | 6 | 0 |
| valid_like | C7_music_software_streaming | 618 | 0.5809 | 0.5939 | 15 | 7 |
| valid_like | true_none | 597 | 0.8040 | 0.8074 | 6 | 4 |
| test_like | C1_3plus | 1117 | 0.7046 | 0.7055 | 18 | 17 |
| test_like | C2_only2 | 100 | 0.2000 | 0.2500 | 7 | 2 |
| test_like | C3_single_only | 122 | 0.0000 | 0.0820 | 10 | 0 |
| test_like | no_plausible_proxy | 64 | 0.0000 | 0.0000 | 0 | 0 |
| test_like | C4_V1_present_wrong_family | 245 | 0.0000 | 0.0571 | 14 | 0 |
| test_like | C5_V1_family_to_none | 194 | 0.0000 | 0.0670 | 13 | 0 |
| test_like | C6_V1_none_to_family | 109 | 0.0000 | 0.0642 | 7 | 0 |
| test_like | C7_music_software_streaming | 618 | 0.5227 | 0.5307 | 15 | 10 |
| test_like | true_none | 597 | 0.8174 | 0.8174 | 7 | 7 |

C4/C5/C6 are defined by V1 errors so their rescue rates can be compared on fixed clients. C1/C2/C3 use inferred semantic/MCC/price eligibility, not annotated true stream membership.

| View | Model | Wrong family | Family to none | None to family |
|---|---|---:|---:|---:|
| original | V1 | 409 | 97 | 122 |
| original | V2 candidate | 384 | 75 | 122 |
| valid_like | V1 | 401 | 144 | 117 |
| valid_like | V2 candidate | 401 | 126 | 115 |
| test_like | V1 | 402 | 194 | 109 |
| test_like | V2 candidate | 398 | 182 | 109 |

Controlled removal on 901 selected train clients (same target, latest k selected-stream card events retained, other events/refunds unchanged):

| Events | V1 accuracy | V2 accuracy | Delta points |
|---|---:|---:|---:|
| 4 | 0.7248 | 0.7137 | -1.1099 |
| 3 | 0.6393 | 0.6315 | -0.7769 |
| 2 | 0.5372 | 0.5594 | +2.2198 |
| 1 | 0.1365 | 0.3296 | +19.3119 |

One-event recovery rises from 13.65% to 32.96%, but four-event displacement violates the fixed gate. Full-sample removal F1 includes clients whose streams were not removed; the affected-cohort table is necessary to see that tradeoff. This is why a global F1 win alone was not accepted.

| View | Family | V1 F1 | V2 F1 |
|---|---|---:|---:|
| original | cloud | 0.6032 | 0.6194 |
| original | gym | 0.6042 | 0.6357 |
| original | insurance | 0.6930 | 0.7159 |
| original | mobile | 0.6421 | 0.6886 |
| original | music | 0.6207 | 0.6316 |
| original | software | 0.6388 | 0.6699 |
| original | streaming | 0.6278 | 0.6637 |
| original | none | 0.8127 | 0.8282 |
| valid_like | cloud | 0.6069 | 0.6069 |
| valid_like | gym | 0.6118 | 0.6256 |
| valid_like | insurance | 0.7067 | 0.6955 |
| valid_like | mobile | 0.6038 | 0.6313 |
| valid_like | music | 0.5633 | 0.5803 |
| valid_like | software | 0.6193 | 0.6300 |
| valid_like | streaming | 0.6009 | 0.6099 |
| valid_like | none | 0.7862 | 0.8000 |
| test_like | cloud | 0.5873 | 0.6174 |
| test_like | gym | 0.6094 | 0.6302 |
| test_like | insurance | 0.6791 | 0.6744 |
| test_like | mobile | 0.6011 | 0.6054 |
| test_like | music | 0.5257 | 0.5416 |
| test_like | software | 0.5827 | 0.5864 |
| test_like | streaming | 0.5569 | 0.5542 |
| test_like | none | 0.7631 | 0.7703 |

These estimates were used for research decisions and are not independent claims of deployment improvement. Next work should address displacement of plausible existing streams and use a new independent stress design; it should not relabel this failed gate as passing.
