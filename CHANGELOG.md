# Changelog

Z2 manager runs switched to the generic configuration in commit `68d8821bca576d6fbd091d7208cb02c7707da495`; among the resulting changes, parameter order now differs between the generic and legacy fixed-copy configs, so pass `--legacy_param_order` when using parameters saved in a legacy order. Legacy-parameter comparisons were performed against the pre-change commit `3679276ce7fe78f781118393d22435016609136a`, where the old config-specific parameter orders were still in use.
