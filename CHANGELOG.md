# Changelog

Z2 manager runs switched to the generic configuration in commit `68d8821bca576d6fbd091d7208cb02c7707da495`; among the resulting changes, parameter order now differs between the generic and legacy fixed-copy configs, so pass `--legacy_param_order` when using parameters saved in a legacy order. The immediately preceding commit, `e818fdb1e7930b3e73332c72fd3918eaa161d34f`, retains the old manager behavior and config-specific parameter orders.
