# Changelog

Notable changes, particularly those that affect numerical reproducibility:

- As part of generalizing the $\mathbb{Z}_2$ ansatz config, the parameter order was changed. The new order was changed to the default in `manager.py` in commit `68d8821`. To use the old order, pass the flag `--legacy_param_order` to `manager.py`. 

- Pfaffians with imaginary coefficients (see derivation of the electric energy calculation) were dropped based on an argument in Appendix C of (Emonts, 2023). However this is unjustified in the multi-layer case. This change was merged into the main branch in commit `d190675`. Passing `drop_imag=True` to the function `generate_gauged_projector_terms()` restores the old behavior. 
