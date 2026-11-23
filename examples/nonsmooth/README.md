> Historical experiment notes. For current entry points and data requirements, see the root README and docs/EXPERIMENTS.md. Generated results and caches are not included in the Git repository.

# Nonsmooth continuous-greedy benchmark

The final report is available as `results_full/REPORT_FINAL.md`,
`results_full/REPORT_FINAL.tex`, and `results_full/REPORT_FINAL.pdf`.

The final headline numbers use `results_full/retuned_results.json`.  The file
`results_full/greedy_24seeds.json` preserves the earlier pre-retuning
replication and is not the source for the final table.

Main commands:

```bash
python nonsmooth_continuous_greedy.py --outdir results_full
python replicate_full_greedy.py
python retuned_evaluation.py
python make_final_report.py
```

The scripts use only NumPy, SciPy, and Matplotlib.  All selection decisions use
the relative dual-residual indicator; truth errors are computed only on the
independent deployment set.
