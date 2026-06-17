"""Statistical bid/keyword optimization engine for Google Ads.

Modules
-------
data_pull  : GAQL query builder + GoogleAdsClient puller (with --demo synthetic data).
stats      : two-proportion z-test, Wilson CI, Empirical-Bayes shrinkage, decision function.
ngram      : 1/2/3-gram performance aggregation, harvesting + negative candidates.
semantic   : sentence-transformers (TF-IDF fallback) similarity to known-bad terms.
sheet_io   : gspread staging-sheet reader/writer (PENDING -> APPROVED -> EXECUTED).
writeback  : build Google Ads mutate operations from APPROVED rows; append audit log.
"""

__version__ = "0.1.0"
