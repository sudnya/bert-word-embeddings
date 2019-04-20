
Goal 2: Use inverse reinforcement learning to discover rewards from different datasets

    Why: Answer: what do (lots of) users care about?
        Hypothesis: Quite a few different things, perhaps including some things that we don't already know about.

    Hypothesis: Clustering on features of good model in 1) will yield interesting classes of
                natural language tasks.

        TODO: Run clustering, consider multiple granularities (sentences, paragraphs, pages, documents).
            TODO: support chunking based on meaningful boundaries

        TODO: Support new data sources (e.g. reddit)

        TODO: Run clustering on new bigger datasets

Goal 3: Find a big deep net that can fit TBs of text data and is fast enough to train on 1 V100.

    TODO: validate that a 10M parameter transformer can do it


Goal 4: Curate a big dataset by training a crawler to find data on the internet that helps interesting
        goals discovered in 2.

        Name: SavantSearch: Mining Human Knowledge to Discover Narrow AIs


=== Finished ===

Goal 1: Implement a LM (embeddings) training pipeline

1. Create a vocab that can handle every word (e.g. using a tokenizer with fallback)
    Hypothesis: Using <UNK> discards information in the tail (and there is a lot of it).
        Main message: Use fallback in deep net LMs, not <UNK>
        TODO: Build NGRAM model (stretch build BERT model)
        TODO: Run on big dataset (all of guttenberg, lots of reddit, Tieba) for vocabs (10, 100, 1k, 10k, 100k, 1M) (ETA is 1TB in 1 day on 30 CPUs)
        TODO: Compare perplexity with and without <UNK> tokenizer
        TODO: Compare compression (LZMA) with and without <UNK> tokenizer
        TODO: Recompute lower bounds on perplexity using Shannon's method rectangular disitrbution, using these models on these datasets. How does <UNK> change the bound? Compare to Shannon's results.
2. Build a simple model for predicting the word using context (e.g. forward/backward using BERT)
    Done
3. Add a class based implementation of the vocab
    Done
4. Train on a basic dataset, report perplexity
    Done
