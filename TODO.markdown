
Goal 1: Create a DiversitySavant. Suggest text to encourage a culture of diversity. Goal: Improve gender equality in society (in companies).

    Task: Is text_A or text_B more appealing from a perspective of diversity?

    TODO: Use the labeling pipeline to label about 100 samples from one of the reddit clusters

    TODO: Write a keyboard web app.

        TODO: Write callbacks from Quill to inference server to draw underlines

Goal 2: Improve current implementation

    TODO: Run clustering, consider multiple granularities (sentences, paragraphs, pages, documents).
        TODO: support chunking based on meaningful boundaries

    TODO: Handle text chunks with different lengths.

Goal 3: Find a big deep net that can fit TBs of text data and is fast enough to train on 1 V100.

    TODO: validate that a 10M parameter transformer can do it
       Test the TURING card


Goal 4: Curate a big dataset by training a crawler to find data on the internet that helps interesting
        goals discovered in 2.

        Name: SavantSearch: Mining Human Knowledge to Discover Narrow AIs


=== Finished ===

Goal 2: Use inverse reinforcement learning to discover rewards from different datasets

    Why: Answer: what do (lots of) users care about?
        Hypothesis: Quite a few different things, perhaps including some things that we don't already know about.

    Hypothesis: Clustering on features of good model in 1) will yield interesting classes of
                natural language tasks. - Done - Conclusion: Yes


        TODO: Support new data sources (e.g. reddit) - Done

        TODO: Run clustering on new bigger datasets - Done
            Hypothesis: clusetrs will obey ZIpf's law : true

Cluster, 2254 (30541) - shills
Cluster, 1094 (30336) - gun law
Cluster, 997 (28751) - societal values *
Cluster, 3192 (26852) - racist bullshit
Cluster, 2143 (26773) - trolling
Cluster, 496 (22911) - public debate
Cluster, 567 (20943) - bias
Cluster, 1946 (18273) - property law
Cluster, 2138 (17372) - england
Cluster, 1320 (16788) - gender *
Cluster, 6234 (16075) - dating
Cluster, 1736 (14823) - immigration
Cluster, 1802 (13590) - young culture
Cluster, 3905 (13517) - jealousy
Cluster, 2096 (11969) - comparisons
Cluster, 7379 (11467) - gun law
Cluster, 6501 (11262) - political policy on jobs
Cluster, 5091 (10497) - EU conflict
Cluster, 2218 (9615) - social responsibility
Cluster, 7087 (9489) - countries
Cluster, 1412 (9389) - attraction
Cluster, 1879 (4268) - toxic heritage

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
