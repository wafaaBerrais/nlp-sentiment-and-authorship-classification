# Data

The corpora are not included in this repository.

## Sentiment: `movies1000`

2,000 English movie reviews (1,000 positive, 1,000 negative), organised as `movies1000/pos/*.txt` and `movies1000/neg/*.txt`.
This is the **Polarity dataset v2.0** by Pang & Lee (2004), publicly available from the
[Cornell movie review data page](https://www.cs.cornell.edu/people/pabo/movie-review-data/).

## Speaker attribution: Chirac vs Mitterrand

French political speeches split into sentences, one per line:

```text
<doc_id:sentence_id:label> sentence text      # label: C (Chirac) or M (Mitterrand)
```

- `corpus.tache1.learn.utf8`: 57,413 labelled sentences from 587 documents
- `corpus.tache1.test.utf8`: unlabelled test sentences (`<doc_id:sentence_id>`), scored on the course's evaluation platform

This corpus was provided by the RITAL course at Sorbonne Université and isn't redistributed here.

## Where the notebooks expect the data

The notebooks were run on Google Colab and read the data from Google Drive (`/content/drive/MyDrive/...`).
To run them elsewhere, change the path variables defined in the first cells of each notebook.
