# Task 1 — EDA & Preprocessing

## Purpose
Explore the raw CFPB complaint dataset and produce a cleaned, filtered dataset
for the RAG pipeline (Tasks 2–4).

## Input
`data/raw/complaints.csv` — full CFPB complaint export (396,542 rows, 18 columns)

## Steps performed
1. Loaded the full dataset and checked column-level missing values.
2. Analyzed complaint volume by product and narrative word-count distribution.
3. Filtered to 4 target products: Credit Card, Personal Loan, Savings Account, Money Transfer.
4. Excluded checking-account complaints from the combined "Checking or savings account" category.
5. Dropped rows with no consumer narrative.
6. Cleaned narrative text: lowercased, removed boilerplate phrases, stripped byte-string artifacts and embedded newline characters.
7. Saved the result to `data/processed/filtered_complaints.csv`.

## Key findings
- Only 15,095 of 396,542 complaints (3.8%) include a narrative.
- After filtering to the 4 target products: **1,343 complaints**.
  - Credit Card: 896
  - Money Transfer: 201
  - Personal Loan: 123
  - Savings Account: 123
- Narrative length: median 137 words, mean 186 words, max 5,307 words.

## Output
`data/processed/filtered_complaints.csv` — used as input for Task 2 (chunking & embedding).

## How to run
Open `01_eda_preprocessing.ipynb` in Google Colab, upload `complaints.csv`, run all cells top to bottom.