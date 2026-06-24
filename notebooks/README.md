# Task 1 — EDA and Preprocessing

**Project:** CrediTrust Financial — RAG-Powered Complaint Chatbot  
**Dataset:** CFPB Consumer Complaint Database  
**Notebook:** `notebooks/01_eda_preprocessing.ipynb`

---

## What This Task Does

Loads the full CFPB complaint dataset, filters it to four target financial products, explores the data through charts and statistics, cleans the complaint narratives, and saves a processed dataset ready for Task 2 embedding.

---

## Dataset Overview

| Metric | Value |
|---|---|
| Raw records in source file | 17,536,785 lines (9,609,797 data rows) |
| After product filter + narrative filter | 477,589 |
| After cleaning pass | **477,114** |
| Final file | `data/processed/filtered_complaints.csv` |

> The raw file was read in chunks of 100,000 rows at a time. Only 9 of the 18 original columns were loaded. This approach kept peak memory usage at **662 MB** — safe on a standard laptop.

---

## Products Covered

| Category | Count |
|---|---|
| Credit Card | 188,400 |
| Savings Account | 154,044 |
| Money Transfer | 98,101 |
| Personal Loan | 37,044 |

The CFPB uses long inconsistent product names across years (e.g. "Payday loan, title loan, or personal loan" and "Consumer Loan" both map to **Personal Loan**). All variants were standardized to the four labels above.

---

## Key EDA Findings

**Narrative length**
- Mean: 207 words — Median: 139 words — Max: 6,469 words
- Most complaints fall between 50–200 words (63% of the dataset)
- 55,365 complaints exceed 400 words — right-skewed distribution

**Boilerplate patterns found in raw text**

| Pattern | Occurrences |
|---|---|
| XXXX redactions | 366,976 |
| `{$amount}` money format | 199,198 |
| "I am writing" opener | 26,744 |
| "To whom it may concern" | 3,304 |
| "Dear CFPB" | 1,187 |

**Complaint volume over time:** Steady growth from 2015 to 2024, with a sharp spike in early 2025 — likely tied to a specific regulatory event worth investigating through the chatbot.

---

## Cleaning Steps Applied

1. Lowercase all text
2. Remove "Dear CFPB", "To whom it may concern", and "I am writing to..." openers
3. Replace `{$2500.00}` style amounts with the token `AMOUNT`
4. Remove XXXX redactions
5. Strip special characters — keep letters, numbers, and basic punctuation
6. Collapse extra whitespace
7. Drop rows where cleaned text is under 10 words (475 rows removed)

---

## Output Columns

```
Date received, Product, Sub-product, Issue, Sub-issue,
Consumer complaint narrative, Company, State, Complaint ID,
product_category, word_count, year_month, clean_narrative
```

---

## How to Run

```bash
# install dependencies
pip install pandas matplotlib seaborn

# open notebook
jupyter notebook notebooks/01_eda_preprocessing.ipynb
```

Make sure your raw CSV is at `data/raw/complaints.csv` — the path is set in the first code cell and can be changed in one place (`RAW_CSV = "..."`).

Plots are saved to `notebooks/eda_plots/` at 150 DPI.

---

## Files Produced

```
data/
└── processed/
    └── filtered_complaints.csv   ← input for Task 2

notebooks/
└── eda_plots/
    ├── product_distribution.png
    ├── word_count_distribution.png
    └── complaints_over_time.png
```
