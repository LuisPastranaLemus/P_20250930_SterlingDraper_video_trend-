# 🧭 Project Analysis
In a role of a video advertising analyst at Sterling & Draper. The objective is to analyze historical YouTube trends using a dashboard in Tableau Public, using data from the trending_by_time table.

The dashboard allows you to:

- Identify trending video categories.
- Analyze the distribution of categories by region.
- Detect the most popular categories in the United States and compare them with other regions.

The tool is designed for advertising planning managers, who use it daily to identify marketing opportunities based on content consumption on YouTube.

---

## 🔍 Project Overview (P-202YMMDD_Name)

Using the trending_by_time dataset, an interactive Tableau Public dashboard was developed to track daily and weekly trends across categories, countries, and timeframes. The dashboard is refreshed every 24 hours and integrates filters by date and region, ensuring flexible exploration of key metrics. The solution is tailored for video planning managers, enabling them to quickly identify high-performing categories, understand geographic differences, and optimize marketing decisions based on evolving audience preferences.

Key questions:

- Absolute and relative counts of trending videos
- Regional distribution of categories
- Cross-country comparisons of category popularity
- Spotlight on the U.S. market versus global patterns

Project Info explanation

The engineers prepared an aggregated table named trending_by_time, hosted in the youtube database and exported as trending_by_time.csv for this project. The table is refreshed daily at midnight UTC, ensuring up-to-date insights.

Considerations

All visualizations carry equal importance. Date and country filters apply to all charts to maintain consistency. Time-series charts must display trending_date on the X-axis, with absolute counts and percentages on the Y-axis. The dashboard is intended for daily use by planning managers.

Development Steps

- Load the dataset trending_by_time.csv into Tableau Public.
- Create the following visualizations:
    - Trending History: absolute count of trending videos by date and category.
    - Trending History (%): relative proportion of categories by date.
    - Regional Events: percentage distribution of categories by country.
    - Category-to-Country Table: absolute counts of categories per region.
- Combine visualizations into a single dashboard with interactive filters for date and region.
- Publish the dashboard on Tableau Public and validate accessibility across browsers.
- Use the dashboard to answer the guiding questions and prepare a short presentation with findings and charts.

Deliverables

- Interactive Tableau Public dashboard accessible online.
- Brief analytical report answering key questions with visual evidence.
- Tool for daily use by Sterling & Draper’s planning managers to support video marketing decisions.

---

## 🧮 Data Dictionary

This project has 1 table.

- `trending_by_time.csv` (video trending)
    `column:record_id`: primary key.
    `column:region`: country or geographic region.
    `column:trending_date`: date and time when the video trended.
    `category_title`: video category (e.g., Entertainment, Music, News & Politics).
    `videos_count`: number of videos trending on that date.
  
---

## 📚 Guided Foundations (Historical Context)

The notebook `00-guided-analysis_foundations.ipynb` reflects an early stage of my data analysis learning journey, guided by TripleTen. It includes data cleaning, basic EDA, and early feature exploration, serving as a foundational block before implementing the improved structure and methodology found in the main analysis.

---

## 📂 Project Structure

```bash
├── data/
│   ├── raw/              # Original dataset(s) in CSV format
│   ├── interim/          # Intermediate cleaned versions
│   └── processed/        # Final, ready-to-analyze dataset
│
├── notebooks/
│   ├── 00-guided-analysis_foundations.ipynb     ← Initial guided project (TripleTen)
│   ├── 01_cleaning.ipynb                        ← Custom cleaning 
│   ├── 02_feature_engineering.ipynb             ← Custom feature engineering
│   ├── 03_eda_and_insights.ipynb                ← Exploratory Data Analysis & visual storytelling
│   └── 04-sda_hypotheses.ipynb                  ← Business insights and hypothesis testing
│
├── src/
│   ├── init.py              # Initialization for reusable functions
│   ├── data_cleaning.py     # Data cleaning and preprocessing functions
│   ├── data_loader.py       # Loader for raw datasets
│   ├── eda.py               # Exploratory data analysis functions
│   ├── features.py          # Creation and transformation functions for new variables to support modeling and EDA
│   └── utils.py             # General utility functions for reusable helpers
│
├── outputs/
│   └── figures/          # Generated plots and visuals
│
├── requirements/
│   └── requirements.txt      # Required Python packages
│
├── .gitignore            # Files and folders to be ignored by Git
└── README.md             # This file
```
---

🛠️ Tools & Libraries

- Python 3.11
- os, pathlib, sys, pandas, NumPy, Matplotlib, seaborn, IPython.display, scipy.stats
- Jupyter Notebook
- Git & GitHub for version control
- Tableau Public

---

## 📌 Notes

This project is part of a personal learning portfolio focused on developing strong skills in data analysis, statistical thinking, and communication of insights. Constructive feedback is welcome.

---

## 👤 Author   
##### Luis Sergio Pastrana Lemus   
##### Engineer pivoting into Data Science | Passionate about insights, structure, and solving real-world problems with data.   
##### [GitHub Profile](https://github.com/LuisPastranaLemus)   
##### 📍 Querétaro, México     
##### 📧 Contact: luis.pastrana.lemus@engineer.com   
---

