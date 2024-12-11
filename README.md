# Premier League Match Prediction
 
## Description

This project aims to predict the outcomes of Premier League matchups based on historical data from past seasons. The dataset, scraped from [fbref.com](https://fbref.com/), contains match data for over 2,281 games, including various attributes relevant to each game, such as team statistics, venues, formations, and match outcomes.

## Dataset

### Source
- Website: [fbref.com](https://fbref.com/)
- Scraping Process: HTML pages were requested and parsed using the `requests` library and `BeautifulSoup` to extract the relevant table data. The scraped data was then organized using `pandas` DataFrames.

### Attributes
The dataset includes the following columns:
- **date, time, comp, round, day, venue, result**: Details about the match, including date, time, competition, round, and venue.
- **gf, ga, opponent**: Goals scored for and against, along with the opposing team.
- **xg, xga, poss, attendance**: Advanced match statistics like expected goals, possession percentage, and attendance.
- **captain, formation, opp formation**: Team composition information such as the captain, team formation, and opponent's formation.
- **referee, match report, notes**: Additional match context, including the referee, match report, and notes.
- **sh, sot, dist, fk, pk, pkatt**: Match statistics related to shots, shot distance, free kicks, penalties, and penalty attempts.
- **season, team**: The season and team involved in the match.

### Classification Task
The project aims to classify the outcome of a future match based on this dataset. The goal is to predict the likelihood of a team winning, drawing, or losing their next match based on historical match data.

## Technical Implementation

### Tools and Libraries
- **requests** and **BeautifulSoup**: Used to scrape and parse data from fbref.com.
- **pandas**: For data manipulation and organization.
  
### Anticipated Challenges
- **Data Imbalance**: Handling potential imbalances between match outcomes (win, lose, draw) to avoid biased predictions.
- **Feature Selection**: Selecting relevant features to improve model accuracy without overfitting.
- **Data Preprocessing**: Cleaning data and dealing with missing or inconsistent values.
  
### Potential Impact
The results of this project could provide insights into key factors that influence match outcomes. By making predictive models available to teams, analysts, and fans, stakeholders gain a better understanding of the dynamics of Premier League matches.

### Stakeholders
- **Sports Analysts and Enthusiasts**: Those interested in data-driven insights into team performance.
- **Sports Teams and Coaches**: May leverage predictions to inform strategy and analyze match factors.
- **Data Scientists and Machine Learning Practitioners**: Those looking to explore sports analytics and model classification tasks on sports data.

### Classifiers
I chose to implement 3 classifiers for this prediction task:
1. random forest
2. k-nearest neighbors
3. decision tree

For my random forest classifier, I based my implementation off previous utility functions from CPSC322, along with the pseudocode provided by **[3]**. Essentially, this classifer builds an ensemble of decision tree classifiers. Each tree is trained on a bootstrapped sample of the training data, with a random subset of features considered for splits to ensure diversity. My random forest classifier creates predictions using majority voting. Key parameters include `max_depth` and `n_estimators` \
\
My decision trees themselves split nodes using the Gini impurity criterion **[4]**, and halts this splitting when the `max_depth` threshold is reached. \
\
Finally, my k-nearest neighbors classifier predicts class labels based on the majority label of the `n_neighbors` closest training samples to a test point. Distances between samples are computed using the Euclidean metric. \
\
To test these classifiers, I created a `pytest` suite, with tests for all classifiers. The unit tests can be found in the `/test` directory. I also created a data preprocessing method to fit my classifiers to the data in order to compare their performance with my actual data. Let's write some code to display these results.

### Summary
I learned quite a few things from this project, across different areas of computer science. Some of my takeaways from this project (I guess you could say "how tos"):
* how to use `Flask` to quickly set up web apps: I did not know you could render HTML so simply with a CSS framework and everything with a `Flask` server
* how to set up directories and import code across them with `.py/ipynb`
* how to create custom classifiers and use them to make predictions: previous projects have only ever involved using prebuilt classifiers
* how to create a pytest suite and integrate it into CI with Github actions, as well as a Python linter

This project also taught me how important it is to have working data preprocessing. The data has to be fine tuned to give good results, and creating a method from scratch for my specific use case was challenging. The concept of label encoding took a while to understand completely, and it took me some time to develop the correct data preprocessing method for my classification task. \
Future work would involve improving the Flask UI, supporting more insight into the model to be displayed on the frontend, further data scraping, and hosting of the `Flask` application

### Sources

[1] [Ryan Kelly - What is xG in football & how is the statistic calculated?](https://www.goal.com/en/news/what-is-xg-football-how-statistic-calculated/h42z0iiv8mdg1ub10iisg1dju) \
[2] [Vikas Paruchuri - Web Scraping Football Matches From The EPL With Python](https://www.youtube.com/watch?v=Nt7WJa2iu0s&t=1s) \
[3] [UW Madison Computer Science Department - Random Forests](https://pages.cs.wisc.edu/~matthewb/pages/notes/pdf/ensembles/RandomForests.pdf) \
[4] [Stack Exchange - When should I use Gini Impurity](https://datascience.stackexchange.com/questions/10228/when-should-i-use-gini-impurity-as-opposed-to-information-gain-entropy) \
[5] [DaisyUI docs](https://daisyui.com/)

data from: [fbref.com](https://fbref.com/), scarped with help of **[2]**

*ai usage: ChatGPT was used to help debug and write comments on code, it also helped rebuild my preprocessing method.* \
*python libraries: sklearn, pandas, numpy, searborn, matplotlib* 



```bash
thank you!
```

flask app hosted on Render @ https://cpsc322-final.onrender.com/ \
Data collection: `scarping.ipynb` at project root \
EDA: `mining.ipynb` at project root \
Classifier implementation: `package/classifiers.py` \
Flask app: `flask/app.py` \
Technical Report: `docs/techreport.ipynb`


