from flask import Flask, render_template, request
import sys
import os

# Add the path to the package to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from package.predictor import FootballMatchPredictor
from package.classifiers import MyRandomForestClassifier



app = Flask(__name__)   

rfc = MyRandomForestClassifier(n_estimators=100, max_depth=5)
predictor = FootballMatchPredictor('../data/premier_league_data2021-24.csv', rfc)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the team names from the form
        home_team = request.form.get('home_team')
        away_team = request.form.get('away_team')
        
        # Use the predictor to make a prediction
        try:
            winner = predictor.predict_match(home_team, away_team)
            return render_template('index.html', winner=winner, home_team=home_team, away_team=away_team)
        except ValueError as e:
            # Handle error if teams are not found in the dataset
            return render_template('index.html', error=str(e))
    
    return render_template('index.html', winner=None)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)