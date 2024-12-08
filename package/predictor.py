import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from classifiers import MyRandomForestClassifier

class FootballMatchPredictor:
    def __init__(self, data_path):
        # Load the data
        self.df = pd.read_csv(data_path)
        
        # Preprocessing features
        self.features = [
            'xg', 'xga', 'poss', 'sh', 'sot', 'dist', 'fk', 'pk', 'pkatt'
        ]
        
        # Preprocessing methods
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        # Prepare the data
        self.prepare_data()
        
        # Train the model
        self.train_model()
    
    def prepare_data(self):
        # Encode results
        self.df['result_encoded'] = (self.df['result'] == 'W').astype(int)
        
        # Group by team and calculate team-level statistics
        team_stats = self.df.groupby('team').agg({
            feature: 'mean' for feature in self.features
        }).reset_index()
        
        # Add performance indicator (win rate)
        team_performance = self.df.groupby('team')['result_encoded'].mean().reset_index()
        team_performance.columns = ['team', 'win_rate']
        
        # Merge team stats
        self.team_stats = pd.merge(team_stats, team_performance, on='team')
    
    def train_model(self):
        # Prepare features and target
        X = self.team_stats[self.features + ['win_rate']].values.astype(float)
        y = np.round(X[:, -1]).astype(int)  # Use win rate as target
        X = X[:, :-1]  # Remove win rate from features
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Train Random Forest Classifier
        self.classifier = MyRandomForestClassifier(n_estimators=100, max_depth=5)
        
        # Ensure X and y are NumPy arrays
        X = np.array(X)
        y = np.array(y)
        
        self.classifier.fit(X, y)
    
    def _get_team_features(self, team_name):
        # Get team features
        team_row = self.team_stats[self.team_stats['team'] == team_name]
        
        if len(team_row) == 0:
            raise ValueError(f"Team {team_name} not found in the dataset")
        
        # Extract features and scale
        team_features = team_row[self.features].values.astype(float)
        team_features = self.scaler.transform(team_features)
        
        return team_features
    
    def predict_match(self, home_team, away_team):
        # Get features for both teams
        home_features = self._get_team_features(home_team)
        away_features = self._get_team_features(away_team)
        
        # Create a combined feature vector
        match_features = np.concatenate([home_features, away_features], axis=1)
        
        # Predict
        prediction = self.classifier.predict(match_features)
        
        # Interpret prediction
        if prediction[0] == 1:
            return home_team
        else:
            return away_team

# Example usage
if __name__ == "__main__":
    # Replace with your actual data path
    predictor = FootballMatchPredictor('../data/premier_league_data2021-24.csv')
    
    # Predict match winner
    winner = predictor.predict_match('Chelsea', 'Manchester City')
    print(f"Predicted Winner: {winner}")