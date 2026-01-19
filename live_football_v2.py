"""
Live Football Under 1.5 Goals Predictor V2
==========================================

Improved predictor that fixes accuracy issues from V1:
1. Collects HISTORICAL finished matches for proper training
2. Uses team scoring statistics (goals per game, clean sheet %)
3. Implements probability calibration to prevent overconfident predictions  
4. Adds confidence bounds and data quality checks
5. Falls back to statistical baselines when insufficient data

Usage: python live_football_v2.py

Output CSVs (with timestamp, e.g., 2026-01-18_13-54):
- data/v2_predictions_YYYY-MM-DD_HH-MM.csv           - Predictions with confidence
- data/v2_predictions_YYYY-MM-DD_HH-MM.result.csv   - Results template for tracking accuracy
- data/v2_historical_matches.csv                     - Historical match data for training
- data/v2_team_stats.csv                             - Team scoring statistics
"""

import asyncio
import pandas as pd
import numpy as np
import time
import json
import os
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict

warnings.filterwarnings('ignore')

# ML imports
try:
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, brier_score_loss
    from sklearn.preprocessing import StandardScaler
    HAS_ML = True
except ImportError:
    HAS_ML = False
    print("⚠️ ML libraries not found. Install with: pip install xgboost lightgbm scikit-learn")

# Sofascore imports
from sofascore_wrapper.api import SofascoreAPI
from sofascore_wrapper.match import Match
from sofascore_wrapper.team import Team

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Generate timestamp for unique filenames (format: YYYY-MM-DD_HH-MM)
RUN_TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H-%M')
FILENAME_PREFIX = f"v2_predictions_{RUN_TIMESTAMP}"

# Time between API requests to avoid rate limiting
REQUEST_DELAY = 0.5

# Minimum training data requirements
MIN_HISTORICAL_MATCHES = 50
MIN_TRAINING_SAMPLES = 30

# Statistical fallbacks (league averages)
DEFAULT_UNDER_1_5_RATE = 0.28  # ~28% of matches have 0 or 1 goals
DEFAULT_UNDER_2_5_RATE = 0.50  # ~50% have under 2.5 goals

# Low-scoring team thresholds
LOW_SCORING_GOALS_THRESHOLD = 1.3  # Teams scoring fewer than this on average are "low-scoring"
MIN_ELAPSED_MINUTES = 30  # Minimum minutes into match
MAX_ELAPSED_MINUTES = 45  # Maximum minutes (exclude second half)


# ============================================================================
# HELPER FUNCTIONS  
# ============================================================================

def safe_get(data: dict, *keys, default=None):
    """Safely get nested dictionary values."""
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key, default)
        else:
            return default
    return data


def parse_value(val):
    """Convert stat value to float, handling percentages."""
    if val is None:
        return 0
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        val = val.strip().replace('%', '')
        try:
            return float(val)
        except:
            return 0
    return 0


def parse_stats(stats_response) -> Dict[str, Any]:
    """Parse match statistics into flat dict."""
    flat = {}
    
    stats_list = []
    if isinstance(stats_response, dict):
        stats_list = stats_response.get('statistics', [])
    elif isinstance(stats_response, list):
        stats_list = stats_response
    
    if not stats_list:
        return flat
    
    # Get ALL period stats, or 1ST if not available
    period_stats = None
    for period in ['ALL', '1ST', '2ND']:
        period_stats = next((s for s in stats_list if s.get('period') == period), None)
        if period_stats:
            break
    
    if not period_stats:
        return flat
    
    for group in period_stats.get('groups', []):
        for item in group.get('statisticsItems', []):
            name = item.get('name', '').lower().replace(' ', '_').replace('-', '_')
            flat[f"home_{name}"] = item.get('homeValue', item.get('home'))
            flat[f"away_{name}"] = item.get('awayValue', item.get('away'))
    
    return flat


# ============================================================================
# DATA COLLECTION - HISTORICAL MATCHES
# ============================================================================

async def collect_team_historical_matches(api: SofascoreAPI, team_id: int, team_name: str, 
                                          num_matches: int = 20) -> List[Dict]:
    """
    Collect historical finished matches for a team.
    Returns list of match dictionaries with goal data.
    """
    matches = []
    
    try:
        team = Team(api, team_id=team_id)
        # Get last fixtures (finished matches) - correct API method
        events_response = await team.last_fixtures()
        # Response is a list directly, not a dict with 'events' key
        events = events_response if isinstance(events_response, list) else events_response.get('events', [])
        
        for event in events[:num_matches]:
            # Only use finished matches
            status = safe_get(event, 'status', 'type', default='')
            if status != 'finished':
                continue
                
            home_score = safe_get(event, 'homeScore', 'current', default=0) or 0
            away_score = safe_get(event, 'awayScore', 'current', default=0) or 0
            total_goals = home_score + away_score
            
            is_home = safe_get(event, 'homeTeam', 'id') == team_id
            
            match_data = {
                'match_id': event.get('id'),
                'team_id': team_id,
                'team_name': team_name,
                'is_home': is_home,
                'opponent_id': safe_get(event, 'awayTeam', 'id') if is_home else safe_get(event, 'homeTeam', 'id'),
                'opponent_name': safe_get(event, 'awayTeam', 'name') if is_home else safe_get(event, 'homeTeam', 'name'),
                'goals_scored': home_score if is_home else away_score,
                'goals_conceded': away_score if is_home else home_score,
                'total_goals': total_goals,
                'under_1_5': 1 if total_goals < 2 else 0,
                'under_2_5': 1 if total_goals < 3 else 0,
                'clean_sheet': 1 if (away_score == 0 if is_home else home_score == 0) else 0,
                'failed_to_score': 1 if (home_score == 0 if is_home else away_score == 0) else 0,
                'tournament': safe_get(event, 'tournament', 'name'),
            }
            matches.append(match_data)
            
    except Exception as e:
        print(f"    ⚠️ Error fetching history for {team_name}: {e}")
    
    return matches


def calculate_team_stats(matches: List[Dict]) -> Dict[str, float]:
    """
    Calculate team statistics from historical matches.
    Returns dict of statistics.
    """
    if not matches:
        return {
            'goals_scored_avg': 1.3,  # Default values
            'goals_conceded_avg': 1.3,
            'clean_sheet_pct': 0.25,
            'failed_to_score_pct': 0.25,
            'under_1_5_pct': DEFAULT_UNDER_1_5_RATE,
            'under_2_5_pct': DEFAULT_UNDER_2_5_RATE,
            'matches_count': 0,
        }
    
    df = pd.DataFrame(matches)
    
    return {
        'goals_scored_avg': df['goals_scored'].mean(),
        'goals_conceded_avg': df['goals_conceded'].mean(),
        'clean_sheet_pct': df['clean_sheet'].mean(),
        'failed_to_score_pct': df['failed_to_score'].mean(),
        'under_1_5_pct': df['under_1_5'].mean(),
        'under_2_5_pct': df['under_2_5'].mean(),
        'matches_count': len(df),
    }


async def collect_historical_training_data(api: SofascoreAPI, live_events: List[Dict]) -> pd.DataFrame:
    """
    Collect historical match data for all teams in live matches.
    This creates proper training data with known outcomes.
    """
    print("\n" + "="*70)
    print("STEP 2: COLLECTING HISTORICAL DATA FOR TRAINING")
    print("="*70)
    
    all_historical = []
    team_stats = {}
    processed_teams = set()
    
    # Get unique teams from live matches
    teams_to_fetch = []
    for event in live_events:
        home_id = safe_get(event, 'homeTeam', 'id')
        home_name = safe_get(event, 'homeTeam', 'name', default='Unknown')
        away_id = safe_get(event, 'awayTeam', 'id')
        away_name = safe_get(event, 'awayTeam', 'name', default='Unknown')
        
        if home_id and home_id not in processed_teams:
            teams_to_fetch.append((home_id, home_name))
            processed_teams.add(home_id)
        if away_id and away_id not in processed_teams:
            teams_to_fetch.append((away_id, away_name))
            processed_teams.add(away_id)
    
    print(f"Fetching historical data for {len(teams_to_fetch)} unique teams...")
    
    for idx, (team_id, team_name) in enumerate(teams_to_fetch):
        print(f"  [{idx+1}/{len(teams_to_fetch)}] {team_name}...")
        
        matches = await collect_team_historical_matches(api, team_id, team_name, num_matches=15)
        all_historical.extend(matches)
        
        # Calculate stats for this team
        team_stats[team_id] = calculate_team_stats(matches)
        team_stats[team_id]['team_name'] = team_name
        
        await asyncio.sleep(REQUEST_DELAY)
    
    # Create DataFrames
    df_historical = pd.DataFrame(all_historical)
    df_team_stats = pd.DataFrame.from_dict(team_stats, orient='index')
    
    # Remove duplicate matches (same match appears for both teams)
    if not df_historical.empty:
        df_historical = df_historical.drop_duplicates(subset=['match_id'])
    
    # Save CSVs
    if not df_historical.empty:
        df_historical.to_csv(f"{OUTPUT_DIR}/v2_historical_matches.csv", index=False)
        print(f"\n✓ Saved {len(df_historical)} historical matches")
    
    if not df_team_stats.empty:
        df_team_stats.to_csv(f"{OUTPUT_DIR}/v2_team_stats.csv", index=True)
        print(f"✓ Saved stats for {len(df_team_stats)} teams")
    
    return df_historical, df_team_stats, team_stats


# ============================================================================
# DATA COLLECTION - LIVE MATCHES
# ============================================================================

async def collect_live_matches() -> Tuple[List[Dict], pd.DataFrame]:
    """
    Collect all live football matches with statistics.
    """
    print("\n" + "="*70)
    print("STEP 1: COLLECTING LIVE MATCHES")
    print("="*70)
    
    api = SofascoreAPI()
    all_matches = []
    
    try:
        print("\nFetching live matches...")
        finder = Match(api)
        live_response = await finder.live_games()
        events = live_response.get('events', [])
        
        # Filter to 0-0 matches that are in first half only (30-45 min elapsed)
        # This excludes second half matches for better predictions
        MAX_MATCHES = 100
        
        target_events = []
        for event in events:
            home_score = safe_get(event, 'homeScore', 'current', default=0) or 0
            away_score = safe_get(event, 'awayScore', 'current', default=0) or 0
            start_ts = event.get('startTimestamp', 0)
            elapsed_min = (time.time() - start_ts) / 60 if start_ts else 0
            
            # Keep 0-0 matches that are 30-45 minutes in (first half only)
            if (home_score == 0 and away_score == 0 and 
                elapsed_min >= MIN_ELAPSED_MINUTES and 
                elapsed_min <= MAX_ELAPSED_MINUTES):
                target_events.append(event)
        
        # Limit to MAX_MATCHES to avoid rate limiting
        if len(target_events) > MAX_MATCHES:
            print(f"⚠️ Limiting from {len(target_events)} to {MAX_MATCHES} matches to avoid rate limiting")
            target_events = target_events[:MAX_MATCHES]
        
        print(f"Found {len(events)} live matches, {len(target_events)} are 0-0 in first half (30-45 min)")
        
        for idx, event in enumerate(target_events):
            match_id = event.get('id')
            home_team = safe_get(event, 'homeTeam', 'name', default='Unknown')
            away_team = safe_get(event, 'awayTeam', 'name', default='Unknown')
            start_ts = event.get('startTimestamp', 0)
            elapsed_min = (time.time() - start_ts) / 60 if start_ts else 0
            
            print(f"  [{idx+1}/{len(target_events)}] {home_team} vs {away_team} ({elapsed_min:.0f} min)")
            
            match_data = {
                'match_id': match_id,
                'tournament': safe_get(event, 'tournament', 'name'),
                'home_team': home_team,
                'away_team': away_team,
                'home_team_id': safe_get(event, 'homeTeam', 'id'),
                'away_team_id': safe_get(event, 'awayTeam', 'id'),
                'home_score': 0,
                'away_score': 0,
                'status': safe_get(event, 'status', 'description', default='Unknown'),
                'elapsed_minutes': round(elapsed_min, 1),
            }
            
            # Get live statistics
            try:
                match_obj = Match(api, match_id=match_id)
                stats = await match_obj.stats()
                parsed = parse_stats(stats)
                match_data.update(parsed)
                await asyncio.sleep(REQUEST_DELAY)
            except Exception:
                pass
            
            # Get H2H
            try:
                match_obj = Match(api, match_id=match_id)
                h2h = await match_obj.h2h()
                team_duel = h2h.get('teamDuel', {})
                match_data['h2h_home_wins'] = team_duel.get('homeWins', 0)
                match_data['h2h_away_wins'] = team_duel.get('awayWins', 0)
                match_data['h2h_draws'] = team_duel.get('draws', 0)
                await asyncio.sleep(REQUEST_DELAY)
            except Exception:
                pass
                
            all_matches.append(match_data)
        
        # Collect historical data for training
        df_historical, df_team_stats, team_stats = await collect_historical_training_data(api, target_events)
        
        await api.close()
        
    except Exception as e:
        print(f"Error during collection: {e}")
        await api.close()
        return [], pd.DataFrame()
    
    df_live = pd.DataFrame(all_matches)
    df_live.to_csv(f"{OUTPUT_DIR}/v2_live_matches.csv", index=False)
    print(f"\n✓ Saved {len(df_live)} live 0-0 matches")
    
    return all_matches, df_live, df_historical, team_stats


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_training_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create features for training from historical match data.
    """
    if df.empty:
        return pd.DataFrame(), []
    
    df = df.copy()
    
    # Basic features from raw data
    feature_cols = [
        'goals_scored', 'goals_conceded', 'total_goals',
    ]
    
    # Only keep columns that exist
    existing = [c for c in feature_cols if c in df.columns]
    
    X = df[existing].copy()
    X = X.fillna(0)
    
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    return X, existing


def create_prediction_features(match_data: Dict, team_stats: Dict) -> Dict[str, float]:
    """
    Create features for prediction using team statistics.
    """
    home_id = match_data.get('home_team_id')
    away_id = match_data.get('away_team_id')
    
    # Get stats for each team (use defaults if not available)
    home_stats = team_stats.get(home_id, {
        'goals_scored_avg': 1.3,
        'goals_conceded_avg': 1.3,
        'clean_sheet_pct': 0.25,
        'failed_to_score_pct': 0.25,
        'under_1_5_pct': DEFAULT_UNDER_1_5_RATE,
        'under_2_5_pct': DEFAULT_UNDER_2_5_RATE,
    })
    away_stats = team_stats.get(away_id, {
        'goals_scored_avg': 1.3,
        'goals_conceded_avg': 1.3,
        'clean_sheet_pct': 0.25,
        'failed_to_score_pct': 0.25,
        'under_1_5_pct': DEFAULT_UNDER_1_5_RATE,
        'under_2_5_pct': DEFAULT_UNDER_2_5_RATE,
    })
    
    # Calculate expected goals in this match
    home_expected_score = (home_stats.get('goals_scored_avg', 1.3) + 
                           away_stats.get('goals_conceded_avg', 1.3)) / 2
    away_expected_score = (away_stats.get('goals_scored_avg', 1.3) + 
                           home_stats.get('goals_conceded_avg', 1.3)) / 2
    expected_total = home_expected_score + away_expected_score
    
    # Probability both teams keep clean sheet (0-0)
    both_cs_prob = (home_stats.get('clean_sheet_pct', 0.25) * 
                    away_stats.get('clean_sheet_pct', 0.25))
    
    # Probability one team fails to score
    one_fts_prob = (home_stats.get('failed_to_score_pct', 0.25) + 
                    away_stats.get('failed_to_score_pct', 0.25)) / 2
    
    # Historical under rates
    combined_under_1_5 = (home_stats.get('under_1_5_pct', DEFAULT_UNDER_1_5_RATE) + 
                          away_stats.get('under_1_5_pct', DEFAULT_UNDER_1_5_RATE)) / 2
    combined_under_2_5 = (home_stats.get('under_2_5_pct', DEFAULT_UNDER_2_5_RATE) + 
                          away_stats.get('under_2_5_pct', DEFAULT_UNDER_2_5_RATE)) / 2
    
    # Time factor - more time passed at 0-0 increases under probability
    elapsed = match_data.get('elapsed_minutes', 30)
    time_factor = min(elapsed / 45, 1.0)  # 0-1 scale, capped at 45 min
    
    # Adjusted under probability based on time
    # If still 0-0 at 40 min, probability of Under increases
    time_adjusted_under = combined_under_1_5 * (1 + time_factor * 0.3)
    time_adjusted_under = min(time_adjusted_under, 0.85)  # Cap at 85%
    
    features = {
        'home_goals_scored_avg': home_stats.get('goals_scored_avg', 1.3),
        'home_goals_conceded_avg': home_stats.get('goals_conceded_avg', 1.3),
        'away_goals_scored_avg': away_stats.get('goals_scored_avg', 1.3),
        'away_goals_conceded_avg': away_stats.get('goals_conceded_avg', 1.3),
        'expected_total_goals': expected_total,
        'home_clean_sheet_pct': home_stats.get('clean_sheet_pct', 0.25),
        'away_clean_sheet_pct': away_stats.get('clean_sheet_pct', 0.25),
        'both_cs_prob': both_cs_prob,
        'home_fts_pct': home_stats.get('failed_to_score_pct', 0.25),
        'away_fts_pct': away_stats.get('failed_to_score_pct', 0.25),
        'one_fts_prob': one_fts_prob,
        'combined_under_1_5': combined_under_1_5,
        'combined_under_2_5': combined_under_2_5,
        'elapsed_minutes': elapsed,
        'time_factor': time_factor,
        'time_adjusted_under': time_adjusted_under,
        'home_historical_matches': home_stats.get('matches_count', 0),
        'away_historical_matches': away_stats.get('matches_count', 0),
    }
    
    return features


# ============================================================================
# MODEL TRAINING (WITH CALIBRATION)
# ============================================================================

class CalibratedUnder15Model:
    """
    Calibrated ensemble model for Under 1.5 goals prediction.
    Uses probability calibration to prevent overconfident predictions.
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        self.training_accuracy = 0
        self.training_samples = 0
    
    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train the calibrated ensemble model."""
        print("\n--- Training Calibrated Under 1.5 Model ---")
        
        if len(df) < MIN_TRAINING_SAMPLES:
            print(f"⚠️ Insufficient data: {len(df)} samples (need {MIN_TRAINING_SAMPLES}+)")
            print("   Model will use statistical fallback for predictions.")
            return {}
        
        # Create features and target
        self.feature_names = ['goals_scored', 'goals_conceded', 'total_goals']
        existing = [c for c in self.feature_names if c in df.columns]
        
        if not existing:
            print("❌ No valid features found")
            return {}
        
        X = df[existing].fillna(0)
        y = (df['total_goals'] < 2).astype(int)  # Under 1.5 = less than 2 goals
        
        print(f"Training on {len(df)} matches with {len(existing)} features")
        print(f"Target distribution: Under 1.5 = {y.sum()} ({y.mean():.1%}), Over 1.5 = {len(y)-y.sum()}")
        
        if len(y.unique()) < 2:
            print("❌ Need both classes for training")
            return {}
        
        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
        except:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create base models with balanced class weights
        xgb = XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            scale_pos_weight=len(y_train[y_train==0]) / max(1, len(y_train[y_train==1])),
            use_label_encoder=False, eval_metric='logloss', verbosity=0
        )
        
        lgb = LGBMClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            class_weight='balanced', verbose=-1
        )
        
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=5, 
            class_weight='balanced', random_state=42
        )
        
        # Create voting ensemble
        ensemble = VotingClassifier(
            estimators=[('xgb', xgb), ('lgb', lgb), ('rf', rf)],
            voting='soft'
        )
        
        # Calibrate the ensemble for reliable probabilities
        self.model = CalibratedClassifierCV(ensemble, cv=3, method='isotonic')
        
        try:
            self.model.fit(X_train_scaled, y_train)
        except Exception as e:
            print(f"⚠️ Calibration failed, using uncalibrated model: {e}")
            ensemble.fit(X_train_scaled, y_train)
            self.model = ensemble
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        brier = brier_score_loss(y_test, y_proba)
        
        # Check for prediction variance (model collapse detection)
        proba_std = np.std(y_proba)
        if proba_std < 0.05:
            print(f"⚠️ WARNING: Model outputs have low variance ({proba_std:.3f})")
            print("   This indicates possible model collapse. Using statistical fallback.")
            self.is_trained = False
            return {}
        
        self.training_accuracy = accuracy
        self.training_samples = len(df)
        self.feature_names = existing
        self.is_trained = True
        
        print(f"✓ Accuracy: {accuracy:.1%}")
        print(f"✓ Brier Score: {brier:.4f} (lower is better, <0.25 is good)")
        print(f"✓ Probability Std Dev: {proba_std:.3f} (>0.05 is good)")
        
        return {
            'accuracy': accuracy,
            'brier_score': brier,
            'proba_std': proba_std,
            'samples': len(df),
        }
    
    def predict(self, features: Dict[str, float]) -> Tuple[float, str]:
        """
        Predict Under 1.5 probability with confidence level.
        Returns (probability, confidence_level)
        """
        if not self.is_trained:
            # Use statistical fallback
            base_prob = features.get('time_adjusted_under', DEFAULT_UNDER_1_5_RATE)
            return base_prob, 'STATISTICAL'
        
        # Create feature vector
        X = pd.DataFrame([{k: features.get(k, 0) for k in self.feature_names}])
        X = X.fillna(0)
        X_scaled = self.scaler.transform(X)
        
        proba = self.model.predict_proba(X_scaled)[0, 1]
        
        # Blend with statistical estimate for robustness
        stat_estimate = features.get('time_adjusted_under', DEFAULT_UNDER_1_5_RATE)
        
        # Weight depends on training data quality
        if self.training_samples >= 100:
            model_weight = 0.7
        elif self.training_samples >= 50:
            model_weight = 0.5
        else:
            model_weight = 0.3
        
        blended_proba = proba * model_weight + stat_estimate * (1 - model_weight)
        
        # Determine confidence
        if self.training_samples >= 100 and self.training_accuracy >= 0.65:
            if blended_proba >= 0.7:
                confidence = 'HIGH'
            elif blended_proba >= 0.5:
                confidence = 'MEDIUM'
            else:
                confidence = 'LOW'
        else:
            confidence = 'LIMITED_DATA'
        
        return blended_proba, confidence


# ============================================================================
# PREDICTION ENGINE
# ============================================================================

def is_low_scoring_team(team_id: int, team_stats: Dict, threshold: float = LOW_SCORING_GOALS_THRESHOLD) -> bool:
    """
    Check if a team is a low-scoring team based on historical goals average.
    Returns True if the team's goals_scored_avg is below the threshold.
    """
    stats = team_stats.get(team_id, {})
    goals_avg = stats.get('goals_scored_avg', 1.3)  # Default to threshold if unknown
    return goals_avg < threshold


def make_predictions(live_matches: List[Dict], team_stats: Dict, 
                    model: CalibratedUnder15Model) -> pd.DataFrame:
    """
    Make predictions for all live 0-0 matches.
    Filters to only include matches with low-scoring teams.
    """
    print("\n" + "="*70)
    print("STEP 3: FILTERING FOR LOW-SCORING TEAMS & MAKING PREDICTIONS")
    print("="*70)
    
    # Filter for matches where BOTH teams are low-scoring
    filtered_matches = []
    for match in live_matches:
        home_id = match.get('home_team_id')
        away_id = match.get('away_team_id')
        
        home_low_scoring = is_low_scoring_team(home_id, team_stats)
        away_low_scoring = is_low_scoring_team(away_id, team_stats)
        
        home_stats = team_stats.get(home_id, {})
        away_stats = team_stats.get(away_id, {})
        home_avg = home_stats.get('goals_scored_avg', 1.3)
        away_avg = away_stats.get('goals_scored_avg', 1.3)
        
        if home_low_scoring and away_low_scoring:
            filtered_matches.append(match)
            print(f"  ✓ {match.get('home_team')} ({home_avg:.2f} gpg) vs {match.get('away_team')} ({away_avg:.2f} gpg) - LOW SCORING")
        else:
            print(f"  ✗ {match.get('home_team')} ({home_avg:.2f} gpg) vs {match.get('away_team')} ({away_avg:.2f} gpg) - HIGH SCORING (excluded)")
    
    print(f"\n→ Filtered from {len(live_matches)} to {len(filtered_matches)} matches with low-scoring teams (threshold: {LOW_SCORING_GOALS_THRESHOLD} gpg)")
    
    if not filtered_matches:
        print("\n⚠️ No matches with low-scoring teams found!")
        return pd.DataFrame()
    
    predictions = []
    
    for match in filtered_matches:
        # Create features
        features = create_prediction_features(match, team_stats)
        
        # Get model prediction
        proba, confidence = model.predict(features)
        
        # Create prediction record
        pred = {
            'match_id': match.get('match_id'),
            'tournament': match.get('tournament'),
            'home_team': match.get('home_team'),
            'away_team': match.get('away_team'),
            'home_score': match.get('home_score', 0),
            'away_score': match.get('away_score', 0),
            'status': match.get('status'),
            'elapsed_minutes': match.get('elapsed_minutes'),
            'under_1_5_probability': round(proba, 4),
            'confidence': confidence,
            'expected_total_goals': round(features.get('expected_total_goals', 2.6), 2),
            'home_goals_avg': round(features.get('home_goals_scored_avg', 0), 2),
            'away_goals_avg': round(features.get('away_goals_scored_avg', 0), 2),
            'home_historical_matches': features.get('home_historical_matches', 0),
            'away_historical_matches': features.get('away_historical_matches', 0),
        }
        
        # Recommendation
        if proba >= 0.6 and confidence in ['HIGH', 'MEDIUM']:
            pred['recommendation'] = 'UNDER 1.5 Goals'
        elif proba >= 0.5 and confidence == 'HIGH':
            pred['recommendation'] = 'LEAN UNDER 1.5 Goals'
        elif proba <= 0.35:
            pred['recommendation'] = 'OVER 1.5 Goals'
        else:
            pred['recommendation'] = 'NO STRONG SIGNAL'
        
        predictions.append(pred)
    
    df_pred = pd.DataFrame(predictions)
    
    if not df_pred.empty:
        # Sort by probability
        df_pred = df_pred.sort_values('under_1_5_probability', ascending=False)
        
        # Save predictions with timestamp
        predictions_file = f"{OUTPUT_DIR}/{FILENAME_PREFIX}.csv"
        df_pred.to_csv(predictions_file, index=False)
        print(f"\n✓ Saved {len(df_pred)} predictions to {predictions_file}")
        
        # Create results template file for tracking actual outcomes
        # This file can be filled in later with actual final scores
        df_results = df_pred[['match_id', 'tournament', 'home_team', 'away_team', 
                               'under_1_5_probability', 'confidence', 'recommendation']].copy()
        df_results['actual_home_score'] = ''  # To be filled after match ends
        df_results['actual_away_score'] = ''  # To be filled after match ends
        df_results['actual_total_goals'] = '' # To be filled after match ends
        df_results['was_under_1_5'] = ''      # To be filled: 1 if total < 2, else 0
        df_results['prediction_correct'] = '' # To be filled: 1 if prediction matched, else 0
        df_results['prediction_timestamp'] = RUN_TIMESTAMP
        
        results_file = f"{OUTPUT_DIR}/{FILENAME_PREFIX}.result.csv"
        df_results.to_csv(results_file, index=False)
        print(f"✓ Created results template: {results_file}")
    
    # Display predictions
    print("\n" + "-"*70)
    print("PREDICTIONS - Matches sorted by Under 1.5 probability:")
    print("-"*70)
    
    for _, row in df_pred.iterrows():
        prob = row['under_1_5_probability']
        conf = row['confidence']
        rec = row['recommendation']
        exp_goals = row['expected_total_goals']
        
        print(f"\n  {row['home_team']} vs {row['away_team']}")
        print(f"    Status: {row['status']} | Elapsed: {row['elapsed_minutes']:.0f} min")
        print(f"    Expected Goals: {exp_goals} | Under 1.5 Prob: {prob:.1%}")
        print(f"    Confidence: {conf} | >>> {rec}")
    
    return df_pred


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main execution."""
    print("\n" + "="*70)
    print("  LIVE FOOTBALL UNDER 1.5 GOALS PREDICTOR V2")
    print("  With Historical Data + Calibrated Models")
    print("="*70)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    # Step 1 & 2: Collect live matches and historical data
    result = await collect_live_matches()
    if len(result) < 4:
        print("\n❌ No live 0-0 matches found. Exiting.")
        return
    
    live_matches, df_live, df_historical, team_stats = result
    
    if not live_matches:
        print("\n❌ No live 0-0 matches found. Exiting.")
        return
    
    # Train model on historical data
    model = CalibratedUnder15Model()
    if not df_historical.empty:
        model.train(df_historical)
    else:
        print("\n⚠️ No historical data. Using statistical predictions only.")
    
    # Step 3: Make predictions
    predictions = make_predictions(live_matches, team_stats, model)
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"\nTotal time: {elapsed:.1f} seconds")
    print(f"\nGenerated files in '{OUTPUT_DIR}/' folder:")
    print(f"  • v2_live_matches.csv             - {len(df_live)} live 0-0 matches")
    print(f"  • v2_historical_matches.csv       - Historical matches for training")
    print(f"  • v2_team_stats.csv               - Team statistics")
    print(f"  • {FILENAME_PREFIX}.csv           - {len(predictions)} predictions")
    print(f"  • {FILENAME_PREFIX}.result.csv    - Results template for tracking accuracy")
    
    if not predictions.empty:
        high_conf = predictions[predictions['confidence'] == 'HIGH']
        under_recs = predictions[predictions['recommendation'].str.contains('UNDER', na=False)]
        print(f"\n📊 High confidence predictions: {len(high_conf)}")
        print(f"🎯 Under 1.5 recommendations: {len(under_recs)}")
        print(f"\n💡 TIP: Fill in {FILENAME_PREFIX}.result.csv after matches end to track accuracy!")


def run():
    """Entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
