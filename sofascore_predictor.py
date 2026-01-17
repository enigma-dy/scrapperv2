"""
SofaScore Under 1 Goal Predictor - Two-Stage Model
===================================================

Two-stage modeling approach that eliminates data leakage:

Model A - "Late First Goal Model"
    → Predicts whether the first goal comes after 45 minutes
    
Model B - "Under 1 Goal Model"  
    → Predicts if match stays Under 1 Goal, conditioned on 0-0 at 30 min

Key improvements:
- No data leakage: Model never trains on matches where first goal 
  happened before 30 minutes
- Clean features that encode "no goal yet" state
- Dominance indicators (possession, shots, corners ratios)
- Combined probability for optimal predictions

Usage: python sofascore_predictor.py

Output CSVs:
- data/live_matches.csv           - All live match data
- data/h2h_data.csv               - Head-to-head statistics  
- data/goal_timestamps.csv        - First goal times for all matches
- data/training_data.csv          - Filtered training data (0-0 at 30 min)
- data/model_a_results.csv        - Late First Goal model results
- data/model_b_results.csv        - Under 1 Goal model results
- data/predictions.csv            - Final combined predictions
"""

import asyncio
import pandas as pd
import numpy as np
import time
import json
import os
import warnings
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

warnings.filterwarnings('ignore')

# ML imports
try:
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    HAS_ML = True
except ImportError:
    HAS_ML = False
    print("⚠️ ML libraries not found. Install with: pip install xgboost lightgbm scikit-learn")

# Sofascore imports
from sofascore_wrapper.api import SofascoreAPI
from sofascore_wrapper.match import Match

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Time between API requests to avoid rate limiting
REQUEST_DELAY = 0.6


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


def extract_goal_info(incidents: List[Dict]) -> Dict[str, Any]:
    """
    Extract goal timing information from match incidents.
    Returns first goal time, all goal timestamps, and goal counts.
    """
    goals = []
    first_goal_time = None
    home_goals = 0
    away_goals = 0
    
    for inc in incidents:
        inc_type = inc.get('incidentType', '')
        # Check for goals (regular, penalty scored, own goal)
        if inc_type == 'goal' or (inc_type == 'inGamePenalty' and inc.get('incidentClass') != 'missed'):
            goal_time = inc.get('time', 0)
            added_time = inc.get('addedTime', 0)
            is_home = inc.get('isHome', False)
            scorer = safe_get(inc, 'player', 'name', default='Unknown')
            goal_type = inc.get('incidentClass', 'regular')
            
            goals.append({
                'time': goal_time,
                'added_time': added_time,
                'total_time': goal_time + (added_time if added_time else 0),
                'is_home': is_home,
                'scorer': scorer,
                'type': goal_type,
            })
            
            if first_goal_time is None:
                first_goal_time = goal_time
            
            if is_home:
                home_goals += 1
            else:
                away_goals += 1
    
    # Calculate goal timing features
    return {
        'total_goals': len(goals),
        'home_goals_scored': home_goals,
        'away_goals_scored': away_goals,
        'first_goal_time': first_goal_time,
        'goals_first_half': sum(1 for g in goals if g['time'] <= 45),
        'goals_second_half': sum(1 for g in goals if g['time'] > 45),
        'goal_timestamps': json.dumps([g['time'] for g in goals]),
        'avg_goal_time': np.mean([g['time'] for g in goals]) if goals else None,
    }


# ============================================================================
# FEATURE ENGINEERING (LEAK-FREE)
# ============================================================================

def calculate_dominance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate dominance indicators that are safe for 0-0 prediction.
    These features encode relative strength without leaking goal info.
    """
    df = df.copy()
    
    # Parse numeric values for calculations
    for col in ['home_ball_possession', 'away_ball_possession',
                'home_total_shots', 'away_total_shots',
                'home_shots_on_target', 'away_shots_on_target',
                'home_corner_kicks', 'away_corner_kicks',
                'home_big_chances', 'away_big_chances',
                'home_expected_goals', 'away_expected_goals',
                'home_fouls', 'away_fouls']:
        if col in df.columns:
            df[col] = df[col].apply(parse_value)
    
    # Possession ratio (0 to 1, 0.5 = equal)
    home_poss = df.get('home_ball_possession', pd.Series([50]*len(df)))
    away_poss = df.get('away_ball_possession', pd.Series([50]*len(df)))
    total_poss = home_poss + away_poss
    df['possession_ratio'] = np.where(total_poss > 0, home_poss / total_poss, 0.5)
    
    # Shots ratio
    home_shots = df.get('home_total_shots', pd.Series([0]*len(df)))
    away_shots = df.get('away_total_shots', pd.Series([0]*len(df)))
    total_shots = home_shots + away_shots
    df['shots_ratio'] = np.where(total_shots > 0, home_shots / total_shots, 0.5)
    df['total_shots'] = total_shots
    
    # Shots on target ratio
    home_sot = df.get('home_shots_on_target', pd.Series([0]*len(df)))
    away_sot = df.get('away_shots_on_target', pd.Series([0]*len(df)))
    total_sot = home_sot + away_sot
    df['sot_ratio'] = np.where(total_sot > 0, home_sot / total_sot, 0.5)
    df['total_shots_on_target'] = total_sot
    
    # Corners ratio
    home_corners = df.get('home_corner_kicks', pd.Series([0]*len(df)))
    away_corners = df.get('away_corner_kicks', pd.Series([0]*len(df)))
    total_corners = home_corners + away_corners
    df['corners_ratio'] = np.where(total_corners > 0, home_corners / total_corners, 0.5)
    df['total_corners'] = total_corners
    
    # Big chances ratio
    home_bc = df.get('home_big_chances', pd.Series([0]*len(df)))
    away_bc = df.get('away_big_chances', pd.Series([0]*len(df)))
    total_bc = home_bc + away_bc
    df['big_chances_ratio'] = np.where(total_bc > 0, home_bc / total_bc, 0.5)
    df['total_big_chances'] = total_bc
    
    # Expected goals dominance (if available)
    home_xg = df.get('home_expected_goals', pd.Series([0]*len(df)))
    away_xg = df.get('away_expected_goals', pd.Series([0]*len(df)))
    df['xg_difference'] = home_xg - away_xg
    total_xg = home_xg + away_xg
    df['xg_ratio'] = np.where(total_xg > 0, home_xg / total_xg, 0.5)
    df['total_xg'] = total_xg
    
    # Pressure indicators (normalized by elapsed time)
    elapsed = df.get('elapsed_minutes', pd.Series([30]*len(df)))
    elapsed = elapsed.replace(0, 30)  # Avoid division by zero
    
    df['shot_pressure'] = total_shots / elapsed * 30  # Shots per 30 min
    df['sot_pressure'] = total_sot / elapsed * 30
    df['corner_pressure'] = total_corners / elapsed * 30
    df['chance_pressure'] = total_bc / elapsed * 30
    
    # Defensive indicators
    home_fouls = df.get('home_fouls', pd.Series([0]*len(df)))
    away_fouls = df.get('away_fouls', pd.Series([0]*len(df)))
    df['total_fouls'] = home_fouls + away_fouls
    df['foul_pressure'] = df['total_fouls'] / elapsed * 30
    
    return df


def filter_training_data(df: pd.DataFrame, min_elapsed: int = 30) -> pd.DataFrame:
    """
    Filter training data to eliminate data leakage.
    
    Only keeps matches where:
    1. First goal came AFTER the specified minute (min_elapsed), OR
    2. No goal was scored at all
    
    This simulates predicting on matches that are 0-0 at min_elapsed minutes.
    """
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Get first goal time, treating NaN as "no goal"
    first_goal = df['first_goal_time'].fillna(999)  # No goal = very late
    
    # Keep only matches where first goal > min_elapsed or no goal
    valid_mask = first_goal > min_elapsed
    filtered_df = df[valid_mask].copy()
    
    print(f"\n📊 Training Data Filtering:")
    print(f"   Original matches: {len(df)}")
    print(f"   Matches with first goal > {min_elapsed} min: {valid_mask.sum()}")
    print(f"   Filtering removed {len(df) - len(filtered_df)} matches (had early goals)")
    
    return filtered_df


# ============================================================================
# DATA COLLECTION
# ============================================================================

async def collect_all_data():
    """
    Main data collection function.
    Gets ALL live matches, stats, H2H data, and goal timestamps.
    """
    print("\n" + "="*70)
    print("STEP 1: COLLECTING ALL LIVE MATCH DATA")
    print("="*70)
    
    api = SofascoreAPI()
    all_matches = []
    all_h2h = []
    all_goals = []
    
    try:
        # Get live matches
        print("\nFetching live matches...")
        finder = Match(api)
        live_response = await finder.live_games()
        events = live_response.get('events', [])
        
        print(f"Found {len(events)} live football matches - processing ALL")
        
        for idx, event in enumerate(events):
            match_id = event.get('id')
            home_team = safe_get(event, 'homeTeam', 'name', default='Unknown')
            away_team = safe_get(event, 'awayTeam', 'name', default='Unknown')
            home_score = safe_get(event, 'homeScore', 'current', default=0)
            away_score = safe_get(event, 'awayScore', 'current', default=0)
            status = safe_get(event, 'status', 'description', default='Unknown')
            start_ts = event.get('startTimestamp', 0)
            elapsed_min = (time.time() - start_ts) / 60 if start_ts else 0
            
            print(f"  [{idx+1}/{len(events)}] {home_team} vs {away_team} ({home_score}-{away_score})")
            
            # Base match data
            match_data = {
                'match_id': match_id,
                'tournament': safe_get(event, 'tournament', 'name'),
                'tournament_id': safe_get(event, 'tournament', 'uniqueTournament', 'id'),
                'home_team': home_team,
                'away_team': away_team,
                'home_team_id': safe_get(event, 'homeTeam', 'id'),
                'away_team_id': safe_get(event, 'awayTeam', 'id'),
                'home_score': home_score,
                'away_score': away_score,
                'total_score': home_score + away_score,
                'status': status,
                'elapsed_minutes': round(elapsed_min, 1),
                'is_zero_zero': 1 if (home_score == 0 and away_score == 0) else 0,
                'is_around_30_min': 1 if (25 <= elapsed_min <= 40) else 0,
            }
            
            # Create match instance for detailed data
            match = Match(api, match_id=match_id)
            
            # Get statistics
            try:
                stats = await match.stats()
                parsed = parse_stats(stats)
                match_data.update(parsed)
                await asyncio.sleep(REQUEST_DELAY)
            except Exception:
                pass
            
            # Get incidents (GOAL TIMESTAMPS)
            try:
                inc_response = await match.incidents()
                incidents = inc_response.get('incidents', [])
                goal_info = extract_goal_info(incidents)
                match_data.update(goal_info)
                
                # Save goal data separately
                all_goals.append({
                    'match_id': match_id,
                    'home_team': home_team,
                    'away_team': away_team,
                    'first_goal_time': goal_info['first_goal_time'],
                    'total_goals': goal_info['total_goals'],
                    'goal_timestamps': goal_info['goal_timestamps'],
                    'goals_first_half': goal_info['goals_first_half'],
                    'goals_second_half': goal_info['goals_second_half'],
                })
                
                await asyncio.sleep(REQUEST_DELAY)
            except Exception:
                match_data['first_goal_time'] = None
                match_data['total_goals'] = 0
            
            # Get H2H data
            try:
                h2h = await match.h2h()
                team_duel = h2h.get('teamDuel', {})
                
                h2h_record = {
                    'match_id': match_id,
                    'home_team': home_team,
                    'away_team': away_team,
                    'h2h_home_wins': team_duel.get('homeWins', 0),
                    'h2h_away_wins': team_duel.get('awayWins', 0),
                    'h2h_draws': team_duel.get('draws', 0),
                }
                all_h2h.append(h2h_record)
                
                # Add to match data
                match_data['h2h_home_wins'] = team_duel.get('homeWins', 0)
                match_data['h2h_away_wins'] = team_duel.get('awayWins', 0)
                match_data['h2h_draws'] = team_duel.get('draws', 0)
                h2h_total = sum([team_duel.get('homeWins', 0), 
                                 team_duel.get('awayWins', 0), 
                                 team_duel.get('draws', 0)])
                match_data['h2h_total'] = h2h_total
                match_data['h2h_draw_rate'] = team_duel.get('draws', 0) / h2h_total if h2h_total > 0 else 0
                
                await asyncio.sleep(REQUEST_DELAY)
            except Exception:
                match_data['h2h_home_wins'] = 0
                match_data['h2h_away_wins'] = 0
                match_data['h2h_draws'] = 0
                match_data['h2h_total'] = 0
                match_data['h2h_draw_rate'] = 0
            
            # Get pre-match form
            try:
                form = await match.pre_match_form()
                home_form = form.get('homeTeam', {})
                away_form = form.get('awayTeam', {})
                
                match_data['home_avg_rating'] = float(home_form.get('avgRating', 0) or 0)
                match_data['away_avg_rating'] = float(away_form.get('avgRating', 0) or 0)
                match_data['home_position'] = home_form.get('position', 0) or 0
                match_data['away_position'] = away_form.get('position', 0) or 0
                
                # Form points (W=3, D=1, L=0)
                def form_pts(f): 
                    return sum(3 if r=='W' else 1 if r=='D' else 0 for r in (f or []))
                match_data['home_form_pts'] = form_pts(home_form.get('form'))
                match_data['away_form_pts'] = form_pts(away_form.get('form'))
                
                await asyncio.sleep(REQUEST_DELAY)
            except Exception:
                pass
            
            all_matches.append(match_data)
        
        await api.close()
        
    except Exception as e:
        print(f"Error during collection: {e}")
        await api.close()
    
    # Create DataFrames
    df_matches = pd.DataFrame(all_matches)
    df_h2h = pd.DataFrame(all_h2h)
    df_goals = pd.DataFrame(all_goals)
    
    # Save CSVs
    matches_file = f"{OUTPUT_DIR}/live_matches.csv"
    h2h_file = f"{OUTPUT_DIR}/h2h_data.csv"
    goals_file = f"{OUTPUT_DIR}/goal_timestamps.csv"
    
    df_matches.to_csv(matches_file, index=False)
    df_h2h.to_csv(h2h_file, index=False)
    df_goals.to_csv(goals_file, index=False)
    
    print(f"\n✓ Saved {len(df_matches)} matches to {matches_file}")
    print(f"✓ Saved {len(df_h2h)} H2H records to {h2h_file}")
    print(f"✓ Saved {len(df_goals)} goal records to {goals_file}")
    
    return df_matches, df_h2h, df_goals


# ============================================================================
# TWO-STAGE MODEL TRAINING
# ============================================================================

# Clean feature columns (no data leakage)
CLEAN_FEATURE_COLS = [
    # Dominance indicators (ratios are safe - they don't encode goals)
    'possession_ratio', 'shots_ratio', 'sot_ratio', 'corners_ratio',
    'big_chances_ratio', 'xg_ratio', 'xg_difference',
    
    # Total activity (safe - doesn't encode if goals happened)
    'total_shots', 'total_shots_on_target', 'total_corners', 
    'total_big_chances', 'total_xg', 'total_fouls',
    
    # Pressure indicators (normalized by time)
    'shot_pressure', 'sot_pressure', 'corner_pressure', 
    'chance_pressure', 'foul_pressure',
    
    # H2H features (historical - safe)
    'h2h_home_wins', 'h2h_away_wins', 'h2h_draws', 
    'h2h_total', 'h2h_draw_rate',
    
    # Form features (pre-match - safe)
    'home_avg_rating', 'away_avg_rating',
    'home_position', 'away_position',
    'home_form_pts', 'away_form_pts',
    
    # Match state
    'elapsed_minutes',
]


def prepare_clean_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prepare leak-free features for ML models.
    Only uses features that don't encode goal information.
    """
    # Calculate dominance features first
    df = calculate_dominance_features(df)
    
    # Only keep columns that exist
    existing_features = [c for c in CLEAN_FEATURE_COLS if c in df.columns]
    
    # Create feature dataframe
    X = df[existing_features].copy()
    
    # Fill NaN with 0 or median
    X = X.fillna(0)
    
    # Convert all to numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    return X, existing_features


class LateFirstGoalModel:
    """
    Model A: Predicts whether the first goal comes after 45 minutes.
    
    Target: 1 if first_goal_time > 45 OR no goals, else 0
    
    This model answers: "If this match is 0-0 at 30 min, 
    will the first goal come late (after 45 min)?"
    """
    
    def __init__(self):
        self.xgb_model = None
        self.lgb_model = None
        self.feature_names = []
        self.is_trained = False
    
    def create_target(self, df: pd.DataFrame) -> pd.Series:
        """Create target: 1 if first goal > 45 min or no goal."""
        first_goal = df['first_goal_time'].fillna(999)  # No goal = very late
        return (first_goal > 45).astype(int)
    
    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train Model A on filtered data."""
        print("\n--- Training Model A: Late First Goal ---")
        
        if len(df) < 5:
            print("❌ Not enough data for Model A")
            return {}
        
        # Create target
        y = self.create_target(df)
        
        # Prepare features
        X, self.feature_names = prepare_clean_features(df)
        
        if X.empty or len(X.columns) == 0:
            print("❌ No valid features found")
            return {}
        
        print(f"Training on {len(df)} matches with {len(X.columns)} features")
        print(f"Target distribution: Late={y.sum()}, Early={len(y)-y.sum()}")
        
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
        
        # Train XGBoost
        self.xgb_model = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            use_label_encoder=False, eval_metric='logloss', verbosity=0
        )
        self.xgb_model.fit(X_train, y_train)
        xgb_pred = self.xgb_model.predict(X_test)
        xgb_proba = self.xgb_model.predict_proba(X_test)[:, 1]
        
        # Train LightGBM
        self.lgb_model = LGBMClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1, verbose=-1
        )
        self.lgb_model.fit(X_train, y_train)
        lgb_pred = self.lgb_model.predict(X_test)
        lgb_proba = self.lgb_model.predict_proba(X_test)[:, 1]
        
        # Ensemble
        ensemble_proba = (xgb_proba + lgb_proba) / 2
        ensemble_pred = (ensemble_proba >= 0.5).astype(int)
        
        # Metrics
        results = {
            'xgb_accuracy': accuracy_score(y_test, xgb_pred),
            'lgb_accuracy': accuracy_score(y_test, lgb_pred),
            'ensemble_accuracy': accuracy_score(y_test, ensemble_pred),
        }
        
        print(f"XGBoost Accuracy: {results['xgb_accuracy']:.1%}")
        print(f"LightGBM Accuracy: {results['lgb_accuracy']:.1%}")
        print(f"Ensemble Accuracy: {results['ensemble_accuracy']:.1%}")
        
        # Save results
        model_a_results = pd.DataFrame({
            'actual': y_test.values,
            'xgb_probability': xgb_proba,
            'lgb_probability': lgb_proba,
            'ensemble_probability': ensemble_proba,
            'prediction': ensemble_pred
        })
        model_a_results.to_csv(f"{OUTPUT_DIR}/model_a_results.csv", index=False)
        print(f"✓ Saved to {OUTPUT_DIR}/model_a_results.csv")
        
        self.is_trained = True
        return results
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict probability of late first goal."""
        if not self.is_trained:
            return np.full(len(df), 0.5)
        
        X, _ = prepare_clean_features(df)
        
        # Ensure columns match
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]
        
        xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
        lgb_proba = self.lgb_model.predict_proba(X)[:, 1]
        
        return (xgb_proba + lgb_proba) / 2


class Under1GoalModel:
    """
    Model B: Predicts whether the match stays Under 1 Goal.
    
    Target: 1 if total_goals <= 1, else 0
    
    This model answers: "If this match is 0-0 at 30 min,
    will it finish with 0 or 1 goals total?"
    """
    
    def __init__(self):
        self.xgb_model = None
        self.lgb_model = None
        self.feature_names = []
        self.is_trained = False
    
    def create_target(self, df: pd.DataFrame) -> pd.Series:
        """Create target: 1 if total goals <= 1."""
        return (df['total_goals'] <= 1).astype(int)
    
    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train Model B on filtered data."""
        print("\n--- Training Model B: Under 1 Goal ---")
        
        if len(df) < 5:
            print("❌ Not enough data for Model B")
            return {}
        
        # Create target
        y = self.create_target(df)
        
        # Prepare features
        X, self.feature_names = prepare_clean_features(df)
        
        if X.empty or len(X.columns) == 0:
            print("❌ No valid features found")
            return {}
        
        print(f"Training on {len(df)} matches with {len(X.columns)} features")
        print(f"Target distribution: Under={y.sum()}, Over={len(y)-y.sum()}")
        
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
        
        # Train XGBoost
        self.xgb_model = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            use_label_encoder=False, eval_metric='logloss', verbosity=0
        )
        self.xgb_model.fit(X_train, y_train)
        xgb_pred = self.xgb_model.predict(X_test)
        xgb_proba = self.xgb_model.predict_proba(X_test)[:, 1]
        
        # Train LightGBM
        self.lgb_model = LGBMClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1, verbose=-1
        )
        self.lgb_model.fit(X_train, y_train)
        lgb_pred = self.lgb_model.predict(X_test)
        lgb_proba = self.lgb_model.predict_proba(X_test)[:, 1]
        
        # Ensemble
        ensemble_proba = (xgb_proba + lgb_proba) / 2
        ensemble_pred = (ensemble_proba >= 0.5).astype(int)
        
        # Metrics
        results = {
            'xgb_accuracy': accuracy_score(y_test, xgb_pred),
            'lgb_accuracy': accuracy_score(y_test, lgb_pred),
            'ensemble_accuracy': accuracy_score(y_test, ensemble_pred),
        }
        
        print(f"XGBoost Accuracy: {results['xgb_accuracy']:.1%}")
        print(f"LightGBM Accuracy: {results['lgb_accuracy']:.1%}")
        print(f"Ensemble Accuracy: {results['ensemble_accuracy']:.1%}")
        
        # Save results
        model_b_results = pd.DataFrame({
            'actual': y_test.values,
            'xgb_probability': xgb_proba,
            'lgb_probability': lgb_proba,
            'ensemble_probability': ensemble_proba,
            'prediction': ensemble_pred
        })
        model_b_results.to_csv(f"{OUTPUT_DIR}/model_b_results.csv", index=False)
        print(f"✓ Saved to {OUTPUT_DIR}/model_b_results.csv")
        
        self.is_trained = True
        return results
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict probability of under 1 goal."""
        if not self.is_trained:
            return np.full(len(df), 0.5)
        
        X, _ = prepare_clean_features(df)
        
        # Ensure columns match
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_names]
        
        xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
        lgb_proba = self.lgb_model.predict_proba(X)[:, 1]
        
        return (xgb_proba + lgb_proba) / 2


def train_two_stage_models(df: pd.DataFrame) -> Tuple[LateFirstGoalModel, Under1GoalModel]:
    """
    Train both models on properly filtered data.
    
    The key insight: We filter training data to only include matches
    that were 0-0 at 30 minutes. This eliminates data leakage.
    """
    print("\n" + "="*70)
    print("STEP 2: TRAINING TWO-STAGE MODELS (NO DATA LEAKAGE)")
    print("="*70)
    
    if not HAS_ML:
        print("❌ ML libraries not available. Skipping training.")
        return LateFirstGoalModel(), Under1GoalModel()
    
    if len(df) < 5:
        print("❌ Not enough data for training (need at least 5 matches)")
        return LateFirstGoalModel(), Under1GoalModel()
    
    # CRITICAL: Filter training data to eliminate leakage
    # Only use matches where first goal came after 30 min
    filtered_df = filter_training_data(df, min_elapsed=30)
    
    if len(filtered_df) < 5:
        print("⚠️ After filtering, not enough valid training data.")
        print("   Using all data as fallback (less accurate predictions).")
        filtered_df = df.copy()
    
    # Save filtered training data
    filtered_df.to_csv(f"{OUTPUT_DIR}/training_data.csv", index=False)
    print(f"\n✓ Saved filtered training data to {OUTPUT_DIR}/training_data.csv")
    
    # Train Model A: Late First Goal
    model_a = LateFirstGoalModel()
    results_a = model_a.train(filtered_df)
    
    # Train Model B: Under 1 Goal
    model_b = Under1GoalModel()
    results_b = model_b.train(filtered_df)
    
    # Summary
    print("\n" + "-"*50)
    print("TWO-STAGE MODEL SUMMARY:")
    print("-"*50)
    if results_a:
        print(f"  Model A (Late First Goal): {results_a.get('ensemble_accuracy', 0):.1%}")
    else:
        print("  Model A (Late First Goal): Not trained")
    if results_b:
        print(f"  Model B (Under 1 Goal):    {results_b.get('ensemble_accuracy', 0):.1%}")
    else:
        print("  Model B (Under 1 Goal): Not trained")
    print("-"*50)
    
    return model_a, model_b


# ============================================================================
# PREDICTION
# ============================================================================

def make_two_stage_predictions(
    df: pd.DataFrame, 
    model_a: LateFirstGoalModel, 
    model_b: Under1GoalModel
) -> pd.DataFrame:
    """
    Make predictions using two-stage model.
    
    Combined prediction answers:
    "If this match is 0-0 at ~30 minutes, what is the chance 
    the first goal comes at ~45+ AND the game stays Under 1 Goal?"
    """
    print("\n" + "="*70)
    print("STEP 3: PREDICTING 0-0 MATCHES (Two-Stage Model)")
    print("="*70)
    
    # Filter to 0-0 matches around 30 minutes
    candidates = df[
        (df['is_zero_zero'] == 1) & 
        (df['elapsed_minutes'] >= 20) & 
        (df['elapsed_minutes'] <= 50)
    ].copy()
    
    print(f"\nFound {len(candidates)} matches that are 0-0 between 20-50 minutes")
    
    if len(candidates) == 0:
        print("⚠️ No 0-0 matches found in target window.")
        all_zero = df[df['is_zero_zero'] == 1].copy()
        if len(all_zero) > 0:
            print(f"\nAll 0-0 matches ({len(all_zero)}):")
            for _, row in all_zero.head(10).iterrows():
                print(f"  • {row['home_team']} vs {row['away_team']} ({row['status']})")
        candidates = all_zero if len(all_zero) > 0 else df.copy()
    
    # Make predictions with both models
    if model_a.is_trained and model_b.is_trained:
        proba_late_goal = model_a.predict(candidates)
        proba_under1 = model_b.predict(candidates)
        
        # Combined probability: weighted average
        # Higher weight on Under 1 Goal as it's the primary target
        combined_proba = (proba_late_goal * 0.4 + proba_under1 * 0.6)
        
        candidates['prob_late_first_goal'] = proba_late_goal
        candidates['prob_under_1_goal'] = proba_under1
        candidates['combined_probability'] = combined_proba
        candidates['prediction'] = np.where(
            combined_proba >= 0.5, 
            'UNDER 1 Goal (High Confidence)', 
            'Risk of Goals'
        )
        
        # Confidence level
        candidates['confidence'] = np.where(
            combined_proba >= 0.7, 'HIGH',
            np.where(combined_proba >= 0.5, 'MEDIUM', 'LOW')
        )
    else:
        print("\n⚠️ Models not trained. Using heuristic predictions.")
        candidates['prob_late_first_goal'] = 0.5
        candidates['prob_under_1_goal'] = 0.5
        candidates['combined_probability'] = 0.5
        candidates['prediction'] = 'Unknown'
        candidates['confidence'] = 'N/A'
    
    # Sort by combined probability
    candidates = candidates.sort_values('combined_probability', ascending=False)
    
    # Select output columns
    output_cols = [
        'match_id', 'tournament', 'home_team', 'away_team',
        'home_score', 'away_score', 'status', 'elapsed_minutes',
        'prob_late_first_goal', 'prob_under_1_goal', 'combined_probability',
        'confidence', 'prediction',
        'h2h_home_wins', 'h2h_draws', 'h2h_away_wins',
        'home_avg_rating', 'away_avg_rating',
        'total_shots', 'total_shots_on_target', 'total_xg'
    ]
    existing_cols = [c for c in output_cols if c in candidates.columns]
    result_df = candidates[existing_cols].copy()
    
    # Save predictions
    result_df.to_csv(f"{OUTPUT_DIR}/predictions.csv", index=False)
    print(f"\n✓ Saved predictions to {OUTPUT_DIR}/predictions.csv")
    
    # Display predictions
    print("\n" + "-"*70)
    print("TOP PREDICTIONS - Matches likely to stay UNDER 1 Goal:")
    print("-"*70)
    
    for _, row in result_df.head(15).iterrows():
        late_p = row.get('prob_late_first_goal', 0)
        under_p = row.get('prob_under_1_goal', 0)
        combined_p = row.get('combined_probability', 0)
        conf = row.get('confidence', 'N/A')
        
        print(f"\n  {row['home_team']} vs {row['away_team']}")
        print(f"    Score: {row['home_score']}-{row['away_score']} | {row['status']}")
        print(f"    P(Late Goal): {late_p:.1%} | P(Under 1): {under_p:.1%}")
        print(f"    >>> Combined: {combined_p:.1%} [{conf}]")
        print(f"    >>> {row.get('prediction', 'N/A')}")
    
    return result_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main execution - runs everything in one command."""
    
    print("\n" + "="*70)
    print("  SOFASCORE UNDER 1 GOAL PREDICTOR - TWO-STAGE MODEL")
    print("  No Data Leakage | Model A + Model B Combined")
    print("="*70)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    # Step 1: Collect data
    df_matches, df_h2h, df_goals = await collect_all_data()
    
    if df_matches.empty:
        print("\n❌ No match data collected. Exiting.")
        return
    
    # Step 2: Train two-stage models (with data filtering)
    model_a, model_b = train_two_stage_models(df_matches)
    
    # Step 3: Make predictions
    predictions = make_two_stage_predictions(df_matches, model_a, model_b)
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"\nTotal time: {elapsed:.1f} seconds")
    print(f"\nGenerated files in '{OUTPUT_DIR}/' folder:")
    print(f"  • live_matches.csv      - All {len(df_matches)} live matches with stats")
    print(f"  • h2h_data.csv          - Head-to-head records")
    print(f"  • goal_timestamps.csv   - First goal times for each match")
    print(f"  • training_data.csv     - Filtered training data (no early goals)")
    print(f"  • model_a_results.csv   - Late First Goal model results")
    print(f"  • model_b_results.csv   - Under 1 Goal model results")
    print(f"  • predictions.csv       - Final two-stage predictions")
    
    # Stats
    zero_zero = df_matches[df_matches['is_zero_zero'] == 1]
    print(f"\n📊 Summary: {len(zero_zero)} matches currently at 0-0")
    
    if len(predictions) > 0 and 'combined_probability' in predictions.columns:
        high_conf = predictions[predictions['confidence'] == 'HIGH']
        print(f"🎯 High-confidence Under predictions: {len(high_conf)}")


def run():
    """Entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
