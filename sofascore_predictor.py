"""
SofaScore Under 1 Goal Predictor - Enhanced
============================================

Single-command script that:
1. Collects ALL live match data with stats, H2H, and goal timestamps
2. Trains HYBRID ensemble model (XGBoost + LightGBM)
3. Identifies 0-0 matches at ~30 min
4. Predicts which will finish with less than 1 goal

Usage: python sofascore_predictor.py

Output CSVs:
- data/live_matches.csv           - All live match data
- data/h2h_data.csv               - Head-to-head statistics  
- data/goal_timestamps.csv        - First goal times for all matches
- data/training_data.csv          - Data used for training
- data/xgboost_results.csv        - XGBoost model results
- data/lightgbm_results.csv       - LightGBM model results
- data/ensemble_results.csv       - Ensemble model results
- data/predictions.csv            - Final predictions for 0-0 matches
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
        'first_goal_before_15': 1 if first_goal_time and first_goal_time <= 15 else 0,
        'first_goal_before_30': 1 if first_goal_time and first_goal_time <= 30 else 0,
        'first_goal_before_45': 1 if first_goal_time and first_goal_time <= 45 else 0,
        'goals_first_half': sum(1 for g in goals if g['time'] <= 45),
        'goals_second_half': sum(1 for g in goals if g['time'] > 45),
        'goals_last_15': sum(1 for g in goals if g['time'] >= 75),
        'goal_timestamps': json.dumps([g['time'] for g in goals]),
        'avg_goal_time': np.mean([g['time'] for g in goals]) if goals else None,
    }


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
# MODEL TRAINING & PREDICTION (HYBRID ENSEMBLE)
# ============================================================================

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Prepare features for ML model including goal timestamp features."""
    
    # Define feature columns - now includes goal timing!
    feature_cols = [
        # Match statistics
        'home_ball_possession', 'away_ball_possession',
        'home_total_shots', 'away_total_shots',
        'home_shots_on_target', 'away_shots_on_target',
        'home_corner_kicks', 'away_corner_kicks',
        'home_fouls', 'away_fouls',
        'home_big_chances', 'away_big_chances',
        # Goal timing features
        'first_goal_time', 'first_goal_before_30', 'first_goal_before_45',
        'goals_first_half', 'goals_second_half', 'goals_last_15',
        # H2H features
        'h2h_home_wins', 'h2h_away_wins', 'h2h_draws', 'h2h_total', 'h2h_draw_rate',
        # Form features
        'home_avg_rating', 'away_avg_rating',
        'home_position', 'away_position',
        'home_form_pts', 'away_form_pts',
        # Match state
        'elapsed_minutes',
    ]
    
    # Keep only columns that exist
    existing_features = [c for c in feature_cols if c in df.columns]
    
    # Create feature dataframe
    X = df[existing_features].copy()
    
    # Convert all to numeric
    for col in X.columns:
        X[col] = X[col].apply(parse_value)
    
    # Fill NaN with median or 0
    X = X.fillna(0)
    
    return X, existing_features


def train_hybrid_ensemble(df: pd.DataFrame) -> Tuple[Optional[object], Optional[object], List[str]]:
    """
    Train HYBRID ensemble model using XGBoost + LightGBM.
    Generates separate CSV for each model.
    """
    print("\n" + "="*70)
    print("STEP 2: TRAINING HYBRID ENSEMBLE MODEL")
    print("="*70)
    
    if not HAS_ML:
        print("❌ ML libraries not available. Skipping training.")
        return None, None, []
    
    if len(df) < 5:
        print("❌ Not enough data for training (need at least 5 matches)")
        return None, None, []
    
    # Create target variable: is current total <= 1 goal?
    df['target'] = (df['total_score'] <= 1).astype(int)
    
    # Prepare features
    X, feature_names = prepare_features(df)
    y = df['target']
    
    if X.empty or len(X.columns) == 0:
        print("❌ No valid features found in data")
        return None, None, []
    
    print(f"\nTraining on {len(df)} matches with {len(X.columns)} features:")
    for i, col in enumerate(X.columns[:10]):
        print(f"  {i+1}. {col}")
    if len(X.columns) > 10:
        print(f"  ... and {len(X.columns)-10} more")
    
    print(f"\nTarget distribution: Under={y.sum()}, Over={len(y)-y.sum()}")
    
    # Check if we have both classes
    if len(y.unique()) < 2:
        print("❌ Need both Under and Over matches for training")
        return None, None, []
    
    # Split data
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
    except:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
    
    # ========== XGBoost ==========
    print("\n--- Training XGBoost ---")
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    
    xgb_pred = xgb_model.predict(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_acc = accuracy_score(y_test, xgb_pred)
    
    print(f"XGBoost Accuracy: {xgb_acc:.1%}")
    
    # Save XGBoost results
    xgb_results = pd.DataFrame({
        'actual': y_test.values,
        'predicted': xgb_pred,
        'probability': xgb_proba
    })
    xgb_results.to_csv(f"{OUTPUT_DIR}/xgboost_results.csv", index=False)
    print(f"✓ Saved to {OUTPUT_DIR}/xgboost_results.csv")
    
    # ========== LightGBM ==========
    print("\n--- Training LightGBM ---")
    lgb_model = LGBMClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    
    lgb_pred = lgb_model.predict(X_test)
    lgb_proba = lgb_model.predict_proba(X_test)[:, 1]
    lgb_acc = accuracy_score(y_test, lgb_pred)
    
    print(f"LightGBM Accuracy: {lgb_acc:.1%}")
    
    # Save LightGBM results
    lgb_results = pd.DataFrame({
        'actual': y_test.values,
        'predicted': lgb_pred,
        'probability': lgb_proba
    })
    lgb_results.to_csv(f"{OUTPUT_DIR}/lightgbm_results.csv", index=False)
    print(f"✓ Saved to {OUTPUT_DIR}/lightgbm_results.csv")
    
    # ========== Ensemble ==========
    print("\n--- HYBRID ENSEMBLE (XGBoost + LightGBM) ---")
    ensemble_proba = (xgb_proba + lgb_proba) / 2
    ensemble_pred = (ensemble_proba >= 0.5).astype(int)
    ensemble_acc = accuracy_score(y_test, ensemble_pred)
    
    print(f"Ensemble Accuracy: {ensemble_acc:.1%}")
    
    # Save Ensemble results
    ensemble_results = pd.DataFrame({
        'actual': y_test.values,
        'xgb_probability': xgb_proba,
        'lgb_probability': lgb_proba,
        'ensemble_probability': ensemble_proba,
        'xgb_prediction': xgb_pred,
        'lgb_prediction': lgb_pred,
        'ensemble_prediction': ensemble_pred
    })
    ensemble_results.to_csv(f"{OUTPUT_DIR}/ensemble_results.csv", index=False)
    print(f"✓ Saved to {OUTPUT_DIR}/ensemble_results.csv")
    
    # Summary
    print("\n" + "-"*50)
    print("MODEL COMPARISON:")
    print(f"  XGBoost:  {xgb_acc:.1%}")
    print(f"  LightGBM: {lgb_acc:.1%}")
    print(f"  Ensemble: {ensemble_acc:.1%} ← Using this for predictions")
    print("-"*50)
    
    return xgb_model, lgb_model, feature_names


def make_predictions(df: pd.DataFrame, xgb_model, lgb_model, feature_names: List[str]):
    """
    Predict which 0-0 matches at ~30 min will stay Under 1 goal.
    Uses hybrid ensemble (XGBoost + LightGBM average).
    """
    print("\n" + "="*70)
    print("STEP 3: PREDICTING 0-0 MATCHES (Under 1 Goal)")
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
    
    if xgb_model is None or lgb_model is None:
        print("\n⚠️ No trained models. Using heuristic.")
        candidates['xgb_probability'] = 0.5
        candidates['lgb_probability'] = 0.5
        candidates['ensemble_probability'] = 0.5
        candidates['prediction'] = 'Unknown'
    else:
        # Prepare features
        X_pred, _ = prepare_features(candidates)
        
        # Ensure columns match
        for col in feature_names:
            if col not in X_pred.columns:
                X_pred[col] = 0
        X_pred = X_pred[feature_names]
        
        # Make predictions with both models
        xgb_proba = xgb_model.predict_proba(X_pred)[:, 1]
        lgb_proba = lgb_model.predict_proba(X_pred)[:, 1]
        ensemble_proba = (xgb_proba + lgb_proba) / 2
        
        candidates['xgb_probability'] = xgb_proba
        candidates['lgb_probability'] = lgb_proba
        candidates['ensemble_probability'] = ensemble_proba
        candidates['prediction'] = np.where(ensemble_proba >= 0.5, 'UNDER 1 Goal', 'OVER 1 Goal')
    
    # Sort by ensemble probability
    candidates = candidates.sort_values('ensemble_probability', ascending=False)
    
    # Select output columns
    output_cols = [
        'match_id', 'tournament', 'home_team', 'away_team',
        'home_score', 'away_score', 'status', 'elapsed_minutes',
        'first_goal_time', 'goals_first_half',
        'xgb_probability', 'lgb_probability', 'ensemble_probability', 'prediction',
        'h2h_home_wins', 'h2h_draws', 'h2h_away_wins',
        'home_avg_rating', 'away_avg_rating'
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
        xgb_p = row.get('xgb_probability', 0)
        lgb_p = row.get('lgb_probability', 0)
        ens_p = row.get('ensemble_probability', 0)
        fg = row.get('first_goal_time', 'N/A')
        
        print(f"\n  {row['home_team']} vs {row['away_team']}")
        print(f"    Score: {row['home_score']}-{row['away_score']} | {row['status']}")
        print(f"    First Goal: {fg if fg else 'No goals yet'}")
        print(f"    XGBoost: {xgb_p:.1%} | LightGBM: {lgb_p:.1%} | Ensemble: {ens_p:.1%}")
        print(f"    >>> {row.get('prediction', 'N/A')}")
    
    return result_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main execution - runs everything in one command."""
    
    print("\n" + "="*70)
    print("  SOFASCORE UNDER 1 GOAL PREDICTOR - ENHANCED")
    print("  Hybrid XGBoost + LightGBM Ensemble with Goal Timestamps")
    print("="*70)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    # Step 1: Collect data
    df_matches, df_h2h, df_goals = await collect_all_data()
    
    if df_matches.empty:
        print("\n❌ No match data collected. Exiting.")
        return
    
    # Step 2: Train hybrid ensemble
    xgb_model, lgb_model, features = train_hybrid_ensemble(df_matches)
    
    # Save training data
    df_matches.to_csv(f"{OUTPUT_DIR}/training_data.csv", index=False)
    print(f"\n✓ Saved training data to {OUTPUT_DIR}/training_data.csv")
    
    # Step 3: Make predictions
    predictions = make_predictions(df_matches, xgb_model, lgb_model, features)
    
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
    print(f"  • training_data.csv     - Full training dataset")
    print(f"  • xgboost_results.csv   - XGBoost model results")
    print(f"  • lightgbm_results.csv  - LightGBM model results")
    print(f"  • ensemble_results.csv  - Hybrid ensemble results")
    print(f"  • predictions.csv       - Final Under 1 goal predictions")
    
    # Stats
    zero_zero = df_matches[df_matches['is_zero_zero'] == 1]
    print(f"\n📊 Summary: {len(zero_zero)} matches currently at 0-0")
    
    if len(predictions) > 0 and 'ensemble_probability' in predictions.columns:
        high_conf = predictions[predictions['ensemble_probability'] >= 0.6]
        print(f"🎯 High-confidence Under predictions (≥60%): {len(high_conf)}")


def run():
    """Entry point."""
    asyncio.run(main())


if __name__ == "__main__":
    run()
