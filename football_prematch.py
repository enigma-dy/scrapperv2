"""
Football Prematch Predictor
===========================

Predicts prematch football outcomes:
- 1X2 (Home Win / Draw / Away Win)
- Over/Under 2.5 Goals

Usage: python football_prematch.py

Output CSVs in data/ folder:
- prematch_fixtures.csv      - Today's upcoming matches
- prematch_h2h.csv           - H2H data for fixtures
- prematch_training.csv      - Training dataset
- prematch_1x2_results.csv   - 1X2 model results
- prematch_o25_results.csv   - Over 2.5 model results
- prematch_predictions.csv   - Final predictions
"""

import asyncio
import pandas as pd
import numpy as np
import time
import os
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional

warnings.filterwarnings('ignore')

# ML imports
try:
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import LabelEncoder
    HAS_ML = True
except ImportError:
    HAS_ML = False
    print("⚠️ Install ML libraries: pip install xgboost lightgbm scikit-learn")

from sofascore_wrapper.api import SofascoreAPI
from sofascore_wrapper.match import Match
from sofascore_wrapper.basketball import Basketball

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
REQUEST_DELAY = 0.6


# ============================================================================
# HELPERS
# ============================================================================

def safe_get(data: dict, *keys, default=None):
    """Safely get nested dict values."""
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key, default)
        else:
            return default
    return data


def parse_value(val):
    """Convert to float."""
    if val is None:
        return 0
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val.replace('%', ''))
        except:
            return 0
    return 0


def parse_stats(stats_response) -> Dict[str, Any]:
    """Parse match statistics."""
    flat = {}
    stats_list = stats_response.get('statistics', []) if isinstance(stats_response, dict) else []
    
    period_stats = None
    for period in ['ALL', '1ST', '2ND']:
        period_stats = next((s for s in stats_list if s.get('period') == period), None)
        if period_stats:
            break
    
    if not period_stats:
        return flat
    
    for group in period_stats.get('groups', []):
        for item in group.get('statisticsItems', []):
            name = item.get('name', '').lower().replace(' ', '_')
            flat[f"home_{name}"] = item.get('homeValue', item.get('home'))
            flat[f"away_{name}"] = item.get('awayValue', item.get('away'))
    
    return flat


# ============================================================================
# DATA COLLECTION
# ============================================================================

async def collect_prematch_data():
    """Collect today's fixtures and their stats for prematch prediction."""
    print("\n" + "="*70)
    print("STEP 1: COLLECTING PREMATCH FIXTURE DATA")
    print("="*70)
    
    api = SofascoreAPI()
    all_fixtures = []
    all_h2h = []
    
    try:
        # Get today and tomorrow's fixtures
        basketball = Basketball(api)  # Has matches_by_date
        
        today = datetime.now().strftime("%Y-%m-%d")
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        
        print(f"\nFetching football fixtures for {today} and {tomorrow}...")
        
        for date in [today, tomorrow]:
            try:
                fixtures = await basketball.matches_by_date(sport='football', date=date)
                events = fixtures.get('events', [])
                
                # Filter to not started matches
                upcoming = [e for e in events if safe_get(e, 'status', 'type') == 'notstarted']
                print(f"  {date}: {len(upcoming)} upcoming matches")
                
                for event in upcoming[:100]:  # Limit per day
                    fixture = {
                        'match_id': event.get('id'),
                        'date': date,
                        'tournament': safe_get(event, 'tournament', 'name'),
                        'home_team': safe_get(event, 'homeTeam', 'name'),
                        'away_team': safe_get(event, 'awayTeam', 'name'),
                        'home_team_id': safe_get(event, 'homeTeam', 'id'),
                        'away_team_id': safe_get(event, 'awayTeam', 'id'),
                        'start_timestamp': event.get('startTimestamp'),
                    }
                    all_fixtures.append(fixture)
                    
            except Exception as e:
                print(f"  Error for {date}: {e}")
        
        print(f"\nTotal fixtures: {len(all_fixtures)}")
        print("Collecting H2H and form data...")
        
        # Get detailed data for each fixture
        for idx, fixture in enumerate(all_fixtures[:50]):  # Detailed for top 50
            match_id = fixture['match_id']
            match = Match(api, match_id=match_id)
            
            print(f"  [{idx+1}/50] {fixture['home_team']} vs {fixture['away_team']}")
            
            # H2H
            try:
                h2h = await match.h2h()
                team_duel = h2h.get('teamDuel', {})
                
                h2h_record = {
                    'match_id': match_id,
                    'home_team': fixture['home_team'],
                    'away_team': fixture['away_team'],
                    'h2h_home_wins': team_duel.get('homeWins', 0),
                    'h2h_away_wins': team_duel.get('awayWins', 0),
                    'h2h_draws': team_duel.get('draws', 0),
                }
                all_h2h.append(h2h_record)
                
                fixture['h2h_home_wins'] = team_duel.get('homeWins', 0)
                fixture['h2h_away_wins'] = team_duel.get('awayWins', 0)
                fixture['h2h_draws'] = team_duel.get('draws', 0)
                h2h_total = sum([team_duel.get('homeWins', 0), 
                                 team_duel.get('awayWins', 0), 
                                 team_duel.get('draws', 0)])
                fixture['h2h_total'] = h2h_total
                
                await asyncio.sleep(REQUEST_DELAY)
            except:
                pass
            
            # Pre-match form
            try:
                form = await match.pre_match_form()
                home_form = form.get('homeTeam', {})
                away_form = form.get('awayTeam', {})
                
                fixture['home_avg_rating'] = float(home_form.get('avgRating', 0) or 0)
                fixture['away_avg_rating'] = float(away_form.get('avgRating', 0) or 0)
                fixture['home_position'] = home_form.get('position', 0) or 0
                fixture['away_position'] = away_form.get('position', 0) or 0
                
                def form_pts(f): 
                    return sum(3 if r=='W' else 1 if r=='D' else 0 for r in (f or []))
                fixture['home_form_pts'] = form_pts(home_form.get('form'))
                fixture['away_form_pts'] = form_pts(away_form.get('form'))
                
                await asyncio.sleep(REQUEST_DELAY)
            except:
                pass
        
        await api.close()
        
    except Exception as e:
        print(f"Error: {e}")
        await api.close()
    
    # Save
    df_fixtures = pd.DataFrame(all_fixtures)
    df_h2h = pd.DataFrame(all_h2h)
    
    df_fixtures.to_csv(f"{OUTPUT_DIR}/prematch_fixtures.csv", index=False)
    df_h2h.to_csv(f"{OUTPUT_DIR}/prematch_h2h.csv", index=False)
    
    print(f"\n✓ Saved {len(df_fixtures)} fixtures to {OUTPUT_DIR}/prematch_fixtures.csv")
    print(f"✓ Saved {len(df_h2h)} H2H records to {OUTPUT_DIR}/prematch_h2h.csv")
    
    return df_fixtures, df_h2h


async def collect_finished_matches():
    """Collect recently finished matches for training."""
    print("\n" + "="*70)
    print("STEP 2: COLLECTING FINISHED MATCHES FOR TRAINING")
    print("="*70)
    
    api = SofascoreAPI()
    finished = []
    
    try:
        # Get matches from past days
        basketball = Basketball(api)
        
        for days_ago in range(1, 4):  # Last 3 days
            date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")
            print(f"  Fetching {date}...")
            
            try:
                fixtures = await basketball.matches_by_date(sport='football', date=date)
                events = fixtures.get('events', [])
                
                # Filter to finished matches
                done = [e for e in events if safe_get(e, 'status', 'type') == 'finished']
                
                for event in done[:30]:
                    home_score = safe_get(event, 'homeScore', 'current', default=0)
                    away_score = safe_get(event, 'awayScore', 'current', default=0)
                    
                    # Determine 1X2 result
                    if home_score > away_score:
                        result_1x2 = 1  # Home win
                    elif home_score < away_score:
                        result_1x2 = 2  # Away win
                    else:
                        result_1x2 = 0  # Draw
                    
                    total_goals = home_score + away_score
                    
                    match_data = {
                        'match_id': event.get('id'),
                        'tournament': safe_get(event, 'tournament', 'name'),
                        'home_team': safe_get(event, 'homeTeam', 'name'),
                        'away_team': safe_get(event, 'awayTeam', 'name'),
                        'home_score': home_score,
                        'away_score': away_score,
                        'total_goals': total_goals,
                        'result_1x2': result_1x2,  # 1=Home, 0=Draw, 2=Away
                        'over_2_5': 1 if total_goals > 2.5 else 0,
                    }
                    
                    # Get H2H
                    match = Match(api, match_id=event.get('id'))
                    try:
                        h2h = await match.h2h()
                        td = h2h.get('teamDuel', {})
                        match_data['h2h_home_wins'] = td.get('homeWins', 0)
                        match_data['h2h_away_wins'] = td.get('awayWins', 0)
                        match_data['h2h_draws'] = td.get('draws', 0)
                        await asyncio.sleep(REQUEST_DELAY)
                    except:
                        pass
                    
                    # Get form
                    try:
                        form = await match.pre_match_form()
                        match_data['home_avg_rating'] = float(form.get('homeTeam', {}).get('avgRating', 0) or 0)
                        match_data['away_avg_rating'] = float(form.get('awayTeam', {}).get('avgRating', 0) or 0)
                        await asyncio.sleep(REQUEST_DELAY)
                    except:
                        pass
                    
                    finished.append(match_data)
                    
            except Exception as e:
                print(f"    Error: {e}")
        
        await api.close()
        
    except Exception as e:
        print(f"Error: {e}")
        await api.close()
    
    df = pd.DataFrame(finished)
    df.to_csv(f"{OUTPUT_DIR}/prematch_training.csv", index=False)
    print(f"\n✓ Collected {len(df)} finished matches for training")
    
    return df


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_prematch_models(df: pd.DataFrame):
    """Train 1X2 and Over 2.5 models."""
    print("\n" + "="*70)
    print("STEP 3: TRAINING PREMATCH MODELS")
    print("="*70)
    
    if not HAS_ML or len(df) < 10:
        print("❌ Not enough data for training")
        return None, None, []
    
    # Feature columns
    feature_cols = [
        'h2h_home_wins', 'h2h_away_wins', 'h2h_draws',
        'home_avg_rating', 'away_avg_rating',
    ]
    existing = [c for c in feature_cols if c in df.columns]
    
    if len(existing) < 2:
        print("❌ Not enough feature columns")
        return None, None, []
    
    X = df[existing].fillna(0).apply(lambda col: col.apply(parse_value))
    
    # ========== 1X2 Model ==========
    print("\n--- Training 1X2 Model (Home/Draw/Away) ---")
    
    if 'result_1x2' not in df.columns:
        print("❌ No result_1x2 column")
        model_1x2 = None
    else:
        y_1x2 = df['result_1x2']
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y_1x2, test_size=0.3, random_state=42)
            
            model_1x2 = XGBClassifier(n_estimators=50, max_depth=3, verbosity=0, use_label_encoder=False)
            model_1x2.fit(X_train, y_train)
            
            y_pred = model_1x2.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"1X2 Accuracy: {acc:.1%}")
            
            results = pd.DataFrame({'actual': y_test, 'predicted': y_pred})
            results.to_csv(f"{OUTPUT_DIR}/prematch_1x2_results.csv", index=False)
            
        except Exception as e:
            print(f"Error: {e}")
            model_1x2 = None
    
    # ========== Over 2.5 Model ==========
    print("\n--- Training Over 2.5 Goals Model ---")
    
    if 'over_2_5' not in df.columns:
        print("❌ No over_2_5 column")
        model_o25 = None
    else:
        y_o25 = df['over_2_5']
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y_o25, test_size=0.3, random_state=42)
            
            model_o25 = LGBMClassifier(n_estimators=50, verbose=-1)
            model_o25.fit(X_train, y_train)
            
            y_pred = model_o25.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"Over 2.5 Accuracy: {acc:.1%}")
            
            results = pd.DataFrame({'actual': y_test, 'predicted': y_pred})
            results.to_csv(f"{OUTPUT_DIR}/prematch_o25_results.csv", index=False)
            
        except Exception as e:
            print(f"Error: {e}")
            model_o25 = None
    
    return model_1x2, model_o25, existing


def make_prematch_predictions(df_fixtures: pd.DataFrame, model_1x2, model_o25, features):
    """Predict upcoming fixtures."""
    print("\n" + "="*70)
    print("STEP 4: PREDICTING UPCOMING MATCHES")
    print("="*70)
    
    if len(df_fixtures) == 0:
        print("No fixtures to predict")
        return pd.DataFrame()
    
    # Prepare features
    X = df_fixtures[features].fillna(0).apply(lambda col: col.apply(parse_value)) if features else pd.DataFrame()
    
    result = df_fixtures.copy()
    
    if model_1x2 is not None and not X.empty:
        for col in features:
            if col not in X.columns:
                X[col] = 0
        
        proba = model_1x2.predict_proba(X)
        result['prob_home'] = proba[:, 1] if proba.shape[1] > 1 else 0
        result['prob_draw'] = proba[:, 0] if proba.shape[1] > 0 else 0
        result['prob_away'] = proba[:, 2] if proba.shape[1] > 2 else 0
        pred = model_1x2.predict(X)
        result['prediction_1x2'] = np.where(pred == 1, 'HOME', np.where(pred == 2, 'AWAY', 'DRAW'))
    
    if model_o25 is not None and not X.empty:
        proba = model_o25.predict_proba(X)[:, 1]
        result['prob_over_2_5'] = proba
        result['prediction_goals'] = np.where(proba >= 0.5, 'OVER 2.5', 'UNDER 2.5')
    
    # Save
    result.to_csv(f"{OUTPUT_DIR}/prematch_predictions.csv", index=False)
    print(f"\n✓ Saved predictions to {OUTPUT_DIR}/prematch_predictions.csv")
    
    # Display
    print("\n" + "-"*70)
    print("TOP PREDICTIONS:")
    print("-"*70)
    
    for _, row in result.head(15).iterrows():
        print(f"\n  {row['home_team']} vs {row['away_team']}")
        print(f"    Tournament: {row.get('tournament', 'N/A')}")
        if 'prediction_1x2' in row:
            print(f"    1X2: {row['prediction_1x2']}")
        if 'prediction_goals' in row:
            print(f"    Goals: {row['prediction_goals']} (prob: {row.get('prob_over_2_5', 0):.1%})")
    
    return result


# ============================================================================
# MAIN
# ============================================================================

async def main():
    print("\n" + "="*70)
    print("  FOOTBALL PREMATCH PREDICTOR")
    print("  Predictions: 1X2 (Home/Draw/Away) + Over/Under 2.5 Goals")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start = time.time()
    
    # Collect upcoming fixtures
    df_fixtures, df_h2h = await collect_prematch_data()
    
    # Collect training data
    df_training = await collect_finished_matches()
    
    # Train models
    model_1x2, model_o25, features = train_prematch_models(df_training)
    
    # Predict
    predictions = make_prematch_predictions(df_fixtures, model_1x2, model_o25, features)
    
    elapsed = time.time() - start
    print("\n" + "="*70)
    print(f"COMPLETE! Time: {elapsed:.1f}s")
    print("="*70)
    print(f"\nFiles in {OUTPUT_DIR}/:")
    print("  • prematch_fixtures.csv")
    print("  • prematch_h2h.csv")
    print("  • prematch_training.csv")
    print("  • prematch_1x2_results.csv")
    print("  • prematch_o25_results.csv")
    print("  • prematch_predictions.csv")


if __name__ == "__main__":
    asyncio.run(main())
