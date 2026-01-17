"""
SofaScore Under 1 Goal Predictor - Refactored (Proper First-Goal Modeling)
============================================================================

Two-stage system:
1) Model A: Predicts whether the FIRST goal will come AFTER 45 minutes
2) Model B: Predicts final UNDER 1 GOAL (conditional on 0-0 at ~30 min)

Usage: python sofascore_predictor.py

Outputs:
- data/live_matches.csv
- data/h2h_data.csv
- data/goal_timestamps.csv
- data/training_data.csv
- data/late_goal_model_results.csv
- data/under_model_results.csv
- data/predictions.csv
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
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    HAS_ML = True
except ImportError:
    HAS_ML = False
    print("⚠️ Install ML: pip install xgboost scikit-learn")

from sofascore_wrapper.api import SofascoreAPI
from sofascore_wrapper.match import Match

# ================= CONFIG =================

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
REQUEST_DELAY = 0.6

# ================= HELPERS =================

def safe_get(data: dict, *keys, default=None):
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key, default)
        else:
            return default
    return data

def parse_stats(stats_response) -> Dict[str, Any]:
    flat = {}
    stats_list = stats_response.get('statistics', []) if isinstance(stats_response, dict) else stats_response

    period_stats = next((s for s in stats_list if s.get('period') == 'ALL'), None)
    if not period_stats:
        period_stats = next((s for s in stats_list if s.get('period') == '1ST'), None)

    if not period_stats:
        return flat

    for group in period_stats.get('groups', []):
        for item in group.get('statisticsItems', []):
            name = item.get('name', '').lower().replace(' ', '_').replace('-', '_')
            flat[f"home_{name}"] = item.get('homeValue', item.get('home'))
            flat[f"away_{name}"] = item.get('awayValue', item.get('away'))

    return flat

def parse_value(val):
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
    goals = []
    first_goal_time = None

    for inc in incidents:
        inc_type = inc.get('incidentType', '')
        if inc_type == 'goal' or (inc_type == 'inGamePenalty' and inc.get('incidentClass') != 'missed'):
            t = inc.get('time', 0)
            goals.append(t)
            if first_goal_time is None:
                first_goal_time = t

    return {
        'first_goal_time': first_goal_time,
        'first_goal_after_45': 1 if first_goal_time and first_goal_time > 45 else 0,
        'total_goals': len(goals),
        'goal_timestamps': json.dumps(goals),
    }

# ================= DATA COLLECTION =================

async def collect_all_data():
    api = SofascoreAPI()
    all_matches, all_h2h, all_goals = [], [], []

    finder = Match(api)
    live_response = await finder.live_games()
    events = live_response.get('events', [])

    for idx, event in enumerate(events):
        match_id = event.get('id')
        home_team = safe_get(event, 'homeTeam', 'name', default='Unknown')
        away_team = safe_get(event, 'awayTeam', 'name', default='Unknown')
        home_score = safe_get(event, 'homeScore', 'current', default=0)
        away_score = safe_get(event, 'awayScore', 'current', default=0)
        start_ts = event.get('startTimestamp', 0)
        elapsed_min = (time.time() - start_ts) / 60 if start_ts else 0

        match_data = {
            'match_id': match_id,
            'home_team': home_team,
            'away_team': away_team,
            'home_score': home_score,
            'away_score': away_score,
            'total_score': home_score + away_score,
            'elapsed_minutes': round(elapsed_min, 1),
            'is_zero_zero': 1 if (home_score == 0 and away_score == 0) else 0,
            'minutes_without_goal': round(elapsed_min, 1),
            'no_goal_so_far': 1 if (home_score + away_score == 0) else 0,
        }

        match = Match(api, match_id=match_id)

        try:
            stats = await match.stats()
            match_data.update(parse_stats(stats))
            await asyncio.sleep(REQUEST_DELAY)
        except:
            pass

        try:
            inc = await match.incidents()
            goal_info = extract_goal_info(inc.get('incidents', []))
            match_data.update(goal_info)

            all_goals.append({
                'match_id': match_id,
                'home_team': home_team,
                'away_team': away_team,
                'first_goal_time': goal_info['first_goal_time'],
                'total_goals': goal_info['total_goals'],
            })
            await asyncio.sleep(REQUEST_DELAY)
        except:
            match_data['first_goal_time'] = None
            match_data['total_goals'] = 0

        try:
            h2h = await match.h2h()
            td = h2h.get('teamDuel', {})

            match_data['h2h_home_wins'] = td.get('homeWins', 0)
            match_data['h2h_away_wins'] = td.get('awayWins', 0)
            match_data['h2h_draws'] = td.get('draws', 0)
            match_data['h2h_total'] = sum([
                td.get('homeWins', 0),
                td.get('awayWins', 0),
                td.get('draws', 0)
            ])
            await asyncio.sleep(REQUEST_DELAY)
        except:
            match_data['h2h_home_wins'] = 0
            match_data['h2h_away_wins'] = 0
            match_data['h2h_draws'] = 0
            match_data['h2h_total'] = 0

        all_matches.append(match_data)

    await api.close()

    df_matches = pd.DataFrame(all_matches)
    df_h2h = pd.DataFrame(all_h2h)
    df_goals = pd.DataFrame(all_goals)

    df_matches.to_csv(f"{OUTPUT_DIR}/live_matches.csv", index=False)
    df_h2h.to_csv(f"{OUTPUT_DIR}/h2h_data.csv", index=False)
    df_goals.to_csv(f"{OUTPUT_DIR}/goal_timestamps.csv", index=False)

    return df_matches, df_h2h, df_goals

# ================= FEATURE ENGINEERING =================

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:

    feature_cols = [
        'home_ball_possession', 'away_ball_possession',
        'home_total_shots', 'away_total_shots',
        'home_shots_on_target', 'away_shots_on_target',
        'home_corner_kicks', 'away_corner_kicks',
        'home_fouls', 'away_fouls',
        'minutes_without_goal',
        'no_goal_so_far',
        'h2h_home_wins', 'h2h_away_wins', 'h2h_draws', 'h2h_total',
        'elapsed_minutes',
    ]

    existing = [c for c in feature_cols if c in df.columns]
    X = df[existing].copy()

    for col in X.columns:
        X[col] = X[col].apply(parse_value)

    X = X.fillna(0)
    return X, existing

# ================= TRAINING (TWO MODELS) =================



def train_models(df: pd.DataFrame):
    if not HAS_ML or len(df) < 10:
        return None, None, []

    # Keep only matches around 30 minutes
    train_df = df[(df['elapsed_minutes'] >= 20) & (df['elapsed_minutes'] <= 40)].copy()

    # ================= MODEL A: LATE FIRST GOAL =================
    train_df['late_first_goal'] = (train_df['first_goal_time'] > 45).astype(int)

    X, features = prepare_features(train_df)
    y_late = train_df['late_first_goal']

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y_late, train_df.index, test_size=0.3, random_state=42
    )

    late_model = XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.1)
    late_model.fit(X_train, y_train)

    late_pred = late_model.predict(X_test)
    late_prob = late_model.predict_proba(X_test)[:, 1]

    late_acc = accuracy_score(y_test, late_pred)
    late_precision = precision_score(y_test, late_pred, zero_division=0)
    late_recall = recall_score(y_test, late_pred, zero_division=0)
    late_f1 = f1_score(y_test, late_pred, zero_division=0)

    print(f"Late-First-Goal Accuracy: {late_acc:.1%}")
    print(f"Late-First-Goal F1: {late_f1:.3f}")

    late_results = pd.DataFrame({
        'match_id': train_df.loc[idx_test, 'match_id'].values,
        'home_team': train_df.loc[idx_test, 'home_team'].values,
        'away_team': train_df.loc[idx_test, 'away_team'].values,
        'elapsed_minutes': train_df.loc[idx_test, 'elapsed_minutes'].values,
        'first_goal_time': train_df.loc[idx_test, 'first_goal_time'].values,
        'actual_late_first_goal': y_test.values,
        'predicted_late_first_goal': late_pred,
        'prob_late_first_goal': late_prob,
        'accuracy': late_acc,
        'precision': late_precision,
        'recall': late_recall,
        'f1_score': late_f1
    })

    late_results.to_csv(f"{OUTPUT_DIR}/late_goal_model_results.csv", index=False)

    # ================= MODEL B: UNDER 1 GOAL =================
    train_df['under_1_goal'] = (train_df['total_goals'] <= 1).astype(int)
    y_under = train_df['under_1_goal']

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y_under, train_df.index, test_size=0.3, random_state=42
    )

    under_model = XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.1)
    under_model.fit(X_train, y_train)

    under_pred = under_model.predict(X_test)
    under_prob = under_model.predict_proba(X_test)[:, 1]

    under_acc = accuracy_score(y_test, under_pred)
    under_precision = precision_score(y_test, under_pred, zero_division=0)
    under_recall = recall_score(y_test, under_pred, zero_division=0)
    under_f1 = f1_score(y_test, under_pred, zero_division=0)

    print(f"Under-1-Goal Accuracy: {under_acc:.1%}")
    print(f"Under-1-Goal F1: {under_f1:.3f}")

    under_results = pd.DataFrame({
        'match_id': train_df.loc[idx_test, 'match_id'].values,
        'home_team': train_df.loc[idx_test, 'home_team'].values,
        'away_team': train_df.loc[idx_test, 'away_team'].values,
        'elapsed_minutes': train_df.loc[idx_test, 'elapsed_minutes'].values,
        'total_goals': train_df.loc[idx_test, 'total_goals'].values,
        'actual_under_1_goal': y_test.values,
        'predicted_under_1_goal': under_pred,
        'prob_under_1_goal': under_prob,
        'accuracy': under_acc,
        'precision': under_precision,
        'recall': under_recall,
        'f1_score': under_f1
    })

    under_results.to_csv(f"{OUTPUT_DIR}/under_model_results.csv", index=False)

    # ================= SINGLE COMBINED FILE =================
    combined = pd.DataFrame({
        'match_id': train_df.loc[idx_test, 'match_id'].values,
        'home_team': train_df.loc[idx_test, 'home_team'].values,
        'away_team': train_df.loc[idx_test, 'away_team'].values,
        'elapsed_minutes': train_df.loc[idx_test, 'elapsed_minutes'].values,
        'first_goal_time': train_df.loc[idx_test, 'first_goal_time'].values,
        'total_goals': train_df.loc[idx_test, 'total_goals'].values,

        # Late goal model results
        'actual_late_first_goal': y_late.loc[idx_test].values,
        'pred_late_first_goal': late_pred,
        'prob_late_first_goal': late_prob,

        # Under 1 goal model results
        'actual_under_1_goal': y_under.loc[idx_test].values,
        'pred_under_1_goal': under_pred,
        'prob_under_1_goal': under_prob,

        # Metrics summary (same for all rows, useful for logs)
        'late_accuracy': late_acc,
        'late_precision': late_precision,
        'late_recall': late_recall,
        'late_f1': late_f1,

        'under_accuracy': under_acc,
        'under_precision': under_precision,
        'under_recall': under_recall,
        'under_f1': under_f1,
    })

    combined.to_csv(f"{OUTPUT_DIR}/combined_model_results.csv", index=False)

    print(f"Saved combined results to {OUTPUT_DIR}/combined_model_results.csv")

    return late_model, under_model, features

# ================= PREDICTION =================

def make_predictions(df, late_model, under_model, features):
    candidates = df[
        (df['is_zero_zero'] == 1) &
        (df['elapsed_minutes'] >= 25) &
        (df['elapsed_minutes'] <= 40)
    ].copy()

    X_pred, _ = prepare_features(candidates)

    for f in features:
        if f not in X_pred.columns:
            X_pred[f] = 0
    X_pred = X_pred[features]

    candidates['prob_late_first_goal'] = late_model.predict_proba(X_pred)[:, 1]
    candidates['prob_under'] = under_model.predict_proba(X_pred)[:, 1]

    candidates['ensemble_score'] = (candidates['prob_late_first_goal'] * 0.6 +
                                    candidates['prob_under'] * 0.4)

    candidates['prediction'] = np.where(candidates['ensemble_score'] >= 0.55,
                                        'STRONG UNDER 1 GOAL',
                                        'WEAK UNDER')

    output = candidates[['match_id', 'home_team', 'away_team',
                         'elapsed_minutes',
                         'prob_late_first_goal',
                         'prob_under',
                         'ensemble_score',
                         'prediction']]

    output.to_csv(f"{OUTPUT_DIR}/predictions.csv", index=False)
    return output

# ================= MAIN =================

async def main():
    df_matches, df_h2h, df_goals = await collect_all_data()

    late_model, under_model, features = train_models(df_matches)

    predictions = make_predictions(df_matches, late_model, under_model, features)

    print("\nTOP UNDER CANDIDATES:")
    print(predictions.sort_values('ensemble_score', ascending=False).head(10))

def run():
    asyncio.run(main())

if __name__ == "__main__":
    run()
