import pandas as pd
import numpy as np
import time
import json
import os
from sofascrape import SofascoreClient

# ML Models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------- SETTINGS ----------------
H2H_MATCHES = 3       # Keep low to avoid timeouts
RECENT_MATCHES = 3
TARGET_MIN = 30
N_SIMULATIONS = 100

# ---------------- LOGGING UTILITIES ----------------
if not os.path.exists("debug_logs"):
    os.makedirs("debug_logs")

if not os.path.exists("results"):
    os.makedirs("results")

def log_api_response(filename_prefix, data):
    try:
        filename = f"debug_logs/{filename_prefix}_{int(time.time())}.json"
        with open(filename, "w", encoding='utf-8') as f:
            if hasattr(data, 'data'):
                json.dump(data.data, f, indent=2)
            else:
                json.dump(data, f, indent=2)
    except:
        pass

# ---------------- MOMENTUM ONLY (Since Stats are missing) ----------------
def get_momentum(client, match_id):
    """
    Calculates attack momentum using get_event_graph() which IS supported.
    """
    momentum = 0
    try:
        graph_data = client.get_event_graph(match_id)
        # log_api_response(f"graph_{match_id}", graph_data)
        
        graph = graph_data.data.get('graphPoints',[])
        if graph:
            # Take average of last 5 minutes of momentum
            recent_vals = [abs(pt.get('value',0)) for pt in graph[-5:]]
            if recent_vals:
                momentum = float(np.mean(recent_vals))
    except Exception:
        pass # Graph might not exist for minor leagues
    return momentum

# ---------------- H2H FEATURES ----------------
def get_h2h_features(client, match_id):
    feats = {"under_1_5_rate":0.5,"goal_before_30":0.5,"zero_at_30":0.5}
    try:
        # Add sleep to prevent timeouts
        time.sleep(1)
        response = client.get_event_h2h(match_id)
        
        # Safely extract events
        h2h_events = response.data.get('events', [])
        if not h2h_events:
             h2h_events = response.data.get('teamDuel', {}).get('previousEvents', [])

        h2h = h2h_events[:H2H_MATCHES]
        if not h2h:
            return feats

        under, early_goal, zero30 = 0,0,0

        for m in h2h:
            mid = m.get('id')
            
            # --- INCIDENT CHECKING ---
            time.sleep(0.5) # Be polite
            try:
                inc_data = client.get_event_incidents(mid)
                incidents = inc_data.data.get('incidents',[])
                first_goal = None
                
                for inc in incidents:
                    if inc.get('type') == 'goal':
                        # --- PRINT REQUESTED BY YOU ---
                        print(f"\n[H2H GOAL] Match: {mid} | Time: {inc.get('time')}'")
                        print(json.dumps(inc, indent=2))
                        # ------------------------------
                        first_goal = inc.get('time')
                        break
                
                if first_goal is not None:
                    if first_goal <= 30:
                        early_goal += 1
                else:
                    zero30 += 1
            except Exception:
                pass 

            # Check Total Score
            h_score = m.get('homeScore',{}).get('display',0)
            a_score = m.get('awayScore',{}).get('display',0)

            if (h_score + a_score) < 1.5:
                under += 1

        feats = {
            "under_1_5_rate": under/len(h2h),
            "goal_before_30": early_goal/len(h2h),
            "zero_at_30": zero30/len(h2h)
        }
    except Exception as e:
        print(f"   > H2H Error for {match_id}: {e}")
        pass
    return feats

# ---------------- BUILD DATASET ----------------
def build_dataset_from_live(client):
    print("Fetching live events...")
    try:
        live_res = client.get_live_events()
        live_events = live_res.data.get('events',[])
    except Exception as e:
        print(f"Failed to get live events: {e}")
        return pd.DataFrame()

    rows = []
    print(f"Found {len(live_events)} live events. Processing...")

    for i, e in enumerate(live_events):
        try:
            match_id = e.get('id')
            home_team = e['homeTeam']['name']
            away_team = e['awayTeam']['name']
            print(f"Processing {i+1}/{len(live_events)}: {home_team} vs {away_team}")
            
            # 1. Get Momentum (Graph)
            mom = get_momentum(client, match_id)
            
            # 2. Get H2H
            h2h_feats = get_h2h_features(client, match_id)

            h_score = e.get('homeScore',{}).get('current',0)
            a_score = e.get('awayScore',{}).get('current',0)

            row = {
                "match_id": match_id,
                "home_team": home_team,
                "away_team": away_team,
                "attack_momentum": mom,
                **{f"h2h_{k}":v for k,v in h2h_feats.items()},
                "is_under_1_5": 1 if (h_score + a_score) < 1.5 else 0
            }
            rows.append(row)
        
        except Exception as e:
            print(f"   > Skipping match {match_id}: {e}")
            continue

    df = pd.DataFrame(rows)
    if not df.empty:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        df.to_csv(f"results/training_dataset_{timestamp}.csv", index=False)
        print(f"Dataset built: {len(df)} rows.")
    return df

# ---------------- TRAINING ----------------
def train_hybrid_ai(df):
    if df.empty or len(df) < 5:
        print("Not enough data to train.")
        return None, None, None

    # Separate match info from features
    match_info_cols = ['match_id', 'home_team', 'away_team']
    feature_cols = [col for col in df.columns if col not in match_info_cols + ['is_under_1_5']]
    
    df_features = df[feature_cols + ['is_under_1_5']].fillna(0)
    
    # Needs variance
    if len(df_features['is_under_1_5'].unique()) < 2:
        print("Data is all Under or all Over. Cannot train classifier.")
        return None, None, None

    X = df_features.drop('is_under_1_5', axis=1)
    y = df_features['is_under_1_5']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train models
    xgb = XGBClassifier(n_estimators=50, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train, y_train)

    lgb = LGBMClassifier(n_estimators=50, verbose=-1)
    lgb.fit(X_train, y_train)

    # Get predictions on test set
    xgb_pred = xgb.predict(X_test)
    lgb_pred = lgb.predict(X_test)
    
    xgb_proba = xgb.predict_proba(X_test)[:, 1]
    lgb_proba = lgb.predict_proba(X_test)[:, 1]
    
    # Ensemble prediction
    ensemble_proba = (xgb_proba + lgb_proba) / 2
    ensemble_pred = (ensemble_proba >= 0.5).astype(int)

    # Calculate metrics
    xgb_acc = accuracy_score(y_test, xgb_pred)
    lgb_acc = accuracy_score(y_test, lgb_pred)
    ensemble_acc = accuracy_score(y_test, ensemble_pred)

    print(f"\nModel Performance:")
    print(f"  XGBoost Accuracy: {xgb_acc:.2%}")
    print(f"  LightGBM Accuracy: {lgb_acc:.2%}")
    print(f"  Ensemble Accuracy: {ensemble_acc:.2%}")

    # Save test results
    test_results = pd.DataFrame({
        'actual': y_test.values,
        'xgb_prediction': xgb_pred,
        'lgb_prediction': lgb_pred,
        'ensemble_prediction': ensemble_pred,
        'xgb_probability': xgb_proba,
        'lgb_probability': lgb_proba,
        'ensemble_probability': ensemble_proba
    })
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    test_results.to_csv(f"results/test_results_{timestamp}.csv", index=False)
    print(f"✓ Test results saved to: results/test_results_{timestamp}.csv")

    # Save training summary
    summary = pd.DataFrame({
        'Model': ['XGBoost', 'LightGBM', 'Ensemble'],
        'Accuracy': [xgb_acc, lgb_acc, ensemble_acc],
        'Test_Size': [len(y_test)] * 3,
        'Train_Size': [len(y_train)] * 3,
        'Timestamp': [timestamp] * 3
    })
    summary.to_csv(f"results/training_summary_{timestamp}.csv", index=False)
    print(f"✓ Training summary saved to: results/training_summary_{timestamp}.csv")

    return xgb, lgb, feature_cols

# ---------------- PREDICTION ----------------
def predict_live_advanced(client, xgb, lgb, feature_cols):
    print("\n--- STARTING PREDICTIONS ---")
    live_events = client.get_live_events().data.get('events',[])
    current_ts = time.time()
    
    predictions = []
    
    for e in live_events:
        try:
            h_score = e.get('homeScore',{}).get('current',0)
            a_score = e.get('awayScore',{}).get('current',0)
            start_ts = e.get('startTimestamp',0)
            elapsed = (current_ts - start_ts) / 60

            # Only look at games around 30 mins that are 0-0
            if h_score == 0 and a_score == 0 and 20 <= elapsed <= 35:
                match_id = e.get('id')
                home_team = e['homeTeam']['name']
                away_team = e['awayTeam']['name']
                
                # Get features
                mom = get_momentum(client, match_id)
                h2h_feats = get_h2h_features(client, match_id)
                
                X_pred = pd.DataFrame([{
                    "attack_momentum": mom,
                    **{f"h2h_{k}":v for k,v in h2h_feats.items()}
                }])
                
                # Ensure columns match training
                X_pred = X_pred[feature_cols]
                
                # Ensemble prediction
                xgb_prob = xgb.predict_proba(X_pred)[0][1]
                lgb_prob = lgb.predict_proba(X_pred)[0][1]
                avg_prob = (xgb_prob + lgb_prob) / 2
                
                prediction = "Under 1.5" if avg_prob >= 0.5 else "Over 1.5"
                
                print(f"\n{home_team} vs {away_team}")
                print(f"  Under 1.5 Probability: {avg_prob:.2%}")
                print(f"  Elapsed: {elapsed:.1f} mins | Score: {h_score}-{a_score}")
                
                # Store prediction
                predictions.append({
                    'match_id': match_id,
                    'home_team': home_team,
                    'away_team': away_team,
                    'elapsed_minutes': round(elapsed, 1),
                    'current_score': f"{h_score}-{a_score}",
                    'xgb_probability': round(xgb_prob, 4),
                    'lgb_probability': round(lgb_prob, 4),
                    'ensemble_probability': round(avg_prob, 4),
                    'prediction': prediction,
                    'attack_momentum': mom,
                    'h2h_under_1_5_rate': h2h_feats['under_1_5_rate'],
                    'h2h_goal_before_30': h2h_feats['goal_before_30'],
                    'h2h_zero_at_30': h2h_feats['zero_at_30'],
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                })
                
        except Exception as ex:
            print(f"Error processing match: {ex}")
            continue
    
    # Save predictions to CSV
    if predictions:
        predictions_df = pd.DataFrame(predictions)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        predictions_df.to_csv(f"results/live_predictions_{timestamp}.csv", index=False)
        print(f"\n✓ Live predictions saved to: results/live_predictions_{timestamp}.csv")
        print(f"✓ Total predictions made: {len(predictions)}")
    else:
        print("\n⚠️ No matches found matching prediction criteria (0-0, 20-35 mins)")

# ---------------- MAIN EXECUTION ----------------
if __name__ == "__main__":
    print("=" * 60)
    print("LIVE FOOTBALL UNDER 1.5 GOALS PREDICTOR")
    print("=" * 60)
    
    print("\nInitializing SofascoreClient...")
    
    # Use context manager to properly initialize the client
    with SofascoreClient() as client:
        print("✓ Client initialized successfully")
        
        print("\n=== STEP 1: Building Dataset from Live Matches ===")
        df = build_dataset_from_live(client)
        
        if df.empty:
            print("\n⚠️ No live matches found or data collection failed.")
            print("This could mean:")
            print("  - No football matches are currently live")
            print("  - API connection issues")
            print("  - Rate limiting from too many requests")
        else:
            print(f"\n✓ Collected {len(df)} matches")
            print("\nDataset Preview:")
            print(df.head())
            
            print("\n=== STEP 2: Training Models ===")
            xgb_model, lgb_model, feature_cols = train_hybrid_ai(df)
            
            if xgb_model is None:
                print("\n⚠️ Training failed. Not enough varied data.")
                print("Need matches with both Under 1.5 and Over 1.5 outcomes.")
            else:
                print("\n=== STEP 3: Making Predictions ===")
                predict_live_advanced(client, xgb_model, lgb_model, feature_cols)
                
                print("\n" + "=" * 60)
                print("✓ All done!")
                print("=" * 60)
                print("\nOutput files created in 'results/' folder:")
                print("  - training_dataset_YYYYMMDD_HHMMSS.csv")
                print("  - test_results_YYYYMMDD_HHMMSS.csv")
                print("  - training_summary_YYYYMMDD_HHMMSS.csv")
                print("  - live_predictions_YYYYMMDD_HHMMSS.csv")