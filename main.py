import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import warnings
warnings.filterwarnings('ignore')

class BSAComplianceTool:
    def __init__(self):
        self.bayesian_network = None
        self.random_forest = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.sar_threshold = 100000
        
    def load_and_preprocess_data(self, file_path):
        print("Loading dataset...")
        df = pd.read_csv(file_path)
        
        df['amount_to_balance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
        df['balance_change_ratio'] = abs(df['newbalanceOrig'] - df['oldbalanceOrg']) / (df['oldbalanceOrg'] + 1)
        df['is_round_number'] = (df['amount'] % 1000 == 0).astype(int)
        df['transaction_hour'] = df['step'] % 24
        df['is_night_transaction'] = ((df['transaction_hour'] >= 22) | (df['transaction_hour'] <= 6)).astype(int)
        df['sar_required'] = (df['amount'] >= self.sar_threshold).astype(int)

        for col in ['type']:
            self.label_encoders[col] = LabelEncoder()
            df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
        
        return df

    def build_bayesian_network(self, df):
        print("Building Bayesian Network for SAR detection...")

        df['amount_category'] = pd.cut(df['amount'], 
            bins=[0, 10000, 50000, 100000, np.inf], 
            labels=['Low', 'Medium', 'High', 'Very_High']
        )
        df['balance_category'] = pd.cut(df['oldbalanceOrg'], 
            bins=[0, 10000, 100000, np.inf], 
            labels=['Low', 'Medium', 'High']
        )

        model = DiscreteBayesianNetwork([
            ('amount_category', 'sar_required'),
            ('type_encoded', 'sar_required'),
            ('is_night_transaction', 'sar_required'),
            ('balance_category', 'sar_required'),
            ('is_round_number', 'sar_required')
        ])

        amount_cpd = TabularCPD(
            variable='amount_category', variable_card=4,
            values=[[0.1], [0.3], [0.4], [0.2]]
        )
        type_card = len(df['type_encoded'].unique())
        type_cpd = TabularCPD(
            variable='type_encoded', variable_card=type_card,
            values=[[1/type_card] for _ in range(type_card)]
        )
        night_cpd = TabularCPD(
            variable='is_night_transaction', variable_card=2,
            values=[[0.8], [0.2]]
        )
        balance_cpd = TabularCPD(
            variable='balance_category', variable_card=3,
            values=[[0.4], [0.4], [0.2]]
        )
        round_cpd = TabularCPD(
            variable='is_round_number', variable_card=2,
            values=[[0.7], [0.3]]
        )

        values = np.random.dirichlet([1, 1], size=(4*5*2*3*2)).T.tolist()

        sar_cpd = TabularCPD(
            variable='sar_required',
            variable_card=2,
            values=values,
            evidence=['amount_category', 'type_encoded', 'is_night_transaction', 
                    'balance_category', 'is_round_number'],
            evidence_card=[4, 5, 2, 3, 2]
)


        model.add_cpds(amount_cpd, type_cpd, night_cpd, balance_cpd, round_cpd, sar_cpd)
        assert model.check_model()

        self.bayesian_network = model
        return model

    def train_random_forest(self, df):
        print("Training Random Forest for risk scoring...")
        features = [
            'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
            'type_encoded', 'amount_to_balance_ratio', 'balance_change_ratio',
            'is_round_number', 'transaction_hour', 'is_night_transaction'
        ]
        X = df[features].fillna(0)
        y = df['isFraud']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.random_forest = RandomForestClassifier(
            n_estimators=100, max_depth=5, min_samples_split=5,
            min_samples_leaf=2, random_state=42, class_weight='balanced'
        )
        self.random_forest.fit(X_train_scaled, y_train)

        y_pred = self.random_forest.predict(X_test_scaled)
        print("\nRandom Forest Performance:")
        print(classification_report(y_test, y_pred))

        return X_test_scaled, y_test

    def calculate_risk_score(self, transaction_data):
        if self.random_forest is None:
            raise ValueError("Random Forest model not trained yet")
        
        features = transaction_data[[
            'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
            'type_encoded', 'amount_to_balance_ratio', 'balance_change_ratio',
            'is_round_number', 'transaction_hour', 'is_night_transaction'
        ]].fillna(0)

        features_scaled = self.scaler.transform(features)
        return self.random_forest.predict_proba(features_scaled)[:, 1]

    def bayesian_sar_decision(self, transaction_data):
        if self.bayesian_network is None:
            raise ValueError("Bayesian Network not built yet")
        
        inference = VariableElimination(self.bayesian_network)
        decisions = []

        for _, row in transaction_data.iterrows():
            evidence = {
                'amount_category': row['amount_category'],
                'type_encoded': row['type_encoded'],
                'is_night_transaction': row['is_night_transaction'],
                'balance_category': row['balance_category'],
                'is_round_number': row['is_round_number']
            }

            try:
                result = inference.query(variables=['sar_required'], evidence=evidence)
                decisions.append(result.values[1] > 0.5)
            except:
                decisions.append(row['amount'] >= self.sar_threshold)
        
        return decisions

    def generate_compliance_report(self, transaction_data):
        risk_scores = self.calculate_risk_score(transaction_data)
        bayesian_sar = self.bayesian_sar_decision(transaction_data)

        recommendations = []
        for risk_score, sar_decision in zip(risk_scores, bayesian_sar):
            if sar_decision or risk_score > 0.8:
                recommendations.append("FILE_SAR")
            elif risk_score > 0.5:
                recommendations.append("INVESTIGATE")
            else:
                recommendations.append("MONITOR")

        results = pd.DataFrame({
            'transaction_id': range(len(transaction_data)),
            'amount': transaction_data['amount'].values,
            'risk_score': risk_scores,
            'bayesian_sar_required': bayesian_sar,
            'recommendation': recommendations,
            'actual_fraud': transaction_data['isFraud'].values if 'isFraud' in transaction_data.columns else None
        })

        return results

def main():
    tool = BSAComplianceTool()
    df = tool.load_and_preprocess_data('paysim_data.csv')
    tool.build_bayesian_network(df)
    tool.train_random_forest(df)

    # Enrich the sample with required features
    sample = df.sample(n=1000, random_state=42).copy()
    sample['amount_category'] = pd.cut(sample['amount'], [0, 10000, 50000, 100000, np.inf], labels=['Low', 'Medium', 'High', 'Very_High'])
    sample['balance_category'] = pd.cut(sample['oldbalanceOrg'], [0, 10000, 100000, np.inf], labels=['Low', 'Medium', 'High'])

    report = tool.generate_compliance_report(sample)

    print("\nCompliance Report Summary:")
    print(f"Total transactions analyzed: {len(report)}")
    print(f"SARs recommended: {sum(report['recommendation'] == 'FILE_SAR')}")
    print(f"Investigations recommended: {sum(report['recommendation'] == 'INVESTIGATE')}")
    print(f"Average risk score: {report['risk_score'].mean():.3f}")

    high_risk = report[report['risk_score'] > 0.7]
    print(f"\nHigh-risk transactions (risk > 0.7): {len(high_risk)}")
    print(high_risk[['transaction_id', 'amount', 'risk_score', 'recommendation']].head())

    return tool, report

if __name__ == "__main__":
    tool, report = main()
