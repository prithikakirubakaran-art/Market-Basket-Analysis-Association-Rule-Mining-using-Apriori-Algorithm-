# Experiment 6 : Market Basket Analysis (Association Rule Mining using Apriori Algorithm)

## Aim
To perform Market Basket Analysis using the Apriori algorithm to identify frequent itemsets and generate association rules from transaction data for decision-making in retail and business analytics.

## Dataset
A sample retail transaction dataset containing **10 transactions** across **4 grocery items**: `Bread`, `Butter`, `Milk`, and `Eggs`.

| Transaction ID | Items Purchased |
| :---: | :--- |
| **T1** | Milk, Bread, Eggs |
| **T2** | Milk, Bread |
| **T3** | Milk, Eggs |
| **T4** | Bread, Butter |
| **T5** | Milk, Bread, Butter, Eggs |
| **T6** | Bread, Eggs |
| **T7** | Milk, Butter |
| **T8** | Milk, Bread, Butter |
| **T9** | Bread, Butter, Eggs |
| **T10** | Milk, Bread, Butter, Eggs |

*The transaction list is transformed into a One-Hot Encoded binary DataFrame prior to mining.*

## Algorithm / Steps Followed
1. Import required Python libraries (`pandas`, `mlxtend`).
2. Create the sample grocery transaction dataset.
3. Convert the dataset into a one-hot encoded binary matrix using `TransactionEncoder`.
4. Apply the `apriori()` algorithm (`min_support=0.3`) to identify frequent itemsets.
5. Generate association rules from frequent itemsets using `association_rules()` with `lift >= 1.0`.
6. Filter strong rules based on thresholds (`confidence >= 0.6` and `lift > 1`).
7. Interpret the rules to uncover cross-selling and store layout insights.

## Functions and Methods Used

| Category | Function / Method | Purpose / Description |
| :--- | :--- | :--- |
| **Data Handling** | `TransactionEncoder()` | Converts a list of transactions into a one-hot encoded binary matrix |
| **Frequent Pattern Mining** | `apriori()` | Finds frequent itemsets meeting the minimum support threshold |
| **Association Rule Generation** | `association_rules()` | Derives association rules using metrics such as support, confidence, and lift |
| **Filtering** | Conditional filtering (`rules[...]`) | Selects rules meeting strong confidence and lift thresholds |
| **Visualization / Display** | `print()` / `DataFrame.head()` | Displays encoded transactions, itemsets, and rules in tabular format |

## Program & Output

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Create transaction dataset
transactions = [
    ['Milk', 'Bread', 'Eggs'],
    ['Milk', 'Bread'],
    ['Milk', 'Eggs'],
    ['Bread', 'Butter'],
    ['Milk', 'Bread', 'Butter', 'Eggs'],
    ['Bread', 'Eggs'],
    ['Milk', 'Butter'],
    ['Milk', 'Bread', 'Butter'],
    ['Bread', 'Butter', 'Eggs'],
    ['Milk', 'Bread', 'Butter', 'Eggs']
]

# Convert transactions into one-hot encoding
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)

df = pd.DataFrame(te_array, columns=te.columns_)

print("Encoded Transaction Data:")
print(df)

# Find frequent itemsets
frequent_itemsets = apriori(
    df, min_support=0.3, use_colnames=True
)

print("\nFrequent Itemsets:")
print(frequent_itemsets)

# Generate association rules
rules = association_rules(
    frequent_itemsets,
    metric="lift",
    min_threshold=1.0
)

# Filter strong association rules
strong_rules = rules[
    (rules['confidence'] >= 0.6) &
    (rules['lift'] > 1)
]

print("\nStrong Association Rules:")
print(strong_rules[
    ['antecedents', 'consequents',
     'support', 'confidence', 'lift']
].to_string(index=False))

print("\nMarket Basket Analysis Completed!")

## Output
    Encoded Transaction Data:
       Bread  Butter   Eggs   Milk
    0   True   False   True   True
    1   True   False  False   True
    2  False   False   True   True
    3   True    True  False  False
    4   True    True   True   True
    5   True   False   True  False
    6  False    True  False   True
    7   True    True  False   True
    8   True    True   True  False
    9   True    True   True   True
    
    Frequent Itemsets:
        support               itemsets
    0       0.8                (Bread)
    1       0.6               (Butter)
    2       0.6                 (Eggs)
    3       0.7                 (Milk)
    4       0.5        (Bread, Butter)
    5       0.5          (Eggs, Bread)
    6       0.5          (Bread, Milk)
    7       0.3         (Eggs, Butter)
    8       0.4         (Butter, Milk)
    9       0.4           (Eggs, Milk)
    10      0.3  (Eggs, Bread, Butter)
    11      0.3  (Bread, Butter, Milk)
    12      0.3    (Eggs, Bread, Milk)
    
    Strong Association Rules:
       antecedents consequents  support  confidence     lift
           (Bread)    (Butter)      0.5    0.625000 1.041667
          (Butter)     (Bread)      0.5    0.833333 1.041667
            (Eggs)     (Bread)      0.5    0.833333 1.041667
           (Bread)      (Eggs)      0.5    0.625000 1.041667
    (Eggs, Butter)     (Bread)      0.3    1.000000 1.250000
    
    Market Basket Analysis Completed!
## Result
Thus, Market Basket Analysis was successfully implemented using the Apriori algorithm in Python. The transaction dataset was converted into a one-hot encoded format, frequent itemsets were identified, and association rules were generated using support, confidence, and lift. The results help identify purchasing patterns and support retail cross-selling and product-placement decisions.
