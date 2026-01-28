import pandas as pd
import numpy as np
from typing import Tuple, Union


def balance_classes(df: pd.DataFrame,
                    label_column: str,
                    method: str = 'value',
                    multiplier: float = 2.0) -> pd.DataFrame:

    class_counts = df[label_column].value_counts().sort_values()

    print("Rozkład klas przed balansowaniem:")
    print(class_counts)
    print(f"\nNajmniej licznych: {class_counts.min()}")
    print(f"Najbardziej licznych: {class_counts.max()}")

    min_count = class_counts.min()

    if method == 'all':
        max_samples_per_class = min_count
        print(f"\nMetoda 'all': Obcinanie wszystkich klas do {max_samples_per_class} próbek")
    elif method == 'value':
        max_samples_per_class = int(min_count * multiplier)
        print(f"\nMetoda 'value': Obcinanie klas do {max_samples_per_class} próbek ({multiplier}*{min_count})")
    else:
        raise ValueError(f"Nieznana metoda: {method}. Użyj 'value' lub 'all'")

    balanced_dfs = []

    for class_label in class_counts.index:
        class_df = df[df[label_column] == class_label]
        current_count = len(class_df)

        if current_count > max_samples_per_class:
            class_df_sampled = class_df.sample(n=max_samples_per_class, random_state=42)
            print(
                f"Klasa {class_label}: {current_count} -> {max_samples_per_class} (obcięto {current_count - max_samples_per_class})")
            balanced_dfs.append(class_df_sampled)
        else:
            print(f"Klasa {class_label}: {current_count} (bez zmian)")
            balanced_dfs.append(class_df)

    balanced_df = pd.concat(balanced_dfs, ignore_index=True)

    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    print("\nRozkład klas po balansowaniu:")
    print(balanced_df[label_column].value_counts().sort_values())
    print(f"\nPrzed: {len(df)} próbek")
    print(f"Po: {len(balanced_df)} próbek")
    print(f"Usunięto: {len(df) - len(balanced_df)} próbek ({100 * (len(df) - len(balanced_df)) / len(df):.2f}%)")

    return balanced_df