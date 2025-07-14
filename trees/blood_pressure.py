from typing import Dict, Any

BP_DECISION_TREE: Dict[str, Any] = {
    "question": "What is your diastolic blood pressure?",
    "branches": {
        # Hypertensive crisis if DBP ≥120 mm Hg regardless of SBP
        ('>=', 120): "Hypertensive crisis - Seek emergency care immediately",
        # Otherwise we still need SBP to finish the classification
        ('<', 120): {
            "question": "What is your systolic blood pressure?",
            "branches": {
                # Crisis if SBP ≥180 mm Hg (even though DBP <120)
                ('>=', 180): "Hypertensive crisis - Seek emergency care immediately",

                # Hypertension Stage 2
                ('>=', 140): "Hypertension Stage 2 - Discuss medication and lifestyle changes with a clinician",

                # Hypertension Stage 1
                ('in', range(130, 140)): "Hypertension Stage 1 - Lifestyle changes and possible medication (clinician‑guided)",

                # Elevated BP (SBP 120‑129 *and* DBP < 80, which we already know here)
                ('in', range(120, 130)): "Elevated blood pressure - Adopt heart‑healthy lifestyle",

                # Normal BP (SBP < 120 and DBP < 80)
                ('<', 120): "Normal blood pressure - Maintain current healthy habits"
            }
        }
    }
}

